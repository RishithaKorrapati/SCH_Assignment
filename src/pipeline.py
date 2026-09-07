"""CLI orchestrator: ingest -> gates -> store -> train-if-ready."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clean import clean
from src.ingest import _as_day, _zone, load_bronze, materialize_gold, store_silver
from src.train import train
from src.utils.config import load_config
from src.validate import GateResult, validate, validate_cleaned


def _gate_label(result: GateResult | None, ran: bool) -> str:
    if not ran:
        return "SKIP"
    if result.passed:
        return "PASS" + (" (soft)" if result.soft_failures else "")
    return "FAIL"


def _train_label(result) -> str:
    if result.trained:
        auc = result.metrics["candidates"][result.selected_model]["roc_auc"]
        return f"TRAINED {result.selected_model} (roc_auc={auc:.4f})"
    return f"SKIPPED ({result.reason})"


def _write_report(day, config, payload):
    reports_dir = _zone(config, "reports_dir")
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{day}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _quarantine(day, config):
    """Copy a rejected drop into Bronze/quarantine. Original stays as received."""
    day_str = _as_day(day)
    src = _zone(config, "bronze_dir") / f"{day_str}.csv"
    dest_dir = _zone(config, "quarantine_dir")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{day_str}.csv"
    shutil.copy2(src, dest)
    print(f"QUARANTINED {src.name} -> {dest}")
    return dest


def bronze_days(config):
    bronze_dir = _zone(config, "bronze_dir")
    return sorted(p.stem for p in bronze_dir.glob("*.csv"))


def run_day(day, config, previous_row_counts):
    """Run the full path for one date. Trains only if both gates passed."""
    day_str = _as_day(day)
    raw = load_bronze(day_str, config)
    input_result = validate(raw, config, previous_row_counts=previous_row_counts)
    next_counts = list(previous_row_counts) + [len(raw)]

    output_result = None
    output_ran = False

    if not input_result.passed:
        _quarantine(day_str, config)
        train_result = train(
            day_str, config, input_passed=False, output_passed=False
        )
    else:
        cleaned = clean(raw)
        output_result = validate_cleaned(cleaned, config, raw_row_count=len(raw))
        output_ran = True
        if output_result.passed:
            store_silver(cleaned, day_str, config)
            materialize_gold(day_str, config)
        train_result = train(
            day_str,
            config,
            input_passed=True,
            output_passed=output_result.passed,
        )

    row = {
        "date": day_str,
        "input": _gate_label(input_result, ran=True),
        "output": _gate_label(output_result, ran=output_ran),
        "training": _train_label(train_result),
    }
    _write_report(
        day_str,
        config,
        {
            "date": day_str,
            "input_gate": asdict(input_result),
            "output_gate": asdict(output_result) if output_result else None,
            "training": {
                "trained": train_result.trained,
                "skipped": train_result.skipped,
                "reason": train_result.reason,
                "selected_model": train_result.selected_model,
            },
        },
    )
    return row, next_counts


def print_summary(rows):
    print()
    print(f"{'date':<12} {'input':<14} {'output':<14} training")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['date']:<12} {row['input']:<14} {row['output']:<14} {row['training']}"
        )


def run_all(config):
    counts = []
    rows = []
    for day in bronze_days(config):
        row, counts = run_day(day, config, counts)
        rows.append(row)
    print_summary(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Churn pipeline: ingest -> validate -> clean -> store -> train"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", metavar="YYYY-MM-DD", help="Run a single day")
    group.add_argument(
        "--all", action="store_true", help="Replay every Bronze day in order"
    )
    args = parser.parse_args(argv)

    config = load_config()
    if args.all:
        return run_all(config)

    days = bronze_days(config)
    day = _as_day(args.date)
    if day not in days:
        parser.error(f"no Bronze file for {day}; have: {', '.join(days)}")
    # Replay prior days' row counts so the soft volume check has history.
    counts = []
    for prior in days:
        if prior >= day:
            break
        counts.append(len(load_bronze(prior, config)))
    row, _ = run_day(day, config, counts)
    print_summary([row])
    return [row]


if __name__ == "__main__":
    main()
