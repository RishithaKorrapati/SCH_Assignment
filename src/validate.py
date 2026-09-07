"""Input gate: is this raw daily drop OK?

Hard failures mean the file is broken. They block the rest of that day
and the orchestrator copies the file to Bronze/quarantine. Soft failures
mean the file is unusual but might be real — logged only, never a block.
"""

from dataclasses import dataclass, field

import pandas as pd

@dataclass
class GateResult:
    passed: bool
    gate: str
    hard_failures: list = field(default_factory=list)
    soft_failures: list = field(default_factory=list)
    row_count: int = 0


def _blank_mask(series):
    """True for NaN or whitespace-only values (CSV empty cells)."""
    as_str = series.astype("string").str.strip()
    return series.isna() | as_str.eq("") | as_str.str.lower().isin(["nan", "none", "<na>"])


def validate(df, config, previous_row_counts=None):
    """Run the input gate on a raw Bronze dataframe.

    previous_row_counts: chronological list of prior days' row counts
    (Bronze arrivals). Used only for the soft volume check. Pass None
    or [] on the first day — no history, no soft flag.
    """
    gate_cfg = config["input_gate"]
    hard = []
    soft = []
    n_rows = len(df)

    missing = [c for c in gate_cfg["required_columns"] if c not in df.columns]
    if missing:
        hard.append(f"missing_columns: {missing}")

    present = [c for c in gate_cfg["required_non_null_columns"] if c in df.columns]
    null_hits = []
    for col in present:
        mask = _blank_mask(df[col])
        # Blank TotalCharges is legitimate (tenure=0, never billed). Garbage is not.
        if col == "TotalCharges":
            continue
        n_null = int(mask.sum())
        if n_null:
            null_hits.append(f"{col} ({n_null} rows)")
    if null_hits:
        hard.append(f"nulls_in_required: {', '.join(null_hits)}")

    if "customerID" in df.columns:
        n_dup = int(df["customerID"].duplicated().sum())
        if n_dup:
            hard.append(f"duplicate_customer_id: {n_dup} duplicate row(s)")

    if "TotalCharges" in df.columns:
        stripped = df["TotalCharges"].astype("string").str.strip()
        is_blank = _blank_mask(df["TotalCharges"])
        parsed = pd.to_numeric(stripped.mask(is_blank), errors="coerce")
        n_garbage = int((~is_blank & parsed.isna()).sum())
        if n_garbage:
            hard.append(
                f"total_charges_not_numeric: {n_garbage} non-blank value(s) "
                "could not be parsed as numeric"
            )

    if "tenure" in df.columns:
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        out = (~tenure.isna()) & (
            (tenure < gate_cfg["tenure_min"]) | (tenure > gate_cfg["tenure_max"])
        )
        n_out = int(out.sum())
        if n_out:
            hard.append(
                f"tenure_out_of_range: {n_out} row(s) outside "
                f"[{gate_cfg['tenure_min']}, {gate_cfg['tenure_max']}]"
            )

    if "MonthlyCharges" in df.columns:
        charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        out = (~charges.isna()) & (
            (charges < gate_cfg["monthly_charges_min"])
            | (charges > gate_cfg["monthly_charges_max"])
        )
        n_out = int(out.sum())
        if n_out:
            hard.append(
                f"monthly_charges_out_of_range: {n_out} row(s) outside "
                f"[{gate_cfg['monthly_charges_min']}, {gate_cfg['monthly_charges_max']}]"
            )

    for col, allowed in gate_cfg["categories"].items():
        if col not in df.columns:
            continue
        allowed_set = set(allowed)
        mask = ~_blank_mask(df[col])
        invalid = mask & ~df[col].astype("string").str.strip().isin(allowed_set)
        n_invalid = int(invalid.sum())
        if n_invalid:
            hard.append(
                f"invalid_category_{col}: {n_invalid} value(s) not in {sorted(allowed_set)}"
            )

    if n_rows < gate_cfg["min_row_count"]:
        hard.append(
            f"row_count_below_floor: {n_rows} < {gate_cfg['min_row_count']}"
        )

    # Soft only: never appended to hard, never flips passed to False.
    if previous_row_counts:
        window = list(previous_row_counts)[-gate_cfg["volume_trailing_days"] :]
        avg = sum(window) / len(window)
        if avg > 0:
            ratio = n_rows / avg
            if ratio < gate_cfg["volume_min_ratio"] or ratio > gate_cfg["volume_max_ratio"]:
                soft.append(
                    f"volume_deviation: today={n_rows} trailing_avg={avg:.1f} "
                    f"ratio={ratio:.2f} (allowed "
                    f"[{gate_cfg['volume_min_ratio']}, {gate_cfg['volume_max_ratio']}])"
                )

    return GateResult(
        passed=len(hard) == 0,
        gate="input",
        hard_failures=hard,
        soft_failures=soft,
        row_count=n_rows,
    )


def validate_cleaned(cleaned_df, config, raw_row_count):
    """Output gate: did our own clean() behave correctly?

    This is not a second opinion on the source file. The input gate already
    answered that. A failure here is a bug in this codebase — same block
    (no Silver, no training), different owner of the fix.
    """
    gate_cfg = config["output_gate"]
    hard = []
    n_rows = len(cleaned_df)

    if raw_row_count <= 0:
        hard.append("row_retention: raw_row_count must be > 0")
    else:
        retention = n_rows / raw_row_count
        if retention < gate_cfg["min_row_retention"]:
            hard.append(
                f"row_retention: {n_rows}/{raw_row_count} = {retention:.4f} "
                f"< {gate_cfg['min_row_retention']}"
            )

    if "TotalCharges" in cleaned_df.columns:
        n_null = int(cleaned_df["TotalCharges"].isna().sum())
        if n_null:
            hard.append(f"nulls_in_total_charges: {n_null} row(s)")
    else:
        hard.append("nulls_in_total_charges: column missing")

    if "Churn" in cleaned_df.columns:
        n_null = int(cleaned_df["Churn"].isna().sum())
        if n_null:
            hard.append(f"nulls_in_churn: {n_null} row(s)")
        allowed = set(gate_cfg["cleaned_churn_values"])
        present = set(cleaned_df["Churn"].dropna().unique())
        if not present.issubset(allowed):
            hard.append(
                f"churn_not_binary: values {sorted(present, key=str)} "
                f"not a subset of {sorted(allowed)}"
            )
    else:
        hard.append("churn_not_binary: column missing")

    for col in gate_cfg["numeric_columns"]:
        if col not in cleaned_df.columns:
            hard.append(f"non_numeric_dtype_{col}: column missing")
            continue
        if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
            hard.append(f"non_numeric_dtype_{col}: dtype={cleaned_df[col].dtype}")

    if "customerID" in cleaned_df.columns:
        n_dup = int(cleaned_df["customerID"].duplicated().sum())
        if n_dup:
            hard.append(f"duplicate_customer_id: {n_dup} duplicate row(s)")

    return GateResult(
        passed=len(hard) == 0,
        gate="output",
        hard_failures=hard,
        soft_failures=[],
        row_count=n_rows,
    )
