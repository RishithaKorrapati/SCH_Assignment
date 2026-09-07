import pandas as pd

from src.clean import clean
from src.validate import validate, validate_cleaned


def test_clean_blank_total_charges_becomes_zero(valid_batch):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("TotalCharges")] = " "
    df.iloc[0, df.columns.get_loc("tenure")] = 0
    cleaned = clean(df)
    assert cleaned.loc[0, "TotalCharges"] == 0
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_encodes_churn_as_01(valid_batch):
    cleaned = clean(valid_batch)
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert cleaned["Churn"].isna().sum() == 0


def test_output_gate_passes_real_clean(valid_batch, config):
    result = validate_cleaned(
        clean(valid_batch), config, raw_row_count=len(valid_batch)
    )
    assert result.passed is True
    assert result.gate == "output"


def test_output_gate_catches_cleaning_bug_input_gate_cannot_see(valid_batch, config):
    """Input gate never inspects cleaned data. A clean() bug is the output gate's job.

    Simulated bug: forget to coerce TotalCharges, leave a NaN. The raw file
    still passes the input gate. The output gate must fail and
    tag itself as gate='output'.
    """
    raw = valid_batch.copy()
    input_result = validate(raw, config)
    assert input_result.passed is True
    assert input_result.gate == "input"

    buggy = clean(raw)
    buggy.loc[buggy.index[0], "TotalCharges"] = pd.NA

    output_result = validate_cleaned(buggy, config, raw_row_count=len(raw))
    assert output_result.passed is False
    assert output_result.gate == "output"
    assert any(f.startswith("nulls_in_total_charges") for f in output_result.hard_failures)
    # The source file did not change — input gate still would not block.
    assert validate(raw, config).passed is True
