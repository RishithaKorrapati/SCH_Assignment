from tests.conftest import failure_codes
from src.validate import validate


def test_valid_batch_passes(valid_batch, config):
    result = validate(valid_batch, config)
    assert result.passed is True
    assert result.gate == "input"
    assert result.hard_failures == []
    assert result.soft_failures == []


def test_blank_total_charges_allowed(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("TotalCharges")] = "   "
    df.iloc[0, df.columns.get_loc("tenure")] = 0
    result = validate(df, config)
    assert result.passed is True
    assert result.hard_failures == []


def test_missing_columns(valid_batch, config):
    df = valid_batch.drop(columns=["Churn"])
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["missing_columns"]


def test_null_required_field(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("gender")] = None
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["nulls_in_required"]


def test_duplicate_customer_id(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[1, df.columns.get_loc("customerID")] = df.iloc[0]["customerID"]
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["duplicate_customer_id"]


def test_total_charges_garbage_text(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("TotalCharges")] = "not-a-number"
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["total_charges_not_numeric"]


def test_tenure_out_of_range(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("tenure")] = 999
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["tenure_out_of_range"]


def test_monthly_charges_out_of_range(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("MonthlyCharges")] = (
        config["input_gate"]["monthly_charges_max"] + 1
    )
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["monthly_charges_out_of_range"]


def test_invalid_category_gender(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("gender")] = "X"
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["invalid_category_gender"]


def test_invalid_category_contract(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("Contract")] = "Three year"
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["invalid_category_Contract"]


def test_invalid_category_internet_service(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("InternetService")] = "Satellite"
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["invalid_category_InternetService"]


def test_invalid_category_payment_method(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("PaymentMethod")] = "Cash"
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["invalid_category_PaymentMethod"]


def test_invalid_category_churn(valid_batch, config):
    df = valid_batch.copy()
    df.iloc[0, df.columns.get_loc("Churn")] = "maybe"
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["invalid_category_Churn"]


def test_row_count_below_floor(valid_batch, config):
    df = valid_batch.head(config["input_gate"]["min_row_count"] - 1).copy()
    result = validate(df, config)
    assert result.passed is False
    assert failure_codes(result) == ["row_count_below_floor"]


def test_volume_deviation_is_soft_not_hard(valid_batch, config):
    """Odd volume logs a soft failure and still passes — it never blocks."""
    previous = [700, 700, 700]
    result = validate(valid_batch, config, previous_row_counts=previous)
    assert result.passed is True
    assert result.hard_failures == []
    assert len(result.soft_failures) == 1
    assert result.soft_failures[0].startswith("volume_deviation")
