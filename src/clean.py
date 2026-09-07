"""Bronze -> Silver transform. Call only after the input gate has passed."""

import pandas as pd

from src.validate import _blank_mask


def clean(df):
    """Prepare a trusted raw batch for storage and training.

    - Strip whitespace on text columns.
    - Coerce TotalCharges to numeric; blanks become 0 (correct for tenure=0).
      Non-numeric garbage is left as NaN so the output gate can catch it.
    - Encode Churn Yes/No as 1/0. Anything else becomes NaN.
    - Drop exact duplicate rows only.
    """
    out = df.copy()

    text_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        out[col] = out[col].astype("string").str.strip()

    if "TotalCharges" in out.columns:
        stripped = out["TotalCharges"].astype("string").str.strip()
        parsed = pd.to_numeric(stripped, errors="coerce")
        is_blank = _blank_mask(out["TotalCharges"])
        # Blanks -> 0; already-numeric values stay; garbage stays NaN.
        out["TotalCharges"] = parsed.where(~is_blank, 0.0)

    if "Churn" in out.columns:
        out["Churn"] = out["Churn"].map({"Yes": 1, "No": 0})

    return out.drop_duplicates().reset_index(drop=True)
