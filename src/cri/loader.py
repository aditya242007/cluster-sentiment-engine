from pathlib import Path
import re
from typing import Set
import pandas as pd

REQUIRED_COLUMNS: Set[str] = {
    "review_id",
    "property_id",
    "cluster_id",
    "review_text",
    "rating",
    "review_date",
}

PII_COLUMN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (r"^reviewer_name$", r"^email$", r"^phone$", r"^user_id$")
]


def load_reviews(filepath: str) -> pd.DataFrame:
    """Load reviews from a CSV or JSON file, dropping PII and parsing dates.

    Args:
        filepath: Path to the .csv or .json file containing reviews.

    Returns:
        pd.DataFrame: Cleaned reviews dataframe with parsed date columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file extension is unsupported or required columns are missing.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Review file not found: {filepath}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Expected .csv or .json.")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(sorted(missing))}. "
            f"Required columns are: {', '.join(sorted(REQUIRED_COLUMNS))}"
        )

    pii_columns = [
        col for col in df.columns
        if any(pattern.match(col.strip()) for pattern in PII_COLUMN_PATTERNS)
    ]
    if pii_columns:
        df = df.drop(columns=pii_columns)

    df["review_date"] = pd.to_datetime(df["review_date"])
    if "stay_date" in df.columns:
        df["stay_date"] = pd.to_datetime(df["stay_date"])

    return df
