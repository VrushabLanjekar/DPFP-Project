"""Transform a daily API batch to a processed CSV file."""
from pathlib import Path
import pandas as pd


def transform_day_data(df: pd.DataFrame, run_date: str, output_dir: str) -> str:
    normalized_date = pd.Timestamp(run_date).date().isoformat()
    processed = df.copy()
    processed["run_date"] = normalized_date
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"processed_{normalized_date}.csv"
    processed.to_csv(path, index=False, encoding="utf-8")
    return str(path)
