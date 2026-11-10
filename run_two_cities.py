"""
run_two_cities.py
---------------------------------
Generate the TWO datasets for cross-dataset validation by running the pipeline
for Pune and Delhi over the same date range.

Produces:
    logs/api_runs/pipeline_runs.csv     (Pune   - config/config.yaml)
    logs/delhi_runs/pipeline_runs.csv   (Delhi  - config/config_delhi.yaml)

Each city uses its own database, log directory, processed-data directory, and
feature-history file, so their rolling baselines never mix.

USAGE
    # full range (recommended): uses each config's historical_start/end
    python run_two_cities.py

    # or a custom range:
    python run_two_cities.py --start 2022-08-05 --end 2026-10-28

Notes:
- Runs are idempotent: a date already completed successfully is skipped, so you
  can re-run safely to resume or to extend the range later.
- Each run is logged on its own run_date timeline (1 run = 1 day), so lead time
  reads as "1 day ahead".
"""
import argparse
import datetime as dt
from pathlib import Path

import yaml

from src.pipeline.runner import run_pipeline

CITY_CONFIGS = [
    ("Pune",  "config/config.yaml"),
    ("Delhi", "config/config_delhi.yaml"),
]


def daterange(start: str, end: str):
    d0 = dt.date.fromisoformat(start)
    d1 = dt.date.fromisoformat(end)
    step = dt.timedelta(days=1)
    while d0 <= d1:
        yield d0.isoformat()
        d0 += step


def run_city(label: str, config_path: str, start: str, end: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    s = start or cfg["periods"]["historical_start"]
    e = end or cfg["periods"]["historical_end"]
    # never run past today
    today = dt.date.today().isoformat()
    if e > today:
        e = today
    print(f"\n{'='*64}\n{label}  ({config_path})\n  range: {s} -> {e}\n{'='*64}")

    ok = fail = skip = err = 0
    for i, date_text in enumerate(daterange(s, e), 1):
        try:
            result = run_pipeline(date_text, config_path=config_path)
            status = result.get("overall_status", "?")
            if result.get("skipped"):
                skip += 1
            elif status == "SUCCESS":
                ok += 1
            elif status == "FAIL":
                fail += 1
        except Exception as exc:
            err += 1
            print(f"  [{date_text}] ERROR: {exc}")
        if i % 100 == 0:
            print(f"  ...{i} days processed (ok={ok} fail={fail} skip={skip} err={err})")
    print(f"  DONE {label}: success={ok} fail={fail} skipped={skip} errors={err}")


def main():
    ap = argparse.ArgumentParser(description="Run the pipeline for two cities (cross-dataset).")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: config historical_start)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: config historical_end, capped at today)")
    args = ap.parse_args()

    for label, cfg in CITY_CONFIGS:
        if not Path(cfg).is_file():
            print(f"SKIP missing config: {cfg}")
            continue
        run_city(label, cfg, args.start, args.end)

    print("\nBoth cities done. Datasets:")
    print("  Pune : logs/api_runs/pipeline_runs.csv")
    print("  Delhi: logs/delhi_runs/pipeline_runs.csv")
    print("\nNext: open the modelling notebook and point it at both CSVs.")


if __name__ == "__main__":
    main()
