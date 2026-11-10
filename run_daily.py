"""
run_daily.py
---------------------------------
Daily automated job (for PythonAnywhere / any scheduler).

What it does, once per day:
  1. Fetches YESTERDAY's air-quality data for both cities (Pune + Delhi),
     runs the pipeline, and appends the result to each city's log CSV.
     (Yesterday, because today's full day of data is not complete yet.)
  2. On Sundays ONLY, also retrains the model on the full updated dataset
     and appends one row to outputs/performance_history.csv, so you can plot
     how the model improves as data grows.

This reuses the existing pipeline unchanged. It never re-fetches a day that
already succeeded (the pipeline skips those), so it is safe to run repeatedly.

USAGE (manual test):
    python run_daily.py                 # fetch yesterday, retrain if Sunday
    python run_daily.py --date 2026-11-01   # fetch a specific day
    python run_daily.py --no-retrain    # never retrain this run

On PythonAnywhere: add ONE daily scheduled task that runs:
    python /home/<you>/dpfp/run_daily.py
(See DEPLOY_PYTHONANYWHERE.md.)
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CITY_CONFIGS = [
    ("Pune",  "config/config.yaml"),
    ("Delhi", "config/config_delhi.yaml"),
]


def fetch_day(date_text: str):
    """Run the pipeline for one date, both cities."""
    from src.pipeline.runner import run_pipeline  # imported here so --help works without deps
    for label, cfg in CITY_CONFIGS:
        if not (HERE / cfg).is_file():
            print(f"  [{label}] SKIP — config not found: {cfg}")
            continue
        try:
            r = run_pipeline(date_text, config_path=str(HERE / cfg))
            status = r.get("overall_status", "?")
            print(f"  [{label}] {date_text}: {status}"
                  + (" (skipped — already done)" if r.get("skipped") else ""))
        except Exception as exc:
            print(f"  [{label}] {date_text}: ERROR {type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description="Daily DPFP fetch + weekly retrain.")
    ap.add_argument("--date", default=None,
                    help="Day to fetch (YYYY-MM-DD). Default: yesterday.")
    ap.add_argument("--no-retrain", action="store_true",
                    help="Do not retrain even if today is Sunday.")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain regardless of weekday.")
    args = ap.parse_args()

    date_text = args.date or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    today = dt.date.today()

    print(f"=== DPFP daily job — {today.isoformat()} ===")
    print(f"Fetching data for: {date_text}")
    fetch_day(date_text)

    # Retrain weekly (Sundays), unless overridden.
    is_sunday = today.weekday() == 6
    do_retrain = args.force_retrain or (is_sunday and not args.no_retrain)
    if do_retrain:
        print("\nWeekly retrain triggered…")
        try:
            import retrain_weekly
            retrain_weekly.retrain_and_log()
        except Exception as exc:
            print(f"  Retrain ERROR: {type(exc).__name__}: {exc}")
    else:
        print("\nNo retrain today (retrain runs on Sundays).")

    print("=== done ===")


if __name__ == "__main__":
    main()
