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
import csv
import datetime as dt
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CITY_CONFIGS = [
    ("Pune",  "config/config.yaml"),
    ("Delhi", "config/config_delhi.yaml"),
]


IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MAX_CATCHUP_DAYS = 60  # safety cap so a long outage can't trigger thousands of calls


def last_logged_date(cfg_path: str):
    """Latest run_date already in this city's log CSV, or None."""
    cfg = yaml.safe_load((HERE / cfg_path).read_text(encoding="utf-8"))
    csv_path = HERE / cfg["logging"]["directory"] / "pipeline_runs.csv"
    if not csv_path.is_file():
        return None
    with csv_path.open(encoding="utf-8", newline="") as fh:
        dates = [row["run_date"][:10] for row in csv.DictReader(fh) if row.get("run_date")]
    return max(dates) if dates else None


def dates_to_fetch(cfg_path: str, end_text: str):
    """Every day after the last logged day, up to and including end_text."""
    last = last_logged_date(cfg_path)
    end = dt.date.fromisoformat(end_text)
    if last is None:
        return [end_text]
    start = dt.date.fromisoformat(last) + dt.timedelta(days=1)
    start = max(start, end - dt.timedelta(days=MAX_CATCHUP_DAYS - 1))
    out, d = [], start
    while d <= end:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def fetch_day(date_text: str, catch_up: bool = True):
    """Run the pipeline for date_text (plus any missed earlier days), both cities."""
    from src.pipeline.runner import run_pipeline  # imported here so --help works without deps
    for label, cfg in CITY_CONFIGS:
        if not (HERE / cfg).is_file():
            print(f"  [{label}] SKIP - config not found: {cfg}")
            continue
        days = dates_to_fetch(cfg, date_text) if catch_up else [date_text]
        if not days:
            print(f"  [{label}] up to date (last logged day is {last_logged_date(cfg)})")
            continue
        if len(days) > 1:
            print(f"  [{label}] catching up {len(days)} days: {days[0]} -> {days[-1]}")
        for day in days:
            try:
                r = run_pipeline(day, config_path=str(HERE / cfg))
                status = r.get("overall_status", "?")
                print(f"  [{label}] {day}: {status}"
                      + (" (skipped - already done)" if r.get("skipped") else ""))
            except Exception as exc:
                print(f"  [{label}] {day}: ERROR {type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description="Daily DPFP fetch + weekly retrain.")
    ap.add_argument("--date", default=None,
                    help="Day to fetch (YYYY-MM-DD). Default: yesterday.")
    ap.add_argument("--no-retrain", action="store_true",
                    help="Do not retrain even if today is Sunday.")
    ap.add_argument("--force-retrain", action="store_true",
                    help="Retrain regardless of weekday.")
    args = ap.parse_args()

    # "Today" is India time, so the schedule behaves the same on any server (GitHub runs in UTC).
    today = dt.datetime.now(IST).date()
    date_text = args.date or (today - dt.timedelta(days=1)).isoformat()

    print(f"=== DPFP daily job - {today.isoformat()} (IST) ===")
    print(f"Fetching data up to: {date_text}")
    fetch_day(date_text, catch_up=args.date is None)

    # Retrain weekly (Sundays), unless overridden.
    is_sunday = today.weekday() == 6
    do_retrain = args.force_retrain or (is_sunday and not args.no_retrain)
    retrain_failed = False
    if do_retrain:
        print("\nWeekly retrain triggered...")
        try:
            import retrain_weekly
            retrain_weekly.retrain_and_log()
        except Exception as exc:
            retrain_failed = True
            print(f"  Retrain ERROR: {type(exc).__name__}: {exc}")
    else:
        print("\nNo retrain today (retrain runs on Sundays).")

    print("=== done ===")
    if retrain_failed:
        sys.exit(1)  # make a failed retrain visible (red run on GitHub)


if __name__ == "__main__":
    main()
