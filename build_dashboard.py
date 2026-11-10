"""
build_dashboard.py
---------------------------------
Builds the Pipeline Radar dashboard from the latest data and model.

  1. Scores every watched pipeline with outputs/dpfp_model.joblib
     (risk that the NEXT day's run is a data-quality failure).
  2. Appends today's forecast to outputs/forecast_log.csv, so every forecast
     is kept exactly as it was made (honest, out-of-sample history).
  3. Writes dashboard/data.json and a self-contained dashboard/index.html
     (data inlined, so the page opens by double-click or on any static host).

Run after run_daily.py (the GitHub Action does this every morning):
    python build_dashboard.py

To watch another pipeline, add one entry to PIPELINES below. Its run log must
have the same columns as logs/*/pipeline_runs.csv (see REQUIRED_COLUMNS).
"""
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from retrain_weekly import _parse_run_dates

HERE = Path(__file__).resolve().parent
OUT = HERE / "dashboard"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))

PIPELINES = [
    {"id": "delhi", "name": "air_quality_delhi", "source": "Open-Meteo air-quality API, Delhi",
     "table": "logs/delhi_runs", "owner": "DPFP project", "config": "config/config_delhi.yaml"},
    {"id": "pune", "name": "air_quality_pune", "source": "Open-Meteo air-quality API, Pune",
     "table": "logs/api_runs", "owner": "DPFP project", "config": "config/config.yaml"},
]

REQUIRED_COLUMNS = [
    "run_date", "start_time", "overall_status", "failure_type", "warning_status", "warning_reason",
    "rows_processed", "drift_psi_mean", "drift_psi_max", "max_hourly_jump", "max_stagnation_run",
    "invalid_value_count", "freshness_lag_hours", "rows_vs_expected_ratio", "hazard_exceed_count",
    "duration_seconds", "duration_vs_baseline", "api_latency_mean", "api_latency_vs_baseline",
    "api_latency_roll7_std",
]

HISTORY_DAYS = 120          # days of history shown on the dashboard
MIN_FAILURES_CONFIDENT = 30  # fewer past failures than this -> "low confidence" label

# Plain-language reason for each model feature, when it pushes the risk UP.
REASON = {
    "max_hourly_jump": "Sudden jumps in readings",
    "max_hourly_jump_roll7": "Many sudden jumps this week",
    "month_cos": "Time of year: failures are more common now",
    "month_sin": "Time of year: failures are more common now",
    "fails_last7": "Failed recently; failures come in groups",
    "days_since_fail": "A failure happened recently",
    "api_latency_vs_baseline": "Source speed changed",
    "api_latency_mean": "Source is slow",
    "api_latency_roll7_std": "Source speed is unstable",
    "duration_vs_baseline": "Run took longer than usual",
    "duration_seconds": "Slow run",
    "drift_psi_mean": "Data looks different from normal",
    "drift_psi_max": "Data looks different from normal",
    "drift_psi_mean_roll7": "Data looks unusual this week",
    "max_stagnation_run": "A sensor got stuck",
    "invalid_value_count": "Invalid values",
    "hazard_exceed_count": "Extreme readings",
    "dq_fail_today": "Today's run failed",
    "sla_fail_today": "Source was too slow today",
    "runtime_warn_last7": "Slow runs this week",
    "freshness_lag_hours": "Data arrived late",
    "rows_vs_expected_ratio": "Fewer rows than expected",
}


def features(df):
    """Same features as retrain_weekly.build(), but keeps the latest day (the one we forecast from)."""
    d = df.copy()
    d["run_date"] = _parse_run_dates(d["run_date"])
    d = (d.sort_values(["run_date", "start_time"])
          .drop_duplicates("run_date", keep="last").reset_index(drop=True))
    ft, wr = d["failure_type"].fillna(""), d["warning_reason"].fillna("")
    d["is_fail"] = (d["overall_status"].astype(str).str.upper() == "FAIL").astype(int)
    d["dq_fail"] = (ft == "DATA_QUALITY_FAILURE").astype(int)
    d["sla_fail"] = (ft == "API_SLA_BREACH").astype(int)
    d["runtime_warn"] = wr.str.contains("HIGH_RUNTIME").astype(int)

    F = pd.DataFrame(index=d.index)
    for c in ["drift_psi_mean", "drift_psi_max", "max_hourly_jump", "max_stagnation_run",
              "invalid_value_count", "freshness_lag_hours", "rows_vs_expected_ratio",
              "hazard_exceed_count", "duration_seconds", "duration_vs_baseline",
              "api_latency_mean", "api_latency_vs_baseline", "api_latency_roll7_std"]:
        F[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["drift_psi_mean", "max_hourly_jump"]:
        F[f"{c}_roll7"] = pd.to_numeric(d[c], errors="coerce").rolling(7, min_periods=1).mean()
    F["runtime_warn_last7"] = d["runtime_warn"].rolling(7, min_periods=1).sum()
    F["dq_fail_today"] = d["dq_fail"]
    F["sla_fail_today"] = d["sla_fail"]
    F["fails_last7"] = d["is_fail"].rolling(7, min_periods=1).sum()
    last = d["run_date"].where(d["is_fail"] == 1).ffill()
    F["days_since_fail"] = (d["run_date"] - last).dt.days.fillna(365).clip(upper=365)
    m = d["run_date"].dt.month
    F["month_sin"] = np.sin(2 * np.pi * m / 12)
    F["month_cos"] = np.cos(2 * np.pi * m / 12)
    return d, F


def num(v, nd=3):
    return None if v is None or pd.isna(v) else round(float(v), nd)


def build_pipeline(p, model, cols, med):
    cfg = yaml.safe_load((HERE / p["config"]).read_text(encoding="utf-8"))
    log = HERE / cfg["logging"]["directory"] / "pipeline_runs.csv"
    raw = pd.read_csv(log)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{p['name']}: run log is missing columns {missing}")
    expected = cfg.get("pipeline", {}).get("expected_records_per_day", 24)

    d, F = features(raw)
    X = F.reindex(columns=cols).fillna(pd.Series(med))
    risk = model.predict_proba(X)[:, 1]

    status = np.where(d["overall_status"].astype(str).str.upper() == "FAIL", "F",
                      np.where(d["warning_status"].fillna("NONE").astype(str).str.upper() == "WARNING", "W", "S"))

    # Reasons: for each recent day, which features push the risk up most
    # (replace one feature at a time with its typical value and see how much the risk drops).
    n = min(HISTORY_DAYS, len(d))
    Xt = X.iloc[-n:].reset_index(drop=True)
    batch = []
    for c in cols:
        x2 = Xt.copy()
        x2[c] = med[c]
        batch.append(x2)
    alt = model.predict_proba(pd.concat(batch, ignore_index=True))[:, 1].reshape(len(cols), n)
    effect = risk[-n:][None, :] - alt

    rows = pd.to_numeric(d["rows_processed"], errors="coerce")
    missing_pct = (100 - rows / expected * 100).clip(lower=0)
    sig_cols = [missing_pct,
                pd.to_numeric(d["drift_psi_mean"], errors="coerce"),
                pd.to_numeric(d["api_latency_mean"], errors="coerce"),
                pd.to_numeric(d["max_hourly_jump"], errors="coerce")]
    norm = [[0.0, 4.0]] + [[num(np.nanpercentile(s, 50)), num(np.nanpercentile(s, 90))] for s in sig_cols[1:]]

    days, off = [], len(d) - n
    for i in range(n):
        j = off + i
        ups = [k for k in np.argsort(-effect[:, i]) if effect[k, i] > 0.002][:3]
        why = []
        for k in ups:
            w = REASON.get(cols[k], cols[k].replace("_", " "))
            if w not in why:
                why.append(w)
        ft = d["failure_type"].iloc[j]
        days.append({
            "d": d["run_date"].iloc[j].date().isoformat(),
            "s": str(status[j]),
            "t": ("slow" if ft == "API_SLA_BREACH" else "bad") if status[j] == "F" else "",
            "p": num(risk[j]),
            "rows": num(rows.iloc[j], 0),
            "sig": [num(s.iloc[j]) for s in sig_cols],
            "why": why,
        })

    dq_failures = int(d["dq_fail"].sum())
    return {
        **{k: p[k] for k in ("id", "name", "source", "table", "owner")},
        "real": True,
        "lowconf": dq_failures < MIN_FAILURES_CONFIDENT,
        "dq_failures": dq_failures,
        "n_total": int(len(d)),
        "fail_total": int((status == "F").sum()),
        "fail_before": int((status[:off] == "F").sum()),
        "norm": norm,
        "days": days,
    }


def log_forecasts(pipes, threshold):
    """Keep every forecast exactly as made. One row per pipeline per data day."""
    path = HERE / "outputs" / "forecast_log.csv"
    log = pd.read_csv(path) if path.is_file() else pd.DataFrame(
        columns=["made_from", "for_date", "pipeline", "risk", "alert", "logged_at"])
    now = dt.datetime.now(IST).isoformat(timespec="seconds")
    new = []
    for p in pipes:
        last = p["days"][-1]
        for_date = (dt.date.fromisoformat(last["d"]) + dt.timedelta(days=1)).isoformat()
        if ((log["pipeline"] == p["id"]) & (log["made_from"] == last["d"])).any():
            continue  # already logged this forecast; never overwrite it
        new.append({"made_from": last["d"], "for_date": for_date, "pipeline": p["id"],
                    "risk": last["p"], "alert": int(last["p"] >= threshold), "logged_at": now})
    if new:
        log = pd.concat([log, pd.DataFrame(new)], ignore_index=True)
        log.to_csv(path, index=False)
    # Use the logged (as-made) forecast wherever one exists.
    for p in pipes:
        made = log[log["pipeline"] == p["id"]].set_index("made_from")["risk"].to_dict()
        for day in p["days"]:
            if day["d"] in made:
                day["p"] = round(float(made[day["d"]]), 3)
                day["logged"] = True
    return len(new)


def main():
    art = joblib.load(HERE / "outputs" / "dpfp_model.joblib")
    model, cols, med = art["model"], art["features"], art["medians"]
    meta = json.loads((HERE / "outputs" / "dpfp_model_meta.json").read_text(encoding="utf-8"))
    threshold = float(meta["threshold"])

    pipes = [build_pipeline(p, model, cols, med) for p in PIPELINES]
    added = log_forecasts(pipes, threshold)

    # Align every pipeline to the same dates (the dashboard replays them together).
    common = sorted(set.intersection(*[{x["d"] for x in p["days"]} for p in pipes]))
    for p in pipes:
        p["days"] = [x for x in p["days"] if x["d"] in set(common)]

    perf = pd.read_csv(HERE / "outputs" / "performance_history.csv").tail(12)
    data = {
        "threshold": threshold,
        "built_at": dt.datetime.now(IST).isoformat(timespec="minutes"),
        "data_through": common[-1],
        "model_trained": art.get("trained_on", meta.get("train_period", ["?"])[-1]),
        "required_columns": REQUIRED_COLUMNS,
        "performance": json.loads(perf.to_json(orient="records")),
        "pipelines": pipes,
    }
    OUT.mkdir(exist_ok=True)
    blob = json.dumps(data, separators=(",", ":"))
    (OUT / "data.json").write_text(blob, encoding="utf-8")
    page = (OUT / "template.html").read_text(encoding="utf-8").replace("__DATA__", blob.replace("</", "<\\/"))
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print(f"Dashboard built: {len(pipes)} pipelines, data through {common[-1]}, "
          f"{added} new forecast(s) logged -> dashboard/index.html")


if __name__ == "__main__":
    main()
