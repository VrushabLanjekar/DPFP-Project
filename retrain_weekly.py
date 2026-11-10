"""
retrain_weekly.py
---------------------------------
Retrain the next-day failure-prediction model on the FULL current dataset
(Delhi = main city), save it, and append one row to the performance history
so you can show the model improving as data grows.

Aligned with the Phase-2 notebook (DPFP_Phase2_Modelling.ipynb):
  - target   = will the NEXT day's run be a DATA_QUALITY failure?
  - features = compact 22-signal set (DATA + INFRA + HISTORY + CALENDAR)
  - model    = RandomForest (imbalance='none', max_features='sqrt')
  - split    = chronological; last 25% held-out test window (>= 30 failures)
"""
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

DELHI_CANDIDATES = [
    HERE / "logs" / "delhi_runs" / "pipeline_runs.csv",
    HERE / "pipeline_runs_delhi.csv",
    HERE / "pipeline_runs_Delhi.csv",
    HERE / "pipeline_runs_d.csv",
]

# Exact compact feature order from Phase-2 Notebook (Section 2)
COMPACT_FEATS = [
    "max_hourly_jump", "drift_psi_mean", "drift_psi_max",
    "max_stagnation_run", "invalid_value_count", "max_hourly_jump_roll7",
    "drift_psi_mean_roll7", "freshness_lag_hours", "rows_vs_expected_ratio",
    "hazard_exceed_count",
    "duration_seconds", "duration_vs_baseline", "api_latency_mean",
    "api_latency_vs_baseline", "api_latency_roll7_std", "runtime_warn_last7",
    "dq_fail_today", "sla_fail_today", "fails_last7", "days_since_fail",
    "month_sin", "month_cos",
]

# True  -> trains on 75% (train+val) before scoring the 25% test tail (PR-AUC ~ 0.3252)
# False -> trains on 50%, Platt-calibrates on 25% val, scores 25% test (PR-AUC ~ 0.2291, matches notebook table)
USE_FULL_PRETEST_TRAIN = True


def _find(cands):
    for c in cands:
        if Path(c).is_file():
            return c
    raise FileNotFoundError(f"Delhi log not found in: {[str(c) for c in cands]}")


def _parse_run_dates(series):
    s = series.astype(str).str.strip()
    is_iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    d_iso = pd.to_datetime(s[is_iso], format="%Y-%m-%d", errors="coerce")
    d_other = pd.to_datetime(s[~is_iso], dayfirst=True, format="mixed", errors="coerce")
    return pd.concat([d_iso, d_other]).sort_index().dt.normalize()


def build(df):
    d = df.copy()
    d["run_date"] = _parse_run_dates(d["run_date"])
    d = (d.sort_values(["run_date", "start_time"])
          .drop_duplicates("run_date", keep="last")
          .reset_index(drop=True))

    ft = d["failure_type"].fillna("")
    wr = d["warning_reason"].fillna("")
    d["is_fail"] = (d["overall_status"].str.upper() == "FAIL").astype(int)
    d["dq_fail"] = (ft == "DATA_QUALITY_FAILURE").astype(int)
    d["sla_fail"] = (ft == "API_SLA_BREACH").astype(int)
    d["runtime_warn"] = wr.str.contains("HIGH_RUNTIME").astype(int)

    F = pd.DataFrame(index=d.index)
    raw_cols = [
        "drift_psi_mean", "drift_psi_max", "max_hourly_jump", "max_stagnation_run",
        "invalid_value_count", "freshness_lag_hours", "rows_vs_expected_ratio",
        "hazard_exceed_count", "duration_seconds", "duration_vs_baseline",
        "api_latency_mean", "api_latency_vs_baseline", "api_latency_roll7_std",
    ]
    for c in raw_cols:
        if c in d:
            F[c] = pd.to_numeric(d[c], errors="coerce")

    for c in ["drift_psi_mean", "max_hourly_jump"]:
        if c in d:
            v = pd.to_numeric(d[c], errors="coerce")
            F[f"{c}_roll7"] = v.rolling(7, min_periods=1).mean()

    F["runtime_warn_last7"] = d["runtime_warn"].rolling(7, min_periods=1).sum()
    F["dq_fail_today"] = d["dq_fail"]
    F["sla_fail_today"] = d["sla_fail"]
    F["fails_last7"] = d["is_fail"].rolling(7, min_periods=1).sum()
    last = d["run_date"].where(d["is_fail"] == 1).ffill()
    F["days_since_fail"] = (d["run_date"] - last).dt.days.fillna(365).clip(upper=365)

    m = d["run_date"].dt.month
    F["month_sin"] = np.sin(2 * np.pi * m / 12)
    F["month_cos"] = np.cos(2 * np.pi * m / 12)

    cols_present = [c for c in COMPACT_FEATS if c in F.columns]
    F = F[cols_present]

    y = d["dq_fail"].shift(-1)
    keep = y.notna()
    return (
        F[keep].reset_index(drop=True),
        y[keep].astype(int).reset_index(drop=True),
        d.loc[keep, "run_date"].reset_index(drop=True),
    )


def retrain_and_log():
    path = _find(DELHI_CANDIDATES)
    df = pd.read_csv(path)
    X, y, dates = build(df)

    n = len(X)
    a, b = int(n * 0.50), int(n * 0.75)

    if USE_FULL_PRETEST_TRAIN:
        Xtr_raw, ytr = X.iloc[:b], y.iloc[:b]
        Xte_raw, yte = X.iloc[b:], y.iloc[b:]
        cols = [c for c in Xtr_raw.columns if Xtr_raw[c].nunique(dropna=True) > 1]
        med = Xtr_raw[cols].median()
        Xtr, Xte = Xtr_raw[cols].fillna(med), Xte_raw[cols].fillna(med)

        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=20, max_depth=6,
            max_features="sqrt", n_jobs=-1, class_weight=None, random_state=42
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
    else:
        Xtr_raw, ytr = X.iloc[:a], y.iloc[:a]
        Xva_raw, yva = X.iloc[a:b], y.iloc[a:b]
        Xte_raw, yte = X.iloc[b:], y.iloc[b:]
        cols = [c for c in Xtr_raw.columns if Xtr_raw[c].nunique(dropna=True) > 1]
        med = Xtr_raw[cols].median()
        Xtr = Xtr_raw[cols].fillna(med)
        Xva = Xva_raw[cols].fillna(med)
        Xte = Xte_raw[cols].fillna(med)

        model = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=20, max_depth=6,
            max_features="sqrt", n_jobs=-1, class_weight=None, random_state=42
        )
        model.fit(Xtr, ytr)
        p_va = np.clip(model.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
        p_te = np.clip(model.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
        lr = LogisticRegression(C=1e3).fit(np.log(p_va / (1 - p_va)).reshape(-1, 1), yva)
        p = lr.predict_proba(np.log(p_te / (1 - p_te)).reshape(-1, 1))[:, 1]

    pr = roc = np.nan
    if yte.sum() >= 3 and yte.nunique() > 1:
        pr = round(float(average_precision_score(yte, p)), 4)
        roc = round(float(roc_auc_score(yte, p)), 4)

    # Retrain on ALL data for the deployed model
    full_med = X[cols].median()
    X_full = X[cols].fillna(full_med)
    model.fit(X_full, y)
    joblib.dump(
        {
            "model": model,
            "features": cols,
            "medians": full_med.to_dict(),
            "trained_on": dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date().isoformat(),
            "n_rows": n,
        },
        OUT / "dpfp_model.joblib",
    )

    row = {
        "retrain_date": dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date().isoformat(),
        "data_through": str(dates.max().date()),
        "n_rows": n,
        "n_failures": int(y.sum()),
        "fail_rate": round(float(y.mean()), 4),
        "test_PR_AUC": pr,
        "test_ROC_AUC": roc,
        "base_rate": round(float(yte.mean()), 4) if len(yte) else np.nan,
    }
    hist_path = OUT / "performance_history.csv"
    hist = pd.read_csv(hist_path) if hist_path.is_file() else pd.DataFrame()
    # One row per retrain day: re-running the same day replaces the row instead of duplicating it.
    if not hist.empty:
        hist = hist[hist["retrain_date"] != row["retrain_date"]]
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(hist_path, index=False)

    print(
        f"  Retrained on {n} rows ({int(y.sum())} failures). "
        f"Test PR-AUC={pr} ROC-AUC={roc} (base_rate={row['base_rate']}). Model + history saved."
    )
    return row


if __name__ == "__main__":
    retrain_and_log()