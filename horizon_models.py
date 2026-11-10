"""
horizon_models.py
---------------------------------
Models for the dashboard's 3-day outlook.

The main model (retrain_weekly.py) answers: will the NEXT run (day t+1) fail?
This file trains the same model for a specific later day:
    h = 2  -> will the run for day t+2 fail?
    h = 3  -> will the run for day t+3 fail?

Same features, same Random Forest settings, trained on Delhi (the main city).
Each model is scored honestly first (chronological split: train on the first
75%, test on the last 25%), then refit on all data and saved with its scores:
    outputs/dpfp_model_h2.joblib, outputs/dpfp_model_h3.joblib
    outputs/horizon_scores.csv   (h = 1, 2, 3: test PR-AUC, base rate, lift)

build_dashboard.py retrains these automatically whenever the main model has been
retrained since (the main model is retrained every Sunday).
"""
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from retrain_weekly import DELHI_CANDIDATES, _find

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
HORIZONS = (2, 3)


def model_path(h):
    return OUT / f"dpfp_model_h{h}.joblib"


def _rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=20, max_depth=6,
                                  max_features="sqrt", n_jobs=-1, class_weight=None, random_state=42)


def train_all(features_fn, cols, main_stamp):
    """features_fn(df) -> (d, F) as in build_dashboard.features. cols = main model's feature list."""
    d, F = features_fn(pd.read_csv(_find(DELHI_CANDIDATES)))
    scores = []
    for h in (1,) + HORIZONS:
        y = d["dq_fail"].shift(-h)
        keep = y.notna()
        X, y = F.loc[keep, cols].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)
        b = int(len(X) * 0.75)
        med = X.iloc[:b].median()
        m = _rf().fit(X.iloc[:b].fillna(med), y.iloc[:b])
        p = m.predict_proba(X.iloc[b:].fillna(med))[:, 1]
        yte = y.iloc[b:]
        pr = float(average_precision_score(yte, p)) if yte.sum() >= 3 else np.nan
        roc = float(roc_auc_score(yte, p)) if yte.nunique() > 1 else np.nan
        base = float(yte.mean())
        scores.append({"days_ahead": h, "test_PR_AUC": round(pr, 4), "test_ROC_AUC": round(roc, 4),
                       "base_rate": round(base, 4), "lift": round(pr / base, 2) if base else np.nan,
                       "test_failures": int(yte.sum()), "n_rows": int(len(X))})
        if h in HORIZONS:
            full_med = X.median()
            m = _rf().fit(X.fillna(full_med), y)
            joblib.dump({"model": m, "features": cols, "medians": full_med.to_dict(), "days_ahead": h,
                         "trained_on": dt.datetime.now(IST).date().isoformat(), "score": scores[-1],
                         "main_model_stamp": main_stamp},
                        model_path(h))
    pd.DataFrame(scores).to_csv(OUT / "horizon_scores.csv", index=False)
    return scores


def ensure(features_fn, cols, main_stamp):
    """Train the outlook models if they are missing or were built from an older main model.
    main_stamp identifies the main model (its train date + rows), so this works on any
    machine regardless of file timestamps."""
    def current(h):
        try:
            return joblib.load(model_path(h)).get("main_model_stamp") == main_stamp
        except Exception:
            return False
    if not (OUT / "horizon_scores.csv").is_file() or not all(current(h) for h in HORIZONS):
        print("  Training 3-day outlook models...")
        train_all(features_fn, cols, main_stamp)
    return pd.read_csv(OUT / "horizon_scores.csv").to_dict(orient="records")
