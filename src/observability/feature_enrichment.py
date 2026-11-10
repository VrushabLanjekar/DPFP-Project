"""
feature_enrichment.py
---------------------------------
Leading-indicator feature enrichment for the air-quality pipeline.

WHY THIS EXISTS
    The original pipeline logged a signal only WHEN something broke (e.g.
    invalid_value_count > 0). A predictor cannot learn an early warning from a
    value that is flat until the moment of failure. This module computes, on
    EVERY run, signals that tend to MOVE BEFORE a failure:

        1. API response-time statistics   -> leads API_SLA_BREACH
        2. Drift score vs recent history   -> leads DATA_QUALITY_FAILURE
        3. Freshness & completeness        -> early quality signals
        4. Rolling sensor stability        -> stagnation / sudden-jump build-up

    All features are computed from the CURRENT run plus a short ROLLING HISTORY
    of recent runs for the same city. Nothing here reads the failure label, so
    it introduces no leakage.

HOW TO USE (see INTEGRATION_GUIDE.md for the exact hook into runner.py)
    enricher = FeatureEnricher(history_path="logs/<city>/feature_history.json")
    extra = enricher.compute(
        day_frame=day_frame,              # the fetched/normalized hourly data
        pollutant_cols=["pm2_5","pm10","nitrogen_dioxide",
                        "sulphur_dioxide","carbon_monoxide","ozone"],
        request_latencies=[0.42, 0.55],   # per-attempt API latency (seconds)
        duration_seconds=run_duration,    # total run time
        rows_processed=len(day_frame),
        expected_rows=24,
        data_timestamp=latest_data_ts,    # max "To Date" in the batch
        fetch_time=now_utc,               # when the run fetched
    )
    # `extra` is a flat dict of new columns -> merge into your log record.
    enricher.commit()                     # persist this run into history

The module keeps its own small JSON history file PER CITY so the two
cross-validation datasets never mix their rolling statistics.
"""

from __future__ import annotations
import json
import math
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ----------------------------- helpers --------------------------------------

def _safe(x, default=0.0):
    try:
        if x is None:
            return default
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.
    PSI < 0.1 = no real shift, 0.1-0.25 = moderate, > 0.25 = large shift."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    # fixed bin edges from the reference distribution
    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_frac = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, None)
    a_frac = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))


# ----------------------------- main class -----------------------------------

class FeatureEnricher:
    """Computes leading-indicator features and keeps a rolling per-city history."""

    # how many recent runs to keep for rolling stats / drift reference
    WINDOW = 7
    DRIFT_REFERENCE = 7  # runs used as the "normal" distribution for PSI

    def __init__(self, history_path: str, window: int | None = None):
        self.history_path = Path(history_path)
        self.window = window or self.WINDOW
        self._history = self._load_history()
        self._pending = None  # the run we just computed, awaiting commit()

    # ---- history persistence ----
    def _load_history(self) -> list[dict]:
        if self.history_path.is_file():
            try:
                return json.loads(self.history_path.read_text())[-50:]
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_history(self):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(self._history[-50:], default=str))

    # ---- the computation ----
    def compute(
        self,
        day_frame: pd.DataFrame,
        pollutant_cols: list[str],
        request_latencies: list[float] | None,
        duration_seconds: float,
        rows_processed: int,
        expected_rows: int = 24,
        data_timestamp=None,
        fetch_time=None,
        hazard_bounds: dict | None = None,
    ) -> dict:
        """Return a flat dict of new feature columns for this run."""
        hist = self._history
        feats: dict[str, float] = {}

        # ---------- 1. API RESPONSE-TIME STATS (leads SLA breach) ----------
        lat = [l for l in (request_latencies or []) if l is not None]
        feats["api_latency_mean"] = _safe(np.mean(lat)) if lat else _safe(duration_seconds)
        feats["api_latency_max"] = _safe(np.max(lat)) if lat else _safe(duration_seconds)
        feats["api_latency_p95"] = _safe(np.percentile(lat, 95)) if len(lat) >= 2 else feats["api_latency_max"]
        # trend vs recent runs
        recent_lat = [h.get("api_latency_mean", 0) for h in hist[-self.window:]]
        base_lat = _safe(np.mean(recent_lat)) if recent_lat else feats["api_latency_mean"]
        feats["api_latency_vs_baseline"] = feats["api_latency_mean"] - base_lat
        feats["api_latency_roll3_mean"] = _safe(np.mean(([*recent_lat, feats["api_latency_mean"]])[-3:]))
        feats["api_latency_roll7_std"] = _safe(np.std(([*recent_lat, feats["api_latency_mean"]])[-7:]))

        # ---------- 2. DRIFT SCORE vs recent history (leads DQ failure) ----------
        # Build a reference distribution from the last DRIFT_REFERENCE runs' means.
        drift_scores = []
        for col in pollutant_cols:
            if col not in day_frame.columns:
                continue
            cur = pd.to_numeric(day_frame[col], errors="coerce").to_numpy(dtype=float)
            ref_means = [h.get(f"{col}_mean") for h in hist[-self.DRIFT_REFERENCE:]
                         if h.get(f"{col}_mean") is not None]
            # compare current sample to a synthetic reference around recent means
            if ref_means and len(cur[~np.isnan(cur)]) >= 2:
                ref = np.random.normal(np.mean(ref_means),
                                       max(np.std(ref_means), 1e-3),
                                       size=200)
                drift_scores.append(_psi(ref, cur))
        feats["drift_psi_mean"] = _safe(np.mean(drift_scores)) if drift_scores else 0.0
        feats["drift_psi_max"] = _safe(np.max(drift_scores)) if drift_scores else 0.0

        # ---------- 3. FRESHNESS & COMPLETENESS ----------
        feats["rows_vs_expected_ratio"] = _safe(rows_processed / max(expected_rows, 1), 1.0)
        feats["row_shortfall"] = _safe(max(expected_rows - rows_processed, 0))
        # freshness lag in hours (data timestamp vs fetch time)
        lag_hours = 0.0
        if data_timestamp is not None and fetch_time is not None:
            try:
                dt_data = pd.to_datetime(data_timestamp, utc=True)
                dt_fetch = pd.to_datetime(fetch_time, utc=True)
                lag_hours = max((dt_fetch - dt_data).total_seconds() / 3600.0, 0.0)
            except (ValueError, TypeError):
                lag_hours = 0.0
        feats["freshness_lag_hours"] = _safe(lag_hours)

        # % of values NEAR (within 80% of) each hazard threshold -> creeping risk
        near = 0
        total = 0
        if hazard_bounds:
            name_map = {"pm2_5": "PM2.5", "pm10": "PM10", "nitrogen_dioxide": "NO2",
                        "sulphur_dioxide": "SO2", "carbon_monoxide": "CO", "ozone": "OZONE"}
            for col in pollutant_cols:
                bound = hazard_bounds.get(name_map.get(col, col))
                if bound is None or col not in day_frame.columns:
                    continue
                vals = pd.to_numeric(day_frame[col], errors="coerce").dropna()
                total += len(vals)
                near += int((vals >= 0.8 * float(bound)).sum())
        feats["pct_near_hazard"] = _safe(100 * near / total) if total else 0.0

        # ---------- 4. ROLLING SENSOR STABILITY ----------
        max_stag = 0
        max_jump = 0.0
        for col in pollutant_cols:
            if col not in day_frame.columns:
                continue
            vals = pd.to_numeric(day_frame[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if len(vals) < 2:
                continue
            # longest run of identical consecutive values (stagnation)
            run = best = 1
            for i in range(1, len(vals)):
                if vals[i] == vals[i - 1]:
                    run += 1
                    best = max(best, run)
                else:
                    run = 1
            max_stag = max(max_stag, best)
            # largest single-step jump
            max_jump = max(max_jump, float(np.max(np.abs(np.diff(vals)))))
            # store this run's mean for the NEXT run's drift reference
            feats[f"{col}_mean"] = _safe(np.mean(vals))
            feats[f"{col}_std"] = _safe(np.std(vals))
        feats["max_stagnation_run"] = float(max_stag)
        feats["max_hourly_jump"] = _safe(max_jump)

        # duration trend (cheap but useful)
        recent_dur = [h.get("duration_seconds", 0) for h in hist[-self.window:]]
        feats["duration_roll3_mean"] = _safe(np.mean(([*recent_dur, duration_seconds])[-3:]))
        feats["duration_roll7_std"] = _safe(np.std(([*recent_dur, duration_seconds])[-7:]))
        feats["duration_vs_baseline"] = _safe(duration_seconds - (np.mean(recent_dur) if recent_dur else duration_seconds))

        # keep a compact record to append to history on commit()
        self._pending = {
            "duration_seconds": _safe(duration_seconds),
            "api_latency_mean": feats["api_latency_mean"],
            **{k: v for k, v in feats.items() if k.endswith("_mean") or k.endswith("_std")},
        }
        return feats

    def commit(self):
        """Persist the last computed run into the rolling history."""
        if self._pending is not None:
            self._history.append(self._pending)
            self._save_history()
            self._pending = None


# The exact list of new columns this module adds (for DB migration / docs).
ENRICHED_COLUMNS = [
    "api_latency_mean", "api_latency_max", "api_latency_p95",
    "api_latency_vs_baseline", "api_latency_roll3_mean", "api_latency_roll7_std",
    "drift_psi_mean", "drift_psi_max",
    "rows_vs_expected_ratio", "row_shortfall", "freshness_lag_hours", "pct_near_hazard",
    "max_stagnation_run", "max_hourly_jump",
    "duration_roll3_mean", "duration_roll7_std", "duration_vs_baseline",
]
