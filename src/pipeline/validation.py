
"""Rule-based data-quality validation for the air-quality API pipeline."""

import pandas as pd

POLLUTANT_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "OZONE",
]

# Core variables selected for this project.
CORE_SOURCE_COLUMNS = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "OZONE",
]

TIMESTAMP_COLUMNS = ["From Date", "To Date"]

# Regulatory severe hazard thresholds (CPCB / WHO Standards for India).
# Values exceeding these indicate extreme pollution emergencies or sensor runaway.
POLLUTANT_UPPER_BOUNDS = {
    "PM2.5": 120.0,    # CPCB Severe hazard limit (µg/m³); Pune peak: 155.5
    "PM10":  200.0,    # CPCB Severe hazard limit (µg/m³); Pune peak: 223.1
    "NO2":   100.0,    # CPCB Severe hazard limit (µg/m³); Pune peak: 105.5
    "SO2":   100.0,    # CPCB Severe hazard limit (µg/m³)
    "CO":    2000.0,   # CPCB Severe hazard limit (µg/m³); Pune peak: 2253.0
    "OZONE": 180.0,    # CPCB 1-hr standard (µg/m³); Pune peak: 204.0
}


def validate_day_data(df: pd.DataFrame, config: dict) -> dict:
    """Apply deterministic quality rules to one day's API data."""

    expected = int(
        config["pipeline"].get("expected_records_per_day", 24)
    )
    interval = int(
        config["pipeline"].get("expected_interval_minutes", 60)
    )
    monitoring = config["monitoring"]

    def failed_result(message, rows=0, schema_missing=None):
        return {
            "validation_status": "FAIL",
            "rows_processed": rows,
            "coverage_percent": 0.0,
            "missing_value_rate_percent": 100.0,
            "core_missing_value_rate_percent": 100.0,
            "unavailable_columns": POLLUTANT_COLUMNS.copy(),
            "all_missing_core_columns": CORE_SOURCE_COLUMNS.copy(),
            "schema_missing_columns": schema_missing or [],
            "duplicate_count": 0,
            "invalid_value_count": 0,
            "missing_intervals": expected,
            "max_gap_minutes": None,
            "quality_error": message,
        }

    # R03: reject an empty response.
    if df is None or df.empty:
        return failed_result("EMPTY_RESPONSE: No records available.")

    # R04: require all expected timestamp and pollutant columns.
    required = TIMESTAMP_COLUMNS + POLLUTANT_COLUMNS
    schema_missing = [c for c in required if c not in df.columns]

    if schema_missing:
        return failed_result(
            "SCHEMA_ERROR: Missing required columns: "
            + ", ".join(schema_missing),
            rows=len(df),
            schema_missing=schema_missing,
        )

    rows = len(df)
    issues = []

    # Parse timestamps.
    start_ts = pd.to_datetime(df["From Date"], errors="coerce")
    end_ts = pd.to_datetime(df["To Date"], errors="coerce")

    invalid_timestamps = start_ts.isna() | end_ts.isna()
    reversed_intervals = (
        start_ts.notna()
        & end_ts.notna()
        & (end_ts <= start_ts)
    )

    valid_start = start_ts.dropna()

    # R05: reject records containing multiple dates.
    mixed_dates = (
        valid_start.dt.date.nunique() > 1
        if not valid_start.empty
        else False
    )

    if mixed_dates:
        issues.append(
            "INVALID_TIMESTAMP: Multiple dates in one-day input."
        )

    # R06: reject records that do not match the requested run date.
    requested_date = config.get("_run_date")
    wrong_date_count = 0

    if requested_date:
        requested_date = pd.Timestamp(requested_date).date()
        wrong_date_count = int(
            sum(ts.date() != requested_date for ts in valid_start)
        )

        if wrong_date_count:
            issues.append(
                "WRONG_RUN_DATE: Records do not match the requested date."
            )

    # R07: detect duplicate start timestamps.
    duplicate_count = int(valid_start.duplicated().sum())

    if duplicate_count:
        issues.append(
            "DUPLICATE_TIMESTAMPS: Duplicate start timestamps."
        )

    # Detect invalid timestamp intervals.
    invalid_date_count = int(
        invalid_timestamps.sum() + reversed_intervals.sum()
    )

    if invalid_timestamps.any():
        issues.append(
            "INVALID_TIMESTAMP: Unparseable timestamp."
        )

    if reversed_intervals.any():
        issues.append(
            "INVALID_TIMESTAMP: Invalid time interval."
        )

    # Convert pollutant fields to numeric.
    numeric = df[POLLUTANT_COLUMNS].apply(
        lambda col: pd.to_numeric(col, errors="coerce")
    )

    # Count negative measurements as invalid.
    negative_count = int((numeric < 0).sum().sum())

    if negative_count:
        issues.append(
            "INVALID_VALUE: Negative pollutant values."
        )

    # F-DQ-05: detect values that exceed regulatory severe hazard bounds.
    # NOTE: A value exceeding a CPCB hazard limit reflects GENUINELY HIGH
    # real-world pollution — it is correct data, not a data/pipeline fault.
    # Therefore it is recorded as a WARNING / leading-indicator signal, NOT a
    # failure. (Genuine data failures are: negatives, impossible spikes,
    # sensor stagnation, missing/short data, and pipeline/API exceptions.)
    bounds = config.get("quality_bounds") or POLLUTANT_UPPER_BOUNDS
    out_of_range_count = 0
    out_of_range_cols = []
    for col, upper in bounds.items():
        if col in numeric.columns:
            exceeded = int((numeric[col] > float(upper)).sum())
            if exceeded:
                out_of_range_count += exceeded
                out_of_range_cols.append(
                    f"{col}(>{upper})"
                )

    warnings = []
    if out_of_range_count:
        warnings.append(
            "HAZARD_THRESHOLD_EXCEEDED: Values exceed regulatory severe limit — "
            + ", ".join(out_of_range_cols)
        )

    # F-DQ-06: Sudden spike / rate-of-change anomaly (unphysical jump).
    max_roc = float(monitoring.get("max_hourly_rate_of_change_pm25", 50.0))
    spike_count = 0
    if "PM2.5" in numeric.columns and len(numeric["PM2.5"].dropna()) > 1:
        pm25_diff = numeric["PM2.5"].dropna().diff().abs()
        max_diff = pm25_diff.max()
        if max_diff > max_roc:
            spike_count = int((pm25_diff > max_roc).sum())
            issues.append(
                f"SUDDEN_SPIKE_ANOMALY: PM2.5 jumped {round(max_diff, 1)} µg/m³ in 1 hour (max allowed {max_roc})"
            )

    # F-DQ-07: Sensor flatline / stagnation check.
    max_stagnant = int(monitoring.get("max_consecutive_stagnant_hours", 6))
    for col in ("PM2.5", "PM10"):
        if col in numeric.columns:
            s = numeric[col].dropna()
            if len(s) >= max_stagnant:
                longest_run = (s != s.shift()).cumsum().value_counts().max()
                if longest_run >= max_stagnant:
                    issues.append(
                        f"SENSOR_STAGNATION: {col} flatlined with identical value for {longest_run} consecutive hours"
                    )

    # invalid_value_count counts GENUINE data errors only (negatives,
    # bad timestamps, impossible spikes) — NOT hazard-threshold crossings,
    # which are real high pollution recorded as a warning below.
    invalid_count = (
        invalid_date_count
        + negative_count
        + spike_count
    )
    # Hazard crossings are tracked separately as a predictive signal.
    hazard_exceed_count = out_of_range_count

    # R08: calculate overall and core pollutant missingness.
    total_values = rows * len(POLLUTANT_COLUMNS)
    missing_values = int(numeric.isna().sum().sum())
    missing_rate = (
        100.0 * missing_values / total_values
        if total_values
        else 100.0
    )

    core = numeric[CORE_SOURCE_COLUMNS]
    core_total = rows * len(CORE_SOURCE_COLUMNS)
    core_missing = int(core.isna().sum().sum())
    core_missing_rate = (
        100.0 * core_missing / core_total
        if core_total
        else 100.0
    )

    unavailable = [
        col for col in POLLUTANT_COLUMNS
        if numeric[col].isna().all()
    ]

    all_missing_core = [
        col for col in CORE_SOURCE_COLUMNS
        if numeric[col].isna().all()
    ]

    # R09: calculate timestamp coverage and gaps.
    unique_ts = (
        valid_start.drop_duplicates().sort_values()
    )
    coverage = min(
        100.0,
        100.0 * len(unique_ts) / expected,
    ) if expected > 0 else 0.0

    if len(unique_ts) > 1:
        gaps = unique_ts.diff().dt.total_seconds().div(60).dropna()
        max_gap = float(gaps.max())
        missing_intervals = int(
            sum(
                max(0, round(float(gap) / interval) - 1)
                for gap in gaps
            )
        ) if interval > 0 else 0
    else:
        max_gap = None
        missing_intervals = max(0, expected - len(unique_ts))

    # R10: fail when coverage is below the configured threshold.
    if coverage < float(monitoring["coverage_failure_percent"]):
        issues.append(
            "LOW_COVERAGE: Coverage below failure threshold."
        )

    # F-DQ-01: fail if ANY core column has even one null value.
    if core_missing_rate > 0.0:
        issues.append(
            f"CORE_DATA_MISSING: {round(core_missing_rate, 2)}% of core "
            f"column values are null (expected 0% for this source). "
            f"Affected columns: "
            + ", ".join(all_missing_core or ["partial nulls in core columns"])
        )

    # R11: fail when core missingness reaches the configured threshold.
    # (Kept as a secondary check for large-scale degradation.)
    elif core_missing_rate >= float(
        monitoring["missing_value_failure_percent"]
    ):
        issues.append(
            "CORE_DATA_MISSING: Missingness reached failure threshold."
        )

    # R12: fail if any core pollutant column is entirely missing.
    if all_missing_core:
        issues.append(
            "CORE_COLUMN_UNAVAILABLE: "
            + ", ".join(all_missing_core)
        )


    return {
        "validation_status": "FAIL" if issues else "SUCCESS",
        "rows_processed": rows,
        "coverage_percent": round(coverage, 2),
        "missing_value_rate_percent": round(missing_rate, 2),
        "core_missing_value_rate_percent": round(
            core_missing_rate, 2
        ),
        "unavailable_columns": unavailable,
        "all_missing_core_columns": all_missing_core,
        "schema_missing_columns": [],
        "duplicate_count": duplicate_count,
        "invalid_value_count": invalid_count,
        "hazard_exceed_count": hazard_exceed_count,
        "missing_intervals": missing_intervals,
        "max_gap_minutes": (
            round(max_gap, 2) if max_gap is not None else None
        ),
        # Genuine data failures (drive validation_status).
        "quality_error": "; ".join(issues) if issues else None,
        # Hazard-threshold crossings = real high pollution, a WARNING signal.
        "quality_warning": "; ".join(warnings) if warnings else None,
    }
