
"""SQLite storage for append-only pipeline execution records."""

import sqlite3
from pathlib import Path

COLUMNS = [
    "run_id",
    "run_date",
    "start_time",
    "end_time",
    "duration_seconds",
    "api_status_code",
    "retry_count",
    "rows_processed",
    "coverage_percent",
    "missing_value_rate_percent",
    "core_missing_value_rate_percent",
    "unavailable_columns",
    "all_missing_core_columns",
    "schema_missing_columns",
    "max_gap_minutes",
    "missing_intervals",
    "duplicate_count",
    "invalid_value_count",
    "validation_status",
    "transformation_status",
    "overall_status",
    "failure_type",
    "failed_stage",
    "error_message",
    "quality_error",
    "processed_file",
    "warning_status",
    "warning_reason",
    # --- leading-indicator features (added for failure prediction) ---
    "api_latency_mean",
    "api_latency_max",
    "api_latency_p95",
    "api_latency_vs_baseline",
    "api_latency_roll3_mean",
    "api_latency_roll7_std",
    "drift_psi_mean",
    "drift_psi_max",
    "rows_vs_expected_ratio",
    "row_shortfall",
    "freshness_lag_hours",
    "pct_near_hazard",
    "max_stagnation_run",
    "max_hourly_jump",
    "duration_roll3_mean",
    "duration_roll7_std",
    "duration_vs_baseline",
    # count of hazard-threshold crossings (real high pollution) = warning signal
    "hazard_exceed_count",
]

# The 17 leading-indicator feature columns, added by the FeatureEnricher.
ENRICHED_COLUMNS = [
    "api_latency_mean", "api_latency_max", "api_latency_p95",
    "api_latency_vs_baseline", "api_latency_roll3_mean", "api_latency_roll7_std",
    "drift_psi_mean", "drift_psi_max",
    "rows_vs_expected_ratio", "row_shortfall", "freshness_lag_hours", "pct_near_hazard",
    "max_stagnation_run", "max_hourly_jump",
    "duration_roll3_mean", "duration_roll7_std", "duration_vs_baseline",
]

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id TEXT PRIMARY KEY,
    run_date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    duration_seconds REAL,
    api_status_code INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    rows_processed INTEGER,
    coverage_percent REAL,
    missing_value_rate_percent REAL,
    core_missing_value_rate_percent REAL,
    unavailable_columns TEXT,
    all_missing_core_columns TEXT,
    schema_missing_columns TEXT,
    max_gap_minutes REAL,
    missing_intervals INTEGER,
    duplicate_count INTEGER,
    invalid_value_count INTEGER,
    validation_status TEXT,
    transformation_status TEXT,
    overall_status TEXT,
    failure_type TEXT,
    failed_stage TEXT,
    error_message TEXT,
    quality_error TEXT,
    processed_file TEXT,
    warning_status TEXT DEFAULT 'NONE',
    warning_reason TEXT,
    api_latency_mean REAL,
    api_latency_max REAL,
    api_latency_p95 REAL,
    api_latency_vs_baseline REAL,
    api_latency_roll3_mean REAL,
    api_latency_roll7_std REAL,
    drift_psi_mean REAL,
    drift_psi_max REAL,
    rows_vs_expected_ratio REAL,
    row_shortfall REAL,
    freshness_lag_hours REAL,
    pct_near_hazard REAL,
    max_stagnation_run REAL,
    max_hourly_jump REAL,
    duration_roll3_mean REAL,
    duration_roll7_std REAL,
    duration_vs_baseline REAL,
    hazard_exceed_count INTEGER
)
"""

# New columns that must also be added to an existing database.
MIGRATION_COLUMNS = {
    "core_missing_value_rate_percent": "REAL",
    "unavailable_columns": "TEXT",
    "all_missing_core_columns": "TEXT",
    "schema_missing_columns": "TEXT",
    "quality_error": "TEXT",
    # leading-indicator feature columns
    "api_latency_mean": "REAL",
    "api_latency_max": "REAL",
    "api_latency_p95": "REAL",
    "api_latency_vs_baseline": "REAL",
    "api_latency_roll3_mean": "REAL",
    "api_latency_roll7_std": "REAL",
    "drift_psi_mean": "REAL",
    "drift_psi_max": "REAL",
    "rows_vs_expected_ratio": "REAL",
    "row_shortfall": "REAL",
    "freshness_lag_hours": "REAL",
    "pct_near_hazard": "REAL",
    "max_stagnation_run": "REAL",
    "max_hourly_jump": "REAL",
    "duration_roll3_mean": "REAL",
    "duration_roll7_std": "REAL",
    "duration_vs_baseline": "REAL",
    "hazard_exceed_count": "INTEGER",
}


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection, creating its parent directory."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, timeout=30)


def initialize_database(db_path: str) -> None:
    """Create the table and add new columns without deleting old records."""
    conn = get_connection(db_path)

    try:
        conn.execute(TABLE_SQL)

        existing_columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(pipeline_runs)"
            ).fetchall()
        }

        for column_name, column_type in MIGRATION_COLUMNS.items():
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE pipeline_runs "
                    f"ADD COLUMN {column_name} {column_type}"
                )

        conn.commit()
    finally:
        conn.close()


def insert_pipeline_run(db_path: str, record: dict) -> None:
    """Insert one execution record without overwriting previous runs."""
    missing = set(COLUMNS) - set(record)
    extra = set(record) - set(COLUMNS)

    if missing or extra:
        raise ValueError(
            "Record/schema mismatch. "
            f"Missing={sorted(missing)}, Extra={sorted(extra)}"
        )

    placeholders = ", ".join("?" for _ in COLUMNS)
    column_names = ", ".join(COLUMNS)

    sql = (
        f"INSERT INTO pipeline_runs ({column_names}) "
        f"VALUES ({placeholders})"
    )

    conn = get_connection(db_path)

    try:
        conn.execute(sql, [record[column] for column in COLUMNS])
        conn.commit()
    finally:
        conn.close()
