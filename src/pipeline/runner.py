
"""API-only pipeline runner with duplicate-date protection and observability logging."""

import json
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.pipeline.ingestion import load_source_from_api, get_day_data
from src.pipeline.validation import (
    validate_day_data,
    POLLUTANT_COLUMNS,
    CORE_SOURCE_COLUMNS,
)
from src.pipeline.transformation import transform_day_data
from src.database.database import (
    COLUMNS,
    ENRICHED_COLUMNS,
    initialize_database,
    insert_pipeline_run,
)
from src.observability.feature_enrichment import FeatureEnricher


def load_config(config_path="config/config.yaml"):
    """Load pipeline configuration."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_existing_successful_run(db_path, date_text):
    """Find a previously successful and validated run for a date."""
    db_file = Path(db_path)

    if not db_file.is_file():
        return None

    try:
        with sqlite3.connect(db_file) as conn:
            row = conn.execute(
                """
                SELECT run_date, overall_status, validation_status,
                       transformation_status, rows_processed,
                       processed_file
                FROM pipeline_runs
                WHERE run_date = ?
                  AND overall_status = 'SUCCESS'
                  AND validation_status = 'SUCCESS'
                  AND transformation_status = 'PASSED'
                LIMIT 1
                """,
                (date_text,),
            ).fetchone()
    except sqlite3.Error:
        # Do not silently skip if the database cannot be checked.
        raise

    if row is None:
        return None

    return {
        "run_date": row[0],
        "overall_status": row[1],
        "validation_status": row[2],
        "transformation_status": row[3],
        "rows_processed": row[4],
        "processed_file": row[5],
    }


def append_json_log(directory: str, record: dict):
    """Append one execution record to JSONL."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    with (path / "pipeline_runs.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def append_csv_log(directory: str, record: dict):
    """Append one execution record to CSV."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / "pipeline_runs.csv"
    csv_record = record.copy()

    for field in (
        "unavailable_columns",
        "all_missing_core_columns",
        "schema_missing_columns",
    ):
        csv_record[field] = json.dumps(
            csv_record.get(field, []), ensure_ascii=False
        )

    pd.DataFrame(
        [{column: csv_record.get(column) for column in COLUMNS}]
    ).to_csv(
        file_path,
        mode="a",
        header=not file_path.exists(),
        index=False,
        encoding="utf-8",
    )


def _api_with_retries(
    api_config: dict,
    run_date: str,
    max_retries: int = 2,
):
    """Fetch a day's data, retrying transient API errors.

    Returns (frame, retries, status_code, latencies) where `latencies` is the
    per-attempt wall-clock request time in seconds (a leading indicator for
    API_SLA_BREACH, used by the feature enricher).
    """
    retries = 0
    latencies = []

    while True:
        attempt_start = time.perf_counter()
        try:
            frame = load_source_from_api(
                start_date=run_date,
                end_date=run_date,
                latitude=float(api_config.get("latitude", 18.5204)),
                longitude=float(api_config.get("longitude", 73.8567)),
                timezone=api_config.get("timezone", "Asia/Kolkata"),
                timeout=int(api_config.get("timeout_seconds", 60)),
                domains=api_config.get("domains", "cams_global"),
            )
            latencies.append(time.perf_counter() - attempt_start)
            return frame, retries, 200, latencies

        except requests.exceptions.HTTPError as exc:
            latencies.append(time.perf_counter() - attempt_start)
            response = exc.response
            status = (
                response.status_code
                if response is not None
                else None
            )

            retryable = (
                status == 429
                or (status is not None and status >= 500)
            )

            if not retryable or retries >= max_retries:
                raise

            retries += 1
            time.sleep(min(2 ** retries, 8))

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ):
            latencies.append(time.perf_counter() - attempt_start)
            if retries >= max_retries:
                raise

            retries += 1
            time.sleep(min(2 ** retries, 8))


def _warning(metrics: dict, config: dict, duration: float):
    """Build warnings without conflating them with execution failure."""
    monitoring = config["monitoring"]
    reasons = []

    if metrics.get("rows_processed", 0) == 0:
        reasons.append("NO_DATA")

    coverage = metrics.get("coverage_percent")
    if (
        coverage is not None
        and coverage < float(monitoring["coverage_warning_percent"])
    ):
        reasons.append("LOW_COVERAGE")

    missing_rate = metrics.get("missing_value_rate_percent")
    if (
        missing_rate is not None
        and missing_rate
        >= float(monitoring["missing_value_warning_percent"])
    ):
        reasons.append("HIGH_MISSINGNESS")

    unavailable = metrics.get("unavailable_columns") or []
    if unavailable:
        reasons.append(
            "SOURCE_COLUMNS_UNAVAILABLE: " + ", ".join(unavailable)
        )

    if duration > float(
        monitoring.get("high_runtime_warning_seconds", 30)
    ):
        reasons.append("HIGH_RUNTIME")

    # Hazard-threshold crossings (real high pollution) are a warning signal,
    # not a failure. Surface them here so they feed the predictor as a
    # leading indicator without labelling the run as failed.
    quality_warning = metrics.get("quality_warning")
    if quality_warning:
        reasons.append(quality_warning)

    return (
        "WARNING" if reasons else "NONE",
        "; ".join(reasons) if reasons else None,
    )


def run_pipeline(
    run_date: str,
    config_path: str = "config/config.yaml",
) -> dict:
    """Run one API execution, skipping previously completed dates."""

    config = load_config(config_path)
    run_day = pd.Timestamp(run_date).date()
    date_text = run_day.isoformat()

    # Confirm the requested date belongs to a configured period.
    periods = config["periods"]

    historical_start = pd.Timestamp(
        periods["historical_start"]
    ).date()
    historical_end = pd.Timestamp(
        periods["historical_end"]
    ).date()

    if not (historical_start <= run_day <= historical_end):
        raise ValueError(
            f"{date_text} is outside the configured historical period."
        )

    processed_dir = Path(config["data"]["processed_directory"])
    existing_file = processed_dir / f"processed_{date_text}.csv"

    # First safeguard: skip a date already confirmed successful in the DB.
    existing_run = get_existing_successful_run(
        config["database"]["path"], date_text
    )

    if existing_run:
        print(
            f"SKIPPED: {date_text} already has a successful, "
            "validated pipeline run. No API call made."
        )
        return {**existing_run, "skipped": True}

    # Second safeguard: never overwrite a CSV when DB history is absent.
    if existing_file.is_file():
        print(
            f"SKIPPED: {date_text} already has a processed CSV, "
            "but no successful database record was found. "
            "No API call made. Review the file before rerunning."
        )
        return {
            "run_date": date_text,
            "overall_status": "SKIPPED",
            "validation_status": "NOT_RUN",
            "transformation_status": "NOT_RUN",
            "rows_processed": 0,
            "processed_file": str(existing_file),
            "skipped": True,
        }

    # Wall-clock marker — used ONLY to measure real processing duration
    # (for SLA-breach detection), never written to the log timeline.
    wall_start = time.perf_counter()

    # Timeline marker — the logged start_time is pinned to the run's own
    # run_date (historical replay), so runs are correctly spaced one day
    # apart and lead time reads as "1 run = 1 day". A fixed 01:00 local
    # time-of-day keeps timestamps ordered and deterministic.
    tz = config["api"].get("timezone", "Asia/Kolkata")
    try:
        start = pd.Timestamp(f"{date_text} 01:00:00").tz_localize(tz)
    except Exception:
        start = pd.Timestamp(f"{date_text} 01:00:00")

    run_id = (
        "RUN_"
        + run_day.strftime("%Y%m%d_")
        + uuid.uuid4().hex[:10]
    )

    api_status_code = None
    retries = 0
    request_latencies = []
    failed_stage = "INGESTION"
    failure_type = None
    error_message = None
    processed_file = None
    transformation_status = "NOT_RUN"

    expected_records = int(
        config["pipeline"].get("expected_records_per_day", 24)
    )

    metrics = {
        "validation_status": "NOT_RUN",
        "rows_processed": 0,
        "coverage_percent": 0.0,
        "missing_value_rate_percent": 100.0,
        "core_missing_value_rate_percent": 100.0,
        "unavailable_columns": POLLUTANT_COLUMNS.copy(),
        "all_missing_core_columns": CORE_SOURCE_COLUMNS.copy(),
        "schema_missing_columns": [],
        "max_gap_minutes": None,
        "missing_intervals": expected_records,
        "duplicate_count": 0,
        "invalid_value_count": 0,
        "hazard_exceed_count": 0,
        "quality_error": None,
        "quality_warning": None,
    }

    # SUCCESS means the execution completed and validation passed.
    overall_status = "SUCCESS"

    # day_frame may not exist if ingestion fails; keep a safe default so the
    # feature enricher can always run (it degrades gracefully on empty data).
    day_frame = pd.DataFrame()

    try:
        frame, retries, api_status_code, request_latencies = _api_with_retries(
            config["api"],
            date_text,
            int(config["api"].get("max_retries", 2)),
        )

        failed_stage = "DATE_FILTER"
        day_frame = get_day_data(frame, date_text)

        failed_stage = "VALIDATION"
        validation_config = dict(config)
        validation_config["_run_date"] = date_text
        metrics.update(
            validate_day_data(day_frame, validation_config)
        )

        if metrics.get("validation_status") == "FAIL":
            overall_status = "FAIL"
            failure_type = "DATA_QUALITY_FAILURE"
            failed_stage = "VALIDATION"
            error_message = metrics.get("quality_error") or (
                "Data-quality validation failed."
            )

        # Do not create a processed file for a validation failure.
        elif not day_frame.empty:
            failed_stage = "TRANSFORMATION"

            # Recheck immediately before transformation to protect the file.
            if existing_file.is_file():
                raise FileExistsError(
                    "Processed file already exists; refusing to overwrite: "
                    + str(existing_file)
                )

            processed_file = transform_day_data(
                day_frame,
                date_text,
                config["data"]["processed_directory"],
            )

            if not Path(processed_file).is_file():
                raise FileNotFoundError(
                    "Processed file was not created: " + processed_file
                )

            # F-STOR-03: reject a zero-byte output file.
            # A partial disk write or silent error can produce an empty file
            # that passes the existence check above.
            written_size = Path(processed_file).stat().st_size
            if written_size == 0:
                raise ValueError(
                    "Processed file is empty (0 bytes) after write: "
                    + processed_file
                )

            # F-STOR-04: row count parity — written CSV must match input.
            written_rows = len(pd.read_csv(processed_file))
            expected_rows = len(day_frame)
            if written_rows != expected_rows:
                raise ValueError(
                    f"Row count mismatch after write: "
                    f"expected {expected_rows} rows, "
                    f"got {written_rows} in {processed_file}"
                )

            transformation_status = "PASSED"
            failed_stage = None


        else:
            # An empty response is already rejected by validation.
            failed_stage = "VALIDATION"

    except Exception as exc:
        overall_status = "FAIL"
        failed_stage = failed_stage or "UNKNOWN"
        error_message = f"{type(exc).__name__}: {exc}"

        if isinstance(exc, requests.exceptions.HTTPError):
            response = exc.response
            api_status_code = (
                response.status_code if response is not None else None
            )

            if api_status_code == 429:
                failure_type = "API_RATE_LIMIT_FAILURE"
            elif api_status_code is not None and api_status_code >= 500:
                failure_type = "API_SERVER_FAILURE"
            elif api_status_code is not None and api_status_code >= 400:
                failure_type = "API_REQUEST_FAILURE"
            else:
                failure_type = "API_HTTP_FAILURE"

        elif isinstance(exc, requests.exceptions.Timeout):
            failure_type = "API_TIMEOUT"

        elif isinstance(exc, requests.exceptions.ConnectionError):
            failure_type = "API_CONNECTION_FAILURE"

        elif failed_stage == "TRANSFORMATION":
            failure_type = "TRANSFORMATION_EXCEPTION"
            transformation_status = "FAILED"

        elif failed_stage == "VALIDATION":
            failure_type = "VALIDATION_EXCEPTION"

        elif failed_stage == "INGESTION":
            failure_type = "API_INGESTION_EXCEPTION"

        else:
            failure_type = f"{failed_stage}_EXCEPTION"

    # Real processing time (wall clock) — this drives SLA-breach detection.
    duration = round(time.perf_counter() - wall_start, 3)
    # Logged end_time stays on the run_date timeline (start + real duration).
    end = start + pd.Timedelta(seconds=duration)

    sla_limit = float(config["monitoring"].get("sla_failure_seconds", 0))
    if sla_limit > 0 and duration > sla_limit and overall_status == "SUCCESS":
        overall_status = "FAIL"
        failed_stage = "INGESTION"
        failure_type = "API_SLA_BREACH"
        error_message = (
            f"Execution duration {duration}s exceeded SLA limit of {sla_limit}s."
        )

    warning_status, warning_reason = _warning(
        metrics, config, duration
    )

    # ---- Leading-indicator feature enrichment (per-city rolling history) ----
    # Computes the 17 data-centric signals (drift/PSI, freshness, % near
    # hazard, API-latency trend, sensor stability, duration trend) that feed
    # the failure-prediction model. Uses only this run + prior runs' history,
    # so it introduces no leakage.
    enrich_cfg = config.get("enrichment", {}) or {}
    history_path = enrich_cfg.get(
        "history_path",
        str(Path(config["logging"]["directory"]) / "feature_history.json"),
    )
    enricher = FeatureEnricher(history_path=history_path)

    # Pollutant columns as they appear in the validated/normalized day_frame
    # (display names, matching config quality_bounds keys).
    pollutant_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "OZONE"]

    # Latest data timestamp in the batch, for freshness lag. The fetch time is
    # pinned to the run_date timeline (matching the time-fix) so freshness is
    # measured consistently in historical replay.
    data_ts = None
    try:
        if not day_frame.empty and "To Date" in day_frame.columns:
            data_ts = pd.to_datetime(day_frame["To Date"], errors="coerce").max()
    except Exception:
        data_ts = None

    enriched = enricher.compute(
        day_frame=day_frame,
        pollutant_cols=pollutant_cols,
        request_latencies=request_latencies,
        duration_seconds=duration,
        rows_processed=metrics.get("rows_processed", len(day_frame)),
        expected_rows=expected_records,
        data_timestamp=data_ts,
        fetch_time=end,
        hazard_bounds=config.get("quality_bounds"),
    )
    enricher.commit()

    record = {
        "run_id": run_id,
        "run_date": date_text,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_seconds": duration,
        "api_status_code": api_status_code,
        "retry_count": retries,
        "rows_processed": metrics.get("rows_processed", 0),
        "coverage_percent": metrics.get("coverage_percent", 0.0),
        "missing_value_rate_percent": metrics.get(
            "missing_value_rate_percent", 100.0
        ),
        "core_missing_value_rate_percent": metrics.get(
            "core_missing_value_rate_percent", 100.0
        ),
        "unavailable_columns": metrics.get(
            "unavailable_columns", []
        ),
        "all_missing_core_columns": metrics.get(
            "all_missing_core_columns", []
        ),
        "schema_missing_columns": metrics.get(
            "schema_missing_columns", []
        ),
        "max_gap_minutes": metrics.get("max_gap_minutes"),
        "missing_intervals": metrics.get(
            "missing_intervals", expected_records
        ),
        "duplicate_count": metrics.get("duplicate_count", 0),
        "invalid_value_count": metrics.get(
            "invalid_value_count", 0
        ),
        "hazard_exceed_count": metrics.get(
            "hazard_exceed_count", 0
        ),
        "validation_status": metrics.get(
            "validation_status", "NOT_RUN"
        ),
        "transformation_status": transformation_status,
        "overall_status": overall_status,
        "failure_type": failure_type,
        "failed_stage": (
            failed_stage if overall_status == "FAIL" else None
        ),
        "error_message": error_message,
        "quality_error": metrics.get("quality_error"),
        "processed_file": processed_file,
        "warning_status": warning_status,
        "warning_reason": warning_reason,
    }

    # Merge only the 17 declared enriched feature columns into the record
    # (the enricher also returns internal per-pollutant history keys, which
    # are persisted to history but are NOT log columns).
    for col in ENRICHED_COLUMNS:
        record[col] = enriched.get(col, 0.0)

    # Save each actual execution to the database and append-only logs.
    db_path = config["database"]["path"]
    initialize_database(db_path)

    db_record = record.copy()

    for field in (
        "unavailable_columns",
        "all_missing_core_columns",
        "schema_missing_columns",
    ):
        db_record[field] = json.dumps(
            db_record[field], ensure_ascii=False
        )

    insert_pipeline_run(db_path, db_record)

    append_json_log(config["logging"]["directory"], record)
    append_csv_log(config["logging"]["directory"], record)

    return record
