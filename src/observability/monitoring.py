"""Helpers for interpreting execution and data-quality outcomes."""

def determine_execution_status(exception_occurred: bool) -> str:
    return "FAIL" if exception_occurred else "SUCCESS"


def determine_data_quality_status(validation_status: str) -> str:
    return "SUCCESS" if validation_status == "SUCCESS" else "FAIL"


def identify_data_quality_issues(metrics: dict, config: dict) -> list[str]:
    issues = []
    if metrics.get("rows_processed", 0) == 0:
        issues.append("NO_DATA")
    if metrics.get("duplicate_count", 0) > 0:
        issues.append("DUPLICATE_DATA")
    if metrics.get("invalid_value_count", 0) > 0:
        issues.append("INVALID_DATA")
    mon = config["monitoring"]
    if metrics.get("coverage_percent", 0) < float(mon["coverage_failure_percent"]):
        issues.append("LOW_COVERAGE")
    if metrics.get("missing_value_rate_percent", 100) >= float(mon["missing_value_failure_percent"]):
        issues.append("HIGH_MISSINGNESS")
    return issues
