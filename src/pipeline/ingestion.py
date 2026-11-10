
"""Open-Meteo API ingestion for the Pune air-quality pipeline."""

import pandas as pd
import requests

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

REQUIRED_COLUMNS = [
    "From Date",
    "To Date",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "OZONE",
]

POLLUTANT_MAP = {
    "PM2.5": "pm2_5",
    "PM10": "pm10",
    "NO2": "nitrogen_dioxide",
    "SO2": "sulphur_dioxide",
    "CO": "carbon_monoxide",
    "OZONE": "ozone",
}


def normalize_source_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize data without removing any required columns."""

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df[REQUIRED_COLUMNS].copy()

    for col in ("From Date", "To Date"):
        result[col] = pd.to_datetime(result[col], errors="coerce")

    for col in POLLUTANT_MAP:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    return result


def load_source_from_api(
    start_date: str,
    end_date: str,
    latitude: float = 18.5204,
    longitude: float = 73.8567,
    timezone: str = "Asia/Kolkata",
    timeout: int = 60,
    domains: str = "cams_global",
) -> pd.DataFrame:
    """Fetch air-quality data from a specified Open-Meteo domain."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "start_date": start_date,
        "end_date": end_date,
        "domains": domains,
        "hourly": list(POLLUTANT_MAP.values()),
    }

    response = requests.get(API_URL, params=params, timeout=timeout)
    response.raise_for_status()

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(
            payload.get("reason", "Open-Meteo API request failed")
        )

    hourly = payload.get("hourly")

    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ValueError(
            "Open-Meteo response did not contain hourly timestamps."
        )

    api_df = pd.DataFrame(hourly)

    if api_df.empty:
        raise ValueError("Open-Meteo returned no hourly records.")

    timestamps = pd.to_datetime(api_df["time"], errors="coerce")

    if timestamps.isna().any():
        raise ValueError("Open-Meteo returned invalid timestamps.")

    result = pd.DataFrame({
        "From Date": timestamps,
        "To Date": timestamps + pd.Timedelta(hours=1),
    })

    for pipeline_col, api_col in POLLUTANT_MAP.items():
        if api_col in api_df.columns:
            result[pipeline_col] = pd.to_numeric(
                api_df[api_col], errors="coerce"
            )
        else:
            # F-SCH-01: an expected core pollutant is absent from the API response.
            raise ValueError(
                f"Core column '{pipeline_col}' (API field '{api_col}') "
                "is absent from the API response. This indicates an API "
                "schema change or upstream feed failure."
            )


    result = normalize_source_data(result)

    print(f"Open-Meteo domain: {domains}")
    print(f"Records fetched: {len(result)}")
    print(f"Location: {latitude}, {longitude}")
    print(f"Timezone: {timezone}")
    print(
        f"Timestamp range: {result['From Date'].min()} "
        f"to {result['From Date'].max()}"
    )

    return result


def get_day_data(df: pd.DataFrame, run_date: str) -> pd.DataFrame:
    """Select and sort one local calendar day's records."""

    target_date = pd.Timestamp(run_date).date()

    return (
        df.loc[df["From Date"].dt.date == target_date]
        .copy()
        .sort_values("From Date")
        .reset_index(drop=True)
    )
