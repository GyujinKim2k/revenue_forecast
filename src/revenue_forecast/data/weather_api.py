"""Korea Meteorological Administration (KMA) ASOS daily-weather client.

Fetches daily surface observations (temperature, humidity, precipitation, wind
speed, sunshine duration) for a given station and caches them to CSV. These
feed the weather-amenity score and the TFT weather covariates.

The API key is read from the ``KMA_SERVICE_KEY`` environment variable — never
hard-code it. Get one at https://www.data.go.kr .

Consolidates the original ``weatherAPI.py``, ``weatherReturn.py`` and
``weather_csv.py`` scripts.
"""
from __future__ import annotations

import csv
import datetime
import os
import time
from pathlib import Path

import requests

from ..features.weather_score import calculate_weather_score

ASOS_DAILY_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
SEOUL_STATION_ID = 108  # KMA ASOS station number for Seoul


def _service_key() -> str:
    key = os.environ.get("KMA_SERVICE_KEY")
    if not key:
        raise RuntimeError(
            "KMA_SERVICE_KEY is not set. Copy .env.example to .env and add your "
            "data.go.kr service key (see README)."
        )
    return key


def get_historical_daily(station_id: int, date: str, service_key: str | None = None) -> dict:
    """Return one day of ASOS observations.

    Args:
        station_id: KMA ASOS station number (e.g. 108 for Seoul).
        date: Query date as ``YYYYMMDD`` (previous day at the latest).
        service_key: Optional override; defaults to ``KMA_SERVICE_KEY`` env var.

    Returns:
        Dict with keys ``date, T, RH, P, W, S`` (temperature C, relative
        humidity %, precipitation mm, wind speed m/s, sunshine hours).
    """
    service_key = service_key or _service_key()
    params = {
        "ServiceKey": service_key,
        "pageNo": "1",
        "numOfRows": "10",
        "dataType": "JSON",
        "dataCd": "ASOS",   # ASOS: surface synoptic observation
        "dateCd": "DAY",    # daily records
        "startDt": date,
        "endDt": date,
        "stnIds": str(station_id),
    }
    resp = requests.get(ASOS_DAILY_URL, params=params)
    resp.raise_for_status()
    items = resp.json()["response"]["body"]["items"]["item"]
    if not items:
        raise ValueError(f"No weather data for {date}")
    d = items[0]

    # Missing precipitation / wind fields come back as empty strings.
    precipitation = float(d["sumRn"]) if d.get("sumRn") not in ("", None) else 0.0
    wind_speed = float(d["avgWs"]) if d.get("avgWs") not in ("", None) else 2.0

    return {
        "date": date,
        "T": float(d.get("avgTa", 0)),
        "RH": float(d.get("avgRhm", 0)),
        "P": precipitation,
        "W": wind_speed,
        "S": float(d.get("ssDur", 0)),
    }


def weather_score(query_date: str, station_id: int = SEOUL_STATION_ID) -> float:
    """Fetch a day's observations and return its weather-amenity score (0-1)."""
    obs = get_historical_daily(station_id, query_date)
    return calculate_weather_score(obs["T"], obs["RH"], obs["P"], obs["W"], obs["S"])


def save_weather_csv(
    start_date: str,
    end_date: str,
    station_id: int,
    filename: str | Path,
    service_key: str | None = None,
    retry_wait: float = 5.0,
    pause: float = 0.2,
) -> None:
    """Download a date range of daily observations to CSV, retrying on failure.

    Args:
        start_date, end_date: ``YYYY-MM-DD`` inclusive bounds.
        station_id: KMA ASOS station number.
        filename: Output CSV path (columns ``date,T,RH,P,W,S``).
    """
    service_key = service_key or _service_key()
    sd = datetime.date.fromisoformat(start_date)
    ed = datetime.date.fromisoformat(end_date)
    dates = [
        (sd + datetime.timedelta(days=i)).strftime("%Y%m%d")
        for i in range((ed - sd).days + 1)
    ]

    fieldnames = ["date", "T", "RH", "P", "W", "S"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dt in dates:
            while True:
                try:
                    row = get_historical_daily(station_id, dt, service_key)
                    break
                except Exception as exc:  # noqa: BLE001 - transient API errors
                    print(f"[{dt}] error: {exc} - retrying in {retry_wait}s")
                    time.sleep(retry_wait)
            writer.writerow(row)
            time.sleep(pause)  # gentle pause between calls


def get_weather_score_from_csv(date: str, csv_file: str | Path) -> float:
    """Look up a cached day (``YYYYMMDD``) in a weather CSV and score it."""
    with open(csv_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["date"] == date:
                return calculate_weather_score(
                    float(row["T"]), float(row["RH"]), float(row["P"]),
                    float(row["W"]), float(row["S"]),
                )
    raise KeyError(f"No data for date {date} in {csv_file}")
