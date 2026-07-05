"""CLI: download KMA daily weather to CSV.

Usage:
    KMA_SERVICE_KEY=... python scripts/collect_weather.py \
        --start 2023-01-01 --end 2023-12-31 --station 108 --out data/external/weather/weather_108.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revenue_forecast.data.weather_api import SEOUL_STATION_ID, save_weather_csv  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--station", type=int, default=SEOUL_STATION_ID)
    p.add_argument("--out", required=True, help="output CSV path")
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_weather_csv(args.start, args.end, args.station, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
