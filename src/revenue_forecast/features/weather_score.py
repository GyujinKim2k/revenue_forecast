"""Weather-amenity score for carsharing demand.

Collapses five daily weather observations into a single 0-1 "how pleasant is it
to drive" score. Humidity and sunshine dominate the weighting, reflecting the
EDA finding that relative humidity correlates strongly with revenue while
same-day temperature does not.
"""
from __future__ import annotations

import numpy as np


def calculate_weather_score(
    temperature: float,
    humidity: float,
    precipitation: float,
    wind_speed: float,
    sunshine_hours: float,
    temp_opt: float = 20.0,
    temp_sigma: float = 5.0,
    weights: tuple[float, float, float, float, float] = (0.05, 0.45, 0.1, 0.1, 0.3),
) -> float:
    """Compute a weather score in ``[0, 1]``.

    Args:
        temperature: Average temperature (C).
        humidity: Average relative humidity (%).
        precipitation: Precipitation (mm).
        wind_speed: Average wind speed (m/s).
        sunshine_hours: Daily sunshine duration (hours).
        temp_opt: Optimal temperature for the Gaussian component.
        temp_sigma: Spread of the temperature Gaussian.
        weights: (temperature, humidity, precipitation, wind, sunshine) weights.
    """
    # Temperature: Gaussian bell peaking at temp_opt.
    temp_score = np.exp(-((temperature - temp_opt) ** 2) / (2 * temp_sigma ** 2))
    # Humidity: linear penalty as it rises.
    humidity_score = max(0.0, 1 - humidity / 100)
    # Precipitation: hyperbolic-tangent decay.
    precip_score = 1 - np.tanh(precipitation / 10)
    # Wind: linear penalty.
    wind_score = max(0.0, 1 - wind_speed / 20)
    # Sunshine: linear reward, saturating at 12h.
    sunshine_score = min(1.0, sunshine_hours / 12)

    w_temp, w_humid, w_precip, w_wind, w_sun = weights
    total = (
        w_temp * temp_score
        + w_humid * humidity_score
        + w_precip * precip_score
        + w_wind * wind_score
        + w_sun * sunshine_score
    )
    return float(max(0.0, min(1.0, total)))


if __name__ == "__main__":
    # Example: T=25C, RH=60%, P=5mm, W=3 m/s, S=8h
    print(f"Weather score: {calculate_weather_score(25.0, 60.0, 5.0, 3.0, 8.0):.2f} / 1")
