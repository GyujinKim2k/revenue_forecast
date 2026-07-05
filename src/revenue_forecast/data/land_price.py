"""MOLIT V-World land-price client.

Resolves parcels (PNU) within a bounding box via the V-World cadastral (WFS)
service, then looks up each parcel's officially published individual land price.
Used to attach a spot-level land-value feature (later found to have low / even
negative feature importance, and excluded from the final model).

The API key is read from the ``VWORLD_API_KEY`` environment variable. Get one
at https://www.vworld.kr .

Converted from ``landPrice.ipynb``.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import requests

LAND_PRICE_URL = "http://api.vworld.kr/ned/data/getIndvdLandPriceAttr"
SVC_LIST_URL = "https://www.vworld.kr/dtna/dtna_apiSvcList_s001.do"
WFS_URL = "https://www.vworld.kr/dtna/dtna_apiSvcFc_s001.do"


def _api_key() -> str:
    key = os.environ.get("VWORLD_API_KEY")
    if not key:
        raise RuntimeError(
            "VWORLD_API_KEY is not set. Copy .env.example to .env and add your "
            "V-World API key (see README)."
        )
    return key


def get_individual_land_price(pnu: str, year: str = "2023", domain: str = "") -> int | None:
    """Published individual land price (KRW/m^2) for a single parcel PNU."""
    params = {"key": _api_key(), "domain": domain, "pnu": pnu,
              "stdrYear": year, "format": "xml"}
    resp = requests.get(LAND_PRICE_URL, params=params)
    resp.raise_for_status()
    val = ET.fromstring(resp.text).findtext(".//pblntfPclnd")
    return int(val) if val else None


def _resolve_parcel_api_num(domain: str = "") -> str:
    """Look up the V-World cadastral-map (지적도) service's apiNum."""
    params = {"key": _api_key(), "domain": domain, "format": "xml",
              "searchKeyword": "지적도", "pageIndex": 1, "perPage": 100}
    resp = requests.get(SVC_LIST_URL, params=params)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    for item in root.findall(".//item"):
        svc_nm = item.findtext("svcNm")
        if svc_nm and "지적" in svc_nm:  # "cadastral" in Korean
            api_num = item.findtext("apiNum")
            if api_num:
                return api_num
    raise RuntimeError("Failed to resolve the cadastral-map service apiNum")


def parcels_in_bbox(lat: float, lon: float, delta: float = 0.001, domain: str = "") -> list[str]:
    """Return the PNUs of all parcels within a lat/lon bounding box."""
    api_num = _resolve_parcel_api_num(domain)
    bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"
    params = {"key": _api_key(), "domain": domain, "apiNum": api_num,
              "format": "xml", "crs": "EPSG:4326", "BBOX": bbox}
    resp = requests.get(WFS_URL, params=params)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    return [el.text for el in root.findall(".//pnu") if el.text]


def average_land_price(lat: float, lon: float, delta: float = 0.001,
                       year: str = "2023", domain: str = "") -> float | None:
    """Average published land price (KRW/m^2) of parcels around a spot."""
    prices = [
        p for pnu in parcels_in_bbox(lat, lon, delta, domain)
        if (p := get_individual_land_price(pnu, year, domain)) is not None
    ]
    return sum(prices) / len(prices) if prices else None


if __name__ == "__main__":
    # Seoul City Hall vicinity.
    print(average_land_price(37.571, 126.976))
