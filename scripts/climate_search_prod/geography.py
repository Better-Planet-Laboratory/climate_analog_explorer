from __future__ import annotations

from functools import lru_cache
import math

import pycountry
import reverse_geocoder as rg


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


@lru_cache(maxsize=100_000)
def reverse_geocode_rounded(lat_4: float, lon_4: float):
    """
    Offline nearest populated-place lookup using reverse_geocoder's
    GeoNames-derived database. This deliberately matches the existing app.
    """
    result = rg.search((lat_4, lon_4), mode=1)[0]

    cc = result.get("cc", "")
    country = cc
    if cc:
        obj = pycountry.countries.get(alpha_2=cc)
        if obj is not None:
            country = obj.name

    city = result.get("name", "") or "Unknown"
    admin1 = result.get("admin1", "") or ""
    city_lat = float(result["lat"])
    city_lon = float(result["lon"])

    return {
        "nearest_city": city,
        "admin1": admin1,
        "country_code": cc,
        "country": country or "Unknown",
        "city_distance_km": haversine_km(lat_4, lon_4, city_lat, city_lon),
    }


def reverse_geocode(lat: float, lon: float):
    return reverse_geocode_rounded(round(float(lat), 4), round(float(lon), 4))


def place_label(row) -> str:
    def get(key):
        if isinstance(row, dict):
            return row.get(key, "")
        return getattr(row, key, "") if hasattr(row, key) else row.get(key, "")

    parts = [get("nearest_city"), get("admin1"), get("country")]
    return ", ".join(str(p) for p in parts if p)
