from __future__ import annotations

from functools import lru_cache
import logging

import requests

from config import GEOCODER_USER_AGENT


LOGGER = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def _user_agent() -> str:
    """
    Nominatim requires an identifying User-Agent.

    If the configured value is missing or too generic, use a descriptive
    fallback for the Climate Analog Explorer.
    """
    value = str(GEOCODER_USER_AGENT or "").strip()

    generic = {
        "",
        "python",
        "python-requests",
        "streamlit",
        "mozilla/5.0",
    }

    if value.lower() in generic:
        return (
            "BetterPlanetLab-ClimateAnalogExplorer/1.0 "
            "(https://github.com/Better-Planet-Laboratory/"
            "climate_analog_explorer)"
        )

    return value


@lru_cache(maxsize=512)
def geocode_place(query: str):
    """
    Geocode a place name using OpenStreetMap Nominatim.

    Returns
    -------
    dict or None
        {
            "lat": float,
            "lon": float,
            "display_name": str
        }

        None is returned only when Nominatim successfully responds but
        contains no matching locations, or when the request fails.

    Notes
    -----
    Errors are logged rather than silently swallowed so failures on
    Streamlit Community Cloud can be diagnosed from the deployment logs.
    """

    query = " ".join(str(query).strip().split())

    if not query:
        return None

    params = {
        "q": query,
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
    }

    headers = {
        "User-Agent": _user_agent(),
        "Accept": "application/json",
        "Accept-Language": "en",
    }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=15,
        )

        LOGGER.info(
            "Nominatim geocode query=%r status=%s",
            query,
            response.status_code,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:
        LOGGER.exception(
            "Nominatim request timed out for query=%r",
            query,
        )
        return None

    except requests.exceptions.HTTPError:
        LOGGER.exception(
            "Nominatim HTTP error for query=%r",
            query,
        )
        return None

    except requests.exceptions.RequestException:
        LOGGER.exception(
            "Nominatim request failed for query=%r",
            query,
        )
        return None

    try:
        payload = response.json()

    except ValueError:
        LOGGER.error(
            "Nominatim returned invalid JSON for query=%r. "
            "Response prefix=%r",
            query,
            response.text[:300],
        )
        return None

    if not payload:
        LOGGER.info(
            "Nominatim returned no matches for query=%r",
            query,
        )
        return None

    x = payload[0]

    try:
        lat = float(x["lat"])
        lon = float(x["lon"])

    except (KeyError, TypeError, ValueError):
        LOGGER.error(
            "Nominatim result missing valid coordinates "
            "for query=%r: %r",
            query,
            x,
        )
        return None

    display_name = str(
        x.get("display_name") or query
    ).strip()

    LOGGER.info(
        "Geocoded %r -> %.5f, %.5f (%s)",
        query,
        lat,
        lon,
        display_name,
    )

    return {
        "lat": lat,
        "lon": lon,
        "display_name": display_name,
    }