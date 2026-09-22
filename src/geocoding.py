from __future__ import annotations
from functools import lru_cache
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
from config import GEOCODER_USER_AGENT

@lru_cache(maxsize=512)
def geocode_place(query: str):
    query = " ".join(query.strip().split())
    if not query:
        return None
    params = urlencode({"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1})
    request = Request(
        f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": GEOCODER_USER_AGENT, "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    if not payload:
        return None
    x = payload[0]
    return {
        "lat": float(x["lat"]),
        "lon": float(x["lon"]),
        "display_name": x.get("display_name", query),
    }
