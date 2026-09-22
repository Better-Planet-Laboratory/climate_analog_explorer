from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SEARCH_DIR = PROJECT_ROOT / "data" / "search_database"
INDEX_DIR = SEARCH_DIR / "indexes"
CHIP_DIR = PROJECT_ROOT / "data" / "chips_64"
NOVELTY_DIR = SEARCH_DIR / "novelty"

HISTORICAL_LABEL = "Historical (1970–2000)"

PERIODS = {
    HISTORICAL_LABEL: {
        "key": "historical_1970_2000",
        "database": SEARCH_DIR / "historical_1970_2000_search.h5",
        "index_json": INDEX_DIR / "historical_1970_2000_index.json",
        "chip_h5": CHIP_DIR / "historical_1970_2000_chips.h5",
        "novelty": NOVELTY_DIR / "historical_1970_2000_novelty.parquet",
    },
    "Mid-century (2041–2060)": {
        "key": "ssp370_2041_2060",
        "database": SEARCH_DIR / "ssp370_2041_2060_search.h5",
        "index_json": INDEX_DIR / "ssp370_2041_2060_index.json",
        "chip_h5": CHIP_DIR / "ssp370_2041_2060_chips.h5",
        "novelty": NOVELTY_DIR / "ssp370_2041_2060_novelty.parquet",
    },
    "Late-century (2081–2100)": {
        "key": "ssp370_2081_2100",
        "database": SEARCH_DIR / "ssp370_2081_2100_search.h5",
        "index_json": INDEX_DIR / "ssp370_2081_2100_index.json",
        "chip_h5": CHIP_DIR / "ssp370_2081_2100_chips.h5",
        "novelty": NOVELTY_DIR / "ssp370_2081_2100_novelty.parquet",
    },
}

GCM_DATABASES = {
    "Mid-century (2041–2060)": SEARCH_DIR / "gcm" / "ssp370_2041_2060_gcm.h5",
    "Late-century (2081–2100)": SEARCH_DIR / "gcm" / "ssp370_2081_2100_gcm.h5",
}

TILE_SERVER_BROWSER_URL = "http://localhost:8765"
DEFAULT_K = 10
MAX_K = 20
DEFAULT_N_CANDIDATES = 250
MAX_ADAPTIVE_CANDIDATES = 5000
DEFAULT_GEO_EXCLUSION_KM = 1000
DEFAULT_ANALOG_SEPARATION_KM = 150
GEOCODER_USER_AGENT = "better-planet-lab-climate-analog-explorer"

# ============================================================
# Vector tile server
# ============================================================

TILE_SERVER_HOST = "127.0.0.1"
TILE_SERVER_PORT = 8765

# At this zoom and above, serve the full climate-cell grid.
# Lower zooms use the tile server's coarser representation.
MIN_FULL_GRID_ZOOM = 4