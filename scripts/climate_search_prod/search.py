from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .common import exact_climate_distance, haversine_km
from .geography import reverse_geocode


BIO_NAMES = [f"BIO{i}" for i in range(1, 20)]


def h5_take(dataset, ids):
    """Read arbitrary HDF5 rows while preserving requested order."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size == 0:
        return np.empty((0,) + dataset.shape[1:], dtype=dataset.dtype)

    sort_order = np.argsort(ids, kind="stable")
    sorted_ids = ids[sort_order]
    unique_ids, inverse = np.unique(sorted_ids, return_inverse=True)
    unique_values = dataset[unique_ids]
    sorted_values = unique_values[inverse]

    restore = np.empty_like(sort_order)
    restore[sort_order] = np.arange(sort_order.size)
    return sorted_values[restore]


def _to_xyz(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def _chord_to_km(chord):
    angle = 2.0 * np.arcsin(np.clip(float(chord) / 2.0, 0.0, 1.0))
    return float(6371.0088 * angle)


class ClimateSearchEngine:
    """
    Production climate-analog backend.

    Pipeline:
      lat/lon -> nearest chip
      -> 64-D candidate retrieval
      -> geographic exclusion
      -> exact corrected shared-valid climate reranking
      -> spatial deduplication
      -> offline place labels + climate statistics
    """

    def __init__(
        self,
        query_database,
        target_database,
        target_index_json,
        query_chip_h5=None,
        target_chip_h5=None,
    ):
        self.query_database = Path(query_database)
        self.target_database = Path(target_database)
        self.target_index_json = Path(target_index_json)

        self.qdb = h5py.File(self.query_database, "r")
        self.tdb = h5py.File(self.target_database, "r")

        self.qchip = h5py.File(query_chip_h5, "r") if query_chip_h5 else None
        self.tchip = h5py.File(target_chip_h5, "r") if target_chip_h5 else None

        self.meta = json.loads(self.target_index_json.read_text())
        self.backend = self.meta["backend"]

        if self.backend == "faiss":
            import faiss
            self.index = faiss.read_index(self.meta["index"])
        elif self.backend == "numpy":
            self.index = np.load(self.meta["index"], mmap_mode="r")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Nearest-chip lookup on a unit sphere: safe at poles/dateline.
        qlat = np.asarray(self.qdb["center_lat"][:], dtype=np.float64)
        qlon = np.asarray(self.qdb["center_lon"][:], dtype=np.float64)
        self.query_spatial = cKDTree(_to_xyz(qlat, qlon))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        for name in ("qdb", "tdb", "qchip", "tchip"):
            obj = getattr(self, name, None)
            if obj is not None:
                try:
                    obj.close()
                finally:
                    setattr(self, name, None)

    def nearest_query_row(self, lat, lon):
        xyz = _to_xyz([lat], [lon])[0]
        chord, row = self.query_spatial.query(xyz, k=1)
        return int(row), _chord_to_km(chord)

    def _retrieve(self, qz, n):
        qz = np.asarray(qz, dtype=np.float32).reshape(1, -1)

        if self.backend == "faiss":
            n = min(int(n), int(self.index.ntotal))
            scores, ids = self.index.search(qz, n)
            ids, scores = ids[0], scores[0]
            good = ids >= 0
            return ids[good].astype(np.int64), scores[good].astype(np.float32)

        scores = np.asarray(self.index @ qz[0], dtype=np.float32)
        n = min(int(n), len(scores))
        if n <= 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)

        if n == len(scores):
            ids = np.argsort(-scores)
        else:
            ids = np.argpartition(-scores, n - 1)[:n]
            ids = ids[np.argsort(-scores[ids])]

        return ids.astype(np.int64), scores[ids]

    @staticmethod
    def _normalization(search_db, chip_db):
        # Prefer copied arrays in search DB if present.
        if "historical_mean" in search_db and "historical_std" in search_db:
            return (
                np.asarray(search_db["historical_mean"][:], np.float32),
                np.asarray(search_db["historical_std"][:], np.float32),
            )
        if chip_db is not None and "historical_mean" in chip_db and "historical_std" in chip_db:
            return (
                np.asarray(chip_db["historical_mean"][:], np.float32),
                np.asarray(chip_db["historical_std"][:], np.float32),
            )
        return None, None

    @classmethod
    def _climate_stats(cls, db, chip_db, row):
        """
        bio_mean/bio_std in the search DB are computed from normalized chip values.

        Return:
          - mean_z: chip mean in historical-SD units
          - spatial_sd_z: within-chip spatial SD in historical-SD units
          - mean_native: reconstructed original WorldClim raster values, when
            historical normalization arrays are available
          - spatial_sd_native: same for within-chip SD
        """
        mean_z = np.asarray(db["bio_mean"][row], dtype=np.float32)
        sd_z = np.asarray(db["bio_std"][row], dtype=np.float32)

        out = {
            "mean_z": {BIO_NAMES[i]: float(mean_z[i]) for i in range(19)},
            "spatial_sd_z": {BIO_NAMES[i]: float(sd_z[i]) for i in range(19)},
        }

        mu, sigma = cls._normalization(db, chip_db)
        if mu is not None and sigma is not None:
            mean_native = mean_z * sigma + mu
            sd_native = sd_z * sigma
            out["mean_native"] = {
                BIO_NAMES[i]: float(mean_native[i]) for i in range(19)
            }
            out["spatial_sd_native"] = {
                BIO_NAMES[i]: float(sd_native[i]) for i in range(19)
            }
        return out

    @staticmethod
    def _place(lat, lon):
        # Reuse the app's offline reverse_geocoder + pycountry approach.
        return reverse_geocode(float(lat), float(lon))

    def search_latlon(self, lat, lon, **kwargs):
        row, snap_km = self.nearest_query_row(lat, lon)
        out = self.search(row, **kwargs)
        out["query"]["requested_lat"] = float(lat)
        out["query"]["requested_lon"] = float(lon)
        out["query"]["click_to_chip_center_km"] = float(snap_km)
        return out

    def search(
        self,
        query_row,
        n_candidates=250,
        n_results=10,
        geo_exclusion_km=0.0,
        min_shared_cells=8,
        oversample_factor=8,
        min_analog_separation_km=150.0,
        include_places=True,
        include_climate_stats=True,
        adaptive_candidates=True,
        max_candidates=5000,
    ):
        """
        Search for exact-reranked, spatially distinct climate analogs.

        Candidate retrieval begins at n_candidates. If spatial deduplication
        leaves fewer than n_results, the embedding candidate pool is expanded
        geometrically up to max_candidates.
        """
        query_row = int(query_row)
        n_candidates = int(n_candidates)
        n_results = int(n_results)
        max_candidates = int(max_candidates)

        n_query = len(self.qdb["chip_id"])
        n_target = len(self.tdb["chip_id"])
        if not 0 <= query_row < n_query:
            raise IndexError(f"query_row={query_row} outside 0..{n_query - 1}")

        qlat = float(self.qdb["center_lat"][query_row])
        qlon = float(self.qdb["center_lon"][query_row])
        qid = int(self.qdb["chip_id"][query_row])
        qz = np.asarray(self.qdb["embedding"][query_row], np.float32)
        qclim = np.asarray(self.qdb["coarse_climate"][query_row], np.float32)
        qvalid = np.asarray(self.qdb["coarse_valid"][query_row], bool)

        query_period = str(self.qdb.attrs.get("period", ""))
        target_period = str(self.tdb.attrs.get("period", ""))
        same_period = query_period == target_period

        pool_size = max(n_candidates, n_results)
        pool_size = min(pool_size, n_target)
        max_candidates = min(max(max_candidates, pool_size), n_target)

        final = None

        while True:
            # Oversample embedding retrieval before geo filtering.
            request = min(
                n_target,
                max(pool_size, pool_size * int(oversample_factor)),
            )
            ids, scores = self._retrieve(qz, request)
            if ids.size == 0:
                raise RuntimeError("Embedding search returned no candidates.")

            lat = np.asarray(h5_take(self.tdb["center_lat"], ids), np.float64)
            lon = np.asarray(h5_take(self.tdb["center_lon"], ids), np.float64)
            geo = np.asarray(haversine_km(qlat, qlon, lat, lon), np.float64)
            keep = geo >= float(geo_exclusion_km)

            if same_period:
                keep &= h5_take(self.tdb["chip_id"], ids) != qid

            ids, scores, geo = ids[keep], scores[keep], geo[keep]

            # If geographic filtering removed too much, exact full embedding scan.
            if len(ids) < pool_size:
                allz = np.asarray(self.tdb["embedding"][:], np.float32)
                allscores = allz @ qz
                all_lat = np.asarray(self.tdb["center_lat"][:], np.float64)
                all_lon = np.asarray(self.tdb["center_lon"][:], np.float64)
                allgeo = np.asarray(
                    haversine_km(qlat, qlon, all_lat, all_lon), np.float64
                )

                allowed = allgeo >= float(geo_exclusion_km)
                if same_period:
                    allowed &= np.asarray(self.tdb["chip_id"][:]) != qid

                allowed_ids = np.flatnonzero(allowed)
                if allowed_ids.size == 0:
                    raise RuntimeError(
                        "No candidates remain after geographic filtering."
                    )

                k = min(pool_size, allowed_ids.size)
                ps = allscores[allowed_ids]
                if k == allowed_ids.size:
                    local = np.argsort(-ps)
                else:
                    local = np.argpartition(-ps, k - 1)[:k]
                    local = local[np.argsort(-ps[local])]

                ids = allowed_ids[local].astype(np.int64)
                scores = allscores[ids].astype(np.float32)
                geo = allgeo[ids]
            else:
                ids = ids[:pool_size]
                scores = scores[:pool_size]
                geo = geo[:pool_size]

            candidate_climate = np.asarray(
                h5_take(self.tdb["coarse_climate"], ids), np.float32
            )
            candidate_valid = np.asarray(
                h5_take(self.tdb["coarse_valid"], ids), bool
            )

            climate_distance = np.asarray(
                exact_climate_distance(
                    qclim,
                    qvalid,
                    candidate_climate,
                    candidate_valid,
                    min_shared_cells=min_shared_cells,
                ),
                np.float32,
            )

            finite_pos = np.flatnonzero(np.isfinite(climate_distance))
            if finite_pos.size == 0:
                raise RuntimeError(
                    "No candidate has sufficient shared-valid climate cells."
                )
            exact_order = finite_pos[np.argsort(climate_distance[finite_pos])]

            selected_positions = []
            selected_lat = []
            selected_lon = []
            sep_km = float(min_analog_separation_km)

            for pos in exact_order:
                r = int(ids[pos])
                rlat = float(self.tdb["center_lat"][r])
                rlon = float(self.tdb["center_lon"][r])

                if sep_km > 0 and selected_positions:
                    separation = np.asarray(
                        haversine_km(
                            rlat,
                            rlon,
                            np.asarray(selected_lat),
                            np.asarray(selected_lon),
                        )
                    )
                    if np.any(separation < sep_km):
                        continue

                selected_positions.append(int(pos))
                selected_lat.append(rlat)
                selected_lon.append(rlon)
                if len(selected_positions) >= n_results:
                    break

            final = (
                ids, scores, geo, climate_distance,
                np.asarray(selected_positions, dtype=np.int64),
                pool_size,
            )

            if len(selected_positions) >= n_results:
                break
            if not adaptive_candidates or pool_size >= max_candidates:
                break

            next_size = min(max_candidates, max(pool_size * 2, pool_size + 1))
            if next_size == pool_size:
                break
            pool_size = next_size

        ids, scores, geo, climate_distance, order, used_pool_size = final

        if order.size == 0:
            raise RuntimeError("No analogs remain after spatial deduplication.")

        rid = ids[order]
        chip_ids = h5_take(self.tdb["chip_id"], rid)
        rlats = h5_take(self.tdb["center_lat"], rid)
        rlons = h5_take(self.tdb["center_lon"], rid)

        query = {
            "row": query_row,
            "chip_id": qid,
            "lat": qlat,
            "lon": qlon,
            "period": query_period,
        }
        if include_places:
            query.update(self._place(qlat, qlon))
        if include_climate_stats:
            query["climate"] = self._climate_stats(
                self.qdb, self.qchip, query_row
            )

        results = []
        qmean = np.asarray(self.qdb["bio_mean"][query_row], np.float32)

        for j, pos in enumerate(order):
            r = int(rid[j])
            item = {
                "rank": j + 1,
                "row": r,
                "chip_id": int(chip_ids[j]),
                "lat": float(rlats[j]),
                "lon": float(rlons[j]),
                "period": target_period,
                "embedding_similarity": float(scores[pos]),
                "embedding_distance": float(1.0 - scores[pos]),
                "climate_distance": float(climate_distance[pos]),
                "geographic_distance_km": float(geo[pos]),
            }

            if include_places:
                item.update(self._place(item["lat"], item["lon"]))

            if include_climate_stats:
                item["climate"] = self._climate_stats(
                    self.tdb, self.tchip, r
                )
                tmean = np.asarray(self.tdb["bio_mean"][r], np.float32)
                delta_z = qmean - tmean
                item["climate"]["query_minus_analog_mean_z"] = {
                    BIO_NAMES[i]: float(delta_z[i]) for i in range(19)
                }

                qmu, qsig = self._normalization(self.qdb, self.qchip)
                tmu, tsig = self._normalization(self.tdb, self.tchip)
                if qmu is not None and tmu is not None:
                    qnative = qmean * qsig + qmu
                    tnative = tmean * tsig + tmu
                    delta_native = qnative - tnative
                    item["climate"]["query_minus_analog_mean_native"] = {
                        BIO_NAMES[i]: float(delta_native[i])
                        for i in range(19)
                    }

            results.append(item)

        return {
            "query": query,
            "search": {
                "backend": self.backend,
                "initial_n_candidates": int(n_candidates),
                "n_candidates_used": int(used_pool_size),
                "max_candidates": int(max_candidates),
                "adaptive_candidates": bool(adaptive_candidates),
                "n_results": int(len(results)),
                "requested_n_results": int(n_results),
                "geo_exclusion_km": float(geo_exclusion_km),
                "min_analog_separation_km": float(min_analog_separation_km),
                "min_shared_cells": int(min_shared_cells),
                "ranking": "exact_corrected_shared_valid_bioclim_distance",
            },
            "results": results,
        }
