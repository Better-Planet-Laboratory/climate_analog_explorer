from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk

BASEMAP_GEOJSON = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)


def _distance_colors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    d = out["climate_distance"].to_numpy(dtype=float)
    finite = np.isfinite(d)
    out["r"], out["g"], out["b"] = 100, 100, 100
    if not finite.any():
        return out

    lo = float(np.nanpercentile(d[finite], 2))
    hi = float(np.nanpercentile(d[finite], 98))
    hi = max(hi, lo + 1e-6)
    t = np.clip((d - lo) / (hi - lo), 0.0, 1.0)

    # Bright plasma-like ramp designed to pop against the dark globe.
    stops = np.asarray([
        [13, 8, 135],
        [126, 3, 168],
        [203, 71, 119],
        [248, 149, 64],
        [240, 249, 33],
    ], dtype=float)

    pos = t * (len(stops) - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.clip(i0 + 1, 0, len(stops) - 1)
    w = pos - i0
    rgb = (stops[i0] * (1-w[:, None]) + stops[i1] * w[:, None]).astype(np.int16)

    out["r"], out["g"], out["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    return out


def build_globe(selected_row, analogs, distance_points, selected_analog_rank=1,
                show_distance=True, show_arcs=True, map_zoom=1.0):
    layers = []

    earth = [[-180,90],[0,90],[180,90],[180,-90],[0,-90],[-180,-90]]
    layers.append(pdk.Layer(
        "SolidPolygonLayer", id="ocean",
        data=[{"polygon": earth}], get_polygon="polygon",
        get_fill_color=[4, 8, 16, 255], stroked=False, filled=True, pickable=False
    ))

    layers.append(pdk.Layer(
        "GeoJsonLayer", id="basemap-countries", data=BASEMAP_GEOJSON,
        filled=True, stroked=True,
        get_fill_color=[18, 25, 31, 255],
        get_line_color=[72, 89, 100, 220],
        line_width_min_pixels=0.55, pickable=False
    ))

    if show_distance and distance_points is not None and len(distance_points):
        dots = _distance_colors(distance_points)
        layers.append(pdk.Layer(
            "ScatterplotLayer", id="climate-cells", data=dots,
            get_position="[center_lon, center_lat]",
            get_fill_color="[r, g, b, 235]",
            get_radius=29000, radius_min_pixels=2.2, radius_max_pixels=7,
            stroked=False, pickable=True, auto_highlight=True,
            highlight_color=[255,255,255,255]
        ))

    q = pd.DataFrame([{
        "center_lat": float(selected_row["center_lat"]),
        "center_lon": float(selected_row["center_lon"])
    }])
    layers.append(pdk.Layer(
        "ScatterplotLayer", id="selected-cell", data=q,
        get_position="[center_lon, center_lat]", get_radius=80000,
        radius_min_pixels=8, radius_max_pixels=15,
        get_fill_color=[255,255,255,255], get_line_color=[255,210,0,255],
        line_width_min_pixels=3, stroked=True, pickable=False
    ))

    markers = analogs.head(10).copy()
    markers["selected"] = markers["rank"].astype(int) == int(selected_analog_rank)
    markers["radius"] = np.where(markers["selected"], 90000, 68000)
    markers["mr"] = np.where(markers["selected"], 255, 255)
    markers["mg"] = np.where(markers["selected"], 210, 75)
    markers["mb"] = np.where(markers["selected"], 0, 75)

    layers.append(pdk.Layer(
        "ScatterplotLayer", id="analog-markers", data=markers,
        get_position="[center_lon, center_lat]", get_radius="radius",
        radius_min_pixels=6, radius_max_pixels=14,
        get_fill_color="[mr, mg, mb, 255]",
        get_line_color=[255,255,255,245], line_width_min_pixels=1.5,
        stroked=True, pickable=True, auto_highlight=True
    ))

    if show_arcs and len(markers):
        arcs = markers.copy()
        arcs["source_lon"] = float(selected_row["center_lon"])
        arcs["source_lat"] = float(selected_row["center_lat"])
        layers.append(pdk.Layer(
            "GreatCircleLayer", id="analog-arcs", data=arcs,
            get_source_position="[source_lon, source_lat]",
            get_target_position="[center_lon, center_lat]",
            # Solid cyan for every arc, source and target identical.
            get_source_color=[0, 220, 255, 245],
            get_target_color=[0, 220, 255, 245],
            get_width=4, width_min_pixels=2,
            pickable=False, parameters={"cullMode": "none"}
        ))

    view = pdk.View(type="_GlobeView", controller=True, id="globe", resolution=5)
    state = pdk.ViewState(
        latitude=float(selected_row["center_lat"]),
        longitude=float(selected_row["center_lon"]),
        zoom=float(map_zoom), pitch=0, bearing=0
    )

    tooltip = {
        "html": (
            "<b>{nearest_city}</b> {admin1} {country}"
            "<br/>Exact climate distance: {climate_distance}"
            "<br/>Geographic distance: {geographic_distance_km} km"
            "<br/>Lat/Lon: {center_lat}, {center_lon}"
        )
    }

    return pdk.Deck(
        layers=layers, views=[view], initial_view_state=state,
        map_provider=None, map_style=None, tooltip=tooltip,
        parameters={"cull": True}
    )
