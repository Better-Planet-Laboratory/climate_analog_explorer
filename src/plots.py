from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

BIOCLIM_DESCRIPTIONS = {
    "BIO1": "Annual Mean Temperature",
    "BIO2": "Mean Diurnal Range",
    "BIO3": "Isothermality",
    "BIO4": "Temperature Seasonality",
    "BIO5": "Max Temperature of Warmest Month",
    "BIO6": "Min Temperature of Coldest Month",
    "BIO7": "Temperature Annual Range",
    "BIO8": "Mean Temperature of Wettest Quarter",
    "BIO9": "Mean Temperature of Driest Quarter",
    "BIO10": "Mean Temperature of Warmest Quarter",
    "BIO11": "Mean Temperature of Coldest Quarter",
    "BIO12": "Annual Precipitation",
    "BIO13": "Precipitation of Wettest Month",
    "BIO14": "Precipitation of Driest Month",
    "BIO15": "Precipitation Seasonality",
    "BIO16": "Precipitation of Wettest Quarter",
    "BIO17": "Precipitation of Driest Quarter",
    "BIO18": "Precipitation of Warmest Quarter",
    "BIO19": "Precipitation of Coldest Quarter",
}
BIO_NAMES = list(BIOCLIM_DESCRIPTIONS)

# Historical normalization statistics used when the climate search chips were built.
BIO_MEAN = np.asarray([
    -4.4446, 10.0962, 34.3702, 891.2146, 13.7880,
    -20.4026, 34.1906, -1.3611, -5.7450, 6.9417,
    -14.4213, 532.1405, 91.4521, 14.4057, 75.8218,
    235.6537, 52.3023, 152.5537, 104.1198,
], dtype=float)

BIO_STD = np.asarray([
    24.7060, 3.1112, 18.7106, 467.2949, 21.7078,
    26.0162, 12.0030, 29.1506, 22.7739, 21.1986,
    26.7963, 630.0364, 102.1227, 27.1151, 44.0509,
    274.2749, 90.5330, 183.2731, 181.2381,
], dtype=float)

BIOCLIM_UNITS = {
    "BIO1": "°C", "BIO2": "°C", "BIO3": "%", "BIO4": "SD × 100",
    "BIO5": "°C", "BIO6": "°C", "BIO7": "°C", "BIO8": "°C",
    "BIO9": "°C", "BIO10": "°C", "BIO11": "°C",
    "BIO12": "mm", "BIO13": "mm", "BIO14": "mm", "BIO15": "%",
    "BIO16": "mm", "BIO17": "mm", "BIO18": "mm", "BIO19": "mm",
}


def comparison_from_bio_vectors(query_bio, analog_bio, variable_names=None):
    """
    bio_mean in the search DB is in the standardized model coordinate system.
    Convert query and analog back to native BIOCLIM units for display, while
    retaining the standardized difference for the comparison bar chart.
    """
    qz = np.asarray(query_bio, dtype=float).reshape(-1)
    az = np.asarray(analog_bio, dtype=float).reshape(-1)
    n = min(len(qz), len(az), 19)
    names = list(variable_names or BIO_NAMES)[:n]

    native_q = qz[:n] * BIO_STD[:n] + BIO_MEAN[:n]
    native_a = az[:n] * BIO_STD[:n] + BIO_MEAN[:n]

    return pd.DataFrame({
        "Variable": names,
        "Description": [BIOCLIM_DESCRIPTIONS.get(x, x) for x in names],
        "Unit": [BIOCLIM_UNITS.get(x, "") for x in names],
        "Selected": native_q,
        "Analog": native_a,
        "Difference": native_q - native_a,
        "Standardized difference": qz[:n] - az[:n],
    })


def sidebar_climate_difference_plot(diff: pd.DataFrame):
    d = diff.copy()
    custom = np.column_stack([
        d["Description"].astype(str),
        d["Unit"].astype(str),
        d["Selected"].astype(float),
        d["Analog"].astype(float),
        d["Difference"].astype(float),
    ])

    fig = go.Figure(go.Bar(
        x=d["Variable"],
        y=d["Standardized difference"],
        customdata=custom,
        hovertemplate=(
            "<b>%{x}: %{customdata[0]}</b><br>"
            "Standardized difference: %{y:.3f}<br>"
            "Selected: %{customdata[2]:.3f} %{customdata[1]}<br>"
            "Analog #1: %{customdata[3]:.3f} %{customdata[1]}<br>"
            "Native difference: %{customdata[4]:.3f} %{customdata[1]}"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_width=1)
    fig.update_layout(
        height=285, margin=dict(l=8, r=8, t=8, b=42),
        xaxis_title=None, yaxis_title="Std. difference",
        bargap=0.13, showlegend=False,
    )
    fig.update_xaxes(tickangle=-65, tickfont=dict(size=8))
    fig.update_yaxes(tickformat=".3f")
    return fig


def novelty_percentile(novelty_df: pd.DataFrame, selected_novelty: float) -> float:
    values = novelty_df["novelty"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0 or not np.isfinite(selected_novelty):
        return np.nan
    return 100.0 * np.mean(values <= float(selected_novelty))


def novelty_distribution_plot(novelty_df: pd.DataFrame, selected_novelty: float):
    """
    Percentile/CDF-style novelty display.

    x = percentile of the reference distribution
    y = exact novelty distance

    This makes a strongly right-skewed novelty distribution much easier to
    read than a conventional histogram while retaining the actual novelty
    distance on the y axis.
    """
    values = novelty_df["novelty"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    values.sort()

    if len(values) == 0:
        return go.Figure()

    # Downsample only for rendering; quantiles are calculated from all cells.
    n_plot = min(2500, len(values))
    probs = np.linspace(0.0, 1.0, n_plot)
    idx = np.clip(
        np.round(probs * (len(values) - 1)).astype(int),
        0,
        len(values) - 1,
    )
    y = values[idx]
    x = probs * 100.0

    pct = novelty_percentile(novelty_df, selected_novelty)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Global distribution",
        hovertemplate=(
            "Percentile: %{x:.1f}<br>"
            "Exact novelty: %{y:.3f}"
            "<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=[pct],
        y=[selected_novelty],
        mode="markers",
        name="Selected",
        marker={"size": 11},
        hovertemplate=(
            "<b>Selected location</b><br>"
            f"Novelty percentile: {pct:.1f}<br>"
            f"Exact novelty: {selected_novelty:.3f}"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        height=235,
        margin=dict(l=8, r=8, t=10, b=38),
        xaxis_title="Climate novelty percentile",
        yaxis_title="Exact novelty",
        showlegend=False,
    )
    fig.update_xaxes(
        range=[0, 100],
        tickvals=[0, 25, 50, 75, 100],
        ticksuffix="%",
    )
    fig.update_yaxes(tickformat=".3f")
    return fig
