from __future__ import annotations

import argparse
import calendar
import html
import json
import webbrowser
from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import requests

from permit_reconciliation import build_permit_rollup, build_public_reconciliation_points


STATE_FIPS = "06"
LA_COUNTY_FIPS = "037"
AREA_CRS = 3310  # California Albers
TIGER_YEAR = 2023
DEFAULT_MONTHS = 12
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Los Angeles supply-demand balance map from LADBS permits and ACS housing indicators."
    )
    parser.add_argument(
        "--months",
        type=int,
        default=DEFAULT_MONTHS,
        help="Trailing number of months to include when explicit dates are not provided. Default: 12.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "la_supply_demand_balance_los_angeles.html",
        help="Output HTML path.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the rendered map in your default browser after export.",
    )
    return parser.parse_args()


def resolve_output_path(path: Path) -> Path:
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def subtract_months(anchor: date, months: int) -> date:
    if months < 0:
        raise ValueError("months must be non-negative")

    year = anchor.year
    month = anchor.month - months
    while month <= 0:
        year -= 1
        month += 12

    day = min(anchor.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def resolve_date_range(args: argparse.Namespace) -> tuple[date, date]:
    if args.start_date or args.end_date:
        end_date = parse_iso_date(args.end_date) if args.end_date else date.today()
        start_date = parse_iso_date(args.start_date) if args.start_date else subtract_months(end_date, args.months)
    else:
        end_date = date.today()
        start_date = subtract_months(end_date, args.months)

    if start_date > end_date:
        raise ValueError("start date must be on or before end date")

    return start_date, end_date


def fetch_acs_market_metrics() -> pd.DataFrame:
    params = {
        "get": (
            "NAME,B19013_001E,B25001_001E,B25004_001E,B25064_001E,"
            "B25070_001E,B25070_007E,B25070_008E,B25070_009E,B25070_010E,B25070_011E"
        ),
        "for": "tract:*",
        "in": f"state:{STATE_FIPS} county:{LA_COUNTY_FIPS}",
    }

    response = requests.get("https://api.census.gov/data/2023/acs/acs5", params=params, timeout=120)
    response.raise_for_status()

    rows = response.json()
    frame = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns={"NAME": "acs_name"})
    numeric_columns = [
        "B19013_001E",
        "B25001_001E",
        "B25004_001E",
        "B25064_001E",
        "B25070_001E",
        "B25070_007E",
        "B25070_008E",
        "B25070_009E",
        "B25070_010E",
        "B25070_011E",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[frame[column] < 0, column] = pd.NA

    frame["tractce"] = frame["tract"]
    frame["median_household_income"] = frame["B19013_001E"]
    frame["housing_units"] = frame["B25001_001E"]
    frame["vacant_units"] = frame["B25004_001E"]
    frame["median_gross_rent"] = frame["B25064_001E"]
    frame["rent_burdened_households"] = frame[
        ["B25070_007E", "B25070_008E", "B25070_009E", "B25070_010E"]
    ].sum(axis=1)
    frame["renter_households_computed"] = frame["B25070_001E"] - frame["B25070_011E"]
    frame["vacancy_rate"] = frame["vacant_units"] / frame["housing_units"].replace({0: pd.NA})
    frame["rent_burden_share"] = frame["rent_burdened_households"] / frame["renter_households_computed"].replace(
        {0: pd.NA}
    )

    return frame[
        [
            "tractce",
            "acs_name",
            "median_household_income",
            "housing_units",
            "vacant_units",
            "median_gross_rent",
            "renter_households_computed",
            "rent_burdened_households",
            "vacancy_rate",
            "rent_burden_share",
        ]
    ]


def load_la_city_tracts() -> gpd.GeoDataFrame:
    tracts = gpd.read_file(
        f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/TRACT/tl_{TIGER_YEAR}_{STATE_FIPS}_tract.zip"
    )
    tracts = tracts[(tracts["STATEFP"] == STATE_FIPS) & (tracts["COUNTYFP"] == LA_COUNTY_FIPS)].copy()

    city = gpd.read_file(
        f"https://www2.census.gov/geo/tiger/TIGER{TIGER_YEAR}/PLACE/tl_{TIGER_YEAR}_{STATE_FIPS}_place.zip"
    )
    city = city.loc[city["NAME"] == "Los Angeles", ["geometry"]].to_crs(AREA_CRS)
    if city.empty:
        raise RuntimeError("Could not locate Los Angeles city boundary in TIGER place geometries.")

    tracts_area = tracts.to_crs(AREA_CRS).copy()
    centroids = tracts_area[["TRACTCE", "geometry"]].copy()
    centroids["geometry"] = centroids.centroid
    city_tract_ids = gpd.sjoin(centroids, city, predicate="within", how="inner")["TRACTCE"]
    return tracts.loc[tracts["TRACTCE"].isin(city_tract_ids)].copy()


def clipped_score(series: pd.Series, *, invert: bool = False, log_scale: bool = False) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float)
    if log_scale:
        clean = np.log1p(clean.clip(lower=0))

    fill_value = clean.dropna().median()
    if pd.isna(fill_value):
        fill_value = 0.0
    clean = clean.fillna(fill_value)

    low = clean.quantile(0.05)
    high = clean.quantile(0.95)
    clean = clean.clip(low, high)

    if invert:
        clean = -clean

    std = clean.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (clean - clean.mean()) / std


def balance_label(imbalance_score: float, demand_score: float, supply_score: float) -> str:
    if imbalance_score >= 1.0 and supply_score <= 0:
        return "High demand, weak supply response"
    if imbalance_score >= 0.35:
        return "Demand leading supply"
    if imbalance_score <= -1.0 and supply_score > 0:
        return "Supply response outpacing demand"
    if imbalance_score <= -0.35:
        return "Supply responding"
    if demand_score > 0.75 and supply_score > 0.75:
        return "High demand, high supply response"
    return "Mixed / roughly balanced"


def interpolate_diverging_color(value: float, min_value: float, max_value: float) -> list[int]:
    palette = [
        (41, 109, 176),
        (88, 194, 205),
        (108, 109, 120),
        (228, 184, 88),
        (228, 101, 68),
    ]

    if pd.isna(value):
        return [72, 74, 84, 110]

    if max_value <= min_value:
        r, g, b = palette[2]
        return [r, g, b, 210]

    clamped = min(max(value, min_value), max_value)
    scaled = (clamped - min_value) / (max_value - min_value)
    position = scaled * (len(palette) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(palette) - 1)
    blend = position - lower_index

    lower = palette[lower_index]
    upper = palette[upper_index]
    rgb = [
        round(lower[channel] + (upper[channel] - lower[channel]) * blend)
        for channel in range(3)
    ]
    return [rgb[0], rgb[1], rgb[2], 215]


def format_int(value: float) -> str:
    if pd.isna(value):
        return "No data"
    return f"{int(round(value)):,.0f}"


def format_signed_int(value: float) -> str:
    if pd.isna(value):
        return "No data"
    rounded = int(round(value))
    return f"{rounded:+,d}" if rounded < 0 else f"{rounded:,d}"


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "No data"
    return f"{value * 100:,.1f}%"


def prepare_map_frame(
    start_date: date,
    end_date: date,
) -> tuple[gpd.GeoDataFrame, dict[str, float | str], dict[str, pd.DataFrame]]:
    rollup = build_permit_rollup(start_date, end_date)
    permit_frame = rollup.tract_metrics
    point_layers = build_public_reconciliation_points(rollup)
    acs_frame = fetch_acs_market_metrics()
    tracts = load_la_city_tracts()

    merged = tracts.merge(permit_frame, left_on="TRACTCE", right_on="tractce", how="left")
    merged = merged.merge(acs_frame, on="tractce", how="left")

    numeric_columns = [
        "all_permits",
        "raw_unit_bearing_permit_rows",
        "housing_projects",
        "net_units",
        "positive_units",
        "duplicate_positive_units_removed",
        "spatial_fallback_positive_units",
        "median_household_income",
        "housing_units",
        "vacant_units",
        "median_gross_rent",
        "renter_households_computed",
        "rent_burdened_households",
        "vacancy_rate",
        "rent_burden_share",
    ]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    merged["all_permits"] = merged["all_permits"].fillna(0)
    merged["raw_unit_bearing_permit_rows"] = merged["raw_unit_bearing_permit_rows"].fillna(0)
    merged["housing_projects"] = merged["housing_projects"].fillna(0)
    merged["net_units"] = merged["net_units"].fillna(0)
    merged["positive_units"] = merged["positive_units"].fillna(0)
    merged["duplicate_positive_units_removed"] = merged["duplicate_positive_units_removed"].fillna(0)
    merged["spatial_fallback_positive_units"] = merged["spatial_fallback_positive_units"].fillna(0)
    merged["other_permits"] = (merged["all_permits"] - merged["raw_unit_bearing_permit_rows"]).clip(lower=0)
    merged["neighborhood_council"] = merged["neighborhood_council"].fillna("No dominant recent permit area")
    merged["community_plan_area"] = merged["community_plan_area"].fillna("Unavailable")

    metric = merged.to_crs(AREA_CRS).copy()
    metric["positive_units_per_1000_homes"] = (
        metric["positive_units"] / metric["housing_units"].replace({0: pd.NA})
    ) * 1000
    metric["net_units_per_1000_homes"] = (
        metric["net_units"] / metric["housing_units"].replace({0: pd.NA})
    ) * 1000
    metric["positive_units_per_1000_homes"] = pd.to_numeric(metric["positive_units_per_1000_homes"], errors="coerce")
    metric["net_units_per_1000_homes"] = pd.to_numeric(metric["net_units_per_1000_homes"], errors="coerce")

    metric["demand_pressure_score"] = (
        0.45 * clipped_score(metric["rent_burden_share"])
        + 0.35 * clipped_score(metric["median_gross_rent"], log_scale=True)
        + 0.20 * clipped_score(metric["vacancy_rate"], invert=True)
    )
    metric["supply_response_score"] = clipped_score(metric["positive_units_per_1000_homes"], log_scale=True)
    metric["imbalance_score"] = metric["demand_pressure_score"] - metric["supply_response_score"]

    metric["demand_pressure_percentile"] = metric["demand_pressure_score"].rank(pct=True, method="average") * 100
    metric["supply_response_percentile"] = metric["supply_response_score"].rank(pct=True, method="average") * 100
    metric["balance_read"] = metric.apply(
        lambda row: balance_label(row["imbalance_score"], row["demand_pressure_score"], row["supply_response_score"]),
        axis=1,
    )

    supply_cap = float(metric["positive_units_per_1000_homes"].quantile(0.98))
    if pd.isna(supply_cap) or supply_cap <= 0:
        supply_cap = float(metric["positive_units_per_1000_homes"].max())
    if pd.isna(supply_cap) or supply_cap <= 0:
        supply_cap = 1.0

    imbalance_low = float(metric["imbalance_score"].quantile(0.05))
    imbalance_high = float(metric["imbalance_score"].quantile(0.95))
    if pd.isna(imbalance_low) or pd.isna(imbalance_high) or imbalance_low == imbalance_high:
        imbalance_low, imbalance_high = -1.0, 1.0

    metric["height_m"] = metric["positive_units_per_1000_homes"].clip(lower=0, upper=supply_cap).fillna(0) * 58
    metric["fill_color"] = metric["imbalance_score"].apply(
        lambda value: interpolate_diverging_color(value, imbalance_low, imbalance_high)
    )

    metric["tract_label"] = metric["NAMELSAD"]
    metric["positive_units_label"] = metric["positive_units"].apply(format_int)
    metric["net_units_label"] = metric["net_units"].apply(format_signed_int)
    metric["all_permits_label"] = metric["all_permits"].apply(format_int)
    metric["raw_unit_bearing_permit_rows_label"] = metric["raw_unit_bearing_permit_rows"].apply(format_int)
    metric["housing_projects_label"] = metric["housing_projects"].apply(format_int)
    metric["other_permits_label"] = metric["other_permits"].apply(format_int)
    metric["duplicate_positive_units_removed_label"] = metric["duplicate_positive_units_removed"].apply(format_int)
    metric["spatial_fallback_positive_units_label"] = metric["spatial_fallback_positive_units"].apply(format_int)
    metric["housing_units_label"] = metric["housing_units"].apply(format_int)
    metric["median_household_income_label"] = metric["median_household_income"].apply(
        lambda value: "No data" if pd.isna(value) else f"${value:,.0f}"
    )
    metric["positive_units_per_1000_label"] = metric["positive_units_per_1000_homes"].apply(
        lambda value: "No data" if pd.isna(value) else f"{value:,.1f}"
    )
    metric["net_units_per_1000_label"] = metric["net_units_per_1000_homes"].apply(
        lambda value: "No data" if pd.isna(value) else f"{value:,.1f}"
    )
    metric["vacancy_rate_label"] = metric["vacancy_rate"].apply(format_pct)
    metric["rent_burden_share_label"] = metric["rent_burden_share"].apply(format_pct)
    metric["median_gross_rent_label"] = metric["median_gross_rent"].apply(
        lambda value: "No data" if pd.isna(value) else f"${value:,.0f}"
    )
    metric["demand_pressure_percentile_label"] = metric["demand_pressure_percentile"].apply(
        lambda value: f"{value:,.0f}"
    )
    metric["supply_response_percentile_label"] = metric["supply_response_percentile"].apply(
        lambda value: f"{value:,.0f}"
    )

    metric["tooltip_html"] = metric.apply(
        lambda row: (
            f"<b>{html.escape(str(row['tract_label']))}</b><br/>"
            f"<b>Neighborhood council:</b> {html.escape(str(row['neighborhood_council']))}<br/>"
            f"<b>Community plan:</b> {html.escape(str(row['community_plan_area']))}<br/>"
            f"<b>Balance read:</b> {html.escape(str(row['balance_read']))}<br/>"
            f"<b>Positive units (reconciled):</b> {row['positive_units_label']}<br/>"
            f"<b>Net units (reconciled):</b> {row['net_units_label']}<br/>"
            f"<b>Housing projects (reconciled):</b> {row['housing_projects_label']}<br/>"
            f"<b>Raw unit-bearing permit rows:</b> {row['raw_unit_bearing_permit_rows_label']}<br/>"
            f"<b>Other permits:</b> {row['other_permits_label']}<br/>"
            f"<b>All permit rows (context):</b> {row['all_permits_label']}<br/>"
            f"<b>Duplicate units removed:</b> {row['duplicate_positive_units_removed_label']}<br/>"
            f"<b>Spatially reassigned units:</b> {row['spatial_fallback_positive_units_label']}<br/>"
            f"<b>Units / 1,000 homes:</b> {row['positive_units_per_1000_label']}<br/>"
            f"<b>Net units / 1,000 homes:</b> {row['net_units_per_1000_label']}<br/>"
            f"<b>Housing units:</b> {row['housing_units_label']}<br/>"
            f"<b>Median HH income:</b> {row['median_household_income_label']}<br/>"
            f"<b>Vacancy rate:</b> {row['vacancy_rate_label']}<br/>"
            f"<b>Rent burdened renter HHs:</b> {row['rent_burden_share_label']}<br/>"
            f"<b>Median gross rent:</b> {row['median_gross_rent_label']}<br/>"
            f"<b>Demand pressure pctile:</b> {row['demand_pressure_percentile_label']}<br/>"
            f"<b>Supply response pctile:</b> {row['supply_response_percentile_label']}"
        ),
        axis=1,
    )

    metric = metric.to_crs(4326)
    stats: dict[str, float | str] = {
        "supply_cap": supply_cap,
        "citywide_positive_units": float(metric["positive_units"].sum()),
        "citywide_net_units": float(metric["net_units"].sum()),
        "citywide_housing_projects": float(metric["housing_projects"].sum()),
        "citywide_raw_unit_bearing_permit_rows": float(metric["raw_unit_bearing_permit_rows"].sum()),
        "citywide_other_permits": float(metric["other_permits"].sum()),
        "citywide_all_permits": float(metric["all_permits"].sum()),
        "citywide_duplicate_positive_units_removed": float(metric["duplicate_positive_units_removed"].sum()),
        "citywide_spatial_fallback_positive_units": float(metric["spatial_fallback_positive_units"].sum()),
        "citywide_unassigned_positive_units": float(rollup.summary["unassigned_positive_units"]),
        "recovered_project_count": float(len(point_layers["recovered_projects"])),
        "unassigned_project_count": float(len(point_layers["unassigned_projects"])),
        "high_imbalance_tracts": float((metric["imbalance_score"] >= 0.75).sum()),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    return metric, stats, point_layers


def build_view_state(gdf: gpd.GeoDataFrame) -> pdk.ViewState:
    minx, miny, maxx, maxy = gdf.total_bounds
    return pdk.ViewState(
        longitude=((minx + maxx) / 2) + 0.02,
        latitude=((miny + maxy) / 2) + 0.01,
        zoom=10.15,
        pitch=52,
        bearing=-24,
    )


def build_deck(gdf: gpd.GeoDataFrame, point_layers: dict[str, pd.DataFrame]) -> pdk.Deck:
    geojson = json.loads(gdf.to_json())
    tract_layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=False,
        opacity=0.9,
        get_fill_color="properties.fill_color",
        get_elevation="properties.height_m",
        get_line_color=[24, 28, 38, 80],
        line_width_min_pixels=0.4,
        auto_highlight=True,
    )

    layers: list[pdk.Layer] = [tract_layer]

    recovered_points = point_layers["recovered_projects"]
    if not recovered_points.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                recovered_points,
                pickable=True,
                opacity=0.9,
                stroked=True,
                filled=True,
                get_position="[lon, lat]",
                get_radius="radius_m",
                radius_units="meters",
                radius_min_pixels=4,
                radius_max_pixels=28,
                get_fill_color=[235, 169, 52, 150],
                get_line_color=[122, 78, 8, 230],
                line_width_min_pixels=1.2,
            )
        )

    unassigned_points = point_layers["unassigned_projects"]
    if not unassigned_points.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                unassigned_points,
                pickable=True,
                opacity=0.95,
                stroked=True,
                filled=True,
                get_position="[lon, lat]",
                get_radius="radius_m",
                radius_units="meters",
                radius_min_pixels=4,
                radius_max_pixels=26,
                get_fill_color=[196, 64, 64, 170],
                get_line_color=[98, 18, 24, 235],
                line_width_min_pixels=1.4,
            )
        )

    tooltip = {
        "html": "{tooltip_html}",
        "style": {
            "backgroundColor": "#10141f",
            "color": "white",
            "fontFamily": "Arial",
            "borderRadius": "8px",
            "fontSize": "12px",
        },
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=build_view_state(gdf),
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK_NO_LABELS,
        tooltip=tooltip,
    )


def build_overlay_html(stats: dict[str, float | str]) -> str:
    start_label = datetime.strptime(str(stats["start_date"]), "%Y-%m-%d").strftime("%b %d, %Y")
    end_label = datetime.strptime(str(stats["end_date"]), "%Y-%m-%d").strftime("%b %d, %Y")
    positive_units = format_int(float(stats["citywide_positive_units"]))
    net_units = format_signed_int(float(stats["citywide_net_units"]))
    housing_projects = format_int(float(stats["citywide_housing_projects"]))
    raw_unit_rows = format_int(float(stats["citywide_raw_unit_bearing_permit_rows"]))
    other_permits = format_int(float(stats["citywide_other_permits"]))
    all_permits = format_int(float(stats["citywide_all_permits"]))
    duplicate_units_removed = format_int(float(stats["citywide_duplicate_positive_units_removed"]))
    spatial_units = format_int(float(stats["citywide_spatial_fallback_positive_units"]))
    unassigned_units = format_int(float(stats["citywide_unassigned_positive_units"]))
    recovered_projects = format_int(float(stats["recovered_project_count"]))
    unassigned_projects = format_int(float(stats["unassigned_project_count"]))
    high_imbalance = format_int(float(stats["high_imbalance_tracts"]))
    supply_cap = f"{float(stats['supply_cap']):,.0f}"

    return f"""
<style>
.map-card {{
  position: absolute;
  z-index: 10;
  background: rgba(12, 16, 26, 0.9);
  color: #eef3ff;
  border: 1px solid rgba(116, 134, 174, 0.28);
  border-radius: 14px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
  font-family: Arial, sans-serif;
  backdrop-filter: blur(8px);
}}
.map-title {{
  top: 18px;
  left: 18px;
  width: 390px;
  padding: 16px 18px;
}}
.map-title h1 {{
  margin: 0 0 8px 0;
  font-size: 22px;
  line-height: 1.1;
}}
.map-title p {{
  margin: 0;
  color: #b7c3dc;
  font-size: 13px;
  line-height: 1.4;
}}
.map-legend {{
  left: 18px;
  bottom: 18px;
  width: 340px;
  padding: 14px 16px;
}}
.legend-row {{
  margin-top: 10px;
  font-size: 12px;
  color: #d6e0f5;
}}
.legend-bar {{
  margin-top: 6px;
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(90deg, rgb(41,109,176), rgb(88,194,205), rgb(108,109,120), rgb(228,184,88), rgb(228,101,68));
}}
.legend-scale {{
  display: flex;
  justify-content: space-between;
  margin-top: 6px;
  color: #aab6d0;
  font-size: 11px;
}}
</style>
<div class="map-card map-title">
  <h1>LA Supply-Demand Balance v3</h1>
  <p>{html.escape(start_label)} to {html.escape(end_label)} permits with 2023 ACS housing context</p>
  <p style="margin-top: 10px;">Height shows <b>reconciled positive units per 1,000 homes</b>. Color shows an <b>imbalance read</b>: blue means supply response is stronger relative to local demand proxies, amber means demand pressure is stronger relative to supply.</p>
  <p style="margin-top: 10px;"><b>Positive units:</b> {html.escape(positive_units)}<br/><b>Net units:</b> {html.escape(net_units)}<br/><b>Housing projects:</b> {html.escape(housing_projects)}<br/><b>Raw unit-bearing permit rows:</b> {html.escape(raw_unit_rows)}<br/><b>Other permits:</b> {html.escape(other_permits)}<br/><b>All permit rows:</b> {html.escape(all_permits)}<br/><b>Recovered projects:</b> {html.escape(recovered_projects)}<br/><b>Spatially reassigned units:</b> {html.escape(spatial_units)}<br/><b>Duplicate units removed:</b> {html.escape(duplicate_units_removed)}<br/><b>Unassigned projects:</b> {html.escape(unassigned_projects)}<br/><b>Still unassigned units:</b> {html.escape(unassigned_units)}<br/><b>High-imbalance tracts:</b> {html.escape(high_imbalance)}</p>
</div>
<div class="map-card map-legend">
  <div class="legend-row"><b>Extrusion</b>: capped near {html.escape(supply_cap)} positive units per 1,000 homes</div>
  <div class="legend-row"><b>Color</b>: supply response stronger to shortage pressure stronger</div>
  <div class="legend-row"><span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:#eba923;border:1px solid #7a4e08;margin-right:6px;"></span>Recovered projects reassigned by lat/lon</div>
  <div class="legend-row"><span style="display:inline-block;width:10px;height:10px;border-radius:999px;background:#c44040;border:1px solid #621218;margin-right:6px;"></span>Still-unassigned positive-unit projects</div>
  <div class="legend-bar"></div>
  <div class="legend-scale">
    <span>Supply-led</span>
    <span>Mixed</span>
    <span>Demand-led</span>
  </div>
</div>
"""


def export_map(deck: pdk.Deck, output_path: Path, stats: dict[str, float | str], open_browser: bool) -> None:
    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_string = deck.to_html(as_string=True, iframe_width="100%", iframe_height=900)
    html_string = html_string.replace(
        "<head>",
        '<head>\n<link rel="stylesheet" href="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.css" />',
        1,
    )
    html_string = html_string.replace("</body>", build_overlay_html(stats) + "\n</body>")
    output_path.write_text(html_string, encoding="utf-8")

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())


def main() -> None:
    args = parse_args()
    start_date, end_date = resolve_date_range(args)
    frame, stats, point_layers = prepare_map_frame(start_date, end_date)
    deck = build_deck(frame, point_layers)
    output_path = resolve_output_path(args.output)
    export_map(deck, output_path, stats, args.open_browser)

    print(f"Saved map to: {output_path}")
    print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"Tracts rendered: {len(frame):,}")
    print(f"Positive units: {int(round(float(stats['citywide_positive_units']))):,}")
    print(f"Net units: {int(round(float(stats['citywide_net_units']))):,}")
    print(f"Housing projects: {int(round(float(stats['citywide_housing_projects']))):,}")
    print(f"Raw unit-bearing permit rows: {int(round(float(stats['citywide_raw_unit_bearing_permit_rows']))):,}")
    print(f"Other permits: {int(round(float(stats['citywide_other_permits']))):,}")
    print(f"All permits: {int(round(float(stats['citywide_all_permits']))):,}")
    print(f"Spatially reassigned units: {int(round(float(stats['citywide_spatial_fallback_positive_units']))):,}")
    print(f"Duplicate units removed: {int(round(float(stats['citywide_duplicate_positive_units_removed']))):,}")
    print(f"Still unassigned to 2023 tracts: {int(round(float(stats['citywide_unassigned_positive_units']))):,}")


if __name__ == "__main__":
    main()
