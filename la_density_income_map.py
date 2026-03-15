from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import requests


STATE_FIPS = "06"
LA_COUNTY_FIPS = "037"
DEFAULT_YEAR = 2023
AREA_CRS = 3310  # California Albers
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

CITY_NEIGHBORHOOD_LABELS = [
    {"name": "Downtown LA", "lon": -118.2437, "lat": 34.0522, "priority": 10},
    {"name": "Hollywood", "lon": -118.3287, "lat": 34.0980, "priority": 9},
    {"name": "Koreatown", "lon": -118.2995, "lat": 34.0618, "priority": 9},
    {"name": "Westwood", "lon": -118.4452, "lat": 34.0635, "priority": 8},
    {"name": "Venice", "lon": -118.4695, "lat": 33.9850, "priority": 8},
    {"name": "Boyle Heights", "lon": -118.2051, "lat": 34.0400, "priority": 8},
    {"name": "South LA", "lon": -118.2875, "lat": 33.9897, "priority": 8},
    {"name": "Van Nuys", "lon": -118.4489, "lat": 34.1867, "priority": 8},
    {"name": "Sherman Oaks", "lon": -118.4492, "lat": 34.1511, "priority": 7},
    {"name": "North Hollywood", "lon": -118.3813, "lat": 34.1722, "priority": 7},
    {"name": "Echo Park", "lon": -118.2606, "lat": 34.0782, "priority": 7},
    {"name": "Silver Lake", "lon": -118.2707, "lat": 34.0900, "priority": 7},
    {"name": "Westchester", "lon": -118.3988, "lat": 33.9597, "priority": 6},
    {"name": "Watts", "lon": -118.2428, "lat": 33.9380, "priority": 6},
    {"name": "San Pedro", "lon": -118.2923, "lat": 33.7361, "priority": 7},
    {"name": "Wilmington", "lon": -118.2617, "lat": 33.7806, "priority": 5},
    {"name": "Encino", "lon": -118.5012, "lat": 34.1598, "priority": 6},
    {"name": "Pacoima", "lon": -118.4112, "lat": 34.2625, "priority": 5},
    {"name": "Highland Park", "lon": -118.1868, "lat": 34.1117, "priority": 6},
    {"name": "Eagle Rock", "lon": -118.2120, "lat": 34.1378, "priority": 5},
    {"name": "Playa del Rey", "lon": -118.4410, "lat": 33.9591, "priority": 4},
    {"name": "Reseda", "lon": -118.5360, "lat": 34.2011, "priority": 4},
    {"name": "Chatsworth", "lon": -118.6059, "lat": 34.2578, "priority": 3},
    {"name": "Sylmar", "lon": -118.4490, "lat": 34.3070, "priority": 4},
]

COUNTY_NEIGHBORHOOD_LABELS = CITY_NEIGHBORHOOD_LABELS + [
    {"name": "Santa Monica", "lon": -118.4912, "lat": 34.0195, "priority": 8},
    {"name": "Pasadena", "lon": -118.1445, "lat": 34.1478, "priority": 8},
    {"name": "Glendale", "lon": -118.2551, "lat": 34.1425, "priority": 7},
    {"name": "Burbank", "lon": -118.3089, "lat": 34.1808, "priority": 7},
    {"name": "Inglewood", "lon": -118.3531, "lat": 33.9617, "priority": 7},
    {"name": "Long Beach", "lon": -118.1937, "lat": 33.7701, "priority": 9},
    {"name": "Torrance", "lon": -118.3406, "lat": 33.8358, "priority": 7},
    {"name": "Pomona", "lon": -117.7490, "lat": 34.0551, "priority": 6},
    {"name": "Santa Clarita", "lon": -118.5426, "lat": 34.3917, "priority": 7},
    {"name": "Lancaster", "lon": -118.1542, "lat": 34.6868, "priority": 7},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 3D Los Angeles tract map with height = density and color = income."
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help="ACS/TIGER vintage year. Default: 2023.",
    )
    parser.add_argument(
        "--region",
        choices=["city", "county"],
        default="city",
        help="Use Los Angeles city tracts or all of Los Angeles County.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "la_density_income_los_angeles.html",
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


def load_acs_tract_data(year: int) -> pd.DataFrame:
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B01003_001E,B19013_001E",
        "for": "tract:*",
        "in": f"state:{STATE_FIPS} county:{LA_COUNTY_FIPS}",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()

    rows = response.json()
    frame = pd.DataFrame(rows[1:], columns=rows[0]).rename(
        columns={
            "NAME": "tract_name",
            "B01003_001E": "population",
            "B19013_001E": "median_income",
        }
    )

    for column in ["population", "median_income"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[frame[column] < 0, column] = pd.NA

    frame["geoid"] = frame["state"] + frame["county"] + frame["tract"]
    return frame[["geoid", "tract_name", "population", "median_income"]]


def load_la_county_tracts(year: int) -> gpd.GeoDataFrame:
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{STATE_FIPS}_tract.zip"
    tracts = gpd.read_file(url)
    tracts = tracts[(tracts["STATEFP"] == STATE_FIPS) & (tracts["COUNTYFP"] == LA_COUNTY_FIPS)].copy()
    tracts["geoid"] = tracts["GEOID"]
    return tracts


def load_los_angeles_city_boundary(year: int) -> gpd.GeoDataFrame:
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/PLACE/tl_{year}_{STATE_FIPS}_place.zip"
    places = gpd.read_file(url)
    city = places.loc[places["NAME"] == "Los Angeles", ["NAME", "geometry"]].copy()
    if city.empty:
        raise RuntimeError(f"Could not find Los Angeles city boundary for {year}.")
    return city


def filter_to_city_tracts(tracts: gpd.GeoDataFrame, year: int) -> gpd.GeoDataFrame:
    city = load_los_angeles_city_boundary(year).to_crs(AREA_CRS)
    tracts_area = tracts.to_crs(AREA_CRS).copy()
    centroids = tracts_area[["geoid", "geometry"]].copy()
    centroids["geometry"] = centroids.centroid
    joined = gpd.sjoin(centroids, city[["geometry"]], predicate="within", how="inner")
    return tracts.loc[tracts["geoid"].isin(joined["geoid"])].copy()


def interpolate_color(value: float, min_value: float, max_value: float) -> list[int]:
    palette = [
        (66, 47, 122),
        (84, 77, 171),
        (73, 130, 182),
        (84, 181, 166),
        (194, 245, 216),
    ]

    if pd.isna(value):
        return [80, 80, 90, 120]

    if max_value <= min_value:
        base = palette[-1]
        return [base[0], base[1], base[2], 210]

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
    return [rgb[0], rgb[1], rgb[2], 210]


def prepare_map_frame(year: int, region: str) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    tracts = load_la_county_tracts(year)
    if region == "city":
        tracts = filter_to_city_tracts(tracts, year)

    acs = load_acs_tract_data(year)
    merged = tracts.merge(acs, on="geoid", how="left")

    metric = merged.to_crs(AREA_CRS).copy()
    metric["area_sqkm"] = metric.geometry.area / 1_000_000
    metric["density_per_sqkm"] = metric["population"] / metric["area_sqkm"]

    metric["median_income"] = metric["median_income"].clip(lower=0)
    metric["density_per_sqkm"] = metric["density_per_sqkm"].replace([pd.NA], pd.NA)
    metric["density_per_sqkm"] = metric["density_per_sqkm"].clip(lower=0)

    density_cap = float(metric["density_per_sqkm"].quantile(0.98))
    income_low = float(metric["median_income"].quantile(0.05))
    income_high = float(metric["median_income"].quantile(0.95))

    metric["height_m"] = metric["density_per_sqkm"].clip(upper=density_cap).fillna(0) * 1.35
    metric["fill_color"] = metric["median_income"].apply(
        lambda value: interpolate_color(value, income_low, income_high)
    )

    metric["density_per_sqmi"] = metric["density_per_sqkm"] * 2.58999
    metric["population"] = metric["population"].fillna(0).round().astype(int)

    metric = metric.to_crs(4326)

    stats = {
        "density_cap": density_cap,
        "income_low": income_low,
        "income_high": income_high,
    }
    return metric, stats


def build_view_state(gdf: gpd.GeoDataFrame, region: str) -> pdk.ViewState:
    minx, miny, maxx, maxy = gdf.total_bounds
    return pdk.ViewState(
        longitude=((minx + maxx) / 2) + (0.02 if region == "city" else 0.0),
        latitude=((miny + maxy) / 2) + (0.01 if region == "city" else 0.0),
        zoom=10.15 if region == "city" else 9.25,
        pitch=52,
        bearing=-24,
    )


def build_neighborhood_points(region: str) -> pd.DataFrame:
    candidates = CITY_NEIGHBORHOOD_LABELS if region == "city" else COUNTY_NEIGHBORHOOD_LABELS
    frame = pd.DataFrame(candidates).copy()
    frame["radius_m"] = frame["priority"].apply(lambda priority: 180 + (priority * 24))
    frame["tooltip_html"] = frame["name"].apply(
        lambda value: f"<b>Neighborhood anchor:</b> {html.escape(str(value))}"
    )
    return frame


def assign_neighborhood_context(gdf: gpd.GeoDataFrame, region: str) -> gpd.GeoDataFrame:
    labeled = gdf.copy()
    anchors = build_neighborhood_points(region)
    anchor_gdf = gpd.GeoDataFrame(
        anchors[["name", "lon", "lat"]],
        geometry=gpd.points_from_xy(anchors["lon"], anchors["lat"]),
        crs=4326,
    ).to_crs(AREA_CRS)

    centroid_gdf = labeled.to_crs(AREA_CRS).copy()
    centroid_gdf["geometry"] = centroid_gdf.geometry.centroid

    anchor_records = list(anchor_gdf[["name", "geometry"]].itertuples(index=False, name=None))
    nearest_names: list[str] = []

    for point in centroid_gdf.geometry:
        nearest_name = min(anchor_records, key=lambda record: point.distance(record[1]))[0]
        nearest_names.append(nearest_name)

    labeled["neighborhood_anchor"] = nearest_names
    return labeled


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "No data"
    return f"${value:,.0f}"


def add_labels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    labeled = gdf.copy()
    labeled["income_label"] = labeled["median_income"].apply(format_currency)
    labeled["density_label"] = labeled["density_per_sqmi"].apply(
        lambda value: "No data" if pd.isna(value) else f"{value:,.0f} people / sq mi"
    )
    labeled["height_label"] = labeled["density_per_sqkm"].apply(
        lambda value: "No data" if pd.isna(value) else f"{value:,.0f} people / sq km"
    )
    labeled["tooltip_html"] = labeled.apply(
        lambda row: (
            f"<b>{html.escape(str(row['tract_name']))}</b><br/>"
            f"<b>Neighborhood anchor:</b> {html.escape(str(row['neighborhood_anchor']))}<br/>"
            f"<b>Median income:</b> {row['income_label']}<br/>"
            f"<b>Density:</b> {row['density_label']}<br/>"
            f"<b>Population:</b> {row['population']}"
        ),
        axis=1,
    )
    return labeled


def build_deck(gdf: gpd.GeoDataFrame, year: int, region: str) -> pdk.Deck:
    gdf = assign_neighborhood_context(gdf, region)
    gdf = add_labels(gdf)
    geojson = json.loads(gdf.to_json())
    view_state = build_view_state(gdf, region)
    neighborhood_points = build_neighborhood_points(region)

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
        get_line_color=[28, 32, 42, 80],
        line_width_min_pixels=0.4,
        auto_highlight=True,
    )

    neighborhood_layer = pdk.Layer(
        "ScatterplotLayer",
        neighborhood_points,
        pickable=True,
        stroked=True,
        filled=True,
        opacity=0.75,
        get_position="[lon, lat]",
        get_radius="radius_m",
        radius_units="meters",
        radius_min_pixels=3,
        radius_max_pixels=7,
        get_fill_color=[208, 235, 255, 36],
        get_line_color=[232, 244, 255, 190],
        line_width_min_pixels=1.5,
    )

    region_label = "Los Angeles City" if region == "city" else "Los Angeles County"
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
        layers=[tract_layer, neighborhood_layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_DARK_NO_LABELS,
        tooltip=tooltip,
    )


def build_overlay_html(year: int, region: str, stats: dict[str, float]) -> str:
    region_label = "Los Angeles City" if region == "city" else "Los Angeles County"
    density_cap = f"{stats['density_cap'] * 2.58999:,.0f}"
    income_low = format_currency(stats["income_low"])
    income_high = format_currency(stats["income_high"])

    title = html.escape(region_label)
    subtitle = html.escape(f"ACS {year} 5-year estimates at the census tract level")

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
  width: 320px;
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
  width: 290px;
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
  background: linear-gradient(90deg, rgb(66,47,122), rgb(84,77,171), rgb(73,130,182), rgb(84,181,166), rgb(194,245,216));
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
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <p style="margin-top: 10px;">Height encodes population density. Color encodes median household income. Hover a tract or neighborhood anchor for details.</p>
</div>
<div class="map-card map-legend">
  <div class="legend-row"><b>Extrusion</b>: capped near {density_cap} people / sq mi for readability</div>
  <div class="legend-row"><b>Income color</b></div>
  <div class="legend-bar"></div>
  <div class="legend-scale">
    <span>{html.escape(income_low)}</span>
    <span>{html.escape(income_high)}</span>
  </div>
</div>
"""


def export_map(
    deck: pdk.Deck,
    output_path: Path,
    year: int,
    region: str,
    stats: dict[str, float],
    open_browser: bool,
) -> None:
    output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_string = deck.to_html(as_string=True, iframe_width="100%", iframe_height=900)
    html_string = html_string.replace(
        "<head>",
        '<head>\n<link rel="stylesheet" href="https://api.tiles.mapbox.com/mapbox-gl-js/v1.13.0/mapbox-gl.css" />',
        1,
    )
    html_string = html_string.replace(
        "</body>", build_overlay_html(year, region, stats) + "\n</body>"
    )
    output_path.write_text(html_string, encoding="utf-8")

    if open_browser:
        webbrowser.open(output_path.resolve().as_uri())


def main() -> None:
    args = parse_args()
    frame, stats = prepare_map_frame(args.year, args.region)
    deck = build_deck(frame, args.year, args.region)
    output_path = resolve_output_path(args.output)
    export_map(deck, output_path, args.year, args.region, stats, args.open_browser)

    print(f"Saved map to: {output_path}")
    print(f"Region: {args.region}")
    print(f"Tracts rendered: {len(frame):,}")


if __name__ == "__main__":
    main()
