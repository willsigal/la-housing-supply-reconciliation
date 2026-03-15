from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import geopandas as gpd
import pandas as pd
import requests


STATE_FIPS = "06"
LA_COUNTY_FIPS = "037"
AREA_CRS = 3310
TIGER_YEAR = 2023
PERMITS_DATASET = "pi9x-tg5x"
PAGE_SIZE = 50_000


@dataclass
class PermitRollupResult:
    tract_metrics: pd.DataFrame
    raw_rows: pd.DataFrame
    project_rows: pd.DataFrame
    summary: dict[str, float]


def _safe_mode(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    mode = clean.mode()
    if not mode.empty:
        return mode.iat[0]
    return clean.iat[0]


def normalize_tract_code(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    match = re.match(r"^(\d+)(?:\.(\d+))?$", str(value).strip())
    if not match:
        return None

    whole = int(match.group(1))
    frac = (match.group(2) or "00")[:2].ljust(2, "0")
    return f"{whole:04d}{frac}"


def build_family_key(value: object) -> str:
    permit_nbr = str(value).strip()
    match = re.match(r"^(\d{5})-(\d{5})-(\d{5})$", permit_nbr)
    if not match:
        return permit_nbr
    return f"{match.group(1)}-{match.group(3)}"


def _fetch_json(url: str, params: dict[str, str]) -> list[dict[str, object]]:
    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload))
    return payload


def fetch_permit_rows(start_date: date, end_date: date) -> pd.DataFrame:
    fields = (
        "permit_nbr,primary_address,issue_date,permit_group,permit_type,permit_sub_type,"
        "use_desc,work_desc,ct,cpa,cnc,lat,lon,du_changed,adu_changed,junior_adu"
    )
    where_clause = (
        f"issue_date between '{start_date.isoformat()}T00:00:00' and "
        f"'{end_date.isoformat()}T23:59:59'"
    )

    rows: list[dict[str, object]] = []
    offset = 0
    url = f"https://data.lacity.org/resource/{PERMITS_DATASET}.json"

    while True:
        params = {
            "$select": fields,
            "$where": where_clause,
            "$limit": str(PAGE_SIZE),
            "$offset": str(offset),
        }
        page = _fetch_json(url, params)
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No permit rows returned for the selected date range.")

    numeric_columns = ["du_changed", "adu_changed", "junior_adu", "lat", "lon"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame[["du_changed", "adu_changed", "junior_adu"]] = frame[
        ["du_changed", "adu_changed", "junior_adu"]
    ].fillna(0)
    frame["net_units"] = frame["du_changed"] + frame["adu_changed"] + frame["junior_adu"]
    frame["positive_units"] = frame["net_units"].clip(lower=0)
    frame["family_key"] = frame["permit_nbr"].apply(build_family_key)
    frame["issue_date_dt"] = pd.to_datetime(frame["issue_date"], errors="coerce")
    frame["raw_tractce"] = frame["ct"].apply(normalize_tract_code)
    frame["has_point"] = frame["lat"].notna() & frame["lon"].notna()
    return frame


@lru_cache(maxsize=1)
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
    return tracts.loc[tracts["TRACTCE"].isin(city_tract_ids)].copy().to_crs(4326)


def assign_current_tracts(frame: pd.DataFrame) -> pd.DataFrame:
    assigned = frame.copy()
    city_tracts = load_la_city_tracts()[["TRACTCE", "geometry"]].copy()
    current_tracts = set(city_tracts["TRACTCE"])

    assigned["resolved_tractce"] = pd.NA
    assigned["tract_assignment_method"] = "unassigned"

    current_mask = assigned["raw_tractce"].isin(current_tracts)
    assigned.loc[current_mask, "resolved_tractce"] = assigned.loc[current_mask, "raw_tractce"]
    assigned.loc[current_mask, "tract_assignment_method"] = "dataset_ct"

    fallback_mask = assigned["resolved_tractce"].isna() & assigned["has_point"]
    if fallback_mask.any():
        points = assigned.loc[fallback_mask, ["lat", "lon"]].copy()
        points_gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points["lon"], points["lat"]),
            crs=4326,
        )
        joined = gpd.sjoin(
            points_gdf,
            city_tracts[["TRACTCE", "geometry"]],
            predicate="within",
            how="left",
        )
        assigned.loc[fallback_mask, "resolved_tractce"] = joined["TRACTCE"].values
        fallback_assigned = assigned.loc[fallback_mask, "resolved_tractce"].notna()
        assigned.loc[fallback_mask & fallback_assigned, "tract_assignment_method"] = "spatial_fallback"

    return assigned


def reconcile_project_rows(frame: pd.DataFrame) -> pd.DataFrame:
    assigned = frame.copy()
    assigned["positive_unit_row"] = assigned["positive_units"] > 0
    assigned["spatial_fallback_row"] = assigned["tract_assignment_method"] == "spatial_fallback"
    assigned["spatial_fallback_positive_units"] = assigned["positive_units"].where(
        assigned["spatial_fallback_row"],
        0,
    )

    family_context = assigned.groupby("family_key", dropna=False).agg(
        resolved_tractce_group=("resolved_tractce", _safe_mode),
        neighborhood_council_group=("cnc", _safe_mode),
        community_plan_area_group=("cpa", _safe_mode),
        family_permit_rows=("permit_nbr", "size"),
        raw_positive_units=("positive_units", "sum"),
        raw_net_units=("net_units", "sum"),
        raw_unit_bearing_rows=("positive_unit_row", "sum"),
        family_spatial_fallback_rows=("spatial_fallback_row", "sum"),
        family_spatial_fallback_positive_units=("spatial_fallback_positive_units", "sum"),
    )

    representative = assigned.sort_values(
        by=["family_key", "positive_units", "issue_date_dt", "permit_nbr"],
        ascending=[True, False, False, False],
    ).drop_duplicates(subset=["family_key"], keep="first")

    representative = representative.merge(
        family_context,
        left_on="family_key",
        right_index=True,
        how="left",
    )

    representative["resolved_tractce"] = representative["resolved_tractce"].fillna(
        representative["resolved_tractce_group"]
    )
    representative["cnc"] = representative["cnc"].fillna(representative["neighborhood_council_group"])
    representative["cpa"] = representative["cpa"].fillna(representative["community_plan_area_group"])
    representative["suppressed_positive_units"] = (
        representative["raw_positive_units"] - representative["positive_units"]
    ).clip(lower=0)
    representative["suppressed_unit_bearing_rows"] = (
        representative["raw_unit_bearing_rows"] - representative["positive_unit_row"].astype(int)
    ).clip(lower=0)
    return representative


def build_neighborhood_context(frame: pd.DataFrame) -> pd.DataFrame:
    context = frame[frame["resolved_tractce"].notna()].copy()
    if context.empty:
        return pd.DataFrame(columns=["tractce", "neighborhood_council", "community_plan_area"])

    for column in ["cnc", "cpa"]:
        context[column] = context[column].astype("string").str.strip()
        context[column] = context[column].replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})

    context["context_present"] = context["cnc"].notna() | context["cpa"].notna()
    context = context[context["context_present"]].copy()
    if context.empty:
        return pd.DataFrame(columns=["tractce", "neighborhood_council", "community_plan_area"])

    grouped = (
        context.groupby(["resolved_tractce", "cpa", "cnc"], dropna=False)
        .agg(
            permit_rows=("permit_nbr", "size"),
            positive_units=("positive_units", "sum"),
        )
        .reset_index()
    )
    grouped["cnc_len"] = grouped["cnc"].fillna("").str.len()
    grouped["cpa_len"] = grouped["cpa"].fillna("").str.len()
    grouped = grouped.sort_values(
        by=["resolved_tractce", "permit_rows", "positive_units", "cnc_len", "cpa_len"],
        ascending=[True, False, False, False, False],
    )
    grouped = grouped.drop_duplicates(subset=["resolved_tractce"], keep="first")
    return grouped.rename(
        columns={
            "resolved_tractce": "tractce",
            "cnc": "neighborhood_council",
            "cpa": "community_plan_area",
        }
    )[["tractce", "neighborhood_council", "community_plan_area"]]


def aggregate_by_tract(raw_rows: pd.DataFrame, project_rows: pd.DataFrame) -> pd.DataFrame:
    raw_assigned = raw_rows[raw_rows["resolved_tractce"].notna()].copy()
    raw_assigned["unit_bearing_permit_row"] = raw_assigned["positive_units"] > 0
    raw_assigned["spatial_fallback_row"] = raw_assigned["tract_assignment_method"] == "spatial_fallback"
    raw_assigned["spatial_fallback_positive_units"] = raw_assigned["positive_units"].where(
        raw_assigned["spatial_fallback_row"],
        0,
    )

    raw_agg = raw_assigned.groupby("resolved_tractce").agg(
        all_permits=("permit_nbr", "size"),
        raw_unit_bearing_permit_rows=("unit_bearing_permit_row", "sum"),
        spatial_fallback_rows=("spatial_fallback_row", "sum"),
        spatial_fallback_positive_units=("spatial_fallback_positive_units", "sum"),
    )
    raw_agg["other_permits"] = raw_agg["all_permits"] - raw_agg["raw_unit_bearing_permit_rows"]

    project_assigned = project_rows[project_rows["resolved_tractce"].notna()].copy()
    project_assigned["housing_project"] = project_assigned["positive_units"] > 0
    project_agg = project_assigned.groupby("resolved_tractce").agg(
        net_units=("net_units", "sum"),
        positive_units=("positive_units", "sum"),
        du_units=("du_changed", "sum"),
        adu_units=("adu_changed", "sum"),
        junior_adu_units=("junior_adu", "sum"),
        housing_projects=("housing_project", "sum"),
        all_project_families=("family_key", "size"),
        duplicate_positive_units_removed=("suppressed_positive_units", "sum"),
        duplicate_unit_rows_removed=("suppressed_unit_bearing_rows", "sum"),
    )

    merged = project_agg.join(raw_agg, how="outer").reset_index().rename(columns={"resolved_tractce": "tractce"})
    numeric_columns = [
        "net_units",
        "positive_units",
        "du_units",
        "adu_units",
        "junior_adu_units",
        "housing_projects",
        "all_project_families",
        "duplicate_positive_units_removed",
        "duplicate_unit_rows_removed",
        "all_permits",
        "raw_unit_bearing_permit_rows",
        "spatial_fallback_rows",
        "spatial_fallback_positive_units",
        "other_permits",
    ]
    for column in numeric_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)

    units_per_project = merged["positive_units"] / merged["housing_projects"].replace({0: pd.NA})
    merged["units_per_housing_project"] = pd.to_numeric(units_per_project, errors="coerce").fillna(0)

    units_per_row = merged["positive_units"] / merged["raw_unit_bearing_permit_rows"].replace({0: pd.NA})
    merged["units_per_unit_bearing_permit_row"] = pd.to_numeric(units_per_row, errors="coerce").fillna(0)

    context = build_neighborhood_context(raw_assigned)
    merged = merged.merge(context, on="tractce", how="left")
    return merged


def build_summary(raw_rows: pd.DataFrame, project_rows: pd.DataFrame) -> dict[str, float]:
    raw_assigned = raw_rows[raw_rows["resolved_tractce"].notna()].copy()
    project_assigned = project_rows[project_rows["resolved_tractce"].notna()].copy()

    raw_positive_units = float(raw_rows["positive_units"].sum())
    assigned_positive_units = float(raw_assigned["positive_units"].sum())
    reconciled_positive_units = float(project_assigned["positive_units"].sum())

    summary = {
        "raw_permit_rows": float(len(raw_rows)),
        "assigned_permit_rows": float(len(raw_assigned)),
        "raw_positive_units": raw_positive_units,
        "assigned_positive_units": assigned_positive_units,
        "reconciled_positive_units": reconciled_positive_units,
        "raw_net_units": float(raw_rows["net_units"].sum()),
        "assigned_net_units": float(raw_assigned["net_units"].sum()),
        "reconciled_net_units": float(project_assigned["net_units"].sum()),
        "raw_unit_bearing_permit_rows": float((raw_assigned["positive_units"] > 0).sum()),
        "reconciled_housing_projects": float((project_assigned["positive_units"] > 0).sum()),
        "duplicate_positive_units_removed": assigned_positive_units - reconciled_positive_units,
        "duplicate_unit_rows_removed": float(
            (raw_assigned["positive_units"] > 0).sum() - (project_assigned["positive_units"] > 0).sum()
        ),
        "spatial_fallback_rows": float((raw_assigned["tract_assignment_method"] == "spatial_fallback").sum()),
        "spatial_fallback_positive_units": float(
            raw_assigned.loc[raw_assigned["tract_assignment_method"] == "spatial_fallback", "positive_units"].sum()
        ),
        "unassigned_permit_rows": float(len(raw_rows) - len(raw_assigned)),
        "unassigned_positive_units": raw_positive_units - assigned_positive_units,
    }
    return summary


def build_permit_rollup(start_date: date, end_date: date) -> PermitRollupResult:
    raw_rows = fetch_permit_rows(start_date, end_date)
    raw_rows = assign_current_tracts(raw_rows)
    project_rows = reconcile_project_rows(raw_rows)
    tract_metrics = aggregate_by_tract(raw_rows, project_rows)
    summary = build_summary(raw_rows, project_rows)
    return PermitRollupResult(
        tract_metrics=tract_metrics,
        raw_rows=raw_rows,
        project_rows=project_rows,
        summary=summary,
    )


def build_public_reconciliation_points(result: PermitRollupResult) -> dict[str, pd.DataFrame]:
    project_rows = result.project_rows.copy()
    project_rows["issue_month_label"] = project_rows["issue_date_dt"].dt.strftime("%b %Y").fillna("Unknown")

    recovered = project_rows[
        (project_rows["family_spatial_fallback_positive_units"] > 0)
        & (project_rows["positive_units"] > 0)
        & project_rows["lat"].notna()
        & project_rows["lon"].notna()
    ].copy()
    recovered = recovered.sort_values(
        by=["family_spatial_fallback_positive_units", "positive_units", "issue_date_dt"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    recovered["public_id"] = [f"REC-{index + 1:03d}" for index in range(len(recovered))]
    recovered["radius_m"] = (
        recovered["family_spatial_fallback_positive_units"].clip(lower=1).pow(0.5) * 180 + 120
    )
    recovered["tooltip_html"] = recovered.apply(
        lambda row: (
            f"<b>Recovered project {row['public_id']}</b><br/>"
            f"<b>Recovered units:</b> {int(round(row['family_spatial_fallback_positive_units'])):,}<br/>"
            f"<b>Project units:</b> {int(round(row['positive_units'])):,}<br/>"
            f"<b>Permit type:</b> {row['permit_type']}<br/>"
            f"<b>Permit subtype:</b> {row['permit_sub_type']}<br/>"
            f"<b>Issue month:</b> {row['issue_month_label']}<br/>"
            f"<b>Method:</b> Assigned by lat/lon to the current tract layer"
        ),
        axis=1,
    )

    unassigned = project_rows[
        project_rows["resolved_tractce"].isna()
        & (project_rows["positive_units"] > 0)
        & project_rows["lat"].notna()
        & project_rows["lon"].notna()
    ].copy()
    unassigned = unassigned.sort_values(
        by=["positive_units", "issue_date_dt"],
        ascending=[False, False],
    ).reset_index(drop=True)
    unassigned["public_id"] = [f"UNA-{index + 1:03d}" for index in range(len(unassigned))]
    unassigned["radius_m"] = (unassigned["positive_units"].clip(lower=1).pow(0.5) * 200 + 140)
    unassigned["tooltip_html"] = unassigned.apply(
        lambda row: (
            f"<b>Unassigned project {row['public_id']}</b><br/>"
            f"<b>Units:</b> {int(round(row['positive_units'])):,}<br/>"
            f"<b>Permit type:</b> {row['permit_type']}<br/>"
            f"<b>Permit subtype:</b> {row['permit_sub_type']}<br/>"
            f"<b>Issue month:</b> {row['issue_month_label']}<br/>"
            f"<b>Status:</b> Still outside the current city tract layer"
        ),
        axis=1,
    )

    return {
        "recovered_projects": recovered[
            ["public_id", "lon", "lat", "radius_m", "tooltip_html", "family_spatial_fallback_positive_units"]
        ].copy(),
        "unassigned_projects": unassigned[
            ["public_id", "lon", "lat", "radius_m", "tooltip_html", "positive_units"]
        ].copy(),
    }
