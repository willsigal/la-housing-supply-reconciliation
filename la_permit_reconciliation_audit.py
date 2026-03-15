from __future__ import annotations

import argparse
import calendar
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from permit_reconciliation import PermitRollupResult, build_permit_rollup


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output" / "reconciliation"
DEFAULT_MONTHS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit LADBS permit-unit reconciliation for Los Angeles housing maps."
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
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for summary artifacts.",
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


def write_outputs(result: PermitRollupResult, output_dir: Path, start_date: date, end_date: date) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        **result.summary,
    }
    summary_path = output_dir / "la_permit_reconciliation_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    duplicate_path = output_dir / "la_duplicate_project_families.csv"
    duplicate_columns = [
        "family_key",
        "permit_nbr",
        "primary_address",
        "issue_date",
        "resolved_tractce",
        "positive_units",
        "raw_positive_units",
        "suppressed_positive_units",
        "family_permit_rows",
        "raw_unit_bearing_rows",
        "cpa",
        "cnc",
        "work_desc",
    ]
    duplicates = result.project_rows[result.project_rows["suppressed_positive_units"] > 0].copy()
    duplicates = duplicates.sort_values(["suppressed_positive_units", "raw_positive_units"], ascending=[False, False])
    duplicates[duplicate_columns].to_csv(duplicate_path, index=False)

    unmatched_path = output_dir / "la_unassigned_positive_unit_rows.csv"
    unmatched_columns = [
        "permit_nbr",
        "primary_address",
        "issue_date",
        "ct",
        "raw_tractce",
        "lat",
        "lon",
        "positive_units",
        "net_units",
        "permit_type",
        "permit_sub_type",
        "cpa",
        "cnc",
        "work_desc",
    ]
    unmatched = result.raw_rows[
        (result.raw_rows["resolved_tractce"].isna()) & (result.raw_rows["positive_units"] > 0)
    ].copy()
    unmatched[unmatched_columns].to_csv(unmatched_path, index=False)

    fallback_path = output_dir / "la_spatial_fallback_positive_rows.csv"
    fallback_columns = [
        "permit_nbr",
        "primary_address",
        "issue_date",
        "ct",
        "raw_tractce",
        "resolved_tractce",
        "positive_units",
        "net_units",
        "permit_type",
        "permit_sub_type",
        "cpa",
        "cnc",
        "work_desc",
    ]
    fallback = result.raw_rows[
        (result.raw_rows["tract_assignment_method"] == "spatial_fallback") & (result.raw_rows["positive_units"] > 0)
    ].copy()
    fallback[fallback_columns].to_csv(fallback_path, index=False)

    return {
        "summary": summary_path,
        "duplicates": duplicate_path,
        "unmatched": unmatched_path,
        "fallback": fallback_path,
    }


def main() -> None:
    args = parse_args()
    start_date, end_date = resolve_date_range(args)
    result = build_permit_rollup(start_date, end_date)
    output_dir = resolve_output_path(args.output_dir)
    outputs = write_outputs(result, output_dir, start_date, end_date)

    print(f"Saved audit artifacts to: {output_dir}")
    print(f"Date range: {start_date.isoformat()} to {end_date.isoformat()}")
    print(f"Raw positive units: {int(round(result.summary['raw_positive_units'])):,}")
    print(f"Assigned positive units: {int(round(result.summary['assigned_positive_units'])):,}")
    print(f"Reconciled positive units: {int(round(result.summary['reconciled_positive_units'])):,}")
    print(f"Raw unit-bearing permit rows: {int(round(result.summary['raw_unit_bearing_permit_rows'])):,}")
    print(f"Reconciled housing projects: {int(round(result.summary['reconciled_housing_projects'])):,}")
    print(f"Spatially reassigned positive units: {int(round(result.summary['spatial_fallback_positive_units'])):,}")
    print(f"Duplicate positive units removed: {int(round(result.summary['duplicate_positive_units_removed'])):,}")
    print(f"Still unassigned positive units: {int(round(result.summary['unassigned_positive_units'])):,}")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
