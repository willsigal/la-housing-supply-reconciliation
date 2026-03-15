# LA Housing Maps

Public-facing Los Angeles housing permit maps with a reconciliation pass for stale tract codes and supplemental permit families.

## What is included

- `docs/`
  - Sanitized GitHub Pages site.
- `permit_reconciliation.py`
  - Shared tract-repair and family-deduping logic.
- `la_permits_units_map.py`
  - Reconciled permitted-units tract map.
- `la_supply_demand_balance_map.py`
  - Reconciled supply-demand context tract map with relative pressure-response classes.
- `build_public_site.py`
  - Rebuilds the published `docs/` site.
- `la_permit_reconciliation_audit.py`
  - Local-only audit export. Its outputs are intentionally ignored by Git.

## Privacy

The public site excludes raw addresses, permit numbers, and other address-level audit artifacts. Those remain local-only under `output/` and are not part of the GitHub Pages build.

## Build

```powershell
python build_public_site.py
```

## Main public outputs

- `docs/index.html`
- `docs/maps/la_permits_units_los_angeles.html`
- `docs/maps/la_supply_demand_balance_los_angeles.html`

## Core data sources

- LADBS permits dataset: `pi9x-tg5x`
- ACS 2023 5-year tract data
- Census TIGER 2023 tract and place geometries
