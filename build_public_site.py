from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from la_permits_units_map import export_map as export_permits_map
from la_permits_units_map import prepare_map_frame as prepare_permits_map_frame
from la_permits_units_map import build_deck as build_permits_deck
from la_supply_demand_balance_map import export_map as export_balance_map
from la_supply_demand_balance_map import prepare_map_frame as prepare_balance_map_frame
from la_supply_demand_balance_map import build_deck as build_balance_deck


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
MAPS_DIR = DOCS_DIR / "maps"
DATA_DIR = DOCS_DIR / "data"
START_DATE = date(2025, 3, 14)
END_DATE = date(2026, 3, 14)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def format_int(value: float) -> str:
    return f"{int(round(float(value))):,}"


def build_index_html(permits_stats: dict[str, float | str], balance_stats: dict[str, float | str]) -> str:
    start_label = datetime.strptime(str(permits_stats["start_date"]), "%Y-%m-%d").strftime("%b %d, %Y")
    end_label = datetime.strptime(str(permits_stats["end_date"]), "%Y-%m-%d").strftime("%b %d, %Y")

    positive_units = format_int(float(permits_stats["citywide_positive_units"]))
    housing_projects = format_int(float(permits_stats["citywide_housing_projects"]))
    raw_unit_rows = format_int(float(permits_stats["citywide_raw_unit_bearing_permit_rows"]))
    duplicate_units_removed = format_int(float(permits_stats["citywide_duplicate_positive_units_removed"]))
    spatial_units = format_int(float(permits_stats["citywide_spatial_fallback_positive_units"]))
    unassigned_units = format_int(float(permits_stats["citywide_unassigned_positive_units"]))
    recovered_projects = format_int(float(permits_stats["recovered_project_count"]))
    unassigned_projects = format_int(float(permits_stats["unassigned_project_count"]))
    high_imbalance = format_int(float(balance_stats["high_imbalance_tracts"]))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Los Angeles Housing Supply Reconciliation</title>
  <meta name="description" content="Public-facing Los Angeles housing permit maps with tract reconciliation and privacy-safe project markers." />
  <link rel="stylesheet" href="assets/site.css" />
</head>
<body>
  <div class="page-shell">
    <header class="header-band">
      <div class="header-inner">
        <p class="eyebrow">LOS ANGELES HOUSING PERMITS | PUBLIC SITE</p>
        <h1>Housing supply appears low because the permit dataset needs tract repair and family-level cleanup, not because a large block of units is missing.</h1>
        <p class="lede">This site reconciles LADBS permit rows issued between {start_label} and {end_label} into a tract-level housing supply view, then pairs that supply signal with ACS market context. Address-level and permit-number details are intentionally excluded from the published pages.</p>
      </div>
    </header>

    <main class="content-grid">
      <section class="takeaway-strip">
        <article class="metric-card">
          <span class="metric-label">Reconciled Positive Units</span>
          <strong class="metric-value">{positive_units}</strong>
          <p>Positive unit additions after tract repair and family-level deduping.</p>
        </article>
        <article class="metric-card">
          <span class="metric-label">Housing Projects</span>
          <strong class="metric-value">{housing_projects}</strong>
          <p>Deduped housing project families represented in the permit rollup.</p>
        </article>
        <article class="metric-card accent-card">
          <span class="metric-label">Recovered Units</span>
          <strong class="metric-value">{spatial_units}</strong>
          <p>{recovered_projects} anonymized projects reassigned by geospatial fallback because the published tract code did not match the current tract layer.</p>
        </article>
        <article class="metric-card danger-card">
          <span class="metric-label">Still Unassigned</span>
          <strong class="metric-value">{unassigned_units}</strong>
          <p>{unassigned_projects} anonymized projects still fall outside the current city tract layer and are marked separately on the maps.</p>
        </article>
      </section>

      <section class="narrative-panel">
        <div class="section-band">KEY READS</div>
        <div class="narrative-grid">
          <article class="note-card">
            <h2>The raw permit file is close on scale, but it is not presentation-ready.</h2>
            <p>The reconciliation removed {duplicate_units_removed} duplicate positive units from supplemental permit families and recovered {spatial_units} units that were being dropped because LADBS tract codes did not line up with the current Census tract layer.</p>
          </article>
          <article class="note-card">
            <h2>The supply map and imbalance map now use the same repaired project rollup.</h2>
            <p>That makes the two views consistent: the supply map shows where units were permitted, while the balance map places that reconciled supply against rent burden, rent level, and vacancy pressure.</p>
          </article>
          <article class="note-card">
            <h2>Privacy-safe publishing required stripping address-level detail.</h2>
            <p>The public site excludes raw addresses, parcel references, and permit numbers. Recovered and unassigned projects are shown with anonymized IDs only.</p>
          </article>
          <article class="note-card">
            <h2>High-imbalance tracts still outnumber obvious supply-led tracts.</h2>
            <p>{high_imbalance} tracts currently rank as high-imbalance areas in the supply-demand view, even after the tract repair and deduping pass.</p>
          </article>
        </div>
      </section>

      <section class="map-section">
        <div class="section-band">MAP 1 | RECONCILED SUPPLY</div>
        <div class="map-shell">
          <div class="map-copy">
            <h2>Reconciled permitted housing units by tract</h2>
            <p>Yellow markers show anonymized recovered projects reassigned by geospatial fallback. Red markers show the few positive-unit projects still outside the current tract layer.</p>
            <p class="micro-note">The raw unit-bearing permit row count is {raw_unit_rows}, but the public map treats those as underlying records, not final project counts.</p>
            <a class="map-link" href="maps/la_permits_units_los_angeles.html" target="_blank" rel="noreferrer">Open full map</a>
          </div>
          <iframe title="Los Angeles reconciled permitted units map" src="maps/la_permits_units_los_angeles.html" loading="lazy"></iframe>
        </div>
      </section>

      <section class="map-section">
        <div class="section-band">MAP 2 | SUPPLY VS DEMAND</div>
        <div class="map-shell">
          <div class="map-copy">
            <h2>Supply-demand balance after permit reconciliation</h2>
            <p>Height shows reconciled positive units per 1,000 homes. Color reads local shortage pressure relative to the observed supply response.</p>
            <p class="micro-note">Inputs combine reconciled LADBS permits with 2023 ACS housing units, vacancy, rent burden, and gross rent.</p>
            <a class="map-link" href="maps/la_supply_demand_balance_los_angeles.html" target="_blank" rel="noreferrer">Open full map</a>
          </div>
          <iframe title="Los Angeles supply-demand balance map" src="maps/la_supply_demand_balance_los_angeles.html" loading="lazy"></iframe>
        </div>
      </section>

      <section class="method-panel">
        <div class="section-band">METHOD</div>
        <div class="method-grid">
          <div>
            <h3>Reconciliation steps</h3>
            <ol>
              <li>Pull live LADBS permit rows for the selected period.</li>
              <li>Compute structured unit changes from dwelling-unit fields rather than text descriptions.</li>
              <li>Repair stale tract codes using geospatial point-in-polygon fallback.</li>
              <li>Deduplicate supplemental permit families into project-level rows.</li>
              <li>Aggregate reconciled projects to Census tracts for mapping.</li>
            </ol>
          </div>
          <div>
            <h3>Publishing note</h3>
            <p>This public site omits address-level audit files. Private QA artifacts remain in the local project folder only and are not part of the Pages build.</p>
          </div>
        </div>
      </section>
    </main>
  </div>
</body>
</html>
"""


def build_site_css() -> str:
    return """\
:root {
  --bg: #f1f1f1;
  --text: #4c4c4c;
  --band: #767662;
  --band-dark: #59594a;
  --stone: #a6a08d;
  --teal: #2babb9;
  --gold: #eba923;
  --amber: #d77d28;
  --purple: #633377;
  --danger: #a14a44;
  --card: #fbfbf8;
  --line: rgba(118, 118, 98, 0.24);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: "Arial Narrow", Arial, Helvetica, sans-serif;
}

.page-shell {
  min-height: 100vh;
}

.header-band {
  background: var(--band);
  color: white;
  padding: 42px 0 36px;
  border-bottom: 8px solid var(--band-dark);
}

.header-inner,
.content-grid {
  width: min(1240px, calc(100vw - 48px));
  margin: 0 auto;
}

.eyebrow {
  margin: 0 0 10px;
  font-size: 13px;
  letter-spacing: 0.12em;
  opacity: 0.82;
}

.header-inner h1 {
  margin: 0;
  max-width: 1020px;
  font-size: clamp(32px, 4vw, 48px);
  line-height: 1.03;
  letter-spacing: -0.02em;
}

.lede {
  margin: 18px 0 0;
  max-width: 900px;
  font-size: 18px;
  line-height: 1.45;
  color: rgba(255, 255, 255, 0.88);
}

.content-grid {
  padding: 28px 0 42px;
  display: grid;
  gap: 22px;
}

.takeaway-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.metric-card,
.note-card,
.map-shell,
.method-panel {
  background: var(--card);
  border: 1px solid var(--line);
}

.metric-card {
  padding: 18px 18px 20px;
}

.metric-label {
  display: block;
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--band);
}

.metric-value {
  display: block;
  margin-top: 10px;
  font-size: clamp(28px, 3vw, 42px);
  line-height: 1;
  color: var(--band-dark);
}

.metric-card p {
  margin: 12px 0 0;
  font-size: 15px;
  line-height: 1.35;
}

.accent-card .metric-value,
.accent-card .metric-label { color: var(--amber); }

.danger-card .metric-value,
.danger-card .metric-label { color: var(--danger); }

.section-band {
  background: var(--band);
  color: white;
  padding: 10px 14px;
  font-size: 14px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.narrative-panel,
.map-section,
.method-panel {
  border: 1px solid var(--line);
  background: var(--card);
}

.narrative-grid,
.method-grid {
  padding: 18px;
  display: grid;
  gap: 16px;
}

.narrative-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.note-card {
  padding: 18px;
}

.note-card h2,
.map-copy h2,
.method-grid h3 {
  margin: 0 0 10px;
  font-size: 24px;
  color: var(--band-dark);
}

.note-card p,
.map-copy p,
.method-grid p,
.method-grid li {
  margin: 0;
  font-size: 16px;
  line-height: 1.45;
}

.map-shell {
  padding: 18px;
  display: grid;
  grid-template-columns: 280px minmax(0, 1fr);
  gap: 18px;
}

.map-copy {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.micro-note {
  color: #737366;
  font-size: 14px;
}

.map-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: fit-content;
  min-width: 148px;
  padding: 10px 14px;
  background: var(--band-dark);
  color: white;
  text-decoration: none;
  font-size: 15px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

iframe {
  width: 100%;
  min-height: 860px;
  border: 1px solid rgba(118, 118, 98, 0.18);
  background: white;
}

.method-grid {
  grid-template-columns: 1.4fr 1fr;
}

ol {
  margin: 0;
  padding-left: 20px;
}

li + li {
  margin-top: 8px;
}

@media (max-width: 1100px) {
  .takeaway-strip,
  .narrative-grid,
  .method-grid,
  .map-shell {
    grid-template-columns: 1fr 1fr;
  }

  .map-shell iframe {
    grid-column: 1 / -1;
  }
}

@media (max-width: 760px) {
  .header-inner,
  .content-grid {
    width: min(100vw - 28px, 1240px);
  }

  .takeaway-strip,
  .narrative-grid,
  .method-grid,
  .map-shell {
    grid-template-columns: 1fr;
  }

  .header-band {
    padding-top: 32px;
  }

  iframe {
    min-height: 680px;
  }
}
"""


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    permits_frame, permits_stats, permits_points = prepare_permits_map_frame(START_DATE, END_DATE)
    permits_deck = build_permits_deck(permits_frame, permits_points)
    export_permits_map(permits_deck, MAPS_DIR / "la_permits_units_los_angeles.html", permits_stats, False)

    balance_frame, balance_stats, balance_points = prepare_balance_map_frame(START_DATE, END_DATE)
    balance_deck = build_balance_deck(balance_frame, balance_points)
    export_balance_map(balance_deck, MAPS_DIR / "la_supply_demand_balance_los_angeles.html", balance_stats, False)

    write_text(DOCS_DIR / "index.html", build_index_html(permits_stats, balance_stats))
    write_text(DOCS_DIR / "assets" / "site.css", build_site_css())
    write_text(DOCS_DIR / ".nojekyll", "")

    summary_json = {
        "start_date": permits_stats["start_date"],
        "end_date": permits_stats["end_date"],
        "positive_units": int(round(float(permits_stats["citywide_positive_units"]))),
        "housing_projects": int(round(float(permits_stats["citywide_housing_projects"]))),
        "raw_unit_bearing_permit_rows": int(round(float(permits_stats["citywide_raw_unit_bearing_permit_rows"]))),
        "duplicate_positive_units_removed": int(round(float(permits_stats["citywide_duplicate_positive_units_removed"]))),
        "spatial_fallback_positive_units": int(round(float(permits_stats["citywide_spatial_fallback_positive_units"]))),
        "unassigned_positive_units": int(round(float(permits_stats["citywide_unassigned_positive_units"]))),
        "high_imbalance_tracts": int(round(float(balance_stats["high_imbalance_tracts"]))),
    }
    write_text(DOCS_DIR / "data" / "summary.json", __import__("json").dumps(summary_json, indent=2))

    print(f"Built public site: {DOCS_DIR}")
    print(f"Maps: {MAPS_DIR}")


if __name__ == "__main__":
    main()
