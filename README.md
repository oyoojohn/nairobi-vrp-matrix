# Nairobi VRP Matrix

A dynamic vehicle-routing engine for Nairobi last-mile distribution, built with Google OR-Tools in Google Colab. It plans a static route, simulates a traffic disruption on a congested corridor (Mombasa Road, Thika Road, Westlands, and others), re-solves mid-day, and benchmarks the result against simulated dispatcher rules. All data is synthetic.

**Start here:** [`REPORT.md`](REPORT.md) (two pages), then the [map gallery](maps/gallery/index.html) (open via GitHub Pages).

## Headline results

Single runs on 13 synthetic instances and 26 placeholder disruption scenarios.

- **Cost:** the optimizer was about 21% cheaper like-for-like than a simulated zone-sweep dispatcher (H1) and about 27% cheaper than a simulated nearest-neighbour dispatcher (H2 v2). It was about 5% cheaper than the solver's own first solution. These dispatchers are rules I wrote. They say nothing about real Nairobi dispatchers.
- **Disruption:** the optimized static plan breached a time window in 21 of 26 scenarios. A 5-second-capped re-solve restored SLA compliance in 14 of those (physical mode).
- **Where it fails:** when cargo is already on a truck on the road, no re-solve can move it. Committed cargo, not compute, is the binding constraint.
- **KPIs:** KPI 1 (reroute while keeping the SLA) met in a majority of scenarios. KPI 2 (dynamic route cheaper than driving through the bottleneck) **not met as written**: a disruption lengthens travel, so the static plan is rarely beaten on cost. KPI 3 (under 10 s) met, but the re-solve has a 5 s time cap, so this passes almost by construction.

## What is in the repo

| Path | Contents |
|---|---|
| `REPORT.md` | Two-page write-up; headline numbers checked against the numbers registry |
| `nairobi_vrp_matrix.ipynb` | Full build, cell by cell (Phases 0 to 5) |
| `engine/` | `vrp_engine.py` (solver, simulator, generator, heuristics, ingestion), `vrp_maps.py`, `vrp_dashboard.py` |
| `outputs/final/` | Results tables, figures, `numbers_registry.json` (every source file fingerprinted with sha256) |
| `maps/gallery/` | Four interactive maps and an index |
| `data/` | Phase 1 network, 13 benchmark instances, and a CSV ingestion template |
| `config.json` | Cost, solver, and disruption parameters |

## Running it

The notebook expects the project at `/content/drive/MyDrive/nairobi_vrp_matrix/` in Google Colab. Copy this repo's contents there, open the notebook, and run Cell 2.0 first (it mounts Drive and installs OR-Tools and Folium). The dashboard needs a Colab runtime. The maps need an internet connection for the Esri basemap.

## Limitations

- All instances, demand, time windows and disruptions are synthetic placeholders. No real SME dataset has been ingested. A CSV ingestion path exists and was tested by round trip only.
- Human dispatchers are simulated by two fixed rules. Savings against them are not savings against real practice.
- The "idealized" dynamic mode lets cargo move between moving trucks. It is a relaxation, so its cost wins are not reported as real savings.
- The re-solver is told the true severity and end time of the disruption (optimistic).
- Fuel (13 L/100 km) and driver cost (KES 200/hour) are assumptions. The diesel price is the EPRA maximum for the Oct 15 to Nov 14 2026 cycle.
- Results are single runs, not a calibrated distribution.
- Built with AI coding assistants. Headline numbers are regenerated from stored files and fingerprinted in the registry, but not every figure in the prose is registered.

## Tech

Python 3.13, OR-Tools 9.15, Pandas, NumPy, Folium, ipywidgets. Maps use Esri basemap tiles (attribution required).