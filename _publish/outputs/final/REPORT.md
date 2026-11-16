# The Nairobi VRP Matrix: delivery report

**Subject:** Dynamic vehicle routing under simulated traffic disruption for Nairobi last-mile distribution
**Basis:** 13 feasibility-gated synthetic instances (10 at 18 delivery nodes, 3 at 30) plus 27 disruption scenarios on the Phase 1 instance and 26 on the benchmark set
**Status:** Alpha. Synthetic data only. Human plans are simulated dispatch rules, not observed dispatcher behaviour.

## 1. The operational problem

A static delivery plan driven through a disrupted Mombasa Road will miss at least one window. That is not a routing failure; it is a planning failure. The optimized static plan that is 10.0% cheaper than the Phase 1 first solution (KES 7225 versus KES 8026) still breaches the SLA in 21 of 26 disruption scenarios on the benchmark set, producing 41 late deliveries and 1297 minutes of lateness. The question this project answers is whether a 5-second re-solve, executed once at the moment the disruption is detected, materially reduces that damage.

## 2. What was built

A constraint-programming engine (Google OR-Tools) solving capacity-constrained, time-windowed VRPs on a synthetic Nairobi network, with a disruption simulator, a dynamic re-solve pipeline, a synthetic dispatcher benchmark, a map gallery, and an interactive dashboard. Every plan is executed by an independent simulator that reports breaches; the solver's own objective is never trusted.

## 3. The three PRD KPIs

| KPI | Verdict | Evidence |
|---|---|---|
| 1. Dynamic reroute holds 100% SLA | Met in a majority of scenarios; not in every scenario | Phase 3C grid: physical 15 of 19, idealized 18 of 19. Benchmark: physical 14 of 21, idealized 16 of 21. |
| 2. Dynamic cost below the static route through the bottleneck | **Not met as written** | 4 of 27 grid scenarios, 3 of 26 benchmark scenarios. A disruption lengthens travel; the static plan driven through it is rarely beaten on cost. The defensible cost result is the ~20.7% like-for-like saving against the H1 comparator (finding 4.1). |
| 3. Solve and re-solve under 10 s | Met, at 18 and 30 nodes | Optimizer solve max 8.04 s (build + 8 s search + verification); re-solve max 5.21 s (grid) and 5.40 s (benchmark). A time limit makes this near-guaranteed; the number measures wall-clock, not solving quality. |

## 4. Three findings

**4.1 The optimizer saves 20.7% against a simulated zone-sweep dispatcher, like-for-like (n=7).**
Against H1 the like-for-like mean is 20.7% (median 23.7%, range 12.6–27.3%); against H2 v2 27.4% (median 26.8%, range 12.0–37.5%, n=7); against the machine first solution 4.8%. Both comparators are simulated rules, not observed dispatcher behaviour. Under the fairness rule that puts every vehicle on the clock from minute 0 (removing the optimizer's late-departure advantage), the saving against H1 is 16.3% (range 7.4–23.0%). Across the six cost-assumption variants the saving over the unoptimized Phase 1 plan ranges from 5.8% to 21.1%.

**4.2 Where the re-solve cannot restore the SLA, the constraint is physical, not algorithmic.**
Four benchmark scenarios fall into class D (neither physical nor idealized restores the SLA). Detecting the disruption at lag 0, the moment it begins, restores the SLA in 1 of 4. Lateness across those four scenarios falls from 113 to 85 minutes, so earlier detection helps, but does not fix. The blocking constraint is committed cargo: a vehicle already out of the depot with a stop it has not yet served cannot hand that cargo to a vehicle still at the depot. The Phase 3C hypothesis ("affected vehicle already out of the depot at detection") is directionally supported but not a rule: the benchmark has 5 counter-examples where an affected vehicle was out and the re-solve still restored the SLA.

**4.3 The gate rule protects against the wrong kind of adoption.**
A naive "always adopt the dynamic plan" policy would accept a re-solve that is cheaper but breaks the SLA. In the benchmark exactly one scenario (seed 108, onset 0) has a dynamic plan at KES 6816 with 1 breach while the static plan held the SLA at KES 7156. Gate G1 (adopt only if breaches fall, or tie at lower cost) correctly rejects it and adopts the dynamic plan in 20 of 26 physical scenarios, 22 of 26 idealized. Over the 26 scenarios: static plan 41 breaches / 1297 min late; always-adopt-dynamic (physical) 10 breaches / 214 min late; G1-gated (physical) 9 breaches / 211 min late; G1-gated (idealized) 6 breaches / 75 min late.

## 5. Limitations (labeled, not findings)

- The human plans are simulated rules, not observed dispatcher behaviour, and say nothing about real Nairobi dispatchers.
- All instances and scenarios are synthetic, generated from the same travel-time assumptions the optimizer uses; no model error is tested.
- The idealized mode is a relaxation: cargo moves between moving trucks. Its cost wins in the disruption benchmark are partly artifacts. No idealized cost win is quoted as a real saving.
- KPI 2 is not met as written. A disruption lengthens travel; the static plan driven through the bottleneck is rarely beaten on cost. The cost headline is the ~20.7% saving versus the H1 comparator, not the 3/26 KPI 2 count.
- KPI 3 is near-guaranteed by the solver's time limit; it measures wall-clock, not solution quality.
- The ingestion path is tested by round-trip only; no real SME dataset has been ingested.
- Utilization is 80–84% only; nothing near the 90% upper bound.
- The 10-area pool and its coordinates are approximate; corridor mix is skewed toward CBD and Ngong_Road.
- The benchmark disruption corridor is the largest-demand congested corridor (a choice, not a finding).
- Breaches are reported, not priced.
- The official optimized plan per instance is the first 8 s run; on one instance it is KES 112 above the second run.

## 6. How to reproduce

1. Open the notebook in Colab and mount Drive; the project directory is /content/drive/MyDrive/nairobi_vrp_matrix/.
2. Run cells in order: 0.1, 0.2, 1.1-1.5 (data), 2.0-2.9 (static baseline), 3A.1-3A.4 (static under disruption), 3B.1-3B.5 (dynamic re-solve), 3C.1-3C.3 (grid sensitivity), 4.1-4.7 (benchmark), 5.0-5.5 (lag test, results book, map gallery, dashboard, report, exit check).
3. Open maps/gallery/index.html in any browser to view the four maps without Python.
4. Open the dashboard's tab 3 to browse the six stored tables and five figures, or read outputs/final/t1_kpi_scorecard.csv.
5. Every number in this report traces to a source file via outputs/final/numbers_registry.json (which records the source path, sha256 fingerprint and byte count of every file the report cites).
