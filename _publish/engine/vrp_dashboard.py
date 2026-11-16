"""vrp_dashboard.py | Nairobi VRP Matrix | unified Colab dashboard (Phase 5)"""
import os
import json
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, Image
import vrp_engine
import vrp_maps

DASH_VERSION = "1.0"
_CACHE = {}
STATUS = {"solve": None, "dis": None, "view": None}
H1, H2 = "H1 zone sweep (simulated)", "H2 v2 nearest neighbour (simulated)"
TABLES = {"KPI scorecard": "t1_kpi_scorecard.csv", "Benchmark instances": "t2_instances.csv",
          "Savings statistics": "t3_savings_stats.csv", "Cost-assumption variants": "t4_cost_variants.csv",
          "Gate policies (benchmark)": "t5_gate_policies_benchmark.csv", "Lag test (class D scenarios)": "t6_lag_test.csv"}
FIGURES = {"Figure 1. Savings by instance": "final_fig1_savings.png", "Figure 2. Cost assumptions": "final_fig2_cost_assumptions.png",
           "Figure 3. Outcome map (27 scenarios)": "final_fig3_outcome_map.png", "Figure 4. Disruption benchmark": "final_fig4_disruption.png",
           "Figure 5. Detection lag": "final_fig5_lag.png"}


def instance_labels(project_dir):
    m = json.load(open(os.path.join(project_dir, "data", "instances", "manifest.json")))
    labels = ["Phase 1 instance (seed 42, 18 nodes)"]
    for r in m["records"]:
        labels.append("Benchmark %d (%s, %d nodes)" % (r["seed"], r["kind"], r["n_nodes"]))
    return labels


def load_instance(project_dir, label):
    assert label in instance_labels(project_dir), "Unknown instance: " + str(label)
    key = (project_dir, label)
    if key in _CACHE:
        return _CACHE[key]
    if label.startswith("Phase 1"):
        st = vrp_engine.load_state(project_dir)
        routes = json.load(open(os.path.join(project_dir, "outputs", "baseline_plan.json")))["routes"]
    else:
        seed = int(label.split()[1])
        folder = os.path.join(project_dir, "data", "instances", "inst_%d" % seed)
        st = vrp_engine.load_state_from(project_dir, folder)
        routes = json.load(open(os.path.join(folder, "optimized_plan.json")))["best_departure"]["routes"]
    _CACHE[key] = (st, routes)
    return _CACHE[key]


def default_corridor(st):
    best = None
    for c in st["config"]["clusters"]:
        if not c["congested"]:
            continue
        d = int(st["nodes"].loc[st["nodes"]["cluster"] == c["name"], "demand_kg"].sum())
        key = (-d, c["name"])
        if best is None or key < best[0]:
            best = (key, c["name"])
    return best[1]


def _check_cap(st, res):
    dem, cap = st["demand"], st["CAP"]
    for f, seq in zip(res["fleet"], res["routes"]):
        rem = [x for x in seq if x != 0 and x not in f["committed"]]
        assert int(sum(int(dem[x]) for x in rem)) <= cap, "Remaining load over capacity"


def _check_physical(routes, res):
    own = {x: v for v, seq in enumerate(routes) for x in seq if x != 0}
    stat = {f["vehicle"]: f["status"] for f in res["fleet"]}
    new = {x: v for v, seq in enumerate(res["routes"]) for x in seq if x != 0}
    for x, v0 in own.items():
        if stat[v0] in ("enroute", "done"):
            assert new[x] == v0, "Cargo rule broken for stop " + str(x)
        else:
            assert stat[new[x]] == "depot", "Cargo rule broken for stop " + str(x)


def solve_view(project_dir, label, overrides=None, use_stored=True, limit=8.0):
    st, stored = load_instance(project_dir, label)
    rates = vrp_engine.cost_rates(st["config"], overrides)
    wall = None
    if use_stored:
        routes = stored
        note = "Stored optimized plan (first 8 s run) repriced at the slider rates; it is not re-optimized for them."
    else:
        res = vrp_engine.solve(st, rates, time_limit_sec=float(limit))
        assert res["routes"] is not None, "No solution within the search limit"
        routes, wall = res["routes"], res["wall_s"]
        note = "Solved now with a %.0f s search under the slider rates (wall-clock %.2f s)." % (float(limit), wall)
    vrp_engine.verify_plan(st, routes, rates)
    r1, _ = vrp_engine.heuristic_zone_sweep(st)
    r2, _ = vrp_engine.heuristic_nearest_neighbour_v2(st)
    plans, rows = {}, []
    for name, rts in (("Optimizer", routes), (H1, r1), (H2, r2)):
        s, sim = vrp_engine.score_plan(st, rts, rates)
        plans[name] = (rts, s, sim)
        rows.append({"plan": name, "cost_kes": s["total_kes"], "fuel_kes": s["fuel_kes"], "time_kes": s["time_kes"],
                     "km": s["km"], "duration_min": s["duration_min"], "breaches": s["breaches"],
                     "minutes_late": s["total_late_min"], "vehicles": s["vehicles_used"]})
    opt = plans["Optimizer"][1]["total_kes"]
    lines = []
    for name in (H1, H2):
        s = plans[name][1]
        flag = "like-for-like" if s["breaches"] == 0 else "not like-for-like: the human plan has %d breach(es)" % s["breaches"]
        lines.append("Optimizer saving vs %s: %.1f%% (%s)" % (name, 100.0 * (s["total_kes"] - opt) / s["total_kes"], flag))
    return {"state": st, "rates": rates, "note": note, "wall_s": wall, "table": pd.DataFrame(rows), "plans": plans, "lines": lines}


def disruption_view(project_dir, label, corridor, severity, onset, duration, lag, budget, idealized=False):
    st, routes = load_instance(project_dir, label)
    rates = vrp_engine.cost_rates(st["config"])
    tick = st["config"]["dynamic"]["tick_min"]
    deps = vrp_engine.static_departures(st, routes)
    sc = vrp_engine.make_scenario(st, corridor, severity, onset, duration)
    t_det = next((t for t in range(0, st["H"] + 1, tick) if t >= onset + lag), None)
    assert t_det is not None, "Detection falls after the shift ends"
    r1, _ = vrp_engine.heuristic_zone_sweep(st)
    s1, _ = vrp_engine.score_plan(st, r1, rates)
    plans = {"H1 (simulated, blind)": (r1, s1["departures"], vrp_engine.simulate_plan(st, r1, rates, sc, s1["departures"])),
             "Optimized static": (routes, deps, vrp_engine.simulate_plan(st, routes, rates, sc, deps))}
    res_p = vrp_engine.resolve_dynamic(st, rates, sc, routes, deps, t_det, time_limit_sec=float(budget), pool_enroute=False)
    assert res_p["solved"], "The physical re-solve returned no solution"
    _check_cap(st, res_p)
    _check_physical(routes, res_p)
    plans["Dynamic, physical"] = (res_p["routes"], res_p["departures"], res_p["sim"])
    walls = {"physical": res_p["wall_s"]}
    ideal_br = None
    if idealized:
        res_i = vrp_engine.resolve_dynamic(st, rates, sc, routes, deps, t_det, time_limit_sec=float(budget), pool_enroute=True)
        assert res_i["solved"], "The idealized re-solve returned no solution"
        _check_cap(st, res_i)
        plans["Dynamic, idealized (relaxation)"] = (res_i["routes"], res_i["departures"], res_i["sim"])
        walls["idealized"] = res_i["wall_s"]
        ideal_br = res_i["sim"]["summary"]["breaches"]
    rows = []
    for name, (rts, dps, sim) in plans.items():
        s = sim["summary"]
        rows.append({"plan": name, "cost_kes": s["total_kes"], "fuel_kes": s["fuel_kes"], "time_kes": s["time_kes"], "km": s["km"],
                     "duration_min": s["duration_min"], "breaches": s["breaches"], "key_breaches": s["key_account_breaches"],
                     "minutes_late": s["total_late_min"]})
    sta, phy = plans["Optimized static"][2]["summary"], plans["Dynamic, physical"][2]["summary"]
    if sta["breaches"] == 0:
        cls = "A (the static plan holds the SLA)"
    elif phy["breaches"] == 0:
        cls = "B (the dynamic re-solve restores the SLA)"
    elif ideal_br is None:
        cls = "C or D (tick the idealized option to tell)"
    else:
        cls = "C (only the idealized relaxation restores it)" if ideal_br == 0 else "D (neither restores it)"
    adopt = (phy["breaches"] < sta["breaches"]) or (phy["breaches"] == sta["breaches"] and phy["total_kes"] < sta["total_kes"])
    return {"state": st, "scenario": sc, "t_detect": t_det, "plans": plans, "table": pd.DataFrame(rows), "class": cls,
            "gate_g1_adopts_dynamic": bool(adopt), "walls": walls}


def benchmark_view(project_dir, name):
    assert name in TABLES or name in FIGURES, "Unknown view: " + str(name)
    base = os.path.join(project_dir, "outputs", "final")
    if name in TABLES:
        path = os.path.join(base, TABLES[name])
        assert os.path.exists(path), "Missing " + TABLES[name] + ": run Cell 5.1"
        return "table", pd.read_csv(path)
    path = os.path.join(base, FIGURES[name])
    assert os.path.exists(path), "Missing " + FIGURES[name] + ": run Cell 5.1"
    return "image", path


def _solve_map(v):
    st = v["state"]
    lines = [vrp_maps.summary_line(n, p[1]) for n, p in v["plans"].items()]
    m = vrp_maps.new_map(st, "Optimizer vs simulated dispatchers (toggle layers, top right)", lines)
    vrp_maps.add_corridors(m, st)
    first = True
    for name, (rts, s, sim) in v["plans"].items():
        vrp_maps.add_plan(m, st, rts, s["departures"], sim, name, None, show=first, dashed=not first)
        first = False
    return vrp_maps.finish_map(m, st)


def _dis_map(v):
    st, sc = v["state"], v["scenario"]
    lines = [vrp_maps.summary_line(n, p[2]["summary"]) for n, p in v["plans"].items()]
    m = vrp_maps.new_map(st, "%s x%s from minute %d for %d min" % (sc["corridor"], sc["severity"], sc["onset"], sc["duration"]), lines)
    vrp_maps.add_corridors(m, st, sc)
    for name, (rts, dps, sim) in v["plans"].items():
        vrp_maps.add_plan(m, st, rts, dps, sim, name, sc, show=(name == "Dynamic, physical"), dashed=name.startswith("H1"))
    return vrp_maps.finish_map(m, st)


def launch(project_dir):
    labels = instance_labels(project_dir)
    c0 = vrp_engine.load_state(project_dir)["config"]["cost"]
    sty, lay = {"description_width": "160px"}, widgets.Layout(width="540px")
    header = widgets.HTML("<h3>Nairobi VRP Matrix: dashboard</h3><p>Synthetic instances and scenarios. The human plans are "
                          "simulated rules, not observed dispatcher behaviour. Every plan is executed by an independent "
                          "simulator; breaches are reported, not priced. The idealized mode is a relaxation.</p>")
    # ----- tab 1: solve -----
    dd1 = widgets.Dropdown(options=labels, value=labels[0], description="Instance", style=sty, layout=lay)
    s_dz = widgets.FloatSlider(value=c0["diesel_price_kes_per_l"], min=150, max=300, step=0.5, description="Diesel (KES/L)",
                               continuous_update=False, style=sty, layout=lay)
    s_fc = widgets.FloatSlider(value=c0["fuel_l_per_100km"], min=8, max=20, step=0.5, description="Fuel (L/100 km)",
                               continuous_update=False, style=sty, layout=lay)
    s_dr = widgets.FloatSlider(value=c0["driver_vehicle_kes_per_hour"], min=100, max=600, step=10, description="Driver+vehicle (KES/h)",
                               continuous_update=False, style=sty, layout=lay)
    cb1 = widgets.Checkbox(value=True, description="Use the stored optimized plan (instant); untick to solve now",
                           style={"description_width": "initial"})
    dl1 = widgets.Dropdown(options=[1, 2, 5, 8], value=8, description="Search limit (s)", style=sty, layout=lay)
    b1 = widgets.Button(description="Run", button_style="primary")
    out1 = widgets.Output()

    def on_solve(_=None):
        with out1:
            clear_output(wait=True)
            print("Working...")
        try:
            ov = {"diesel_price_kes_per_l": s_dz.value, "fuel_l_per_100km": s_fc.value, "driver_vehicle_kes_per_hour": s_dr.value}
            v = solve_view(project_dir, dd1.value, ov, cb1.value, dl1.value)
            m = _solve_map(v)
        except Exception as e:
            with out1:
                clear_output(wait=True)
                print("Could not run: %s" % e)
            STATUS["solve"] = ("error", str(e))
            return
        STATUS["solve"] = ("ok", v["lines"][0])
        with out1:
            clear_output(wait=True)
            print(v["note"])
            display(v["table"])
            for ln in v["lines"]:
                print(ln)
            display(m)

    b1.on_click(on_solve)
    # ----- tab 2: disruption -----
    dd2 = widgets.Dropdown(options=labels, value=labels[1], description="Instance", style=sty, layout=lay)
    dc = widgets.Dropdown(options=[""], value="", description="Corridor", style=sty, layout=lay)
    ss = widgets.FloatSlider(value=2.0, min=1.0, max=4.0, step=0.1, description="Severity (x)", continuous_update=False, style=sty, layout=lay)
    so = widgets.IntSlider(value=0, min=0, max=540, step=15, description="Onset (min)", continuous_update=False, style=sty, layout=lay)
    sd = widgets.IntSlider(value=240, min=30, max=600, step=30, description="Duration (min)", continuous_update=False, style=sty, layout=lay)
    sl = widgets.IntSlider(value=10, min=0, max=60, step=5, description="Detection lag (min)", continuous_update=False, style=sty, layout=lay)
    db = widgets.Dropdown(options=[2, 3, 5], value=5, description="Re-solve budget (s)", style=sty, layout=lay)
    ci = widgets.Checkbox(value=False, description="Also run the idealized mode (adds about 5 s; a relaxation)",
                          style={"description_width": "initial"})
    b2 = widgets.Button(description="Run scenario", button_style="primary")
    out2 = widgets.Output()

    def set_corridors(_=None):
        st, _routes = load_instance(project_dir, dd2.value)
        names = [c["name"] for c in st["config"]["clusters"] if c["congested"]]
        dc.options = names
        dc.value = default_corridor(st)

    set_corridors()
    dd2.observe(set_corridors, names="value")

    def on_dis(_=None):
        with out2:
            clear_output(wait=True)
            print("Solving...")
        try:
            v = disruption_view(project_dir, dd2.value, dc.value, ss.value, so.value, sd.value, sl.value, db.value, ci.value)
            m = _dis_map(v)
        except Exception as e:
            with out2:
                clear_output(wait=True)
                print("Could not run: %s" % e)
            STATUS["dis"] = ("error", str(e))
            return
        STATUS["dis"] = ("ok", v["class"])
        with out2:
            clear_output(wait=True)
            sc = v["scenario"]
            print("%s x%s | onset %d | duration %d min | detection at minute %d | re-solve wall-clock %s s" %
                  (sc["corridor"], sc["severity"], sc["onset"], sc["duration"], v["t_detect"],
                   {k: round(x, 2) for k, x in v["walls"].items()}))
            display(v["table"])
            print("Outcome class: %s" % v["class"])
            print("Gate G1 would adopt the dynamic plan: %s" % ("yes" if v["gate_g1_adopts_dynamic"] else "no"))
            print("Plans run as planned under the disruption (no reactive human re-dispatch simulated); the re-solve is told the true severity and end time.")
            display(m)

    b2.on_click(on_dis)
    # ----- tab 3: benchmark -----
    options = list(TABLES) + list(FIGURES)
    dd3 = widgets.Dropdown(options=options, value=options[0], description="Show", style=sty, layout=lay)
    out3 = widgets.Output()

    def on_view(_=None):
        with out3:
            clear_output(wait=True)
            try:
                kind, obj = benchmark_view(project_dir, dd3.value)
            except Exception as e:
                print("Could not load: %s" % e)
                STATUS["view"] = ("error", str(e))
                return
            STATUS["view"] = ("ok", kind)
            if kind == "table":
                with pd.option_context("display.max_colwidth", 400, "display.max_columns", 30):
                    display(obj)
            else:
                display(Image(filename=obj))
            print("Map gallery: open maps/gallery/index.html from Drive.")

    dd3.observe(on_view, names="value")
    tab = widgets.Tab(children=[widgets.VBox([dd1, s_dz, s_fc, s_dr, cb1, dl1, b1, out1]),
                                widgets.VBox([dd2, dc, ss, so, sd, sl, db, ci, b2, out2]),
                                widgets.VBox([dd3, out3])])
    for i, t in enumerate(("1. Solve", "2. Disruption", "3. Benchmark")):
        tab.set_title(i, t)
    display(header, tab)
    return {"tab": tab, "on_solve": on_solve, "on_dis": on_dis, "on_view": on_view, "dd_inst1": dd1, "dd_inst2": dd2, "dd_corridor": dc}
