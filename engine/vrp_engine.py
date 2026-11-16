"""vrp_engine.py | Nairobi VRP Matrix engine | Phase 2"""
import json, math, os, time
import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ENGINE_VERSION = "2.8"


def load_state(project_dir):
    """Load every Phase 0-1 artifact from Drive into one state dict."""
    P = lambda *a: os.path.join(project_dir, *a)
    with open(P("config.json")) as f:
        config = json.load(f)
    nodes = pd.read_csv(P("data", "nodes.csv")).sort_values("node_id").reset_index(drop=True)
    T = np.load(P("data", "travel_matrix.npy"))
    KM = np.load(P("data", "road_km_matrix.npy"))
    n = len(nodes)
    assert (nodes.node_id.to_numpy() == np.arange(n)).all(), "node_id must be 0..N-1 in order"
    assert T.shape == (n, n) and KM.shape == (n, n), "Matrix shape does not match node count"
    return {
        "project_dir": project_dir, "config": config, "nodes": nodes, "T": T, "KM": KM, "n": n,
        "V": config["network"]["n_vehicles"], "CAP": config["network"]["vehicle_capacity_kg"],
        "H": config["time"]["shift_end_min"],
        "demand": nodes.demand_kg.astype(int).to_numpy(),
        "service": nodes.service_min.astype(int).to_numpy(),
        "open": nodes.tw_open.astype(int).to_numpy(),
        "close": nodes.tw_close.astype(int).to_numpy(),
    }


def cost_rates(config, overrides=None):
    """Return (fuel KES per km, time KES per minute). overrides lets the GUI change inputs."""
    c = dict(config["cost"])
    if overrides:
        c.update(overrides)
    fuel_per_km = c["diesel_price_kes_per_l"] * c["fuel_l_per_100km"] / 100.0
    time_per_min = c["driver_vehicle_kes_per_hour"] / 60.0
    return fuel_per_km, time_per_min


def arc_cost_matrix(state, rates):
    """Integer fuel cost per arc, in KES-cents (cost_scale)."""
    fuel_per_km, _ = rates
    scale = state["config"]["cost"]["cost_scale"]
    return np.rint(state["KM"] * fuel_per_km * scale).astype(np.int64)


def best_schedule(state, seq):
    """Minimum-duration feasible schedule for a FIXED sequence, over all departure minutes."""
    T, svc, op, cl, H = state["T"], state["service"], state["open"], state["close"], state["H"]
    best = None
    for d in range(0, H + 1):
        t, wait, ok = d, 0, True
        for a, b in zip(seq[:-1], seq[1:]):
            t += int(T[a, b])
            if b != 0:
                if t > cl[b]:
                    ok = False
                    break
                if t < op[b]:
                    wait += int(op[b] - t)
                    t = int(op[b])
                t += int(svc[b])
        if ok and t <= H:
            span = t - d
            if best is None or span < best["span"]:
                best = {"span": span, "depart": d, "mid_wait": wait, "return": t}
    return best


def verify_plan(state, routes, rates):
    """Independent verification + pricing. Raises AssertionError on ANY violation."""
    n, V, CAP = state["n"], state["V"], state["CAP"]
    T, KM, svc, dem = state["T"], state["KM"], state["service"], state["demand"]
    fuel_per_km, time_per_min = rates
    assert len(routes) == V, f"Expected {V} routes, got {len(routes)}"
    # Structural checks first (cheap, deterministic error priority), then per-vehicle checks
    for v, seq in enumerate(routes):
        assert seq[0] == 0 and seq[-1] == 0, f"Vehicle {v}: route must start and end at the depot"
    all_stops = sorted(x for seq in routes for x in seq if x != 0)
    assert all_stops == list(range(1, n)), "Nodes missed or duplicated"
    visited, rows = [], []
    for v, seq in enumerate(routes):
        stops = [x for x in seq if x != 0]
        visited += stops
        if not stops:
            rows.append({"vehicle": v, "route": "0-0", "stops": 0, "load_kg": 0, "utilization": 0.0,
                         "depart_min": 0, "return_min": 0, "duration_min": 0, "travel_min": 0,
                         "service_min": 0, "mid_route_wait_min": 0, "km": 0.0,
                         "fuel_kes": 0, "time_kes": 0})
            continue
        load = int(sum(dem[x] for x in stops))
        assert load <= CAP, f"Vehicle {v}: overload {load} > {CAP}"
        s = best_schedule(state, seq)
        assert s is not None, f"Vehicle {v}: no feasible schedule (window, return or shift violation)"
        km = float(sum(KM[a, b] for a, b in zip(seq[:-1], seq[1:])))
        travel = int(sum(T[a, b] for a, b in zip(seq[:-1], seq[1:])))
        rows.append({"vehicle": v, "route": "-".join(map(str, seq)), "stops": len(stops),
                     "load_kg": load, "utilization": round(load / CAP, 3),
                     "depart_min": s["depart"], "return_min": s["return"], "duration_min": s["span"],
                     "travel_min": travel, "service_min": int(sum(svc[x] for x in stops)),
                     "mid_route_wait_min": s["mid_wait"], "km": round(km, 1),
                     "fuel_kes": round(km * fuel_per_km), "time_kes": round(s["span"] * time_per_min)})
    assert sorted(visited) == list(range(1, n)), "Nodes missed or duplicated"
    df = pd.DataFrame(rows)
    df["total_kes"] = df.fuel_kes + df.time_kes
    used = df[df.stops > 0]
    summary = {
        "total_kes": int(df.total_kes.sum()), "fuel_kes": int(df.fuel_kes.sum()),
        "time_kes": int(df.time_kes.sum()), "km": round(float(df.km.sum()), 1),
        "duration_min": int(df.duration_min.sum()), "depot_hold_min": int(used.depart_min.sum()),
        "mid_route_wait_min": int(df.mid_route_wait_min.sum()), "vehicles_used": int(len(used)),
        "max_utilization": float(used.utilization.max()), "min_utilization": float(used.utilization.min()),
        "sla_ok": True,
    }
    return df, summary


def solve(state, rates, time_limit_sec=None, strategy=None, metaheuristic=None):
    """Optimized static plan. Returns a dict; routes is None if no solution was found."""
    cfg = state["config"]
    s = cfg["solver"]
    time_limit = float(time_limit_sec if time_limit_sec is not None else s["time_limit_sec"])
    strategy = strategy or s["first_solution_strategy"]
    meta = metaheuristic or s["metaheuristic"]
    scale = cfg["cost"]["cost_scale"]
    _, time_per_min = rates
    n, V, CAP, H = state["n"], state["V"], state["CAP"], state["H"]

    t0 = time.perf_counter()
    manager = pywrapcp.RoutingIndexManager(n, V, 0)
    routing = pywrapcp.RoutingModel(manager)

    fuel = arc_cost_matrix(state, rates).tolist()
    Tl = state["T"].astype(int).tolist()
    svc = state["service"].tolist()
    dem = state["demand"].tolist()
    op, cl = state["open"].tolist(), state["close"].tolist()

    def fuel_cb(i, j):
        return fuel[manager.IndexToNode(i)][manager.IndexToNode(j)]
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(fuel_cb))

    def time_cb(i, j):
        a, b = manager.IndexToNode(i), manager.IndexToNode(j)
        return Tl[a][b] + svc[a]
    tcb = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(tcb, H, H, False, "Time")          # False: vehicles may depart late
    tdim = routing.GetDimensionOrDie("Time")
    tdim.SetSpanCostCoefficientForAllVehicles(int(round(time_per_min * scale)))

    def dem_cb(i):
        return dem[manager.IndexToNode(i)]
    routing.AddDimensionWithVehicleCapacity(routing.RegisterUnaryTransitCallback(dem_cb),
                                            0, [CAP] * V, True, "Capacity")

    for i in range(1, n):
        tdim.CumulVar(manager.NodeToIndex(i)).SetRange(op[i], cl[i])
    for v in range(V):
        tdim.CumulVar(routing.Start(v)).SetRange(0, H)
        tdim.CumulVar(routing.End(v)).SetRange(0, H)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, strategy)
    p.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, meta)
    p.time_limit.FromMilliseconds(int(time_limit * 1000))
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    sol = routing.SolveWithParameters(p)
    solve_s = time.perf_counter() - t1

    out = {"routes": None, "build_s": build_s, "solve_s": solve_s, "wall_s": build_s + solve_s,
           "strategy": strategy, "metaheuristic": meta, "time_limit_s": time_limit,
           "objective_kes_approx": None}
    if sol is None:
        return out
    routes = []
    for v in range(V):
        idx, seq = routing.Start(v), []
        while not routing.IsEnd(idx):
            seq.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        seq.append(0)
        routes.append(seq)
    out["routes"] = routes
    out["objective_kes_approx"] = sol.ObjectiveValue() / scale
    return out


# ======================= PHASE 3A: DISRUPTION SIMULATOR (engine 2.2) =======================
def make_scenario(state, corridor, severity, onset, duration):
    """A disruption: legs touching `corridor` are slowed by `severity` for legs departing in [onset, onset+duration)."""
    names = {c["name"] for c in state["config"]["clusters"]}
    assert corridor in names, f"Unknown corridor: {corridor}"
    assert severity >= 1.0, "Severity must be >= 1.0 (a disruption cannot speed traffic up)"
    assert onset >= 0 and duration >= 0, "Onset and duration must be non-negative"
    members = sorted(int(x) for x in state["nodes"].loc[state["nodes"].cluster == corridor, "node_id"])
    return {"corridor": corridor, "severity": float(severity), "onset": int(onset),
            "duration": int(duration), "members": members}


def leg_minutes(state, scenario, a, b, t_dep):
    """Travel minutes for leg a->b leaving at minute t_dep (speed is fixed at departure)."""
    base = int(state["T"][a, b])
    if scenario is not None and (a in scenario["members"] or b in scenario["members"]) \
            and scenario["onset"] <= t_dep < scenario["onset"] + scenario["duration"]:
        return int(math.ceil(round(base * scenario["severity"], 6)))
    return base


def simulate_plan(state, routes, rates, scenario=None, departures=None):
    """Execute a FIXED plan: fixed sequences and departures, wait for windows, deliver late if late.
    Returns {'vehicles': DataFrame, 'stops': DataFrame, 'summary': dict}."""
    fuel_per_km, time_per_min = rates
    n, V, H = state["n"], state["V"], state["H"]
    KM, svc, op, cl = state["KM"], state["service"], state["open"], state["close"]
    nodes = state["nodes"]
    assert len(routes) == V, f"Expected {V} routes"
    for v, seq in enumerate(routes):
        assert seq[0] == 0 and seq[-1] == 0, f"Vehicle {v}: route must start and end at the depot"
    assert sorted(x for seq in routes for x in seq if x != 0) == list(range(1, n)), "Nodes missed or duplicated"

    vrows, srows = [], []
    for v, seq in enumerate(routes):
        if len(seq) <= 2:
            vrows.append({"vehicle": v, "route": "0-0", "depart_min": 0, "return_min": 0, "duration_min": 0,
                          "km": 0.0, "mid_route_wait_min": 0, "late_stops": 0, "late_min": 0,
                          "shift_overrun_min": 0, "fuel_kes": 0, "time_kes": 0})
            continue
        if departures is not None:
            d = int(departures[v])
        else:
            s0 = best_schedule(state, seq)
            assert s0 is not None, f"Vehicle {v}: plan infeasible without disruption"
            d = s0["depart"]
        t, km, wait_tot, late_stops, late_tot = d, 0.0, 0, 0, 0
        for a, b in zip(seq[:-1], seq[1:]):
            t += leg_minutes(state, scenario, a, b, t)
            km += float(KM[a, b])
            if b != 0:
                arrive = t
                start = max(arrive, int(op[b]))
                wait = start - arrive
                late = max(0, start - int(cl[b]))
                wait_tot += wait
                late_stops += int(late > 0)
                late_tot += late
                srows.append({"vehicle": v, "node": b, "name": nodes.iloc[b]["name"],
                              "tight": bool(nodes.iloc[b]["tight_window"]), "arrive": arrive, "start": start,
                              "open": int(op[b]), "close": int(cl[b]), "wait": wait, "late_min": late})
                t = start + int(svc[b])
        vrows.append({"vehicle": v, "route": "-".join(map(str, seq)), "depart_min": d, "return_min": t,
                      "duration_min": t - d, "km": round(km, 1), "mid_route_wait_min": wait_tot,
                      "late_stops": late_stops, "late_min": late_tot, "shift_overrun_min": max(0, t - H),
                      "fuel_kes": round(km * fuel_per_km), "time_kes": round((t - d) * time_per_min)})
    vdf, sdf = pd.DataFrame(vrows), pd.DataFrame(srows)
    vdf["total_kes"] = vdf.fuel_kes + vdf.time_kes
    late = sdf[sdf.late_min > 0]
    summary = {
        "total_kes": int(vdf.total_kes.sum()), "fuel_kes": int(vdf.fuel_kes.sum()), "time_kes": int(vdf.time_kes.sum()),
        "km": round(float(vdf.km.sum()), 1), "duration_min": int(vdf.duration_min.sum()),
        "mid_route_wait_min": int(vdf.mid_route_wait_min.sum()),
        "breaches": int(len(late)), "breached_nodes": sorted(int(x) for x in late.node),
        "key_account_breaches": int(late.tight.sum()), "total_late_min": int(late.late_min.sum()),
        "max_late_min": int(late.late_min.max()) if len(late) else 0,
        "vehicles_breaching": int((vdf.late_stops > 0).sum()),
        "shift_overrun_min": int(vdf.shift_overrun_min.sum()),
    }
    summary["sla_ok"] = summary["breaches"] == 0 and summary["shift_overrun_min"] == 0
    return {"vehicles": vdf, "stops": sdf, "summary": summary}


# ======================= PHASE 3B: DYNAMIC RE-SOLVE (engine 2.3) =======================
def static_departures(state, routes):
    """Minimum-cost departure minute of each vehicle's static route (0 for an empty route)."""
    out = []
    for r in routes:
        if not [x for x in r if x != 0]:
            out.append(0)
        else:
            s = best_schedule(state, r)
            assert s is not None, "Static plan is infeasible without disruption"
            out.append(int(s["depart"]))
    return out


def _run(state, scenario, seq, t0):
    """Exact execution of a fixed sequence leaving seq[0] at minute t0. Returns (stop rows, end time)."""
    op, cl, svc = state["open"], state["close"], state["service"]
    t, rows = int(t0), []
    for a, b in zip(seq[:-1], seq[1:]):
        leave_a = t
        t += leg_minutes(state, scenario, a, b, leave_a)
        if b != 0:
            arrive = t
            start = max(arrive, int(op[b]))
            leave = start + int(svc[b])
            rows.append({"node": int(b), "arrive": int(arrive), "start": int(start), "leave": int(leave),
                         "late": int(max(0, start - int(cl[b]))), "pred_leave": int(leave_a)})
            t = leave
    return rows, int(t)


def fleet_state_at(state, routes, departures, scenario, t_star):
    """Replay the static plan up to minute t_star. A stop is COMMITTED once its predecessor has been left.
    status: depot (not departed), enroute (committed to its current stop), done (returning to the depot)."""
    H = state["H"]
    assert 0 <= t_star <= H, "t_star must lie within the shift"
    dem = state["demand"]
    fleet = []
    for v, seq in enumerate(routes):
        stops = [int(x) for x in seq if x != 0]
        base = {"vehicle": v, "status": "depot", "pos": 0, "ready": int(t_star), "committed": [],
                "remaining": stops[:], "depart": None, "delivered_kg": 0}
        if not stops or t_star < int(departures[v]):
            fleet.append(base)
            continue
        d = int(departures[v])
        rows, end = _run(state, scenario, seq, d)
        k = 0
        while k < len(rows) and rows[k]["pred_leave"] <= t_star:
            k += 1
        assert k >= 1, "A departed vehicle must have at least one committed stop"
        committed = [r["node"] for r in rows[:k]]
        last = rows[k - 1]
        if k == len(rows) and last["leave"] <= t_star:
            status, pos, ready, remaining = "done", 0, int(end), []
        else:
            status, pos, ready, remaining = "enroute", int(last["node"]), int(last["leave"]), stops[k:]
        fleet.append({"vehicle": v, "status": status, "pos": pos, "ready": ready, "committed": committed,
                      "remaining": remaining, "depart": d, "delivered_kg": int(sum(dem[x] for x in committed))})
    return fleet


def best_departure_dyn(state, scenario, seq, d_min):
    """Departure minute in [d_min, H] minimizing (shift overrun, lateness, duration, departure) for a fixed sequence."""
    H = state["H"]
    if not [x for x in seq if x != 0]:
        return int(d_min)
    best = None
    for d in range(int(d_min), H + 1):
        rows, end = _run(state, scenario, seq, d)
        key = (max(0, end - H), sum(r["late"] for r in rows), end - d, d)
        if best is None or key < best[0]:
            best = (key, d)
    return int(best[1])


def _model_minutes(state, scenario, a, b, e_dep, l_dep):
    """Conservative model travel time: slowed whenever a departure inside the disruption window is possible."""
    base = int(state["T"][a, b])
    if scenario is None or not (a in scenario["members"] or b in scenario["members"]):
        return base
    w0, w1 = scenario["onset"], scenario["onset"] + scenario["duration"]
    if e_dep < w1 and l_dep >= w0:
        return int(math.ceil(round(base * scenario["severity"], 6)))
    return base


def resolve_from_state(state, rates, scenario, fleet, t_star, time_limit_sec=None, soft=True,
                       penalty_kes_per_min=100.0, pool_enroute=False, initial_suffixes=None):
    """Re-plan the unserved stops from each vehicle's position. Returns suffixes {vehicle: [stops]} (None if unsolved)."""
    cfg = state["config"]
    scale = cfg["cost"]["cost_scale"]
    fuel_per_km, time_per_min = rates
    CAP, H = state["CAP"], state["H"]
    limit = float(time_limit_sec if time_limit_sec is not None else cfg["solver"]["resolve_limit_sec"])
    t0 = time.perf_counter()

    active = [f for f in fleet if f["status"] in ("depot", "enroute")]
    stops = sorted({s for f in fleet for s in f["remaining"]})
    out = {"suffixes": {}, "active": [f["vehicle"] for f in active], "build_s": 0.0, "solve_s": 0.0,
           "wall_s": 0.0, "soft": soft, "warm_started": False, "objective_kes_approx": None, "solved": True,
           "pool_enroute": pool_enroute}
    if not stops:
        out["suffixes"] = {f["vehicle"]: [] for f in active}
        out["wall_s"] = time.perf_counter() - t0
        return out
    assert active, "Unserved stops remain but no vehicle is available"

    enroute = [f for f in active if f["status"] == "enroute"]
    first_stop = 1 + len(enroute)
    local_global = [0] + [f["pos"] for f in enroute] + stops
    loc_of = {g: first_stop + i for i, g in enumerate(stops)}
    pos_loc = {f["vehicle"]: 1 + j for j, f in enumerate(enroute)}
    n_loc, nv = len(local_global), len(active)
    starts = [pos_loc[f["vehicle"]] if f["status"] == "enroute" else 0 for f in active]
    ends = [0] * nv

    op, cl, svc, dem = state["open"], state["close"], state["service"], state["demand"]
    e_dep, l_dep, svc_l, dem_l = [0] * n_loc, [0] * n_loc, [0] * n_loc, [0] * n_loc
    e_dep[0], l_dep[0] = int(t_star), H
    for f in enroute:
        i = pos_loc[f["vehicle"]]
        e_dep[i] = l_dep[i] = int(f["ready"])
    for g in stops:
        i = loc_of[g]
        e_dep[i] = max(int(t_star), int(op[g]) + int(svc[g]))
        l_dep[i] = H if soft else int(cl[g]) + int(svc[g])
        svc_l[i], dem_l[i] = int(svc[g]), int(dem[g])

    Tl = [[0] * n_loc for _ in range(n_loc)]
    Fl = [[0] * n_loc for _ in range(n_loc)]
    for i in range(n_loc):
        for j in range(n_loc):
            if i == j:
                continue
            a, b = local_global[i], local_global[j]
            Tl[i][j] = _model_minutes(state, scenario, a, b, e_dep[i], l_dep[i])
            Fl[i][j] = int(round(float(state["KM"][a, b]) * fuel_per_km * scale))

    manager = pywrapcp.RoutingIndexManager(n_loc, nv, starts, ends)
    routing = pywrapcp.RoutingModel(manager)
    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(
        lambda i, j: Fl[manager.IndexToNode(i)][manager.IndexToNode(j)]))
    tcb = routing.RegisterTransitCallback(
        lambda i, j: Tl[manager.IndexToNode(i)][manager.IndexToNode(j)] + svc_l[manager.IndexToNode(i)])
    routing.AddDimension(tcb, H, H, False, "Time")
    tdim = routing.GetDimensionOrDie("Time")
    tdim.SetSpanCostCoefficientForAllVehicles(int(round(time_per_min * scale)))
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(lambda i: dem_l[manager.IndexToNode(i)]), 0, [CAP] * nv, True, "Capacity")

    pen = int(round(penalty_kes_per_min * scale))
    dep_ids = [k for k, f in enumerate(active) if f["status"] == "depot"]
    owner = {s: f for f in fleet for s in f["remaining"]}
    for g in stops:
        idx = manager.NodeToIndex(loc_of[g])
        if soft:
            tdim.CumulVar(idx).SetRange(int(op[g]), H)
            tdim.SetCumulVarSoftUpperBound(idx, int(cl[g]), pen)
        else:
            tdim.CumulVar(idx).SetRange(int(op[g]), int(cl[g]))
        if pool_enroute:
            allowed = list(range(nv))
        elif owner[g]["status"] == "depot":
            allowed = dep_ids
        else:
            allowed = [k for k, f in enumerate(active) if f["vehicle"] == owner[g]["vehicle"]]
        _restrict_vehicles(routing, idx, allowed)
    for k, f in enumerate(active):
        if f["status"] == "enroute":
            tdim.CumulVar(routing.Start(k)).SetRange(int(f["ready"]), int(f["ready"]))
        else:
            tdim.CumulVar(routing.Start(k)).SetRange(int(t_star), H)
        tdim.CumulVar(routing.End(k)).SetRange(0, H)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, cfg["solver"]["first_solution_strategy"])
    p.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, cfg["solver"]["metaheuristic"])
    p.time_limit.FromMilliseconds(int(limit * 1000))
    routing.CloseModelWithParameters(p)

    init = None
    if initial_suffixes is not None:
        try:
            routes_loc = [[manager.NodeToIndex(loc_of[g]) for g in initial_suffixes.get(f["vehicle"], [])] for f in active]
            init = routing.ReadAssignmentFromRoutes(routes_loc, True)
        except Exception:
            init = None
    out["build_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    sol = routing.SolveFromAssignmentWithParameters(init, p) if init is not None else routing.SolveWithParameters(p)
    out["solve_s"] = time.perf_counter() - t1
    out["warm_started"] = init is not None
    out["wall_s"] = out["build_s"] + out["solve_s"]
    if sol is None:
        out["solved"], out["suffixes"] = False, None
        return out
    for k, f in enumerate(active):
        idx, seq = sol.Value(routing.NextVar(routing.Start(k))), []
        while not routing.IsEnd(idx):
            seq.append(int(local_global[manager.IndexToNode(idx)]))
            idx = sol.Value(routing.NextVar(idx))
        out["suffixes"][f["vehicle"]] = seq
    out["objective_kes_approx"] = sol.ObjectiveValue() / scale
    return out


def assemble_and_simulate(state, rates, scenario, static_routes, static_deps, fleet, suffixes, t_star):
    """Stitch frozen prefixes and new suffixes into full routes, then execute them with the exact simulator."""
    full_routes, deps = [], []
    for f in fleet:
        v = f["vehicle"]
        if f["status"] == "done":
            full_routes.append(list(static_routes[v])); deps.append(int(static_deps[v]))
        elif f["status"] == "enroute":
            full_routes.append([0] + list(f["committed"]) + list(suffixes.get(v, [])) + [0]); deps.append(int(static_deps[v]))
        else:
            seq = [0] + list(suffixes.get(v, [])) + [0]
            full_routes.append(seq); deps.append(best_departure_dyn(state, scenario, seq, t_star))
    return full_routes, deps, simulate_plan(state, full_routes, rates, scenario, deps)


def resolve_dynamic(state, rates, scenario, routes, departures, t_star, time_limit_sec=None,
                    penalty_kes_per_min=None, pool_enroute=None, warm_start=True):
    """Full dynamic pipeline: freeze state, re-solve, stitch, simulate. wall_s = state + solve + assemble + simulate."""
    dyn = state["config"].get("dynamic", {})
    penalty = penalty_kes_per_min if penalty_kes_per_min is not None else dyn.get("soft_window_penalty_kes_per_min", 100.0)
    pool = pool_enroute if pool_enroute is not None else dyn.get("pool_enroute_cargo", False)
    t0 = time.perf_counter()
    fleet = fleet_state_at(state, routes, departures, scenario, t_star)
    init = {f["vehicle"]: list(f["remaining"]) for f in fleet if f["status"] in ("depot", "enroute")} if warm_start else None
    res = resolve_from_state(state, rates, scenario, fleet, t_star, time_limit_sec, True, penalty, pool, init)
    out = {"solved": res["solved"], "fleet": fleet, "resolve": res, "t_star": int(t_star)}
    if res["solved"]:
        out["routes"], out["departures"], out["sim"] = assemble_and_simulate(
            state, rates, scenario, routes, departures, fleet, res["suffixes"], t_star)
    out["wall_s"] = time.perf_counter() - t0
    return out


def _restrict_vehicles(routing, idx, allowed):
    """Limit which vehicles may visit `idx`. The accepted argument type of the direct API varies by
    OR-Tools build, so try several types, then fall back to the stop's vehicle variable domain."""
    allowed = [int(k) for k in allowed]
    for conv in (lambda a: a, lambda a: tuple(a), lambda a: np.array(a, dtype=np.int32)):
        try:
            routing.SetAllowedVehiclesForIndex(conv(allowed), idx)
            return "SetAllowedVehiclesForIndex"
        except TypeError:
            continue
    routing.VehicleVar(idx).SetValues(allowed)
    return "VehicleVar.SetValues"


# ======================= PHASE 3C: COST-MODEL VARIANTS (engine 2.4) =======================
def schedule_cost_min(state, seq, time_per_min, idle_per_min=0.0, clocked=False):
    """Cheapest feasible schedule for a FIXED sequence under a cost-model variant.
    clocked=True: the driver is on the clock from minute 0 (departure 0; the return time is the duration)."""
    T, svc, op, cl, H = state["T"], state["service"], state["open"], state["close"], state["H"]
    best = None
    for d in ([0] if clocked else range(0, H + 1)):
        t, wait, ok = d, 0, True
        for a, b in zip(seq[:-1], seq[1:]):
            t += int(T[a, b])
            if b != 0:
                if t > cl[b]:
                    ok = False
                    break
                if t < op[b]:
                    wait += int(op[b] - t)
                    t = int(op[b])
                t += int(svc[b])
        if ok and t <= H:
            dur = t if clocked else t - d
            key = (dur * time_per_min + wait * idle_per_min, d)
            if best is None or key < best["key"]:
                best = {"key": key, "span": dur, "depart": d, "mid_wait": wait, "return": t}
    return best


def price_plan(state, routes, rates, clocked=False, idle_kes_per_min=0.0):
    """Independent verification + pricing under a cost-model variant. Raises AssertionError on ANY violation."""
    n, V, CAP = state["n"], state["V"], state["CAP"]
    KM, dem = state["KM"], state["demand"]
    fuel_per_km, time_per_min = rates
    assert len(routes) == V, f"Expected {V} routes, got {len(routes)}"
    for v, seq in enumerate(routes):
        assert seq[0] == 0 and seq[-1] == 0, f"Vehicle {v}: route must start and end at the depot"
    assert sorted(x for seq in routes for x in seq if x != 0) == list(range(1, n)), "Nodes missed or duplicated"
    rows = []
    for v, seq in enumerate(routes):
        stops = [x for x in seq if x != 0]
        if not stops:
            rows.append({"vehicle": v, "route": "0-0", "stops": 0, "load_kg": 0, "depart_min": 0, "return_min": 0,
                         "duration_min": 0, "mid_route_wait_min": 0, "km": 0.0, "fuel_kes": 0, "time_kes": 0, "idle_kes": 0})
            continue
        load = int(sum(dem[x] for x in stops))
        assert load <= CAP, f"Vehicle {v}: overload {load} > {CAP}"
        s = schedule_cost_min(state, seq, time_per_min, idle_kes_per_min, clocked)
        assert s is not None, f"Vehicle {v}: no feasible schedule (window, return or shift violation)"
        km = float(sum(KM[a, b] for a, b in zip(seq[:-1], seq[1:])))
        rows.append({"vehicle": v, "route": "-".join(map(str, seq)), "stops": len(stops), "load_kg": load,
                     "depart_min": s["depart"], "return_min": s["return"], "duration_min": s["span"],
                     "mid_route_wait_min": s["mid_wait"], "km": round(km, 1),
                     "fuel_kes": round(km * fuel_per_km), "time_kes": round(s["span"] * time_per_min),
                     "idle_kes": round(s["mid_wait"] * idle_kes_per_min)})
    df = pd.DataFrame(rows)
    df["total_kes"] = df.fuel_kes + df.time_kes + df.idle_kes
    used = df[df.stops > 0]
    summary = {"total_kes": int(df.total_kes.sum()), "fuel_kes": int(df.fuel_kes.sum()), "time_kes": int(df.time_kes.sum()),
               "idle_kes": int(df.idle_kes.sum()), "km": round(float(df.km.sum()), 1), "duration_min": int(df.duration_min.sum()),
               "mid_route_wait_min": int(df.mid_route_wait_min.sum()), "depot_hold_min": int(used.depart_min.sum()),
               "vehicles_used": int(len(used)), "clocked": bool(clocked), "idle_kes_per_min": float(idle_kes_per_min)}
    return df, summary


def solve_variant(state, rates, time_limit_sec=None, strategy=None, metaheuristic=None, clocked=False, idle_kes_per_min=0.0):
    """Optimized static plan under a cost-model variant (same model as solve(), plus clocked start and idle slack cost)."""
    cfg = state["config"]
    s = cfg["solver"]
    time_limit = float(time_limit_sec if time_limit_sec is not None else s["time_limit_sec"])
    strategy = strategy or s["first_solution_strategy"]
    meta = metaheuristic or s["metaheuristic"]
    scale = cfg["cost"]["cost_scale"]
    _, time_per_min = rates
    n, V, CAP, H = state["n"], state["V"], state["CAP"], state["H"]

    t0 = time.perf_counter()
    manager = pywrapcp.RoutingIndexManager(n, V, 0)
    routing = pywrapcp.RoutingModel(manager)
    fuel = arc_cost_matrix(state, rates).tolist()
    Tl = state["T"].astype(int).tolist()
    svc, dem = state["service"].tolist(), state["demand"].tolist()
    op, cl = state["open"].tolist(), state["close"].tolist()

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(
        lambda i, j: fuel[manager.IndexToNode(i)][manager.IndexToNode(j)]))
    tcb = routing.RegisterTransitCallback(
        lambda i, j: Tl[manager.IndexToNode(i)][manager.IndexToNode(j)] + svc[manager.IndexToNode(i)])
    routing.AddDimension(tcb, H, H, bool(clocked), "Time")
    tdim = routing.GetDimensionOrDie("Time")
    tdim.SetSpanCostCoefficientForAllVehicles(int(round(time_per_min * scale)))
    if idle_kes_per_min and idle_kes_per_min > 0:
        tdim.SetSlackCostCoefficientForAllVehicles(int(round(idle_kes_per_min * scale)))
    routing.AddDimensionWithVehicleCapacity(
        routing.RegisterUnaryTransitCallback(lambda i: dem[manager.IndexToNode(i)]), 0, [CAP] * V, True, "Capacity")
    for i in range(1, n):
        tdim.CumulVar(manager.NodeToIndex(i)).SetRange(op[i], cl[i])
    for v in range(V):
        tdim.CumulVar(routing.Start(v)).SetRange(0, H)
        tdim.CumulVar(routing.End(v)).SetRange(0, H)

    p = pywrapcp.DefaultRoutingSearchParameters()
    p.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, strategy)
    p.local_search_metaheuristic = getattr(routing_enums_pb2.LocalSearchMetaheuristic, meta)
    p.time_limit.FromMilliseconds(int(time_limit * 1000))
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    sol = routing.SolveWithParameters(p)
    solve_s = time.perf_counter() - t1

    out = {"routes": None, "build_s": build_s, "solve_s": solve_s, "wall_s": build_s + solve_s, "strategy": strategy,
           "metaheuristic": meta, "time_limit_s": time_limit, "clocked": bool(clocked),
           "idle_kes_per_min": float(idle_kes_per_min), "objective_kes_approx": None}
    if sol is None:
        return out
    routes = []
    for v in range(V):
        idx, seq = routing.Start(v), []
        while not routing.IsEnd(idx):
            seq.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        seq.append(0)
        routes.append(seq)
    out["routes"] = routes
    out["objective_kes_approx"] = sol.ObjectiveValue() / scale
    return out


# ======================= PHASE 4: INSTANCE GENERATION AND GATE (engine 2.5) =======================
ANCHOR_BUFFER_MIN = 20
AREA_POOL = [
    {"name": "Westlands", "lat": -1.2673, "lon": 36.8110, "jitter_km": 1.5, "congested": True},
    {"name": "Kilimani_UpperHill", "lat": -1.2940, "lon": 36.8000, "jitter_km": 1.5, "congested": False},
    {"name": "Eastlands", "lat": -1.2880, "lon": 36.8930, "jitter_km": 2.0, "congested": False},
    {"name": "Thika_Road", "lat": -1.2200, "lon": 36.8900, "jitter_km": 2.0, "congested": True},
    {"name": "Mombasa_Road", "lat": -1.3350, "lon": 36.8900, "jitter_km": 3.0, "congested": True},
    {"name": "Karen_Langata", "lat": -1.3350, "lon": 36.7300, "jitter_km": 2.5, "congested": False},
    {"name": "CBD", "lat": -1.2850, "lon": 36.8230, "jitter_km": 1.0, "congested": True},
    {"name": "Ngong_Road", "lat": -1.3000, "lon": 36.7600, "jitter_km": 2.0, "congested": True},
    {"name": "Parklands_Pangani", "lat": -1.2620, "lon": 36.8350, "jitter_km": 1.2, "congested": False},
    {"name": "Ruaka_Kiambu_Road", "lat": -1.2050, "lon": 36.8000, "jitter_km": 2.0, "congested": False},
]


def _jitter(lat, lon, radius_km, rng):
    r = radius_km * np.sqrt(rng.uniform())
    theta = rng.uniform(0, 2 * np.pi)
    dlat = (r * np.sin(theta)) / 111.0
    dlon = (r * np.cos(theta)) / (111.0 * np.cos(np.radians(lat)))
    return lat + dlat, lon + dlon


def _hav_vec(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _clock(m):
    h, mm = divmod(7 * 60 + int(m), 60)
    return "%02d:%02d" % (h, mm)


def _instance_config(base_config, spec):
    cfg = json.loads(json.dumps(base_config))
    cfg["network"]["n_delivery_nodes"] = int(spec["n_nodes"])
    cfg["network"]["n_vehicles"] = int(spec["n_vehicles"])
    cfg["network"]["vehicle_capacity_kg"] = int(spec["capacity"])
    cfg["clusters"] = json.loads(json.dumps(spec["clusters"]))
    cfg["demand"]["total_range_kg"] = [int(spec["total_range_kg"][0]), int(spec["total_range_kg"][1])]
    return cfg


def make_spec(base_config, seed, n_nodes=18, n_vehicles=4, nodes_per_cluster=3,
              center_shift_km=0.8, utilization=(0.80, 0.90), attempt=0):
    """Build an instance specification: which areas, shifted centres, scaled demand range."""
    assert n_nodes % nodes_per_cluster == 0, "n_nodes must be a multiple of nodes_per_cluster"
    eff = int(seed) + 10000 * int(attempt)
    k = n_nodes // nodes_per_cluster
    assert k <= len(AREA_POOL), "Not enough areas in the pool"
    rng = np.random.default_rng(eff + 7)
    while True:
        if k < len(AREA_POOL):
            idx = sorted(int(i) for i in rng.choice(len(AREA_POOL), size=k, replace=False))
        else:
            idx = list(range(len(AREA_POOL)))
        if sum(AREA_POOL[i]["congested"] for i in idx) >= 3:
            break
    clusters = []
    for i in idx:
        a = AREA_POOL[i]
        lat, lon = a["lat"], a["lon"]
        if center_shift_km > 0:
            lat, lon = _jitter(lat, lon, center_shift_km, rng)
        clusters.append({"name": a["name"], "lat": round(float(lat), 6), "lon": round(float(lon), 6),
                         "nodes": int(nodes_per_cluster), "jitter_km": a["jitter_km"], "congested": a["congested"]})
    cap = int(base_config["network"]["vehicle_capacity_kg"])
    fleet = n_vehicles * cap
    return {"seed": eff, "base_seed": int(seed), "attempt": int(attempt), "clusters": clusters,
            "n_vehicles": int(n_vehicles), "capacity": cap, "n_nodes": int(n_nodes),
            "total_range_kg": [int(round(utilization[0] * fleet)), int(round(utilization[1] * fleet))]}


def generate_instance(base_config, spec):
    """Spatial draw, demand resampling, windows and travel matrix, with the same random streams as Phase 1."""
    cfg = _instance_config(base_config, spec)
    tm, tr, dm = cfg["time"], cfg["travel"], cfg["demand"]
    n_del = int(spec["n_nodes"])
    assert sum(c["nodes"] for c in cfg["clusters"]) == n_del, "Cluster nodes do not sum to the node count"
    seed = int(spec["seed"])

    # --- spatial (stream: seed) ---
    rng = np.random.default_rng(seed)
    rows = [{"node_id": 0, "name": cfg["depot"]["name"], "cluster": "Depot", "lat": cfg["depot"]["lat"],
             "lon": cfg["depot"]["lon"], "congested_corridor": False}]
    nid = 1
    for c in cfg["clusters"]:
        for k in range(1, c["nodes"] + 1):
            lat, lon = _jitter(c["lat"], c["lon"], c["jitter_km"], rng)
            rows.append({"node_id": nid, "name": c["name"] + "_" + str(k), "cluster": c["name"],
                         "lat": round(lat, 6), "lon": round(lon, 6), "congested_corridor": c["congested"]})
            nid += 1
    nodes = pd.DataFrame(rows)
    inside = nodes["lat"].between(-1.45, -1.15).all() and nodes["lon"].between(36.65, 37.00).all()
    assert inside, "A node fell outside the Nairobi bounding box"

    # --- demand (same stream, continues) ---
    tiers = dm["tiers"]
    lo_t, hi_t = dm["total_range_kg"]
    counts = [int(round(t["share"] * n_del)) for t in tiers]
    counts[0] += n_del - sum(counts)
    assert sum(counts) == n_del and min(counts) >= 0, "Tier counts do not sum to the node count"
    tier_labels = np.repeat([t["label"] for t in tiers], counts)
    accepted = None
    for attempt in range(1, 20001):
        labels = rng.permutation(tier_labels)
        kg = np.zeros(n_del, dtype=int)
        for i, lab in enumerate(labels):
            t = next(x for x in tiers if x["label"] == lab)
            kg[i] = int(round(rng.uniform(t["min_kg"], t["max_kg"]) / 10.0) * 10)
        if lo_t <= kg.sum() <= hi_t:
            accepted = (labels, kg, attempt)
            break
    assert accepted is not None, "No demand vector landed in the target range"
    labels, kg, attempts = accepted
    nodes["demand_kg"] = 0
    nodes["demand_tier"] = "depot"
    nodes.loc[nodes["node_id"] > 0, "demand_kg"] = kg
    nodes.loc[nodes["node_id"] > 0, "demand_tier"] = labels
    nodes["demand_kg"] = nodes["demand_kg"].astype(int)

    # --- windows and service (stream: seed + 1) ---
    rng2 = np.random.default_rng(seed + 1)
    depot = nodes.iloc[0]
    d_km = _hav_vec(depot["lat"], depot["lon"], nodes["lat"].to_numpy(), nodes["lon"].to_numpy())
    est = d_km * tr["road_circuity_factor"] / tr["avg_speed_kmh"] * 60
    max_mult = tr["congestion_multiplier_range"][1]
    est = np.where(nodes["congested_corridor"].to_numpy(), est * max_mult, est)
    nodes["est_direct_min"] = np.ceil(est).astype(int)
    s_lo, s_hi = tm["service_time_min"]
    d_lo = min(t["min_kg"] for t in tiers)
    d_hi = max(t["max_kg"] for t in tiers)
    svc = s_lo + (s_hi - s_lo) * (nodes["demand_kg"] - d_lo) / (d_hi - d_lo)
    nodes["service_min"] = np.clip(np.round(svc), s_lo, s_hi).astype(int)
    nodes.loc[nodes["node_id"] == 0, "service_min"] = 0
    delivery_ids = nodes.loc[nodes["node_id"] > 0, "node_id"].to_numpy()
    n_tight = int(round(tm["tight_window_share"] * len(delivery_ids)))
    tight_ids = set(rng2.choice(delivery_ids, size=n_tight, replace=False).tolist())
    nodes["tight_window"] = nodes["node_id"].isin(tight_ids)
    H = tm["shift_end_min"]
    opens, closes = [], []
    for i in range(len(nodes)):
        if i == 0:
            opens.append(tm["shift_start_min"])
            closes.append(H)
            continue
        tight = bool(nodes.at[i, "tight_window"])
        w_lo, w_hi = tm["window_tight_width_min"] if tight else tm["window_standard_width_min"]
        width = int(rng2.integers(w_lo, w_hi + 1))
        est_i = int(nodes.at[i, "est_direct_min"])
        svc_i = int(nodes.at[i, "service_min"])
        earliest_close = max(est_i + ANCHOR_BUFFER_MIN, width)
        latest_close = H - svc_i - est_i
        close = int(rng2.integers(earliest_close, latest_close + 1))
        opens.append(close - width)
        closes.append(close)
    nodes["tw_open"], nodes["tw_close"] = opens, closes
    nodes["open_clock"] = nodes["tw_open"].map(_clock)
    nodes["close_clock"] = nodes["tw_close"].map(_clock)

    # --- travel matrix (stream: seed + 2) ---
    rng3 = np.random.default_rng(seed + 2)
    lo, hi = tr["congestion_multiplier_range"]
    corridor_mult = {c["name"]: round(float(rng3.uniform(lo, hi)), 2) for c in cfg["clusters"] if c["congested"]}
    n = len(nodes)
    lat_r = np.radians(nodes["lat"].to_numpy())[:, None]
    lon_r = np.radians(nodes["lon"].to_numpy())[:, None]
    dphi, dlam = lat_r.T - lat_r, lon_r.T - lon_r
    a = np.sin(dphi / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlam / 2) ** 2
    hav_km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
    road_km = hav_km * tr["road_circuity_factor"]
    np.fill_diagonal(road_km, 0.0)
    node_mult = nodes["cluster"].map(corridor_mult).fillna(1.0).to_numpy()
    leg_mult = np.maximum(node_mult[:, None], node_mult[None, :])
    base_min = road_km / tr["avg_speed_kmh"] * 60
    time_min = np.ceil(base_min * leg_mult).astype(int)
    off = ~np.eye(n, dtype=bool)
    time_min[off] = np.maximum(time_min[off], 1)
    np.fill_diagonal(time_min, 0)
    return {"seed": seed, "spec": spec, "config": cfg, "nodes": nodes, "T": time_min, "KM": road_km,
            "corridor_mult": corridor_mult, "demand_attempts": int(attempts)}


def validate_instance(inst):
    """Check every Phase 1 invariant. Raises AssertionError with the reason on the first failure."""
    cfg, nodes, T, KM = inst["config"], inst["nodes"], inst["T"], inst["KM"]
    n = len(nodes)
    V, CAP = cfg["network"]["n_vehicles"], cfg["network"]["vehicle_capacity_kg"]
    H, tm, dm = cfg["time"]["shift_end_min"], cfg["time"], cfg["demand"]
    assert (nodes["node_id"].to_numpy() == np.arange(n)).all(), "node_id must be 0..N-1 in order"
    assert nodes["lat"].between(-1.45, -1.15).all() and nodes["lon"].between(36.65, 37.00).all(), "A node lies outside the Nairobi bounding box"
    d = nodes[nodes["node_id"] > 0]
    total = int(d["demand_kg"].sum())
    lo, hi = dm["total_range_kg"]
    assert lo <= total <= hi, "Total demand %d kg is outside the target range %d-%d" % (total, lo, hi)
    assert int(d["demand_kg"].max()) <= CAP and total <= V * CAP, "A drop or the total exceeds capacity"
    counts = [int(round(t["share"] * len(d))) for t in dm["tiers"]]
    counts[0] += len(d) - sum(counts)
    expected = {t["label"]: c for t, c in zip(dm["tiers"], counts)}
    actual = d["demand_tier"].value_counts().to_dict()
    assert all(actual.get(k, 0) == v for k, v in expected.items()), "Demand tier mix does not match the configured shares"
    widths = d["tw_close"] - d["tw_open"]
    n_tight = int(round(tm["tight_window_share"] * len(d)))
    assert int(d["tight_window"].sum()) == n_tight, "Wrong number of tight-window nodes"
    ok_t = widths[d["tight_window"]].between(*tm["window_tight_width_min"]).all()
    ok_s = widths[~d["tight_window"]].between(*tm["window_standard_width_min"]).all()
    assert ok_t and ok_s, "Window widths outside the configured ranges"
    assert (d["tw_close"] >= d["est_direct_min"] + ANCHOR_BUFFER_MIN).all(), "A window closes before plausible arrival"
    assert (d["tw_open"] >= 0).all() and (d["tw_close"] <= H).all(), "A window lies outside the shift"
    assert (d["tw_close"] + d["service_min"] + d["est_direct_min"] <= H).all(), "A node leaves no time to return before shift end"
    s_lo, s_hi = tm["service_time_min"]
    assert d["service_min"].between(s_lo, s_hi).all(), "Service time outside the configured range"
    assert T.shape == (n, n) and np.issubdtype(T.dtype, np.integer), "Travel matrix must be an integer (n, n) array"
    assert (T == T.T).all(), "Travel matrix is not symmetric"
    off = ~np.eye(n, dtype=bool)
    assert (np.diag(T) == 0).all() and (T[off] > 0).all(), "Travel matrix diagonal or off-diagonal values are invalid"
    assert (T[0, 1:] <= d["est_direct_min"].to_numpy()).all(), "Depot legs exceed the anchoring estimate"
    assert KM.shape == (n, n) and np.allclose(KM, KM.T), "Road-km matrix is not symmetric"
    return {"n_nodes": n - 1, "vehicles": V, "total_kg": total, "utilization": round(total / (V * CAP), 3),
            "tight_nodes": sorted(int(x) for x in d.loc[d["tight_window"], "node_id"]),
            "largest_drop_kg": int(d["demand_kg"].max()), "tiers": actual,
            "matrix_min": int(T[off].min()), "matrix_mean": round(float(T[off].mean()), 1), "matrix_max": int(T[off].max()),
            "depot_leg_min": int(T[0, 1:].min()), "depot_leg_max": int(T[0, 1:].max())}


def state_from_instance(inst, project_dir=None):
    """Build the same state dict that load_state returns, from an instance."""
    cfg = inst["config"]
    nodes = inst["nodes"].sort_values("node_id").reset_index(drop=True)
    return {"project_dir": project_dir, "config": cfg, "nodes": nodes, "T": inst["T"], "KM": inst["KM"], "n": len(nodes),
            "V": cfg["network"]["n_vehicles"], "CAP": cfg["network"]["vehicle_capacity_kg"], "H": cfg["time"]["shift_end_min"],
            "demand": nodes["demand_kg"].astype(int).to_numpy(), "service": nodes["service_min"].astype(int).to_numpy(),
            "open": nodes["tw_open"].astype(int).to_numpy(), "close": nodes["tw_close"].astype(int).to_numpy()}


def gate_instance(state, rates, time_limit_sec=2.0):
    """Three-layer feasibility gate: necessary conditions, first-solution solve, independent verification."""
    n, V, CAP, H = state["n"], state["V"], state["CAP"], state["H"]
    T, dem, svc, op, cl = state["T"], state["demand"], state["service"], state["open"], state["close"]
    total = int(dem.sum())
    min_veh = -(-total // CAP)
    failures = []
    if total > V * CAP:
        failures.append("Total demand %d exceeds fleet capacity %d" % (total, V * CAP))
    if int(dem.max()) > CAP:
        failures.append("A single drop exceeds vehicle capacity")
    if min_veh > V:
        failures.append("Need %d vehicles, have %d" % (min_veh, V))
    for i in range(1, n):
        earliest = max(int(op[i]), int(T[0, i]))
        if earliest > cl[i]:
            failures.append("Node %d unreachable inside its window" % i)
        if earliest + int(svc[i]) + int(T[i, 0]) > H:
            failures.append("Node %d cannot return before shift end" % i)
    out = {"gate_passed": False, "failures": failures, "total_kg": total, "utilization": round(total / (V * CAP), 3),
           "min_vehicles": min_veh, "layer2": [], "strategy": None, "routes": None, "first_solution_kes": None}
    if failures:
        return out
    res = None
    for strat in ("PARALLEL_CHEAPEST_INSERTION", "LOCAL_CHEAPEST_INSERTION"):
        res = solve(state, rates, time_limit_sec=time_limit_sec, strategy=strat, metaheuristic="GREEDY_DESCENT")
        out["layer2"].append((strat, res["routes"] is not None, round(res["wall_s"], 3)))
        if res["routes"] is not None:
            break
    if res is None or res["routes"] is None:
        out["failures"].append("No first-solution strategy found a feasible plan")
        return out
    try:
        df, summ = verify_plan(state, res["routes"], rates)
    except AssertionError as e:
        out["failures"].append("Independent verification failed: " + str(e))
        return out
    out.update({"gate_passed": True, "strategy": out["layer2"][-1][0], "routes": res["routes"],
                "first_solution_kes": summ["total_kes"], "vehicles_used": summ["vehicles_used"]})
    return out


def save_instance(inst, folder):
    os.makedirs(folder, exist_ok=True)
    inst["nodes"].to_csv(os.path.join(folder, "nodes.csv"), index=False)
    np.save(os.path.join(folder, "travel_matrix.npy"), inst["T"])
    np.save(os.path.join(folder, "road_km_matrix.npy"), inst["KM"])
    with open(os.path.join(folder, "corridor_multipliers.json"), "w") as f:
        json.dump(inst["corridor_mult"], f, indent=2)
    with open(os.path.join(folder, "instance.json"), "w") as f:
        json.dump({"spec": inst["spec"], "demand_attempts": inst["demand_attempts"]}, f, indent=2)


def load_state_from(project_dir, instance_dir):
    """Load an instance folder into the same state dict as load_state, using the project's config for rates and solver."""
    with open(os.path.join(project_dir, "config.json")) as f:
        base = json.load(f)
    with open(os.path.join(instance_dir, "instance.json")) as f:
        meta = json.load(f)
    inst = {"seed": meta["spec"]["seed"], "spec": meta["spec"], "config": _instance_config(base, meta["spec"]),
            "nodes": pd.read_csv(os.path.join(instance_dir, "nodes.csv")),
            "T": np.load(os.path.join(instance_dir, "travel_matrix.npy")),
            "KM": np.load(os.path.join(instance_dir, "road_km_matrix.npy")),
            "demand_attempts": meta["demand_attempts"]}
    return state_from_instance(inst, project_dir)


# ======================= PHASE 4: SIMULATED DISPATCHER HEURISTICS (engine 2.6) =======================
def _loads(state, routes):
    dem = state["demand"]
    return [int(sum(int(dem[x]) for x in seq if x != 0)) for seq in routes]


def heuristic_zone_sweep(state):
    """H1 (simulated dispatcher): sweep around the depot, balanced zones within capacity,
    each zone sequenced by earliest window close. Returns (routes, unplaced)."""
    nodes, V, CAP = state["nodes"], state["V"], state["CAP"]
    dem, cl = state["demand"], state["close"]
    lat0, lon0 = float(nodes.at[0, "lat"]), float(nodes.at[0, "lon"])
    stops = list(range(1, state["n"]))
    ang = {i: math.atan2(float(nodes.at[i, "lat"]) - lat0, float(nodes.at[i, "lon"]) - lon0) for i in stops}
    srt = sorted(stops, key=lambda i: (ang[i], i))
    gaps = []
    for k in range(len(srt)):
        a, b = ang[srt[k]], ang[srt[(k + 1) % len(srt)]]
        gaps.append((b - a) % (2 * math.pi))
    start = (gaps.index(max(gaps)) + 1) % len(srt)
    order = srt[start:] + srt[:start]
    target = sum(int(dem[i]) for i in stops) / float(V)
    groups = [[]]
    for i in order:
        cur = groups[-1]
        load = sum(int(dem[x]) for x in cur)
        if cur and (load + int(dem[i]) > CAP or (load >= target and len(groups) < V)):
            groups.append([])
        groups[-1].append(i)
    unplaced = []
    if len(groups) > V:
        extra = [i for g in groups[V:] for i in g]
        groups = groups[:V]
        for i in sorted(extra, key=lambda x: (-int(dem[x]), x)):
            loads = [sum(int(dem[x]) for x in g) for g in groups]
            fits = [(CAP - loads[k] - int(dem[i]), k) for k in range(V) if loads[k] + int(dem[i]) <= CAP]
            if fits:
                groups[max(fits)[1]].append(i)
            else:
                unplaced.append(i)
    while len(groups) < V:
        groups.append([])
    routes = [[0] + sorted(g, key=lambda x: (int(cl[x]), x)) + [0] for g in groups]
    return routes, sorted(unplaced)


def heuristic_nearest_neighbour(state):
    """H2 (simulated dispatcher): each vehicle in turn leaves at minute 0 and goes to the nearest stop it can
    still serve in time and capacity. Leftovers go to the vehicle with the most spare capacity.
    Returns (routes, unplaced)."""
    n, V, CAP, H = state["n"], state["V"], state["CAP"], state["H"]
    T, dem, svc = state["T"], state["demand"], state["service"]
    op, cl = state["open"], state["close"]
    unserved = set(range(1, n))
    routes = []
    for v in range(V):
        t, cur, load, seq = 0, 0, 0, []
        while True:
            cands = []
            for b in unserved:
                if load + int(dem[b]) > CAP:
                    continue
                arr = t + int(T[cur, b])
                if arr > int(cl[b]):
                    continue
                if max(arr, int(op[b])) + int(svc[b]) + int(T[b, 0]) > H:
                    continue
                cands.append((int(T[cur, b]), int(cl[b]), b))
            if not cands:
                break
            _, _, b = min(cands)
            t = max(t + int(T[cur, b]), int(op[b])) + int(svc[b])
            load += int(dem[b])
            seq.append(b)
            unserved.discard(b)
            cur = b
        routes.append([0] + seq + [0])
    unplaced = []
    for b in sorted(unserved, key=lambda x: (-int(dem[x]), x)):
        loads = _loads(state, routes)
        fits = [(CAP - loads[k] - int(dem[b]), k) for k in range(V) if loads[k] + int(dem[b]) <= CAP]
        if fits:
            k = max(fits)[1]
            routes[k] = routes[k][:-1] + [b, 0]
        else:
            unplaced.append(b)
    return routes, sorted(unplaced)


def score_plan(state, routes, rates, clocked=False):
    """Execute a plan without raising on SLA violations. clocked=True: every vehicle leaves at minute 0;
    otherwise each route gets the departure that minimizes lateness, then duration. Returns (summary, sim)."""
    deps = []
    for seq in routes:
        if clocked or not [x for x in seq if x != 0]:
            deps.append(0)
        else:
            deps.append(best_departure_dyn(state, None, seq, 0))
    sim = simulate_plan(state, routes, rates, None, deps)
    loads = _loads(state, routes)
    CAP = state["CAP"]
    s = dict(sim["summary"])
    s.update({"loads": loads, "max_load_kg": max(loads), "overload_kg": int(sum(max(0, x - CAP) for x in loads)),
              "vehicles_used": int(sum(1 for seq in routes if [x for x in seq if x != 0])),
              "departures": deps, "clocked": bool(clocked)})
    return s, sim


# ======================= PHASE 4: H2 REVISION (engine 2.7) =======================
def _nn_construct(state):
    """Construction step of H2: vehicles built one at a time by nearest feasible stop.
    Returns (routes, leftover): leftover are stops no vehicle could reach in time and capacity."""
    n, V, CAP, H = state["n"], state["V"], state["CAP"], state["H"]
    T, dem, svc = state["T"], state["demand"], state["service"]
    op, cl = state["open"], state["close"]
    unserved = set(range(1, n))
    routes = []
    for v in range(V):
        t, cur, load, seq = 0, 0, 0, []
        while True:
            cands = []
            for b in unserved:
                if load + int(dem[b]) > CAP:
                    continue
                arr = t + int(T[cur, b])
                if arr > int(cl[b]):
                    continue
                if max(arr, int(op[b])) + int(svc[b]) + int(T[b, 0]) > H:
                    continue
                cands.append((int(T[cur, b]), int(cl[b]), b))
            if not cands:
                break
            _, _, b = min(cands)
            t = max(t + int(T[cur, b]), int(op[b])) + int(svc[b])
            load += int(dem[b])
            seq.append(b)
            unserved.discard(b)
            cur = b
        routes.append([0] + seq + [0])
    return routes, sorted(unserved)


def heuristic_nearest_neighbour_v2(state):
    """H2 v2 (simulated dispatcher): same construction as H2. Each leftover stop goes to the vehicle chosen exactly
    as in H2 v1 (largest drop first, most spare capacity), but is slotted into that route ahead of the first stop
    whose window closes later, not at the end. Returns (routes, unplaced)."""
    V, CAP = state["V"], state["CAP"]
    dem, cl = state["demand"], state["close"]
    routes, left = _nn_construct(state)
    unplaced = []
    for b in sorted(left, key=lambda x: (-int(dem[x]), x)):
        loads = _loads(state, routes)
        fits = [(CAP - loads[k] - int(dem[b]), k) for k in range(V) if loads[k] + int(dem[b]) <= CAP]
        if not fits:
            unplaced.append(b)
            continue
        k = max(fits)[1]
        stops = [x for x in routes[k] if x != 0]
        pos = len(stops)
        for j, x in enumerate(stops):
            if int(cl[x]) > int(cl[b]):
                pos = j
                break
        stops.insert(pos, b)
        routes[k] = [0] + stops + [0]
    return routes, sorted(unplaced)


# ======================= PHASE 4: INGESTION PATH (engine 2.8) =======================
INGEST_SCHEMA = [
    ("nodes_input.csv", "name", "required", "text, unique; the depot row may have any name"),
    ("nodes_input.csv", "is_depot", "required", "0 or 1; exactly one row must be 1"),
    ("nodes_input.csv", "lat", "required", "decimal degrees"),
    ("nodes_input.csv", "lon", "required", "decimal degrees"),
    ("nodes_input.csv", "demand_kg", "required", "number >= 0 and at most the vehicle capacity; depot row 0 or blank"),
    ("nodes_input.csv", "tw_open", "required (customers)", "HH:MM, 24 h clock, same day"),
    ("nodes_input.csv", "tw_close", "required (customers)", "HH:MM, later than tw_open"),
    ("nodes_input.csv", "service_min", "optional", "minutes; default is 10-20 scaled by demand"),
    ("nodes_input.csv", "cluster", "optional", "area label; default All"),
    ("nodes_input.csv", "congested", "optional", "0 or 1; default 0 (used when travel times are estimated and for disruptions)"),
    ("fleet.json", "n_vehicles, capacity_kg, shift_start, shift_end", "required", "integers and HH:MM clock strings"),
    ("travel_minutes.csv", "(square matrix)", "optional", "node names as first column and header; minutes, rounded up; estimated if absent"),
    ("road_km.csv", "(square matrix)", "optional", "same layout, kilometres; estimated from coordinates if absent"),
    ("routes_input.csv", "vehicle, seq, name, depart", "optional", "historical routing; one row per stop; depart (HH:MM) on the first stop of each vehicle"),
]


def _hhmm(s):
    parts = str(s).strip().split(":")
    ok = len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
    assert ok, "Invalid clock time '%s' (use HH:MM, 24 h)" % str(s)
    h, m = int(parts[0]), int(parts[1])
    assert 0 <= h <= 23 and 0 <= m <= 59, "Invalid clock time '%s' (use HH:MM, 24 h)" % str(s)
    return h * 60 + m


def _fmt_clock(abs_min):
    h, m = divmod(int(abs_min), 60)
    return "%02d:%02d" % (h, m)


def _road_km(lat_deg, lon_deg, circuity):
    lat_r = np.radians(np.asarray(lat_deg, dtype=float))[:, None]
    lon_r = np.radians(np.asarray(lon_deg, dtype=float))[:, None]
    dphi, dlam = lat_r.T - lat_r, lon_r.T - lon_r
    a = np.sin(dphi / 2) ** 2 + np.cos(lat_r) * np.cos(lat_r.T) * np.sin(dlam / 2) ** 2
    km = 2 * 6371.0 * np.arcsin(np.sqrt(a)) * circuity
    np.fill_diagonal(km, 0.0)
    return km


def _read_matrix(path, names, label):
    tm = pd.read_csv(path, index_col=0)
    tm.index = [str(x).strip() for x in tm.index]
    tm.columns = [str(x).strip() for x in tm.columns]
    assert set(tm.index) == set(names) and set(tm.columns) == set(names), label + " names must match the node names"
    vals = tm.loc[names, names].to_numpy(dtype=float)
    assert not np.isnan(vals).any() and (vals >= 0).all(), label + " has missing or negative values"
    return vals


def ingest_instance(base_config, nodes_csv, n_vehicles, capacity_kg, shift_start="07:00", shift_end="17:00",
                    travel_csv=None, road_km_csv=None, congestion_multiplier=None):
    """Load a real-data instance in the documented schema into the structure the engine uses."""
    warnings = []
    df = pd.read_csv(nodes_csv)
    required = ["name", "is_depot", "lat", "lon", "demand_kg", "tw_open", "tw_close"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, "Missing required column(s): " + ", ".join(missing)
    df["name"] = df["name"].astype(str).str.strip()
    dup = sorted(df["name"][df["name"].duplicated()].unique().tolist())
    assert not dup, "Duplicate node name(s): " + ", ".join(dup)
    flag = pd.to_numeric(df["is_depot"], errors="coerce").fillna(0).astype(int)
    assert int((flag == 1).sum()) == 1, "The file must contain exactly one depot row (is_depot = 1)"
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    assert not (lat.isna().any() or lon.isna().any()), "Invalid or missing lat/lon value"
    assert lat.between(-90, 90).all() and lon.between(-180, 180).all(), "lat/lon outside the valid range"
    dem = pd.to_numeric(df["demand_kg"], errors="coerce").fillna(0.0)
    assert (dem >= 0).all(), "Negative demand"
    s0 = _hhmm(shift_start)
    H = _hhmm(shift_end) - s0
    assert H > 0, "shift_end must be later than shift_start"
    cfg = json.loads(json.dumps(base_config))
    tiers = cfg["demand"]["tiers"]
    s_lo, s_hi = cfg["time"]["service_time_min"]
    d_lo, d_hi = min(t["min_kg"] for t in tiers), max(t["max_kg"] for t in tiers)
    order = [int(i) for i in df.index[flag == 1]] + [int(i) for i in df.index[flag != 1]]
    rows = []
    for nid, i in enumerate(order):
        r = df.loc[i]
        depot = nid == 0
        d = 0 if depot else int(round(float(dem.loc[i])))
        assert d <= capacity_kg, "Demand of %s (%d kg) exceeds vehicle capacity %d kg" % (r["name"], d, capacity_kg)
        if depot:
            o, c = 0, H
        else:
            o, c = _hhmm(r["tw_open"]) - s0, _hhmm(r["tw_close"]) - s0
            assert c > o, "The window of %s closes before it opens" % r["name"]
            assert not (c < 0 or o > H), "The window of %s lies entirely outside the shift" % r["name"]
            if o < 0 or c > H:
                warnings.append("Window of %s clipped to the shift" % r["name"])
            o, c = max(o, 0), min(c, H)
        if "service_min" in df.columns and not pd.isna(r["service_min"]) and not depot:
            sv = int(round(float(r["service_min"])))
        elif depot:
            sv = 0
        else:
            sv = int(np.clip(round(s_lo + (s_hi - s_lo) * (d - d_lo) / float(d_hi - d_lo)), s_lo, s_hi))
        cong = False
        if "congested" in df.columns and not pd.isna(r["congested"]):
            cong = bool(int(r["congested"]))
        cl = "Depot" if depot else (str(r["cluster"]).strip() if "cluster" in df.columns and not pd.isna(r["cluster"]) else "All")
        tier = "depot" if depot else ("small" if d <= tiers[0]["max_kg"] else "medium" if d <= tiers[1]["max_kg"] else "large")
        rows.append({"node_id": nid, "name": r["name"], "cluster": cl, "lat": float(lat.loc[i]), "lon": float(lon.loc[i]),
                     "congested_corridor": False if depot else cong, "demand_kg": d, "demand_tier": tier,
                     "est_direct_min": 0, "service_min": sv, "tight_window": False if depot else (c - o) <= 60,
                     "tw_open": int(o), "tw_close": int(c), "open_clock": _fmt_clock(s0 + o), "close_clock": _fmt_clock(s0 + c)})
    nodes = pd.DataFrame(rows)
    names = nodes["name"].tolist()
    n = len(nodes)
    tr = cfg["travel"]
    if road_km_csv:
        KM = _read_matrix(road_km_csv, names, "Road-km matrix")
    else:
        KM = _road_km(nodes["lat"], nodes["lon"], tr["road_circuity_factor"])
        warnings.append("Road distances estimated from coordinates")
    off = ~np.eye(n, dtype=bool)
    if travel_csv:
        T = np.ceil(_read_matrix(travel_csv, names, "Travel matrix")).astype(int)
        np.fill_diagonal(T, 0)
        T[off] = np.maximum(T[off], 1)
    else:
        lo, hi = tr["congestion_multiplier_range"]
        mult = float(congestion_multiplier) if congestion_multiplier else (lo + hi) / 2.0
        nm = np.where(nodes["congested_corridor"].to_numpy(), mult, 1.0)
        leg = np.maximum(nm[:, None], nm[None, :])
        T = np.ceil(_road_km(nodes["lat"], nodes["lon"], tr["road_circuity_factor"]) / tr["avg_speed_kmh"] * 60 * leg).astype(int)
        np.fill_diagonal(T, 0)
        T[off] = np.maximum(T[off], 1)
        warnings.append("Travel times estimated (speed %s km/h, congestion multiplier %.2f on congested nodes)" % (tr["avg_speed_kmh"], mult))
    nodes["est_direct_min"] = T[0, :].astype(int)
    cust = nodes[nodes["node_id"] > 0]
    clusters = []
    for name, g in cust.groupby("cluster", sort=False):
        clusters.append({"name": name, "lat": float(g["lat"].mean()), "lon": float(g["lon"].mean()), "nodes": int(len(g)),
                         "jitter_km": 0.0, "congested": bool(g["congested_corridor"].any())})
    cfg["network"].update({"n_delivery_nodes": n - 1, "n_vehicles": int(n_vehicles), "vehicle_capacity_kg": int(capacity_kg)})
    cfg["time"].update({"shift_start_min": 0, "shift_end_min": int(H), "shift_clock_start": shift_start})
    cfg["clusters"] = clusters
    cfg["demand"]["total_range_kg"] = [0, int(n_vehicles * capacity_kg)]
    cfg["depot"] = {"name": nodes.at[0, "name"], "lat": float(nodes.at[0, "lat"]), "lon": float(nodes.at[0, "lon"])}
    return {"seed": None, "spec": {"source": "ingested"}, "config": cfg, "nodes": nodes, "T": T, "KM": KM,
            "corridor_mult": {}, "demand_attempts": 0, "warnings": warnings}


def validate_ingested(inst):
    """Sanity checks for ingested data (the Phase 1 synthetic invariants do not apply). Returns stats and warnings."""
    cfg, nodes, T, KM = inst["config"], inst["nodes"], inst["T"], inst["KM"]
    n = len(nodes)
    V, CAP, H = cfg["network"]["n_vehicles"], cfg["network"]["vehicle_capacity_kg"], cfg["time"]["shift_end_min"]
    d = nodes[nodes["node_id"] > 0]
    assert n >= 2 and (nodes["node_id"].to_numpy() == np.arange(n)).all(), "Need a depot and at least one customer"
    assert (d["tw_open"] >= 0).all() and (d["tw_close"] <= H).all() and (d["tw_close"] > d["tw_open"]).all(), "Invalid windows"
    assert int(d["demand_kg"].max()) <= CAP, "A drop exceeds vehicle capacity"
    assert T.shape == (n, n) and np.issubdtype(T.dtype, np.integer), "Travel matrix must be an integer (n, n) array"
    off = ~np.eye(n, dtype=bool)
    assert (np.diag(T) == 0).all() and (T[off] > 0).all(), "Travel matrix diagonal or off-diagonal values are invalid"
    assert KM.shape == (n, n) and (KM >= 0).all(), "Road-km matrix is invalid"
    total = int(d["demand_kg"].sum())
    warns = list(inst.get("warnings", []))
    if total > 0.95 * V * CAP:
        warns.append("Utilization above 95%: an instance this tight may be infeasible")
    if total > V * CAP:
        warns.append("Total demand exceeds fleet capacity: infeasible")
    return {"n_customers": n - 1, "vehicles": V, "total_kg": total, "utilization": round(total / float(V * CAP), 3),
            "tight_windows": int(d["tight_window"].sum()), "shift_min": int(H), "warnings": warns}


def export_instance_csv(inst, folder, routes=None, departures=None, shift_start="07:00", shift_end="17:00"):
    """Write an instance in the documented ingestion schema (also used to make the template files)."""
    os.makedirs(folder, exist_ok=True)
    s0 = _hhmm(shift_start)
    nodes, cfg = inst["nodes"], inst["config"]
    rows = []
    for r in nodes.itertuples(index=False):
        depot = int(r.node_id) == 0
        rows.append({"name": r.name, "is_depot": 1 if depot else 0, "lat": r.lat, "lon": r.lon, "demand_kg": int(r.demand_kg),
                     "tw_open": "" if depot else _fmt_clock(s0 + int(r.tw_open)),
                     "tw_close": "" if depot else _fmt_clock(s0 + int(r.tw_close)),
                     "service_min": int(r.service_min), "cluster": r.cluster, "congested": int(bool(r.congested_corridor))})
    pd.DataFrame(rows).to_csv(os.path.join(folder, "nodes_input.csv"), index=False)
    names = nodes["name"].tolist()
    pd.DataFrame(inst["T"], index=names, columns=names).to_csv(os.path.join(folder, "travel_minutes.csv"))
    pd.DataFrame(inst["KM"], index=names, columns=names).to_csv(os.path.join(folder, "road_km.csv"))
    with open(os.path.join(folder, "fleet.json"), "w") as f:
        json.dump({"n_vehicles": cfg["network"]["n_vehicles"], "capacity_kg": cfg["network"]["vehicle_capacity_kg"],
                   "shift_start": shift_start, "shift_end": shift_end}, f, indent=2)
    if routes is not None:
        recs = []
        for v, seq in enumerate(routes):
            stops = [x for x in seq if x != 0]
            for k, x in enumerate(stops, start=1):
                dep = _fmt_clock(s0 + int(departures[v])) if (departures is not None and k == 1) else ""
                recs.append({"vehicle": v, "seq": k, "name": names[x], "depart": dep})
        pd.DataFrame(recs).to_csv(os.path.join(folder, "routes_input.csv"), index=False)


def ingest_routes(state, routes_csv, shift_start="07:00"):
    """Load historical routing. Returns (routes, departures or None). Every customer must appear exactly once."""
    df = pd.read_csv(routes_csv)
    missing = [c for c in ("vehicle", "seq", "name") if c not in df.columns]
    assert not missing, "Missing required column(s) in routes file: " + ", ".join(missing)
    df["name"] = df["name"].astype(str).str.strip()
    ids = {nm: i for i, nm in enumerate(state["nodes"]["name"])}
    unknown = sorted(set(df["name"]) - set(ids))
    assert not unknown, "Unknown node name(s) in routes: " + ", ".join(unknown)
    vehicles = sorted(df["vehicle"].unique().tolist())
    assert len(vehicles) <= state["V"], "The routes file uses more vehicles than the fleet"
    s0 = _hhmm(shift_start)
    routes, deps = [], []
    for v in vehicles:
        g = df[df["vehicle"] == v].sort_values("seq")
        seq = [ids[x] for x in g["name"]]
        assert 0 not in seq, "A route lists the depot as a stop; list customer stops only"
        routes.append([0] + seq + [0])
        dep = g["depart"].dropna().tolist() if "depart" in g.columns else []
        deps.append(_hhmm(dep[0]) - s0 if dep else None)
    while len(routes) < state["V"]:
        routes.append([0, 0])
        deps.append(0)
    served = sorted(x for r in routes for x in r if x != 0)
    assert served == list(range(1, state["n"])), "Routes must visit every customer exactly once"
    if any(d is None for d in deps):
        deps = None
    return routes, deps
