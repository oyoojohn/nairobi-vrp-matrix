"""vrp_maps.py | Nairobi VRP Matrix | map helpers (Phase 5)"""
import numpy as np
import pandas as pd
import folium
import vrp_engine

MAPS_VERSION = "1.0"
VEH_COLORS = ["#1F3A93", "#D35400", "#1E8449", "#7B241C", "#6C3483", "#117A65"]
PALETTE = ["#2E86C1", "#27AE60", "#8E44AD", "#E74C3C", "#E67E22", "#16A085", "#7F8C8D", "#D4AC0D", "#5D6D7E", "#A04000"]
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services"


def clock(m):
    h, mm = divmod(7 * 60 + int(m), 60)
    return "%02d:%02d" % (h, mm)


def add_caption(m, title, lines=None):
    html = ('<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:8px 12px;'
            'border:1px solid #888;border-radius:6px;font-family:Arial;font-size:12px;max-width:560px;">')
    html += "<b>" + title + "</b>"
    for ln in (lines or []):
        html += "<br>" + ln
    html += "</div>"
    m.get_root().html.add_child(folium.Element(html))


def new_map(state, title=None, lines=None):
    depot = state["nodes"].iloc[0]
    m = folium.Map(location=[float(depot["lat"]), float(depot["lon"])], zoom_start=11, tiles=None, control_scale=True)
    folium.TileLayer(tiles=ESRI + "/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri", name="Esri Light Gray", max_zoom=16).add_to(m)
    folium.TileLayer(tiles=ESRI + "/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
                     attr="Tiles &copy; Esri", name="Esri Street Map", max_zoom=19).add_to(m)
    m.get_root().header.add_child(folium.Element(
        '<meta name="referrer" content="strict-origin-when-cross-origin"><style>.leaflet-container{background:#eef1f4;}</style>'))
    folium.Marker([float(depot["lat"]), float(depot["lon"])], tooltip="Depot: " + str(depot["name"]),
                  icon=folium.Icon(color="black", icon="home")).add_to(m)
    if title:
        add_caption(m, title, lines)
    return m


def add_corridors(m, state, scenario=None):
    fg = folium.FeatureGroup(name="Congested corridors", show=True)
    for c in state["config"]["clusters"]:
        radius = c["jitter_km"] * 1000 + 600
        if scenario is not None and c["name"] == scenario["corridor"]:
            tip = "DISRUPTED: %s x%s from %s for %d min" % (c["name"], scenario["severity"], clock(scenario["onset"]), scenario["duration"])
            folium.Circle([c["lat"], c["lon"]], radius=radius, color="#E67E22", weight=3, fill=True,
                          fill_opacity=0.18, tooltip=tip).add_to(fg)
        elif c["congested"]:
            folium.Circle([c["lat"], c["lon"]], radius=radius, color="#C0392B", weight=1, fill=True,
                          fill_opacity=0.06, tooltip=c["name"] + " (usual congestion)").add_to(fg)
    fg.add_to(m)


def add_nodes(m, state):
    nodes = state["nodes"]
    names = [c["name"] for c in state["config"]["clusters"]]
    col = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    fg = folium.FeatureGroup(name="Customers", show=True)
    for r in nodes[nodes["node_id"] > 0].to_dict("records"):
        c = col.get(r["cluster"], "#34495E")
        html = ("<b>%s</b><br>Demand: %d kg<br>Service: %d min<br>Window: %s to %s<br>%s" %
                (r["name"], r["demand_kg"], r["service_min"], r["open_clock"], r["close_clock"],
                 "KEY ACCOUNT (tight window)" if r["tight_window"] else "Standard window"))
        folium.CircleMarker([r["lat"], r["lon"]], radius=5 + r["demand_kg"] / 40.0,
                            color="black" if r["tight_window"] else c, weight=3 if r["tight_window"] else 1,
                            fill=True, fill_color=c, fill_opacity=0.85,
                            tooltip="%s | %d kg" % (r["name"], r["demand_kg"]), popup=folium.Popup(html, max_width=260)).add_to(fg)
    fg.add_to(m)


def add_plan(m, state, routes, deps, sim, name, scenario=None, show=True, dashed=False):
    nodes = state["nodes"].set_index("node_id")
    T, op, svc = state["T"], state["open"], state["service"]
    st = sim["stops"].set_index(["vehicle", "node"])
    fg = folium.FeatureGroup(name=name, show=show)
    for v, seq in enumerate(routes):
        stops = [x for x in seq if x != 0]
        if not stops:
            continue
        col = VEH_COLORS[v % len(VEH_COLORS)]
        t = int(deps[v])
        for a, b in zip(seq[:-1], seq[1:]):
            w = vrp_engine.leg_minutes(state, scenario, a, b, t)
            slowed = w != int(T[a, b])
            A, B = nodes.loc[a], nodes.loc[b]
            kw = {"color": "#C0392B" if slowed else col, "weight": 6 if slowed else 3, "opacity": 0.9 if slowed else 0.75,
                  "tooltip": "%s | V%d: %s to %s | %d min%s" % (name, v, A["name"], B["name"], w, " (SLOWED)" if slowed else "")}
            if dashed and not slowed:
                kw["dash_array"] = "6,6"
            folium.PolyLine([[float(A["lat"]), float(A["lon"])], [float(B["lat"]), float(B["lon"])]], **kw).add_to(fg)
            t += w
            if b != 0:
                t = max(t, int(op[b])) + int(svc[b])
        for k, b in enumerate(stops, start=1):
            r, s = nodes.loc[b], st.loc[(v, b)]
            late = int(s["late_min"]) > 0
            border = "3px solid black" if r["tight_window"] else "1px solid white"
            icon = folium.DivIcon(
                html=('<div style="background:%s;color:white;border-radius:50%%;width:22px;height:22px;line-height:20px;'
                      'text-align:center;font-size:11px;font-weight:bold;border:%s;">%d</div>' % ("#C0392B" if late else col, border, k)),
                icon_size=(22, 22), icon_anchor=(11, 11))
            html = ("<b>%s</b> (%s: vehicle %d, stop %d)<br>Arrive %s | wait %d min<br>Window %s to %s<br><b>%s</b><br>%s" %
                    (r["name"], name, v, k, clock(s["arrive"]), int(s["wait"]), r["open_clock"], r["close_clock"],
                     "LATE by %d min" % int(s["late_min"]) if late else "On time",
                     "KEY ACCOUNT" if r["tight_window"] else "Standard window"))
            folium.Marker([float(r["lat"]), float(r["lon"])], icon=icon, popup=folium.Popup(html, max_width=280),
                          tooltip="V%d-%d: %s%s" % (v, k, r["name"], " | LATE" if late else "")).add_to(fg)
    fg.add_to(m)


def finish_map(m, state):
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds(state["nodes"][["lat", "lon"]].values.tolist(), padding=(20, 20))
    return m


def summary_line(label, s):
    return "%s: KES %s | %s km | %s min | %d breach(es), %d min late" % (
        label, format(int(s["total_kes"]), ","), s["km"], s["duration_min"], s["breaches"], s["total_late_min"])
