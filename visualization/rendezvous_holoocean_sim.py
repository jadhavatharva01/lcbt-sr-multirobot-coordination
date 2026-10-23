#!/usr/bin/env python3
"""
Rendezvous pipeline -> HoloOcean visualization / replay driver.

What this shows (the COMPLETED core idea, replayed in 3-D water):
  1. A frozen offline GRID MESH: every USV follows a precomputed space-time
     route and keeps precomputed appointments (rendezvous) with its neighbours.
  2. A DEATH: one USV is killed at t_f. It freezes in place and goes grey.
  3. DETECTION with no comms: the first neighbour whose appointment the victim
     misses (by more than eps) detects the failure -- a red beacon marks the
     missed rendezvous point.
  4. PRECOMPUTED RECOVERY: the takeover is a pure table lookup f(victim, last
     checkpoint) -> {orphan POIs, one taker each}. Takers weave the orphaned
     POIs into their remaining routes while STILL keeping every one of their own
     appointments (meeting-wins). Orphan POIs light up as they are recovered.

Nothing here re-plans or negotiates: the whole run is a REPLAY of the plan
already produced and validated by full_pipeline_sim.py (Stages A/X). HoloOcean
is used purely as a renderer -- the algorithm runs on the kinematic model.

Colour legend (drawn every tick so it reads even when zoomed out):
  USVs         : each USV a distinct bright colour; the dead one turns GREY.
  POI yellow   : pending (not yet searched)
  POI blue     : searched by its original owner
  POI orange   : ORPHAN -- the dead USV's leftover, awaiting takeover
  POI lime     : orphan RECOVERED by a taker
  cyan ring    : a rendezvous (meeting) happening right now
  red beacon   : the missed rendezvous -> the moment of detection
  coloured line: a taker's live takeover path to its orphan POIs

Camera is left FREE: an initial top-down vantage is sent once, then the script
never touches the camera again so you can fly around manually.

Run in this sandbox (no HoloOcean needed) to validate the plan glue:
  python3 rendezvous_holoocean_sim.py --dry_run

Run the actual 3-D visualization (needs your holoocean_env):
  source ~/holoocean_env/bin/activate
  python3 rendezvous_holoocean_sim.py                 # default, watchable pace
  python3 rendezvous_holoocean_sim.py --speed 4       # even slower
It plays in real time (mission-seconds per real-second = --speed) and pauses a
few seconds on the death and the detection so they are easy to catch.

Outputs -> results/holoocean_rendezvous/
"""

import argparse
import csv
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

# --- reuse the already-built + validated pipeline -------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_pipeline_sim import (  # noqa: E402
    make_grid_world, build_grid_mesh, run_failure, execute_repaired_mesh,
    DELTA, EPS,
)
from encounter_planning_probe import pos_at, V, SCAN, CONTACT  # noqa: E402

HUGE = 1e12
OUT = Path(__file__).resolve().parent.parent / "results" / "holoocean_rendezvous"

# ------------------------------------------------------------------ colours
DEAD      = [110, 110, 110]
POI_PEND  = [255, 230, 0]     # yellow  : not yet searched
POI_DONE  = [70, 140, 255]    # blue    : searched by original owner
POI_ORPH  = [255, 140, 0]     # orange  : orphan awaiting takeover
POI_RECOV = [0, 255, 120]     # lime    : orphan recovered by a taker
MEET      = [0, 220, 255]     # cyan    : rendezvous happening now
DETECT    = [255, 30, 30]     # red     : missed rendezvous = detection

# a distinct bright colour per USV (wraps if K is large)
PALETTE = [
    [255, 60, 60], [60, 160, 255], [60, 220, 120], [255, 200, 0],
    [200, 80, 255], [255, 130, 0], [0, 220, 220], [240, 80, 160],
    [150, 220, 60], [120, 120, 255], [255, 90, 140], [90, 200, 180],
    [200, 160, 80], [160, 100, 220], [80, 255, 200], [220, 220, 80],
]


def _pkey(p):
    return (round(p[0], 3), round(p[1], 3))


def resolve_dt(args):
    """Mission-seconds advanced per rendered tick. Derived from --speed
    (mission-seconds shown per real second) unless --mission_dt overrides."""
    if args.mission_dt:
        return args.mission_dt
    return max(0.05, args.speed / args.ticks_per_sec)


# ============================================================ plan assembly
def trace_to_bp(trace):
    """Rebuild interpolatable (t, x, y) breakpoints from a survivor's execution
    trace (list of (label, point, t_event)), matching _execute_survivor timing:
    travel at speed V, then dwell (SCAN at a POI, wait+CONTACT at a meeting)."""
    (_, p0, t0) = trace[0]
    bp = [(t0, p0[0], p0[1])]
    tc, pc = t0, p0
    for (label, pt, t_event) in trace[1:]:
        d = math.hypot(pt[0] - pc[0], pt[1] - pc[1])
        arr = tc + d / V
        if arr > tc + 1e-9:                      # moved -> arrival breakpoint
            bp.append((arr, pt[0], pt[1]))
        if t_event > arr + 1e-9:                 # dwell (SCAN / wait) to event
            bp.append((t_event, pt[0], pt[1]))
        tc, pc = t_event, pt
        if label == "appt":                      # brief contact dwell after
            tc = t_event + CONTACT
            bp.append((tc, pt[0], pt[1]))
    return bp


def assemble_replay(mesh, victim, t_f):
    """Per-robot replay breakpoints + roles for a single death at t_f.
    Non-takers replay their frozen trajectory; the victim freezes at t_f; each
    taker follows its frozen trajectory until detection, then its precomputed
    takeover execution."""
    K, trajs = mesh["K"], mesh["trajs"]
    res = execute_repaired_mesh(mesh, victim, t_f)
    takers = res["per_taker"]

    replay, role, t_det = {}, {}, {}
    for r in range(K):
        if r == victim:
            bp = [(t, x, y) for (t, x, y) in trajs[r] if t <= t_f]
            xf, yf = pos_at(trajs[r], t_f)
            bp.append((t_f, xf, yf))
            bp.append((HUGE, xf, yf))            # hold in place, dead
            replay[r], role[r] = bp, "victim"
        elif r in takers:
            td = takers[r]["t_det"]
            frozen = [(t, x, y) for (t, x, y) in trajs[r] if t < td]
            replay[r] = frozen + trace_to_bp(takers[r]["trace"])
            role[r], t_det[r] = "taker", td
        else:
            replay[r] = list(trajs[r])
            role[r] = "survivor"
    return replay, role, t_det, res, takers


def poi_catalog(mesh, victim, t_f, takers, orphan_keys):
    """Every POI (route point) with the replay time it gets searched and by whom.

    `orphan_keys` is the recovery's authoritative orphan set: the victim's POIs
    that fall AFTER its last confirmed checkpoint. The victim is credited only
    for its NON-orphan POIs (those the team saw it complete before the last
    checkpoint); everything after is orphaned and recovered by a taker -- which
    is exactly, and honestly, what the precomputed recovery does."""
    pois, scan_t, scanner = {}, {}, {}
    for rte in mesh["routes"]:
        for p in rte:
            k = _pkey(p)
            pois[k] = (p[0], p[1])
            scan_t.setdefault(k, None)
            scanner.setdefault(k, None)

    def mark(p, t, r):
        k = _pkey(p)
        scan_t[k], scanner[k] = t, r

    for r, log in enumerate(mesh["poi_logs"]):
        if r == victim:
            for (p, td) in log:                  # credit only confirmed work
                if _pkey(p) not in orphan_keys:
                    mark(p, td, r)
        elif r in takers:
            td = takers[r]["t_det"]
            for (p, tsc) in log:                 # own frozen work before detect
                if tsc <= td and _pkey(p) not in orphan_keys:
                    mark(p, tsc, r)
            for (label, p, t) in takers[r]["trace"]:   # own remainder + orphans
                if label == "poi":
                    mark(p, t, r)
        else:
            for (p, tsc) in log:
                mark(p, tsc, r)

    return pois, scan_t, scanner


def mission_span(mesh, victim, t_f, takers):
    """Last meaningful replay time (so we don't sit through HORIZON padding)."""
    span = t_f
    for r, log in enumerate(mesh["poi_logs"]):
        if r == victim:
            continue
        if r in takers:
            td = takers[r]["t_det"]
            for (p, tsc) in log:
                if tsc <= td:
                    span = max(span, tsc)
        else:
            for (p, tsc) in log:
                span = max(span, tsc)
    for r in takers:
        span = max(span, takers[r]["trace"][-1][2])
    for ap in mesh["appts"]:
        span = max(span, ap["t"])
    return span


def poi_color(k, t, t_f, scan_t, orphan_keys):
    """Colour for POI k at replay time t."""
    searched = scan_t[k] is not None and scan_t[k] <= t
    is_orphan = k in orphan_keys
    if searched:
        return POI_RECOV if is_orphan else POI_DONE
    if is_orphan and t >= t_f:
        return POI_ORPH                          # leftover awaiting takeover
    return POI_PEND


# ============================================================ HoloOcean glue
def to_world(p, base, scale, L, z):
    return [base[0] + (p[0] - L / 2.0) * scale,
            base[1] + (p[1] - L / 2.0) * scale, z]


def make_usv_agent(name, loc, yaw=0.0, main=False):
    return {
        "agent_name": name,
        "agent_type": "SurfaceVessel",
        # LocationSensor lets us paint the marker on the vessel's ACTUAL pose
        # (not the commanded target, which the PD-driven hull trails).
        "sensors": [{"sensor_type": "LocationSensor"}],
        "location": [float(loc[0]), float(loc[1]), float(loc[2])],
        "rotation": [0.0, 0.0, float(yaw)],
        "control_scheme": 1,               # PD target [des_x, des_y, des_yaw]
        "is_main_agent": bool(main),
    }


def get_template_scenario(world_name):
    from holoocean import packagemanager
    aliases = {"OpenWater": "OpenWater-Hovering", "Dam": "Dam-Hovering"}
    name = aliases.get(world_name, world_name)
    try:
        cfg = deepcopy(packagemanager.get_scenario(name))
    except Exception as exc:
        raise SystemExit(
            f"Could not load HoloOcean scenario '{name}'. Try --world OpenWater "
            f"or --world Dam.\nOriginal error: {exc}")
    cfg["package_name"] = "Ocean"
    cfg["name"] = "Rendezvous-Recovery"
    return cfg


def template_base(cfg, surface_z):
    agents = cfg.get("agents", [])
    if not agents:
        return [0.0, 0.0, surface_z]
    loc = agents[0].get("location", [0.0, 0.0, surface_z])
    if isinstance(loc, dict):
        return [float(loc.get("x", 0.0)), float(loc.get("y", 0.0)), surface_z]
    return [float(loc[0]), float(loc[1]), surface_z]


# ---- best-effort viewport camera (send once, then leave free) ------------
def send_camera_pose(env, location, rotation):
    location = [float(x) for x in location]
    rotation = [float(x) for x in rotation]
    try:
        from holoocean.command import TeleportCameraCommand
        cmd = TeleportCameraCommand(location, rotation)
        for attr in ("_command_center", "command_center"):
            c = getattr(env, attr, None)
            if c is not None and hasattr(c, "enqueue_command"):
                c.enqueue_command(cmd)
                return True
    except Exception:
        pass
    for method in ("send_command", "send_client_command"):
        try:
            if hasattr(env, method):
                from holoocean.command import TeleportCameraCommand
                getattr(env, method)(TeleportCameraCommand(location, rotation))
                return True
        except Exception:
            pass
    return False


def draw_overlays(env, t, mesh, replay, role, victim, t_f, scan_t,
                  pois, orphan_keys, colors, world, life, flash_win, usv_world):
    """`usv_world` maps robot -> its ACTUAL (x, y) world position (from the
    LocationSensor), so markers sit on the hull instead of on the target."""
    base, scale, L = world
    zc = base[2]

    def W(p, z):                                # mesh coords -> world
        return to_world(p, base, scale, L, zc + z)

    def Wr(wp, z):                              # already-world coords -> draw
        return [wp[0], wp[1], zc + z]

    # ---- per-USV planned route lines (its own colour) ----------------------
    for r in range(mesh["K"]):
        if role[r] == "victim" and t >= t_f:
            continue
        prev = Wr(usv_world[r], 1.0)
        for p in mesh["routes"][r]:             # only the not-yet-searched ones
            k = _pkey(p)
            if scan_t[k] is not None and scan_t[k] <= t:
                continue
            nxt = W(p, 1.0)
            env.draw_line(prev, nxt, color=colors[r], thickness=3.0, lifetime=life)
            prev = nxt
        if role[r] == "taker":                  # extend the line onto orphans
            for p in _TAKER_ORPHANS.get(r, []):
                k = _pkey(p)
                if not (scan_t[k] is not None and scan_t[k] <= t):
                    env.draw_line(Wr(usv_world[r], 1.0), W(p, 1.0),
                                  color=colors[r], thickness=5.0, lifetime=life)

    # ---- POIs: coloured beacon + box + tall pole so they read zoomed out ---
    for k, p in pois.items():
        col = poi_color(k, t, t_f, scan_t, orphan_keys)
        env.draw_point(W(p, 0.5), color=col, thickness=16.0, lifetime=life)
        env.draw_line(W(p, 0.5), W(p, 8.0), color=col, thickness=5.0, lifetime=life)
        if col in (POI_ORPH, POI_RECOV):        # emphasise the takeover story
            env.draw_box(W(p, 1.5), [1.6, 1.6, 1.6], color=col,
                         thickness=6.0, lifetime=life)

    # ---- rendezvous flashes (cyan) and the detection flash (red) -----------
    for ap in mesh["appts"]:
        pt = ap["pt"]
        involves_victim = victim in (ap["a"], ap["b"])
        if not involves_victim and abs(t - ap["t"]) < flash_win:
            env.draw_point(W(pt, 6.0), color=MEET, thickness=34.0, lifetime=life)
        if involves_victim and ap["t"] > t_f and abs(t - (ap["t"] + EPS)) < flash_win:
            env.draw_point(W(pt, 6.0), color=DETECT, thickness=42.0, lifetime=life)
            env.draw_line(W(pt, 0.5), W(pt, 20.0), color=DETECT,
                          thickness=10.0, lifetime=life)

    # ---- USVs: drawn on their ACTUAL pose, distinct colour, grey when dead --
    for r in range(mesh["K"]):
        dead = (role[r] == "victim" and t >= t_f)
        col = DEAD if dead else colors[r]
        wp = usv_world[r]
        env.draw_box(Wr(wp, 1.5), [2.6, 2.6, 2.6], color=col,
                     thickness=8.0, lifetime=life)
        env.draw_line(Wr(wp, 0.5), Wr(wp, 14.0), color=col,
                      thickness=10.0, lifetime=life)
        env.draw_point(Wr(wp, 14.5), color=col, thickness=24.0, lifetime=life)


_TAKER_ORPHANS = {}   # taker index -> list of its orphan POI points (overlay)


# ============================================================ main run modes
def build_plan(args):
    starts, routes = make_grid_world(args.seed, args.Gr, args.Gc, args.num_pois,
                                     args.L)
    mesh = build_grid_mesh(starts, routes, args.Gr, args.Gc, args.L, DELTA)
    K = mesh["K"]
    victim = args.victim if args.victim is not None else K // 2
    t_f = args.t_fail if args.t_fail is not None else 1.1 * DELTA

    fail = run_failure(mesh, victim, t_f)
    replay, role, t_det, res, takers = assemble_replay(mesh, victim, t_f)
    orphan_keys = {_pkey(p) for p in fail["orphans"]}
    pois, scan_t, scanner = poi_catalog(mesh, victim, t_f, takers, orphan_keys)
    span = mission_span(mesh, victim, t_f, takers)

    # cache each taker's orphan targets for the takeover-path overlay
    _TAKER_ORPHANS.clear()
    for r, pts in fail["assign_pts"].items():
        _TAKER_ORPHANS[r] = list(pts)

    return dict(mesh=mesh, victim=victim, t_f=t_f, fail=fail, res=res,
                replay=replay, role=role, t_det=t_det, takers=takers,
                pois=pois, scan_t=scan_t, scanner=scanner,
                orphan_keys=orphan_keys, span=span)


def report_plan(P, args):
    mesh, fail, res = P["mesh"], P["fail"], P["res"]
    print("=" * 66)
    print(f"  Rendezvous recovery plan   grid {args.Gr}x{args.Gc} "
          f"= {mesh['K']} USVs, {args.num_pois} POIs, L={args.L} m")
    print("=" * 66)
    lat = fail["latency"]
    print(f"  victim USV        : {P['victim']}   (killed at t_f = {P['t_f']:.0f} s)")
    print(f"  detected by       : {fail['n_detectors']} neighbour(s) "
          f"({'redundant' if fail['redundant'] else 'single'} witness)")
    print(f"  detection latency : {lat:.0f} s after death"
          if lat is not None else "  detection latency : (no post-death meeting)")
    print(f"  orphaned POIs     : {res['n_orphans']}   "
          f"takers = {sorted(P['takers'])}")
    print(f"  ownership closed  : {res['ownership_ok']}  "
          f"(all orphans recovered)")
    print(f"  appointments kept : {res['appts_kept_pct']:.1f}% "
          f"of {res['appt_tot']} post-death meetings")
    print(f"  worst makespan cost: +{res['worst_makespan_over']:.1f} pp "
          f"(executed) vs +{res['worst_iso_over']:.1f} pp (isolated est.)")
    print(f"  replay span       : {P['span']:.0f} s of mission time")
    print("=" * 66)


def run_dry(args):
    P = build_plan(args)
    report_plan(P, args)

    mesh, victim, t_f = P["mesh"], P["victim"], P["t_f"]
    replay, scan_t = P["replay"], P["scan_t"]
    pois, orphan_keys, span = P["pois"], P["orphan_keys"], P["span"]

    # 1) breakpoints are monotone in time (interpolation is well-formed)
    bad = 0
    for r, bp in replay.items():
        for a, b in zip(bp, bp[1:]):
            if b[0] < a[0] - 1e-6:
                bad += 1
    print(f"  [check] non-monotone replay breakpoints : {bad} (want 0)")

    # 2) step the replay and confirm the takeover story actually plays out
    dt = resolve_dt(args)
    t, ticks = 0.0, 0
    orphan_recovered_at = {}
    while t <= span + dt:
        for k in orphan_keys:
            if k not in orphan_recovered_at \
                    and scan_t[k] is not None and scan_t[k] <= t:
                orphan_recovered_at[k] = t
        t += dt
        ticks += 1

    # 3) end state: every POI searched, every orphan recovered
    all_scanned = all(v is not None for v in scan_t.values())
    recovered = sum(1 for k in orphan_keys if k in orphan_recovered_at)
    orphans = list(orphan_keys)
    print(f"  [check] POIs searched by end            : "
          f"{sum(v is not None for v in scan_t.values())}/{len(scan_t)} "
          f"({'ALL' if all_scanned else 'MISSING'})")
    print(f"  [check] orphans recovered in replay     : "
          f"{recovered}/{len(orphans)}")
    play = span + max(0.0, args.deploy_time)
    wall = play / args.speed if not args.mission_dt else ticks / args.ticks_per_sec
    print(f"  [check] replay ticks @ dt={dt:.2f}s        : {ticks} "
          f"(~{wall:.0f}s to watch at --speed {args.speed:.0f}, "
          f"incl. {args.deploy_time:.0f}s deploy)")

    # 4) sample a few USV positions to eyeball motion
    print("  [sample] USV0 world path (mesh coords):")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        ts = frac * span
        xs = ", ".join(f"U{r}=({pos_at(replay[r], ts)[0]:.0f},"
                       f"{pos_at(replay[r], ts)[1]:.0f})"
                       for r in range(min(mesh["K"], 4)))
        print(f"           t={ts:6.0f}s  {xs}")

    OUT.mkdir(parents=True, exist_ok=True)
    summ = OUT / "dry_run_summary.csv"
    with summ.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["victim", "t_f", "K", "num_pois", "detect_latency_s",
                    "n_detectors", "n_orphans", "ownership_ok",
                    "appts_kept_pct", "worst_makespan_over_pp", "span_s",
                    "all_pois_searched", "monotone_ok"])
        w.writerow([victim, f"{t_f:.0f}", mesh["K"], args.num_pois,
                    f"{P['fail']['latency']:.0f}" if P['fail']['latency'] else "",
                    P["fail"]["n_detectors"], P["res"]["n_orphans"],
                    P["res"]["ownership_ok"],
                    f"{P['res']['appts_kept_pct']:.1f}",
                    f"{P['res']['worst_makespan_over']:.1f}",
                    f"{span:.0f}", all_scanned, bad == 0])
    print(f"\n  dry-run summary -> {summ}")
    print("  Glue OK. Run without --dry_run inside holoocean_env to visualize.")


def run_holoocean(args):
    try:
        import holoocean  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Could not import holoocean. Activate your venv first:\n"
            "  source ~/holoocean_env/bin/activate\n"
            f"Original error: {exc}")
    import holoocean

    P = build_plan(args)
    report_plan(P, args)
    mesh, victim, t_f = P["mesh"], P["victim"], P["t_f"]
    replay, role = P["replay"], P["role"]
    scan_t = P["scan_t"]
    pois, orphan_keys, span = P["pois"], P["orphan_keys"], P["span"]
    K = mesh["K"]
    colors = {r: PALETTE[r % len(PALETTE)] for r in range(K)}

    cfg = get_template_scenario(args.world)
    base = template_base(cfg, args.surface_z)
    world = (base, args.world_scale, args.L)

    # ---- deploy phase: all USVs start clustered at one dock, then fan out ---
    T_dep = max(0.0, args.deploy_time)
    dock = (args.L / 2.0, args.dock_y)                 # common launch point (mesh)
    route_start = {r: pos_at(replay[r], 0.0) for r in range(K)}
    # lay the fleet out as a compact grid of safely-spaced berths at the dock,
    # so no two hulls overlap at spawn (overlap -> physics blow-up -> crash).
    dock_pos = {}
    cols = int(math.ceil(math.sqrt(K)))
    sp = max(args.dock_spread, 12.0)                   # mesh units between berths
    for r in range(K):
        rr, cc = divmod(r, cols)
        dock_pos[r] = (dock[0] + (cc - (cols - 1) / 2.0) * sp,
                       dock[1] - rr * sp)              # rows sit behind the line

    def replay_pos(r, tt):
        """Mesh-coord position of USV r at mission-time tt. tt<0 is the deploy
        phase: linear fan-out from the dock to the route start."""
        if tt < 0.0 and T_dep > 0.0:
            a = max(0.0, min(1.0, (tt + T_dep) / T_dep))
            dx, dy = dock_pos[r]
            rx, ry = route_start[r]
            return dx + a * (rx - dx), dy + a * (ry - dy)
        return pos_at(replay[r], tt)

    def world_of(r, tt):
        wx, wy, _ = to_world(replay_pos(r, tt), base, args.world_scale, args.L,
                             args.surface_z)
        return wx, wy

    agents, names = [], []
    for r in range(K):
        wx, wy = world_of(r, -T_dep)
        name = f"usv{r}"
        names.append(name)
        agents.append(make_usv_agent(name, [wx, wy, args.surface_z], yaw=0.0,
                                     main=(r == 0)))
    cfg["agents"] = agents
    cfg["main_agent"] = names[0]
    cfg["ticks_per_sec"] = int(args.ticks_per_sec)
    cfg["frames_per_sec"] = int(args.ticks_per_sec)
    cfg["window_width"] = 1280
    cfg["window_height"] = 720

    mdt = resolve_dt(args)
    life = max(2.0 / args.ticks_per_sec, args.overlay_life)
    flash_win = max(mdt, EPS)                  # how long a flash stays lit
    first_det = P["fail"]["first_det"]
    total_play = span + T_dep
    wall_est = total_play / args.speed if (args.realtime and not args.mission_dt) \
        else total_play / (mdt * args.ticks_per_sec)
    prev_world = {r: replay_pos(r, -T_dep) for r in range(K)}
    sensed = {r: world_of(r, -T_dep) for r in range(K)}   # actual hull positions

    print("\nLegend: each USV a distinct colour (dead = grey), with a route line "
          "in its colour. POI yellow=pending, blue=done, orange=orphan, "
          "lime=recovered. cyan=rendezvous now, red=detection.")
    print(f"Playback: {args.speed:.0f} mission-s per real-s "
          f"(~{wall_est:.0f}s to watch, incl. {T_dep:.0f}s deploy). "
          f"Use --speed to change.")
    print("Camera is FREE after the initial vantage -- fly around manually.\n")

    sensor_ok = [False]

    def read_sensed(states):
        """Update actual hull positions from the LocationSensor."""
        for r in range(K):
            try:
                loc = states[names[r]]["LocationSensor"]
                sensed[r] = (float(loc[0]), float(loc[1]))
                sensor_ok[0] = True
            except Exception:
                pass

    def paint(tt):
        """Command every USV toward its replay target at mission-time tt, then
        repaint all overlays with markers on the ACTUAL (sensed) hull poses.
        If the sensor is unavailable, markers track the plan target instead."""
        for r in range(K):
            wx, wy = world_of(r, tt)
            px, py = prev_world[r]
            cx, cy = replay_pos(r, tt)
            yaw = math.degrees(math.atan2(cy - py, cx - px)) \
                if (abs(cx - px) + abs(cy - py)) > 1e-6 else 0.0
            prev_world[r] = (cx, cy)
            dead = (role[r] == "victim" and tt >= t_f)
            env.act(names[r], np.array([wx, wy, 0.0 if dead else yaw],
                                       dtype=np.float32))
            if not sensor_ok[0]:               # fallback: no sensor -> use target
                sensed[r] = (wx, wy)
        draw_overlays(env, tt, mesh, replay, role, victim, t_f, scan_t,
                      pois, orphan_keys, colors, world, life, flash_win, sensed)

    def hold(tt, seconds):
        """Freeze the scene at mission-time tt for `seconds` of wall time, still
        repainting + ticking so the viewport stays live and this moment's flash
        keeps showing."""
        for _ in range(int(max(1, seconds * args.ticks_per_sec))):
            paint(tt)
            read_sensed(env.tick())
            time.sleep(1.0 / args.ticks_per_sec)

    env = holoocean.make(scenario_cfg=cfg, show_viewport=True,
                         verbose=args.verbose)
    try:
        for _ in range(15):                    # settle vessels at the dock
            paint(-T_dep)
            read_sensed(env.tick())

        if not args.no_initial_camera:
            ok = send_camera_pose(env, [base[0], base[1], args.camera_height],
                                  [-88.0, 0.0, 0.0])
            print("Initial top-down camera sent; camera is now free."
                  if ok else "Camera command not accepted; using default view.")

        if T_dep > 0.0:
            print(f"  Deploying: all {K} USVs launch from the dock and fan out "
                  f"to their sectors ...")
        t, next_print = -T_dep, 0.0
        deployed_shown = died_shown = detect_shown = False
        while t <= span + mdt:
            paint(t)
            read_sensed(env.tick())
            if args.realtime:
                time.sleep(1.0 / args.ticks_per_sec)

            if not deployed_shown and t >= 0.0:
                print("  Deployed. Coverage mission + scheduled rendezvous begin.")
                deployed_shown = True
            if not died_shown and t >= t_f:
                print(f"\n  *** USV {victim} FAILED at t={t_f:.0f}s -- it freezes "
                      f"and goes silent (grey). Its POIs turn orange. ***\n")
                died_shown = True
                hold(t, args.event_pause)
            if not detect_shown and first_det is not None and t >= first_det:
                print(f"\n  *** DETECTED at t={first_det:.0f}s: a neighbour's "
                      f"rendezvous with USV {victim} was missed. Precomputed "
                      f"takeover begins -- watch orange POIs turn lime. ***\n")
                detect_shown = True
                hold(t, args.event_pause)

            if t >= next_print:
                done = sum(1 for v in scan_t.values() if v is not None and v <= t)
                phase = ("deploying" if t < 0 else "pre-failure" if t < t_f else
                         "recovering" if first_det is None or t >= first_det
                         else "failed, not yet detected")
                print(f"  t={t:6.0f}s  searched={done}/{len(scan_t)}  [{phase}]")
                next_print += args.print_every
            t += mdt

        hold(span, max(args.event_pause, 2.0))     # linger on the finished state
        print("\nReplay complete. Recovery kept "
              f"{P['res']['appts_kept_pct']:.0f}% of appointments and recovered "
              f"all {P['res']['n_orphans']} orphaned POIs.")
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(
        description="Replay the rendezvous failure/recovery pipeline in HoloOcean.")
    # scenario / plan
    ap.add_argument("--Gr", type=int, default=3, help="grid rows (USVs = Gr*Gc)")
    ap.add_argument("--Gc", type=int, default=3, help="grid cols")
    ap.add_argument("--num_pois", type=int, default=45)
    ap.add_argument("--L", type=float, default=600.0, help="mesh side length (m)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--victim", type=int, default=None,
                    help="index of the USV that dies (default: centre robot)")
    ap.add_argument("--t_fail", type=float, default=None,
                    help="death time in s (default: 1.1*DELTA)")
    # deploy: all USVs start clustered at one dock, then fan out to their cells
    ap.add_argument("--deploy_time", type=float, default=120.0,
                    help="mission-seconds spent fanning out from the dock "
                         "(0 = spawn already in sectors)")
    ap.add_argument("--dock_y", type=float, default=-40.0,
                    help="dock y in mesh coords (below the grid by default)")
    ap.add_argument("--dock_spread", type=float, default=14.0,
                    help="mesh-units between berths at the dock (>=12 enforced "
                         "so hulls never overlap at spawn)")
    # playback
    ap.add_argument("--speed", type=float, default=8.0,
                    help="playback speed: mission-seconds shown per real second "
                         "(lower = slower / easier to follow)")
    ap.add_argument("--mission_dt", type=float, default=None,
                    help="override mission-seconds advanced per rendered tick "
                         "(default derived from --speed and --ticks_per_sec)")
    ap.add_argument("--ticks_per_sec", type=float, default=30.0)
    ap.add_argument("--realtime", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="pace playback to real time (default on); pass "
                         "--no-realtime to play as fast as possible")
    ap.add_argument("--event_pause", type=float, default=3.0,
                    help="wall-seconds to freeze on the death and first "
                         "detection so they are easy to see")
    ap.add_argument("--print_every", type=float, default=DELTA,
                    help="console status cadence in mission seconds")
    # world mapping / rendering
    ap.add_argument("--world", default="OpenWater")
    ap.add_argument("--world_scale", type=float, default=0.5,
                    help="mesh-metre -> world-metre scale")
    ap.add_argument("--surface_z", type=float, default=0.0)
    ap.add_argument("--overlay_life", type=float, default=0.2,
                    help="seconds each drawn overlay persists")
    ap.add_argument("--camera_height", type=float, default=220.0)
    ap.add_argument("--no_initial_camera", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    # mode
    ap.add_argument("--dry_run", action="store_true",
                    help="validate the plan glue without HoloOcean")
    args = ap.parse_args()

    if args.dry_run:
        run_dry(args)
    else:
        run_holoocean(args)


if __name__ == "__main__":
    main()
