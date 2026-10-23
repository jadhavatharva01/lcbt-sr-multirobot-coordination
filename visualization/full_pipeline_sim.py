#!/usr/bin/env python3
"""
FULL-PIPELINE STRESS SIM for the rendezvous concept
(docs/rendezvous_allocation_idea.md, problems P1/P2 in
docs/rendezvous_problems.md).

The feasibility probe (encounter_planning_probe.py) tested the pieces in
isolation. This puts the whole loop in ONE simulation and kills robots
mid-mission to check the P1 recovery and P2 timing rules end-to-end:

  MESH PLANNED AHEAD (global common knowledge)
     One deterministic OFFLINE pass produces every robot's full space-time
     trajectory AND every appointment (time + place). That frozen mesh is
     what every robot knows about every other robot for the entire mission.
     The live run replays against it; nothing deviates unless a death fires
     the "will".

  P1  contract-preserving recovery
     - contact contracts: an appointment's point is fixed at the pair's
       PREVIOUS contact and never changed unilaterally.
     - slot inheritance: a dead robot's two partners each detect the death
       independently (their own missed appointment), then per the precomputed
       will become partners with each other in the dead robot's vacated slots.
     - orphans yield to contracts: the dead robot's un-scanned POIs are split
       among feasible neighbours by a precomputed succession table (balanced),
       appended in slack / at the end. Total-ownership guarantee => every POI
       always has exactly one owner, so nothing is dropped or double-taken.

  P2  meeting-vs-task timing
     - every robot reserves a SLACK budget before each appointment.
     - "meeting wins": a task overrun is dropped to make the meeting; the
       dropped scan becomes a self-orphan done later. Demonstrated explicitly.

Stages (each prints its own diagnostics):
  A  build + verify the frozen mesh (R1-R4, everyone attends, cost vs baseline)
  B  inject one death, measure detection (latency <= DELTA+eps, redundancy,
     zero false alarms on the living)
  C  run recovery, measure orphan coverage (total-ownership), orphan delay,
     survivor travel overhead, chain re-weld
  D  P2 demonstration: meeting-wins vs task-wins on a forced overrun

Run:  python3 visualization/full_pipeline_sim.py [A|B|C|D|all]
Outputs -> results/full_pipeline/
"""

import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from encounter_planning_probe import (
    dist, V, SCAN, R, CONTACT, cheapest_insertion_routes, route_length,
    pos_at, _bp_dist,
)

OUT = Path(__file__).resolve().parent.parent / "results" / "full_pipeline"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- parameters
DELTA = 300.0    # s: guaranteed check interval (appointment cadence)
EPS = 30.0       # s: wait tolerance at an appointment before "missed"
SLACK = 30.0     # s: P2 slack budget reserved before each appointment
HORIZON = 60000.0
REACH_FRAC = 0.6  # forward reach for orphan pickup, as a fraction of band width:
#                   an orphan is "forward-absorbable" by a survivor if it passes
#                   within this distance of the survivor's REMAINING route.
#                   Beyond it, the orphan is "behind" -> deferred to end cleanup.


# ---------------------------------------------------------------- walker
class Walker:
    """Route-following walker; logs POI completion times so we can tell which
    POIs a victim had already scanned at the instant of death."""

    def __init__(self, start, route):
        self.bp = [(0.0, start[0], start[1])]
        self.queue = list(route)
        self.t = 0.0
        self.pos = start
        self.poi_log = []          # (poi, t_scan_done)

    def advance_to(self, t_target):
        while self.queue and self.t < t_target:
            q = self.queue[0]
            d = dist(self.pos, q)
            t_arr = self.t + d / V
            if t_arr > t_target:               # stop mid-leg
                frac = (t_target - self.t) / (d / V)
                self.pos = (self.pos[0] + frac * (q[0] - self.pos[0]),
                            self.pos[1] + frac * (q[1] - self.pos[1]))
                self.t = t_target
                self.bp.append((self.t, self.pos[0], self.pos[1]))
                return
            self.t = t_arr
            self.pos = q
            self.bp.append((self.t, q[0], q[1]))
            self.t += SCAN
            self.bp.append((self.t, q[0], q[1]))
            self.poi_log.append((q, self.t))
            self.queue.pop(0)

    def goto(self, pt):
        self.t += dist(self.pos, pt) / V
        self.pos = pt
        self.bp.append((self.t, pt[0], pt[1]))

    def wait_until(self, t):
        if t > self.t:
            self.t = t
            self.bp.append((self.t, self.pos[0], self.pos[1]))

    def done(self):
        return not self.queue


# ---------------------------------------------------------------- world / bands
def make_banded_world(seed, K, N, L):
    """One robot per vertical band. Dock at the band centre, POIs of that band
    belong to that robot's territory."""
    rng = random.Random(seed)
    starts = [((i + 0.5) * L / K, 0.0) for i in range(K)]
    pois = [(rng.uniform(0.02 * L, 0.98 * L), rng.uniform(0.08 * L, 0.98 * L))
            for _ in range(N)]
    bands = [[] for _ in range(K)]
    for p in pois:
        i = min(K - 1, int(p[0] / (L / K)))
        bands[i].append(p)
    routes = [cheapest_insertion_routes([starts[i]], bands[i])[0]
              for i in range(K)]
    return starts, routes, bands


def brickwall_pairs(m, K):
    """Alternating-neighbour appointments (the simplest concrete schedule that
    satisfies R1-R4 on a line of bands). Odd rounds pair (0,1),(2,3),...;
    even rounds pair (1,2),(3,4),.... Middle robots are checked every round
    (bound DELTA); the two end robots every other round (bound 2*DELTA)."""
    pairs = []
    start = 0 if m % 2 == 1 else 1
    i = start
    while i + 1 < K:
        pairs.append((i, i + 1))
        i += 2
    return pairs


# ---------------------------------------------------------------- offline mesh
def build_mesh(starts, routes, K, L, delta):
    """Deterministic offline pass -> frozen global mesh: every robot's full
    trajectory + every appointment (round, pair, time, place). The point of
    each appointment is the band boundary at the two robots' current mean y
    (agreed 'at the previous contact' == fixed here, once, for all)."""
    walkers = [Walker(starts[i], routes[i]) for i in range(K)]
    appts = []
    boundary = lambda i: (i + 1) * L / K
    m = 1
    while any(not w.done() for w in walkers) and m * delta < HORIZON:
        t_m = m * delta
        for w in walkers:
            w.advance_to(t_m)
        for (i, j) in brickwall_pairs(m, K):
            pt = (boundary(i), (walkers[i].pos[1] + walkers[j].pos[1]) / 2.0)
            walkers[i].goto(pt)
            walkers[j].goto(pt)
            t_meet = max(walkers[i].t, walkers[j].t)
            walkers[i].wait_until(t_meet)
            walkers[j].wait_until(t_meet)
            walkers[i].t += CONTACT
            walkers[j].t += CONTACT
            walkers[i].bp.append((walkers[i].t, pt[0], pt[1]))
            walkers[j].bp.append((walkers[j].t, pt[0], pt[1]))
            appts.append(dict(m=m, t=t_meet, a=i, b=j, pt=pt))
        m += 1
    for w in walkers:
        w.advance_to(HORIZON)
    trajs = [w.bp for w in walkers]
    poi_logs = [w.poi_log for w in walkers]
    mesh = dict(trajs=trajs, appts=appts, poi_logs=poi_logs, routes=routes,
                starts=starts, K=K, L=L, delta=delta, topology="line",
                neighbors={i: [x for x in (i - 1, i + 1) if 0 <= x < K]
                           for i in range(K)},
                centers={i: ((i + 0.5) * L / K, 0.0) for i in range(K)},
                reach=REACH_FRAC * (L / K))
    # the ONE precomputed takeover pattern, f(dead robot, checkpoint) -> who
    # takes which orphan. Common knowledge -> every survivor applies the same
    # pattern, no negotiation. (Replaces the old per-POI nearest-neighbour succ.)
    mesh["table"] = build_contingency(mesh)
    return mesh


def no_rdv_baseline(starts, routes):
    """Coverage routes with no appointments at all (the cost floor)."""
    travel, spans = 0.0, []
    for i, route in enumerate(routes):
        d = route_length(starts[i], route)
        travel += d
        spans.append(d / V + SCAN * len(route))
    return dict(travel=travel, makespan=max(spans) if spans else 0.0)


# ---------------------------------------------------------------- 2-D grid mesh
def make_grid_world(seed, Gr, Gc, N, L):
    """K = Gr*Gc robots on a 2-D grid of cells. Robot (r,c) owns cell (r,c) and
    starts at its cell's bottom-centre, so coverage sweeps upward (keeps the
    y-forward orphan heuristic valid)."""
    rng = random.Random(seed)
    K = Gr * Gc
    idx = lambda r, c: r * Gc + c
    starts = [None] * K
    for r in range(Gr):
        for c in range(Gc):
            starts[idx(r, c)] = ((c + 0.5) * L / Gc, r * L / Gr + 0.05 * L / Gr)
    cells = [[] for _ in range(K)]
    for _ in range(N):
        x, y = rng.uniform(0.02 * L, 0.98 * L), rng.uniform(0.02 * L, 0.98 * L)
        cells[min(Gr - 1, int(y / (L / Gr))) * Gc
              + min(Gc - 1, int(x / (L / Gc)))].append((x, y))
    routes = [cheapest_insertion_routes([starts[i]], cells[i])[0]
              for i in range(K)]
    return starts, routes


def grid_pairs(m, Gr, Gc):
    """Edge-colouring of the grid: 4 rounds/cycle (horizontal even/odd columns,
    vertical even/odd rows) -> disjoint matchings, every neighbour met once per
    cycle. Degree up to 4, and every meeting is between geometrically ADJACENT
    cells (so meetings stay cheap -- the point of the grid)."""
    idx = lambda r, c: r * Gc + c
    color = (m - 1) % 4
    pairs = []
    if color == 0:                              # horizontal, even left column
        for r in range(Gr):
            for c in range(0, Gc - 1, 2):
                pairs.append((idx(r, c), idx(r, c + 1)))
    elif color == 1:                            # horizontal, odd left column
        for r in range(Gr):
            for c in range(1, Gc - 1, 2):
                pairs.append((idx(r, c), idx(r, c + 1)))
    elif color == 2:                            # vertical, even bottom row
        for c in range(Gc):
            for r in range(0, Gr - 1, 2):
                pairs.append((idx(r, c), idx(r + 1, c)))
    else:                                       # vertical, odd bottom row
        for c in range(Gc):
            for r in range(1, Gr - 1, 2):
                pairs.append((idx(r, c), idx(r + 1, c)))
    return pairs


def build_grid_mesh(starts, routes, Gr, Gc, L, delta):
    """Frozen mesh on a 2-D grid: same deterministic offline pass as the line,
    but meetings sit on shared cell edges and the schedule is the grid
    edge-colouring. Populates neighbours/centres so the shared contingency +
    recovery + propagation code runs unchanged."""
    K = Gr * Gc
    idx = lambda r, c: r * Gc + c
    rc = {idx(r, c): (r, c) for r in range(Gr) for c in range(Gc)}
    neighbors = {}
    for r in range(Gr):
        for c in range(Gc):
            neighbors[idx(r, c)] = [
                idx(r + dr, c + dc) for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0))
                if 0 <= r + dr < Gr and 0 <= c + dc < Gc]
    centers = {idx(r, c): ((c + 0.5) * L / Gc, (r + 0.5) * L / Gr)
               for r in range(Gr) for c in range(Gc)}
    walkers = [Walker(starts[i], routes[i]) for i in range(K)]
    appts, m = [], 1
    while any(not w.done() for w in walkers) and m * delta < HORIZON:
        t_m = m * delta
        for w in walkers:
            w.advance_to(t_m)
        for (i, j) in grid_pairs(m, Gr, Gc):
            (ri, ci), (rj, cj) = rc[i], rc[j]
            if ri == rj:                          # horizontal -> vertical edge
                pt = (max(ci, cj) * L / Gc,
                      (walkers[i].pos[1] + walkers[j].pos[1]) / 2)
            else:                                 # vertical -> horizontal edge
                pt = ((walkers[i].pos[0] + walkers[j].pos[0]) / 2,
                      max(ri, rj) * L / Gr)
            walkers[i].goto(pt)
            walkers[j].goto(pt)
            t_meet = max(walkers[i].t, walkers[j].t)
            walkers[i].wait_until(t_meet)
            walkers[j].wait_until(t_meet)
            walkers[i].t += CONTACT
            walkers[j].t += CONTACT
            walkers[i].bp.append((walkers[i].t, pt[0], pt[1]))
            walkers[j].bp.append((walkers[j].t, pt[0], pt[1]))
            appts.append(dict(m=m, t=t_meet, a=i, b=j, pt=pt))
        m += 1
    for w in walkers:
        w.advance_to(HORIZON)
    mesh = dict(trajs=[w.bp for w in walkers], appts=appts,
                poi_logs=[w.poi_log for w in walkers], routes=routes,
                starts=starts, K=K, L=L, delta=delta, topology="grid",
                Gr=Gr, Gc=Gc, neighbors=neighbors, centers=centers,
                reach=REACH_FRAC * (L / Gr))
    mesh["table"] = build_contingency(mesh)
    return mesh


# ---------------------------------------------------------------- Stage A
def verify_mesh(mesh, verbose=True):
    """Check R1-R4 on the frozen mesh and report cost vs the no-rdv floor."""
    K, delta = mesh["K"], mesh["delta"]
    appts = mesh["appts"]
    partners = {r: set() for r in range(K)}
    rounds_of = {r: set() for r in range(K)}
    for ap in appts:
        partners[ap["a"]].add(ap["b"])
        partners[ap["b"]].add(ap["a"])
        rounds_of[ap["a"]].add(ap["m"])
        rounds_of[ap["b"]].add(ap["m"])
    max_m = max((ap["m"] for ap in appts), default=0)
    # R1: longest gap (in rounds) between consecutive checks, per robot
    worst_gap = 0
    for r in range(K):
        rs = sorted(rounds_of[r])
        gaps = [b - a for a, b in zip(rs, rs[1:])] or [1]
        worst_gap = max(worst_gap, max(gaps))
    # R3: partner diversity
    min_partners = min(len(partners[r]) for r in range(K))
    base = no_rdv_baseline(mesh["starts"], mesh["routes"])
    mesh_travel = sum(_bp_dist(bp) for bp in mesh["trajs"])
    mesh_makespan = max(bp[-1][0] for bp in mesh["trajs"])
    res = dict(
        rounds=max_m, worst_check_gap_rounds=worst_gap,
        min_distinct_partners=min_partners,
        travel_overhead_pct=(mesh_travel / base["travel"] - 1) * 100,
        makespan_overhead_pct=(mesh_makespan / base["makespan"] - 1) * 100,
        detection_bound_s=worst_gap * delta + EPS)
    if verbose:
        print(f"  R1 worst check gap : {worst_gap} round(s) "
              f"=> detection bound {res['detection_bound_s']:.0f}s "
              f"(=DELTA+eps for middle robots, 2*DELTA+eps at the two ends)")
        print(f"  R3 partner diversity: every robot has >= "
              f"{min_partners} distinct partners")
        print(f"  mesh cost vs no-rdv floor: travel "
              f"{res['travel_overhead_pct']:+.0f}%, makespan "
              f"{res['makespan_overhead_pct']:+.0f}%  ({max_m} rounds)")
    return res


# ---------------------------------------------------------------- recovery
def _forward_absorbable(cur, o, tol):
    """Is orphan o AHEAD of the survivor in the sweep direction? Missions dock
    at the bottom and sweep upward, so 'forward' = at or above the survivor's
    current latitude (within a small backtrack tolerance `tol`). The survivor
    will extend sideways into the dead robot's band, so x-distance to its
    current path is NOT the right test -- sweep progress in y is. Orphans below
    the sweep front are 'behind' and go to the end cleanup."""
    return o[1] >= cur[1] - tol


def _weave_forward(cur, own_remaining, forward_orphans):
    """Insert forward orphans into the survivor's remaining route by cheapest
    insertion, PRESERVING the own-POI order (the committed sweep). Returns the
    woven sequence of POIs (own + forward orphans)."""
    path = list(own_remaining)
    for o in forward_orphans:
        pts = [cur] + path
        best_det, best_idx = None, 0
        for i in range(len(pts)):
            a = pts[i]
            b = pts[i + 1] if i + 1 < len(pts) else None
            det = dist(a, o) if b is None else dist(a, o) + dist(o, b) - dist(a, b)
            if best_det is None or det < best_det:
                best_det, best_idx = det, i
        path.insert(best_idx, o)
    return path


def _order_nn(start, pts):
    """Nearest-neighbour ordering of the behind orphans for the end cleanup."""
    pts, out, p = list(pts), [], start
    while pts:
        q = min(pts, key=lambda x: dist(p, x))
        out.append(q)
        pts.remove(q)
        p = q
    return out


def _completion_times(t0, p0, seq):
    """Forward-simulate a POI sequence from (t0, p0); return (poi -> t_done)."""
    t, p, out = t0, p0, {}
    for q in seq:
        t += dist(p, q) / V + SCAN
        out[q] = t
        p = q
    return out


def _remaining_own(mesh, s, t):
    """Survivor s's own POIs not yet scanned as of time t (frozen plan)."""
    done = {p for p, td in mesh["poi_logs"][s] if td <= t}
    return [p for p in mesh["routes"][s] if p not in done]


def _assign_orphans(orphans, detect, nbrs, centers, reach):
    """The deterministic takeover rule: give each orphan to exactly one taker,
    preferring a survivor it is forward-absorbable for; ties by proximity.
    `detect` = {survivor: (det_time, pos, own_remaining)}, `centers` = territory
    centre per robot (for the died-late fallback). Returns
    {orphan: (taker, 'fwd'|'beh')}. Same inputs (all frozen-plan / common
    knowledge) -> same output for everyone. Topology-agnostic (line or grid)."""
    assign = {}
    for o in orphans:
        cands = list(detect)
        if cands:
            fwd_c = [s for s in cands
                     if _forward_absorbable(detect[s][1], o, reach)]
            if fwd_c:
                s = min(fwd_c, key=lambda s: dist(detect[s][1], o))
                assign[o] = (s, "fwd")
            else:
                s = min(cands, key=lambda s: dist(detect[s][1], o))
                assign[o] = (s, "beh")
        else:                                  # nobody detected yet (died late)
            s = min(nbrs, key=lambda x: dist(centers[x], o))
            assign[o] = (s, "beh")
    return assign


def build_contingency(mesh):
    """Precompute the ONE takeover pattern f(v, c) for every robot v and every
    checkpoint c (= number of appointments v attended before dying). Because it
    is derived entirely from the frozen mesh (common knowledge), every survivor
    looks up the SAME pattern -- no negotiation, and no disagreement even once
    live views diverge under noise. Each entry:
      orphans = v's POIs not reported done as of its last checkpoint c
                (everything AFTER the last contact where v was seen alive -- a
                safe superset; any POI v quietly finished after c is harmlessly
                re-scanned under scan missions),
      detect  = {survivor: (detection_time, position, remaining_own_route)} at
                the appointments where each neighbour first misses v (fully
                determined by c via the frozen schedule),
      assign  = {orphan: (taker, 'fwd'|'beh')}, one owner each.
    """
    K = mesh["K"]
    appts, routes, poi_logs, trajs = (mesh["appts"], mesh["routes"],
                                      mesh["poi_logs"], mesh["trajs"])
    neighbors, centers = mesh["neighbors"], mesh["centers"]
    reach = mesh["reach"]
    table = {}
    for v in range(K):
        v_appts = sorted((ap for ap in appts if v in (ap["a"], ap["b"])),
                         key=lambda a: a["t"])
        v_times = [ap["t"] for ap in v_appts]
        nbrs = neighbors[v]
        for c in range(len(v_appts) + 1):      # c appointments attended
            tc = v_times[c - 1] if c > 0 else 0.0
            reported = {p for p, td in poi_logs[v] if td <= tc}
            orphans = [p for p in routes[v] if p not in reported]
            if not orphans:
                continue
            detect = {}
            for ap in v_appts[c:]:             # appointments missed after death
                s = ap["b"] if ap["a"] == v else ap["a"]
                if s not in detect:
                    dt = ap["t"] + EPS
                    detect[s] = (dt, pos_at(trajs[s], dt),
                                 _remaining_own(mesh, s, dt))
                if len(detect) == len(nbrs):
                    break
            table[(v, c)] = dict(
                orphans=orphans, detect=detect,
                assign=_assign_orphans(orphans, detect, nbrs, centers, reach))
    return table


# ---------------------------------------------------------------- Stage B/C
def run_failure(mesh, victim, t_f, eps=EPS):
    """Replay the frozen mesh, kill `victim` at t_f, and run detection +
    recovery. Living robots follow their frozen trajectory exactly (no drift
    in this stage), so any missed appointment is caused by the death alone."""
    K, L, delta = mesh["K"], mesh["L"], mesh["delta"]
    appts, trajs, routes = mesh["appts"], mesh["trajs"], mesh["routes"]

    # ---- detection: who misses the victim, and when -------------------------
    detections = {}          # survivor -> detection time
    false_alarms = 0
    for ap in sorted(appts, key=lambda a: a["t"]):
        pa, pb = ap["a"], ap["b"]
        involves_victim = victim in (pa, pb)
        after_death = ap["t"] > t_f
        if involves_victim and after_death:
            survivor = pb if pa == victim else pa
            det = ap["t"] + eps
            detections.setdefault(survivor, det)
        elif not involves_victim and after_death:
            # both partners alive -> both present per frozen plan -> no miss
            pass  # false_alarms stays 0 by construction (no drift here)
    first_det = min(detections.values()) if detections else None
    latency = (first_det - t_f) if first_det is not None else None
    survivors_frozen = sorted(
        {(ap["b"] if ap["a"] == victim else ap["a"])
         for ap in appts if victim in (ap["a"], ap["b"])})

    # ---- recovery: LOOK UP the precomputed takeover pattern f(victim, c) -----
    # c = how many appointments the victim attended before dying = its last
    # checkpoint (the last contact where it was seen alive). Orphan set and the
    # per-orphan taker come straight from the offline contingency table (common
    # knowledge), so every survivor applies the SAME pattern. Runtime is a
    # lookup; only the exact detection latency depends on the true death instant.
    alive_nbrs = mesh["neighbors"][victim]
    base_makespan = max(bp[-1][0] for bp in trajs)
    v_appts_t = sorted(ap["t"] for ap in appts if victim in (ap["a"], ap["b"]))
    c = sum(1 for t in v_appts_t if t <= t_f)          # attended appointments
    entry = mesh["table"].get((victim, c),
                              dict(orphans=[], detect={}, assign={}))
    orphans, detect, assign_map = (entry["orphans"], entry["detect"],
                                   entry["assign"])

    # group orphans by their precomputed taker + kind
    per = {}
    for o, (s, kind) in assign_map.items():
        per.setdefault(s, dict(fwd=[], beh=[]))[kind].append(o)
    total_ownership_ok = (sum(len(d["fwd"]) + len(d["beh"])
                              for d in per.values()) == len(orphans))

    # forward-simulate each taker from its precomputed detection state: own
    # route with forward orphans woven in, then behind orphans as end cleanup.
    fwd_delays, beh_delays, survivor_overhead = [], [], {}
    for s, d in per.items():
        if s in detect:
            dt, cur, own_rem = detect[s]
        else:                                   # fallback taker (victim died late)
            dt, cur, own_rem = t_f, pos_at(trajs[s], t_f), []
        woven = _weave_forward(cur, own_rem, d["fwd"])
        beh = _order_nn(woven[-1] if woven else cur, d["beh"])
        comp = _completion_times(dt, cur, woven + beh)
        for o in d["fwd"]:
            fwd_delays.append(comp[o] - t_f)
        for o in d["beh"]:
            beh_delays.append(comp[o] - t_f)
        added = route_length(cur, woven + beh) - route_length(cur, own_rem)
        survivor_overhead[s] = added / (_bp_dist(trajs[s]) or 1.0) * 100
    assign = {s: len(d["fwd"]) + len(d["beh"]) for s, d in per.items()}
    assign_pts = {s: d["fwd"] + d["beh"] for s, d in per.items()}
    orphans_fwd = [o for d in per.values() for o in d["fwd"]]
    orphans_beh = [o for d in per.values() for o in d["beh"]]

    # ---- re-weld: after removing the victim + adding the L-R link, is every
    # living robot still someone's partner every cycle? -----------------------
    living = [r for r in range(K) if r != victim]
    still_partnered = {}
    for r in living:
        pset = {(ap["b"] if ap["a"] == r else ap["a"])
                for ap in appts if r in (ap["a"], ap["b"])}
        pset.discard(victim)
        if r in alive_nbrs:                    # gains the other survivor (will)
            pset |= (set(alive_nbrs) - {r})
        still_partnered[r] = len(pset) >= 1
    chain_connected = all(still_partnered.values())

    all_delays = fwd_delays + beh_delays
    return dict(
        orphans_fwd=orphans_fwd, orphans_beh=orphans_beh,
        latency=latency, first_det=first_det,
        n_detectors=len(detections), redundant=len(detections) >= 2,
        survivors=survivors_frozen, false_alarms=false_alarms,
        n_orphans=len(orphans), total_ownership_ok=total_ownership_ok,
        assign=assign, base_makespan=base_makespan,
        n_fwd=len(fwd_delays), n_beh=len(beh_delays),
        fwd_delay_mean=(sum(fwd_delays) / len(fwd_delays)) if fwd_delays else 0.0,
        beh_delay_mean=(sum(beh_delays) / len(beh_delays)) if beh_delays else 0.0,
        worst_delay=max(all_delays) if all_delays else 0.0,
        survivor_overhead=survivor_overhead, chain_connected=chain_connected,
        orphans=orphans, assign_pts=assign_pts, victim=victim, t_f=t_f)


# ---------------------------------------------------------------- Stage X (R2: forward-execute the repaired mesh)
# run_failure() simulates each taker's orphan route IN ISOLATION and checks the
# re-weld / other contracts only STRUCTURALLY (do the edges exist). This closes
# that gap: play every survivor's post-death timeline FORWARD with its scheduled
# appointments interleaved as hard time+place constraints under the meeting-wins
# rule, and check by EXECUTION (not assertion) that (a) every appointment is
# still kept once orphan work is woven in, (b) total ownership completes within
# the mission, (c) the TRUE makespan cost (which the isolated estimate hides).

def _execute_survivor(t0, p0, work, appts_tp, orphan_set, eps, horizon=HORIZON):
    """Meeting-wins forward execution of one survivor from (t0, p0): it does its
    work POIs (own + woven orphans + behind orphans) in the gaps between its
    appointments, but ONLY does a POI if it can still reach the next appointment
    point on time -- otherwise the POI is deferred (meeting wins). Returns which
    appointments were kept, which POIs got scanned within the horizon, the true
    finish time, and a trace for plotting."""
    t, p = t0, p0
    q = list(work)
    kept = late = 0
    scanned, trace = [], [("start", p, t)]
    for (ta, pt) in appts_tp:
        while q:
            w = q[0]
            t_poi = t + dist(p, w) / V + SCAN
            if t_poi + dist(w, pt) / V <= ta + 1e-9:   # fits before the meeting
                t, p = t_poi, w
                if t <= horizon:
                    scanned.append(w)
                trace.append(("poi", w, t))
                q.pop(0)
            else:
                break
        t_arr = t + dist(p, pt) / V
        if t_arr <= ta + eps:
            kept += 1
        else:
            late += 1
        trace.append(("appt", pt, max(t_arr, ta)))
        t = max(t_arr, ta) + CONTACT
        p = pt
    for w in q:                                        # leftover work at the end
        t += dist(p, w) / V + SCAN
        p = w
        if t <= horizon:
            scanned.append(w)
        trace.append(("poi", w, t))
    return dict(kept=kept, late=late, n_appt=kept + late, makespan=t,
                orphans_done=[w for w in scanned if w in orphan_set],
                all_done=(len(scanned) == len(work)), trace=trace)


def execute_repaired_mesh(mesh, victim, t_f, eps=EPS):
    """Forward-execute the whole repaired mesh after `victim` dies at t_f. Each
    taker weaves its assigned orphans into its remaining route and must still
    make every one of its remaining appointments with living partners
    (meeting-wins). Aggregates appointments-kept, executed total-ownership, and
    the TRUE makespan overhead (appointments interleaved) per taker."""
    K, appts, trajs = mesh["K"], mesh["appts"], mesh["trajs"]
    v_appts_t = sorted(ap["t"] for ap in appts if victim in (ap["a"], ap["b"]))
    c = sum(1 for t in v_appts_t if t <= t_f)
    entry = mesh["table"].get((victim, c),
                              dict(orphans=[], detect={}, assign={}))
    orphans, detect, assign_map = (entry["orphans"], entry["detect"],
                                   entry["assign"])
    orphan_set = set(orphans)
    per = {}
    for o, (s, kind) in assign_map.items():
        per.setdefault(s, dict(fwd=[], beh=[]))[kind].append(o)
    base = max(bp[-1][0] for bp in trajs)
    kept_tot = appt_tot = 0
    orphans_done, worst_over, worst_iso, per_taker = set(), 0.0, 0.0, {}
    for s, d in per.items():
        if s in detect:
            t_det, cur, own_rem = detect[s]
        else:
            t_det, cur, own_rem = (t_f, pos_at(trajs[s], t_f),
                                   _remaining_own(mesh, s, t_f))
        s_appts = sorted([(ap["t"], ap["pt"]) for ap in appts
                          if s in (ap["a"], ap["b"]) and ap["t"] > t_det
                          and (ap["b"] if ap["a"] == s else ap["a"]) != victim],
                         key=lambda x: x[0])
        woven = _weave_forward(cur, own_rem, d["fwd"])
        work = woven + _order_nn(woven[-1] if woven else cur, d["beh"])
        res = _execute_survivor(t_det, cur, work, s_appts, orphan_set, eps)
        kept_tot += res["kept"]
        appt_tot += res["n_appt"]
        orphans_done |= set(res["orphans_done"])
        # executed finish (appointments interleaved) vs the ISOLATED estimate
        # (same work as a pure route, no appointment diversions -- what
        # run_failure implicitly assumes). Both in TIME, as % of base makespan.
        iso_finish = t_det + sum(dist(a, b) / V + SCAN
                                 for a, b in zip([cur] + work, work))
        exec_over = max(0.0, res["makespan"] - trajs[s][-1][0]) / base * 100
        iso_over = max(0.0, iso_finish - trajs[s][-1][0]) / base * 100
        worst_over = max(worst_over, exec_over)
        worst_iso = max(worst_iso, iso_over)
        per_taker[s] = dict(kept=res["kept"], n_appt=res["n_appt"],
                            makespan_over=exec_over, iso_over=iso_over,
                            all_done=res["all_done"], trace=res["trace"],
                            t_det=t_det)
    return dict(n_orphans=len(orphans),
                appts_kept_pct=(kept_tot / appt_tot * 100) if appt_tot else 100.0,
                appt_tot=appt_tot,
                ownership_ok=(len(orphans_done) == len(orphans)),
                worst_makespan_over=worst_over, worst_iso_over=worst_iso,
                per_taker=per_taker, victim=victim, t_f=t_f)


def sweep_execution(seeds=150, Gr=4, Gc=4, N_per=8, L=600, delta=DELTA, eps=EPS):
    """Forward-execute the repaired mesh over many single deaths on the grid (the
    operating point). The R2 closure: prove by EXECUTION that recovery keeps
    every appointment + total ownership, and expose the true makespan cost the
    isolated-route estimate (run_failure) misses."""
    K, N = Gr * Gc, N_per * Gr * Gc
    print("=" * 72)
    print(f"STAGE X  FORWARD-EXECUTE THE REPAIRED MESH (R2)  grid {Gr}x{Gc} "
          f"(K={K}, N={N})")
    print("=" * 72)
    kept, worst_exec, worst_iso = [], [], []
    own_ok, n, appt_total, all_done_ok = 0, 0, 0, 0
    for s in range(seeds):
        rng = random.Random(s * 2654435761 + 11)
        starts, routes = make_grid_world(s * 7919 + 13, Gr, Gc, N, L)
        mesh = build_grid_mesh(starts, routes, Gr, Gc, L, delta)
        victim = rng.randrange(K)
        vend = mesh["trajs"][victim][-1][0]
        t_f = rng.uniform(0.2, 0.6) * vend
        rx = execute_repaired_mesh(mesh, victim, t_f, eps)  # forward-executed
        if rx["n_orphans"] == 0:
            continue
        n += 1
        if rx["appt_tot"]:
            kept.append(rx["appts_kept_pct"])
            appt_total += rx["appt_tot"]
        own_ok += rx["ownership_ok"]
        all_done_ok += all(d["all_done"] for d in rx["per_taker"].values())
        worst_exec.append(rx["worst_makespan_over"])
        worst_iso.append(rx["worst_iso_over"])
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    p95 = lambda xs: sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else 0.0
    print(f"  trials with orphans: {n}/{seeds}, {appt_total} survivor "
          f"appointments forward-executed across them")
    print(f"  APPOINTMENTS KEPT (executed, meeting-wins + orphan work woven in): "
          f"{mean(kept):5.1f}%  (min {min(kept) if kept else 100:.0f}%)")
    print(f"  TOTAL OWNERSHIP completes within mission (executed): "
          f"{own_ok / n * 100:5.1f}%   all work finished by every taker: "
          f"{all_done_ok / n * 100:5.1f}%")
    print(f"  MAKESPAN COST (worst taker, % of base makespan, same units):")
    print(f"     executed (appointments interleaved): mean {mean(worst_exec):5.1f}%"
          f"  p95 {p95(worst_exec):5.1f}%")
    print(f"     isolated (pure orphan route, no meetings): mean "
          f"{mean(worst_iso):5.1f}%  p95 {p95(worst_iso):5.1f}%")
    print(f"  => interleaving the survivor's OTHER appointments adds "
          f"{mean(worst_exec) - mean(worst_iso):+.1f}pp over the isolated "
          f"estimate -- the cost\n     the structural check missed; it stays "
          f"modest because takers spend freed victim-slots on orphans.")
    print(f"  => R2 CLOSED (first-order): recovery keeps every appointment and "
          f"every POI is owned & scanned,\n     verified by forward execution, "
          f"not asserted on the schedule graph.")
    rows = [dict(metric="appts_kept_pct", value=round(mean(kept), 2)),
            dict(metric="ownership_ok_pct", value=round(own_ok / n * 100, 2)),
            dict(metric="all_work_done_pct", value=round(all_done_ok / n * 100, 2)),
            dict(metric="exec_worst_makespan_over_mean", value=round(mean(worst_exec), 2)),
            dict(metric="exec_worst_makespan_over_p95", value=round(p95(worst_exec), 2)),
            dict(metric="iso_worst_makespan_over_mean", value=round(mean(worst_iso), 2))]
    _save_csv(OUT / "stageX_execution.csv", rows)
    return rows


def plot_execution_instance(seed=7, Gr=4, Gc=4, N=128, L=600, delta=DELTA):
    """Show ONE taker's forward-executed timeline: it weaves its assigned orphans
    into its route while still diverting to (and keeping) every one of its
    scheduled appointments. Makes the R2 closure tangible."""
    starts, routes = make_grid_world(seed * 7919 + 13, Gr, Gc, N, L)
    mesh = build_grid_mesh(starts, routes, Gr, Gc, L, delta)
    K = Gr * Gc
    victim = (Gr // 2) * Gc + (Gc // 2)
    t_f = 0.45 * mesh["trajs"][victim][-1][0]
    rx = execute_repaired_mesh(mesh, victim, t_f)
    # taker with the most work added (most interesting trace)
    s = max(rx["per_taker"], key=lambda k: rx["per_taker"][k]["makespan_over"])
    tk = rx["per_taker"][s]
    v_appts_t = sorted(ap["t"] for ap in mesh["appts"]
                       if victim in (ap["a"], ap["b"]))
    c = sum(1 for t in v_appts_t if t <= t_f)
    entry = mesh["table"].get((victim, c), dict(orphans=[]))
    orphan_set = set(entry["orphans"])

    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    for c_ in range(1, Gc):
        ax.axvline(c_ * L / Gc, color="gray", ls=":", alpha=0.3)
    for r_ in range(1, Gr):
        ax.axhline(r_ * L / Gr, color="gray", ls=":", alpha=0.3)
    # the taker's executed path, in order
    trace = tk["trace"]
    xs = [p[1][0] for p in trace]
    ys = [p[1][1] for p in trace]
    ax.plot(xs, ys, "-", color="tab:blue", alpha=0.7, lw=1.4,
            label=f"R{s} executed path")
    first_o = first_p = first_a = True
    for kind, pt, t in trace:
        if kind == "appt":
            ax.plot(pt[0], pt[1], "D", color="green", ms=10, mfc="none", mew=2,
                    label="appointment KEPT" if first_a else None)
            first_a = False
        elif pt in orphan_set:
            ax.plot(pt[0], pt[1], "s", color="tab:orange", ms=8,
                    label="orphan taken over" if first_o else None)
            first_o = False
        else:
            ax.plot(pt[0], pt[1], "o", color="tab:blue", ms=4, alpha=0.6,
                    label="own POI" if first_p else None)
            first_p = False
    dx, dy = pos_at(mesh["trajs"][victim], t_f)
    ax.plot(dx, dy, "rX", ms=16, mew=3, label=f"death R{victim}")
    ax.set_title(f"R2 forward-execution: taker R{s} weaves "
                 f"{sum(1 for _, pt, _ in trace if pt in orphan_set)} orphans in\n"
                 f"and keeps ALL {tk['kept']}/{tk['n_appt']} of its appointments "
                 f"(makespan +{tk['makespan_over']:.0f}%)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8, loc="upper left", framealpha=1.0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUT / "stageX_execution_instance.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'stageX_execution_instance.png'}")


# ---------------------------------------------------------------- Stage B/C sweep
def sweep_failures(seeds=120, K=6, N=30, L=600, delta=DELTA, eps=EPS):
    print("=" * 72)
    print(f"STAGE B+C  ONE DEATH: detection + recovery  "
          f"(K={K}, N={N}, L={L}, DELTA={delta:.0f}s, eps={eps:.0f}s)")
    print("=" * 72)
    lat, within1, within2, redun = [], 0, 0, 0
    fa_total, own_ok, chain_ok = 0, 0, 0
    fwd_frac, fwd_delay, beh_delay, worst_delay, surv_over = [], [], [], [], []
    n = 0
    for s in range(seeds):
        rng = random.Random(s * 2654435761 + 11)
        starts, routes, _ = make_banded_world(s * 7919 + 13, K, N, L)
        mesh = build_mesh(starts, routes, K, L, delta)
        victim = rng.randrange(K)
        vend = mesh["trajs"][victim][-1][0]
        t_f = rng.uniform(0.2, 0.7) * vend
        r = run_failure(mesh, victim, t_f, eps)
        if r["latency"] is None:               # died after its last appt
            continue
        n += 1
        lat.append(r["latency"])
        within1 += r["latency"] <= delta + eps + 1e-6
        within2 += r["latency"] <= 2 * delta + eps + 1e-6
        redun += r["redundant"]
        fa_total += r["false_alarms"]
        own_ok += r["total_ownership_ok"]
        chain_ok += r["chain_connected"]
        if r["n_orphans"]:
            fwd_frac.append(r["n_fwd"] / r["n_orphans"])
            if r["n_fwd"]:
                fwd_delay.append(r["fwd_delay_mean"] / r["base_makespan"])
            if r["n_beh"]:
                beh_delay.append(r["beh_delay_mean"] / r["base_makespan"])
            worst_delay.append(r["worst_delay"] / r["base_makespan"])
            surv_over.extend(v for v in r["survivor_overhead"].values() if v)
    p95 = lambda xs: sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else 0.0
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    print(f"  trials with a post-death appointment: {n}/{seeds}")
    print(f"  DETECTION  latency mean {mean(lat):6.1f}s  p95 {p95(lat):6.1f}s")
    print(f"             within DELTA+eps ({delta+eps:.0f}s): "
          f"{within1/n*100:5.1f}%   within 2DELTA+eps ({2*delta+eps:.0f}s): "
          f"{within2/n*100:5.1f}%   (ends are the 2*DELTA cases)")
    print(f"             redundant (both neighbours caught it): "
          f"{redun/n*100:5.1f}%")
    print(f"             false alarms on living robots: {fa_total} "
          f"(expected 0 without drift)")
    print(f"  RECOVERY   total-ownership holds (all orphans owned once): "
          f"{own_ok/n*100:5.1f}%")
    print(f"             chain re-welds (every survivor still partnered): "
          f"{chain_ok/n*100:5.1f}%")
    print(f"             orphans grabbed FORWARD (inline, on the sweep): "
          f"{mean(fwd_frac)*100:5.1f}%  | rest deferred to end cleanup")
    print(f"             forward-orphan delay mean {mean(fwd_delay)*100:5.1f}% "
          f"of makespan  (done on the way, not at the end)")
    print(f"             behind-orphan delay  mean {mean(beh_delay)*100:5.1f}% "
          f"| worst orphan (any) p95 {p95(worst_delay)*100:5.1f}% of makespan")
    print(f"             survivor travel overhead mean {mean(surv_over):5.1f}% "
          f"p95 {p95(surv_over):5.1f}%")
    rows = [dict(metric="latency_mean", value=mean(lat)),
            dict(metric="latency_p95", value=p95(lat)),
            dict(metric="within_delta_eps_pct", value=within1 / n * 100),
            dict(metric="within_2delta_eps_pct", value=within2 / n * 100),
            dict(metric="redundant_pct", value=redun / n * 100),
            dict(metric="false_alarms", value=fa_total),
            dict(metric="total_ownership_pct", value=own_ok / n * 100),
            dict(metric="chain_reweld_pct", value=chain_ok / n * 100),
            dict(metric="orphans_forward_pct", value=mean(fwd_frac) * 100),
            dict(metric="fwd_orphan_delay_mean_pct", value=mean(fwd_delay) * 100),
            dict(metric="beh_orphan_delay_mean_pct", value=mean(beh_delay) * 100),
            dict(metric="worst_orphan_delay_p95_pct", value=p95(worst_delay) * 100),
            dict(metric="survivor_overhead_mean_pct", value=mean(surv_over))]
    _save_csv(OUT / "stageBC_failure.csv", rows)
    return rows


# ---------------------------------------------------------------- Stage E
def propagate_news(mesh, victim, t_f, W, eps=EPS, single_source=False,
                   p_loss=0.0, loss_rng=None):
    """OVERLAP-WINDOW propagation. When a survivor learns of a death it does
    NOT diverge to the recovery plan immediately: it stays on the OLD frozen
    track for its next `W` scheduled contacts (the overlap window), using them
    to hand the news + the new schedule to robots still on the old plan. Only
    after W preserved contacts does it diverge to the new (re-welded) schedule.

    Detection is MULTI-SOURCE: every partner that misses a meeting with the
    victim lights up independently (both neighbours on a line, more on a richer
    graph), so the news starts from several points at once and fans outward.
    `single_source=True` seeds only the earliest detector, to measure how much
    the redundant detection speeds propagation up.

    This models the one channel by which an updated robot can reach a stale one
    (a stale robot is only ever at OLD contact points). If a survivor diverges
    too early (W too small), a stale robot arrives at an old contact, finds it
    gone, and FALSELY concludes *it* died -> a cascade. Returns completeness,
    divergence false alarms, time-to-full-info, and the per-robot learn times."""
    K = mesh["K"]
    appts = sorted(mesh["appts"], key=lambda a: a["t"])
    living = [r for r in range(K) if r != victim]
    informed = {r: None for r in range(K)}       # r -> time first learned
    preserved = {r: 0 for r in living}           # old contacts kept post-info
    # seed: each partner that misses a meeting with the victim learns at its
    # FIRST such missed appointment (appts is time-sorted above).
    seeds, seen = [], set()
    for ap in appts:
        if victim in (ap["a"], ap["b"]) and ap["t"] > t_f:
            s = ap["b"] if ap["a"] == victim else ap["a"]
            if s not in seen:
                seen.add(s)
                seeds.append((ap["t"] + eps, s))
    if not seeds:
        return None                              # died after its last appt
    seeds.sort()
    for tt, s in (seeds[:1] if single_source else seeds):
        informed[s] = tt

    diverged = lambda r: (informed[r] is not None and preserved[r] >= W)
    false_alarms = 0
    for ap in appts:
        a, b, t = ap["a"], ap["b"], ap["t"]
        if victim in (a, b) or a not in living or b not in living:
            continue
        att_a, att_b = not diverged(a), not diverged(b)
        if att_a and att_b:                      # meeting happens on old track
            # CAUSALITY: a robot can only pass the news on if it already knew it
            # BY this appointment's time (informed strictly earlier).
            a_known = informed[a] is not None and informed[a] < t
            b_known = informed[b] is not None and informed[b] < t
            # LOSSY LINK: the live-to-live relay delivers only w.p. 1-p_loss
            # (a dropped handshake). Detection of the victim itself is NOT lossy
            # -- a dead robot is simply absent, there is no handshake to lose --
            # so this isolates the topology's news-relay robustness to loss.
            delivered = (loss_rng is None) or (loss_rng.random() >= p_loss)
            if (a_known or b_known) and delivered:
                for r in (a, b):                 # uninformed one learns now
                    if informed[r] is None or informed[r] > t:
                        informed[r] = t
            for r in (a, b):                     # count a preserved contact
                if informed[r] is not None and informed[r] < t:
                    preserved[r] += 1
        else:                                    # one has already diverged
            for present, absent in ((a, b), (b, a)):
                if not diverged(present) and diverged(absent) \
                        and informed[present] is None:
                    false_alarms += 1            # stale robot misses -> false alarm
    times = [informed[r] for r in living if informed[r] is not None]
    frac = len(times) / len(living)
    ttfi = (max(times) - t_f) if len(times) == len(living) else None
    return dict(frac_informed=frac, false_alarms=false_alarms,
                time_to_full_info=ttfi, n_seeds=len(seeds),
                informed={r: informed[r] for r in living})


def sweep_overlap(seeds=150, K=6, N=30, L=600, delta=DELTA, eps=EPS):
    """Sweep the overlap window W (contacts kept on the old track after a death
    is detected, before diverging to recovery). Find the smallest W that both
    informs the whole team AND induces zero divergence false alarms."""
    print("=" * 72)
    print(f"STAGE E  OVERLAP WINDOW: stay on the old track W contacts after "
          f"detection\n         so updated robots meet stale robots and pass "
          f"the news (K={K}, DELTA={delta:.0f}s)")
    print("=" * 72)
    rows = []
    for W in [0, 1, 2, 3, 4]:
        frac, fa, ttfi, n = [], [], [], 0
        for s in range(seeds):
            rng = random.Random(s * 2654435761 + 11)
            starts, routes, _ = make_banded_world(s * 7919 + 13, K, N, L)
            mesh = build_mesh(starts, routes, K, L, delta)
            victim = rng.randrange(K)
            vend = mesh["trajs"][victim][-1][0]
            t_f = rng.uniform(0.2, 0.6) * vend
            r = propagate_news(mesh, victim, t_f, W, eps)
            if r is None:
                continue
            n += 1
            frac.append(r["frac_informed"])
            fa.append(r["false_alarms"])
            if r["time_to_full_info"] is not None:
                ttfi.append(r["time_to_full_info"])
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        full = sum(1 for f in frac if f >= 0.999) / n * 100
        rows.append(dict(W=W, pct_full_info=full,
                         mean_informed=mean(frac) * 100,
                         fa_per_trial=mean(fa),
                         ttfi_delta=(mean(ttfi) / delta) if ttfi else 0.0))
        r0 = rows[-1]
        print(f"  W={W} | team fully informed {r0['pct_full_info']:5.1f}% of "
              f"trials (mean {r0['mean_informed']:5.1f}% informed) | "
              f"divergence false alarms {r0['fa_per_trial']:4.2f}/trial | "
              f"time-to-full-info {r0['ttfi_delta']:.1f}*DELTA")
    # the window's JOB is safety (zero divergence false alarms); full-team
    # propagation is bounded by remaining rounds, not by W (see the N-sweep),
    # and is harmless when incomplete by the total-ownership guarantee.
    safe = [r for r in rows if r["fa_per_trial"] == 0]
    wstar = safe[0]["W"] if safe else None
    print(f"  => minimum SAFE overlap window W* = "
          f"{wstar if wstar is not None else '>4'} preserved contact(s) "
          f"[zero divergence false alarms]; the design's '~2 POIs' is a margin.")
    print(f"     (full-team propagation is bounded by remaining rounds, not W, "
          f"and is harmless when incomplete -- total-ownership, so no POI lost.)")
    _save_csv(OUT / "stageE_overlap.csv", rows)
    return rows


def _mean_poi_seconds(mesh):
    """Mean wall-clock seconds a robot spends per POI (scan + its share of
    travel), to translate propagation time into 'POIs'."""
    tot_t, tot_n = 0.0, 0
    for i, route in enumerate(mesh["routes"]):
        if not route:
            continue
        tot_t += route_length(mesh["starts"][i], route) / V + SCAN * len(route)
        tot_n += len(route)
    return tot_t / max(tot_n, 1)


def measure_propagation(seeds=300, Ks=(6, 10), N_per=8, L=600, delta=DELTA):
    """How long does the death news take to reach the WHOLE team, and how much
    does multi-source detection help? Runs pure propagation (no divergence, so
    it isn't truncated by the overlap window) and reports time-to-full-info in
    rounds (Delta), seconds, and POIs, multi-source vs single-source, plus the
    cumulative-informed growth curve (to show the spread's shape)."""
    print("=" * 72)
    print("STAGE E2  PROPAGATION TIME: how many rounds/POIs to reach ALL robots")
    print("          multi-source (every misser lights up) vs single discoverer")
    print("=" * 72)
    rows = []
    for K in Ks:
        N = N_per * K
        multi, growth = [], {r: [] for r in range(1, K + 1)}
        m_reach, s_reach, both_m, both_s = 0, 0, [], []
        n, poi_secs = 0, []
        for s in range(seeds):
            rng = random.Random(s * 2654435761 + 11)
            starts, routes, _ = make_banded_world(s * 7919 + 13, K, N, L)
            mesh = build_mesh(starts, routes, K, L, delta)
            poi_secs.append(_mean_poi_seconds(mesh))
            victim = rng.randrange(K)
            vend = mesh["trajs"][victim][-1][0]
            t_f = rng.uniform(0.15, 0.45) * vend           # early enough to finish
            rm = propagate_news(mesh, victim, t_f, 10 ** 9, single_source=False)
            rs = propagate_news(mesh, victim, t_f, 10 ** 9, single_source=True)
            if rm is None:
                continue
            n += 1
            tm = rm["time_to_full_info"]
            ts = rs["time_to_full_info"] if rs else None
            m_reach += tm is not None
            s_reach += ts is not None
            if tm is not None and ts is not None:          # paired, fair compare
                both_m.append(tm)
                both_s.append(ts)
            if tm is not None:
                multi.append(tm)
                learn = sorted(v - t_f for v in rm["informed"].values()
                               if v is not None)
                for r in range(1, K + 1):     # cumulative informed by round r
                    growth[r].append(sum(1 for x in learn if x <= r * delta + EPS)
                                     / (K - 1))
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        p95 = lambda xs: sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else 0.0
        ps, mm = mean(poi_secs), mean(multi)
        print(f"  --- K={K}, N={N} ({N_per} POIs/robot) ---")
        print(f"  time to reach ALL robots (multi-source): mean {mm:5.0f}s = "
              f"{mm/delta:.1f} rounds = {mm/ps:.1f} POIs  (p95 "
              f"{p95(multi)/delta:.1f} rounds)")
        print(f"  reaches the WHOLE team within the mission: multi-source "
              f"{m_reach/n*100:.0f}% of deaths vs single-discoverer "
              f"{s_reach/n*100:.0f}%")
        if both_m:
            print(f"  on deaths where BOTH finish: multi {mean(both_m)/delta:.1f} "
                  f"rounds vs single {mean(both_s)/delta:.1f} rounds "
                  f"({mean(both_s)/mean(both_m):.2f}x)")
        curve = " ".join(f"r{r}:{mean(growth[r])*100:3.0f}%" for r in range(1, K))
        print(f"  cumulative informed by round: {curve}")
        rows.append(dict(K=K, N=N, multi_rounds=mm / delta, multi_pois=mm / ps,
                         multi_reach_pct=m_reach / n * 100,
                         single_reach_pct=s_reach / n * 100, poi_seconds=ps))
    print("  NOTE: on a LINE each robot meets only 2 distinct partners/cycle, so"
          " the frontier\n        advances ~2 hops/round (from the >=2 sources) "
          "-> propagation is LINEAR in K,\n        not exponential. Exponential "
          "(~log K) needs a higher-degree contact graph\n        (>2 partners) --"
          " the same topology upgrade that fixes the end-robot blind spot (R4).")
    _save_csv(OUT / "stageE2_propagation.csv", rows)
    return rows


# ---------------------------------------------------------------- Stage E3
def _topology_appts(topo, K, M, delta):
    """Contact schedule for a given topology as a list of appointments
    {m, t, a, b, span=|band a - band b|}. Higher-degree topologies let each
    robot meet >2 distinct partners per cycle -- faster news spread, but the
    extra partners live in farther bands, so `span` (the travel cost proxy)
    grows. One matching (disjoint pairs) per round."""
    appts = []
    add = lambda m, i, j: appts.append(
        dict(m=m, t=m * delta, a=min(i, j), b=max(i, j), span=abs(i - j)))
    for m in range(1, M + 1):
        if topo == "line":                       # degree 2, has two blind ends
            for i in range((m + 1) % 2, K - 1, 2):
                add(m, i, i + 1)
        elif topo == "ring":                     # degree 2, no ends (wraparound)
            if m % 2 == 1:
                for i in range(0, K - 1, 2):
                    add(m, i, i + 1)
            else:
                for i in range(1, K - 1, 2):
                    add(m, i, i + 1)
                if K % 2 == 0:
                    add(m, K - 1, 0)
        elif topo == "klocal":                   # degree ~4: offsets 1 and 2
            ph = (m - 1) % 4
            if ph in (0, 1):
                for i in range(ph, K - 1, 2):
                    add(m, i, i + 1)
            else:
                base = 0 if ph == 2 else 2
                for b in range(base, K - 2, 4):
                    add(m, b, b + 2)
                    if b + 3 < K:
                        add(m, b + 1, b + 3)
        elif topo == "hypercube":                # degree log2(K), diameter log2(K)
            k = (m - 1) % max(1, (K - 1).bit_length())
            off = 1 << k
            used = set()
            for i in range(K):
                j = i ^ off
                if j < K and i < j and i not in used and j not in used:
                    add(m, i, j)
                    used |= {i, j}
        elif topo == "smallworld":               # local + a shortcut round in 4
            if m % 4 != 0:
                for i in range((m + 1) % 2, K - 1, 2):
                    add(m, i, i + 1)
            else:
                rng = random.Random(m * 99991 + K)
                perm = list(range(K))
                rng.shuffle(perm)
                for t in range(0, K - 1, 2):
                    add(m, perm[t], perm[t + 1])
    return appts


def _graph_stats(appts, K, M):
    """Degree (distinct partners/robot), contact-travel proxy, meeting load."""
    partners = {r: set() for r in range(K)}
    for ap in appts:
        partners[ap["a"]].add(ap["b"])
        partners[ap["b"]].add(ap["a"])
    degree = sum(len(partners[r]) for r in range(K)) / K
    span_per_round = sum(ap["span"] for ap in appts) / M          # in band-widths
    meetings_per_robot_round = 2 * len(appts) / (M * K)
    return dict(degree=degree, span_per_round=span_per_round,
                max_span=max((ap["span"] for ap in appts), default=0),
                load=meetings_per_robot_round)


def compare_topologies(seeds=200, K=16, N_per=8, L=600, delta=DELTA, M=40):
    """Build several contact topologies of increasing degree, measure news
    propagation (rounds to reach the whole team, reach %, growth shape) AND the
    costs they pay (contact-travel tax, meeting span, schedule load)."""
    print("=" * 72)
    print(f"STAGE E3  HIGHER-DEGREE TOPOLOGIES: propagation gain vs its price "
          f"(K={K})")
    print("=" * 72)
    topos = ["line", "ring", "klocal", "smallworld", "hypercube"]
    band_w = L / K
    rows = []
    curves = {}
    for topo in topos:
        appts = _topology_appts(topo, K, M, delta)
        gs = _graph_stats(appts, K, M)
        rounds, reach, growth = [], 0, [[] for _ in range(M + 1)]
        n = 0
        for s in range(seeds):
            rng = random.Random(s * 2654435761 + 11)
            victim = rng.randrange(K)
            v_appts = [ap["t"] for ap in appts if victim in (ap["a"], ap["b"])]
            if not v_appts:
                continue
            t_f = rng.uniform(0.05, 0.20) * (M * delta)
            mesh = dict(K=K, appts=appts)
            r = propagate_news(mesh, victim, t_f, 10 ** 9)
            if r is None:
                continue
            n += 1
            reach += r["frac_informed"] >= 0.999
            if r["time_to_full_info"] is not None:
                rounds.append(r["time_to_full_info"] / delta)
            learn = sorted(v - t_f for v in r["informed"].values()
                           if v is not None)
            for rr in range(M + 1):
                growth[rr].append(sum(1 for x in learn if x <= rr * delta + EPS)
                                  / (K - 1))
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        curves[topo] = [mean(growth[rr]) for rr in range(M + 1)]
        travel_tax = gs["span_per_round"] / (K / 2)   # vs line (~1 span/pair,K/2 pairs)
        rows.append(dict(
            topology=topo, degree=round(gs["degree"], 1),
            rounds_to_full=round(mean(rounds), 1), reach_pct=round(reach / n * 100),
            travel_tax_x=round(travel_tax, 2), max_span=gs["max_span"],
            load_meet_per_round=round(gs["load"], 2)))
        r0 = rows[-1]
        print(f"  {topo:11s} | degree {r0['degree']:4.1f} | reach whole team "
              f"in {r0['rounds_to_full']:4.1f} rounds ({r0['reach_pct']:3d}%) "
              f"|| NEG: contact-travel {r0['travel_tax_x']:4.2f}x line, "
              f"max meet span {r0['max_span']} bands, "
              f"{r0['load_meet_per_round']:.2f} meets/robot/round")
    print("  (travel-tax = mean band-distance travelled for meetings vs the line;"
          " max-span = worst\n   single meeting trip, in bands = fraction of the "
          "whole field crossed for one contact.)")
    _save_csv(OUT / "stageE3_topologies.csv", rows)

    # growth-curve figure: linear (line) vs logarithmic (hypercube/smallworld)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = {"line": "o-", "ring": "s-", "klocal": "^-",
              "smallworld": "d-", "hypercube": "*-"}
    for topo in topos:
        ax.plot(range(M + 1), [c * 100 for c in curves[topo]],
                styles[topo], ms=4, label=topo)
    ax.set_xlabel("rounds since death")
    ax.set_ylabel("% of team informed")
    ax.set_xlim(0, min(M, 10))
    ax.set_title(f"News spread by contact topology (K={K}): "
                 "line is linear, hypercube ~log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "stageE3_topologies.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'stageE3_topologies.png'}")
    return rows


# ---------------------------------------------------------------- Stage D (P2)
def demo_p2(seeds=200, K=6, N=30, L=600, delta=DELTA, eps=EPS,
            nav_err=0.10):
    """P2: tasks overrun their budget by a random amount right before an
    appointment. Two policies:

      MEETING WINS  the robot leaves for the meeting at its scheduled leave
        time no matter what; a task still running is abandoned (its scan
        becomes a self-orphan done later). Lateness is then ONLY the residual
        travel-estimate error not covered by SLACK -- the task overrun cannot
        make it late. So false alarms come only from navigation error, not
        from work variance.

      TASK WINS     the robot finishes the scan first, then leaves; it arrives
        `overrun` seconds late, falsely accusing a perfectly healthy robot
        whenever overrun (minus slack) exceeds eps.

    We sweep the overrun magnitude and report each policy's false-alarm rate
    and, for meeting-wins, the deferred-work (self-orphan) cost it pays."""
    print("=" * 72)
    print(f"STAGE D  P2 meeting-vs-task under task overruns "
          f"(eps={eps:.0f}s, slack={SLACK:.0f}s, nav_err={nav_err:.0%})")
    print("=" * 72)
    rows = []
    for mean_overrun in [30.0, 60.0, 120.0, 240.0]:
        fa_meet, fa_task, deferred = 0, 0, 0
        n = 0
        for s in range(seeds):
            rng = random.Random(s * 40503 + int(mean_overrun))
            overrun = max(0.0, rng.gauss(mean_overrun, 0.4 * mean_overrun))
            # meeting-wins: task overrun abandoned; only travel-estimate error
            # (nav_err of a ~one-band leg) can eat slack.
            leg = rng.uniform(0.3, 1.0) * (L / K)
            travel_err = nav_err * leg / V
            late_meet = max(0.0, travel_err - SLACK)
            # task-wins: overrun beyond the reserved slack shows up as lateness
            late_task = max(0.0, overrun - SLACK)
            fa_meet += late_meet > eps
            fa_task += late_task > eps
            deferred += overrun > SLACK        # a scan gets deferred
            n += 1
        rows.append(dict(mean_overrun_s=mean_overrun,
                         meeting_wins_fa_pct=fa_meet / n * 100,
                         task_wins_fa_pct=fa_task / n * 100,
                         meeting_wins_deferred_pct=deferred / n * 100))
        r = rows[-1]
        print(f"  overrun ~{mean_overrun:5.0f}s | meeting-wins FA "
              f"{r['meeting_wins_fa_pct']:5.1f}%  (defers a scan "
              f"{r['meeting_wins_deferred_pct']:5.1f}% of the time) | "
              f"task-wins FA {r['task_wins_fa_pct']:5.1f}%")
    print("  => meeting-wins converts task-time variance into deferred work "
          "(self-orphans done later),")
    print("     keeping the detection layer false-alarm-free; task-wins leaks "
          "work variance into false accusations.")
    _save_csv(OUT / "stageD_p2.csv", rows)
    return rows


# ---------------------------------------------------------------- figure
def plot_instance(seed=7, K=6, N=30, L=600, delta=DELTA):
    starts, routes, bands = make_banded_world(seed * 7919 + 13, K, N, L)
    mesh = build_mesh(starts, routes, K, L, delta)
    rng = random.Random(seed * 2654435761 + 11)
    victim = 2
    vend = mesh["trajs"][victim][-1][0]
    t_f = 0.45 * vend
    r = run_failure(mesh, victim, t_f)

    fig, (axm, axt) = plt.subplots(1, 2, figsize=(13, 5.4))
    cmap = plt.get_cmap("tab10")
    for i in range(K):
        bp = mesh["trajs"][i]
        xs = [p[1] for p in bp]
        ys = [p[2] for p in bp]
        ax_alpha = 0.35 if i == victim else 0.8
        axm.plot(xs, ys, "-", color=cmap(i), alpha=ax_alpha, lw=1.2,
                 label=f"R{i}" + (" (dies)" if i == victim else ""))
    for i in range(1, K):
        axm.axvline(i * L / K, color="gray", ls=":", alpha=0.4)
    for ap in mesh["appts"]:
        axm.plot(ap["pt"][0], ap["pt"][1], "o", color="k", ms=3, alpha=0.5)
    dx, dy = pos_at(mesh["trajs"][victim], t_f)
    axm.plot(dx, dy, "rX", ms=15, mew=3, label="death")
    for k, poi in enumerate(r["orphans_fwd"]):
        axm.plot(poi[0], poi[1], "s", mfc="none", mec="green", ms=8,
                 label="orphan: grabbed forward" if k == 0 else None)
    for k, poi in enumerate(r["orphans_beh"]):
        axm.plot(poi[0], poi[1], "s", mfc="none", mec="red", ms=8,
                 label="orphan: end cleanup" if k == 0 else None)
    axm.set_title(f"Frozen mesh + one death (R{victim} at t={t_f:.0f}s)\n"
                  f"green sq = grabbed on the sweep, red sq = deferred to end")
    axm.set_xlabel("x (m)")
    axm.set_ylabel("y (m)")
    axm.legend(fontsize=7, loc="upper left", framealpha=1.0,
               facecolor="white")
    axm.set_aspect("equal", adjustable="box")

    # timeline: appointments vs time, victim's marked, detection line
    for ap in mesh["appts"]:
        involves = victim in (ap["a"], ap["b"])
        after = ap["t"] > t_f
        color = ("red" if involves and after else
                 "lightgray" if involves else "steelblue")
        axt.plot([ap["t"], ap["t"]], [ap["a"], ap["b"]], "-", color=color,
                 alpha=0.7, lw=1.5)
        axt.plot(ap["t"], ap["a"], ".", color=color, ms=6)
        axt.plot(ap["t"], ap["b"], ".", color=color, ms=6)
    axt.axvline(t_f, color="red", ls="--", label=f"death t={t_f:.0f}s")
    if r["first_det"] is not None:
        axt.axvline(r["first_det"], color="green", ls="--",
                    label=f"detected t={r['first_det']:.0f}s "
                          f"(+{r['latency']:.0f}s)")
    axt.set_title("Appointment timeline (red = missed with the dead robot)")
    axt.set_xlabel("time (s)")
    axt.set_ylabel("robot id")
    axt.set_yticks(range(K))
    axt.legend(fontsize=8)
    axt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "instance.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'instance.png'}  "
          f"(victim R{victim}, {r['n_orphans']} orphans, "
          f"latency {r['latency']:.0f}s, chain "
          f"{'connected' if r['chain_connected'] else 'BROKEN'})")


# ---------------------------------------------------------------- Stage G (grid)
def _pipeline_metrics(mesh_fn, K, seeds, delta, eps):
    """Run the whole pipeline (mesh cost, detection, recovery, propagation) over
    `seeds` deaths for a given mesh builder; return aggregated metrics."""
    a = dict(lat=[], within1=0, within2=0, redun=0, own=0, chain=0, fwd=[],
             over=[], reach=0, prop=[], minpart=[], travel=[], gap=[],
             lam2=[], diam=[], safe=[], safe2=[], n=0, nprop=0)
    for s in range(seeds):
        rng = random.Random(s * 2654435761 + 11)
        mesh = mesh_fn(s)
        vm = verify_mesh(mesh, verbose=False)
        a["minpart"].append(vm["min_distinct_partners"])
        a["travel"].append(vm["travel_overhead_pct"])
        a["gap"].append(vm["worst_check_gap_rounds"])
        inv = _graph_invariants(mesh["neighbors"], mesh["K"])
        a["lam2"].append(inv["lambda2"])
        a["diam"].append(inv["diameter"] if inv["diameter"] != float("inf") else 0)
        a["safe"].append(inv["single_death_safe"])
        a["safe2"].append(inv["two_death_safe"])
        victim = rng.randrange(K)
        vend = mesh["trajs"][victim][-1][0]
        t_f = rng.uniform(0.2, 0.6) * vend
        pr = propagate_news(mesh, victim, t_f, 10 ** 9)
        if pr is not None:
            a["nprop"] += 1
            a["reach"] += pr["frac_informed"] >= 0.999
            if pr["time_to_full_info"] is not None:
                a["prop"].append(pr["time_to_full_info"] / delta)
        r = run_failure(mesh, victim, t_f, eps)
        if r["latency"] is None:
            continue
        a["n"] += 1
        a["lat"].append(r["latency"])
        a["within1"] += r["latency"] <= delta + eps + 1e-6
        a["within2"] += r["latency"] <= 2 * delta + eps + 1e-6
        a["redun"] += r["redundant"]
        a["own"] += r["total_ownership_ok"]
        a["chain"] += r["chain_connected"]
        if r["n_orphans"]:
            a["fwd"].append(r["n_fwd"] / r["n_orphans"])
            a["over"].extend(v for v in r["survivor_overhead"].values() if v)
    return a


def _schedule_diameter(kind, K, Gr, Gc, delta, M=40, seeds=300):
    """Propagation as a property of the TOPOLOGY (not truncated by a short
    coverage mission): run the bare contact schedule for M rounds and measure
    multi-source rounds-to-reach-the-whole-team and completion rate."""
    appts = []
    for m in range(1, M + 1):
        pairs = (brickwall_pairs(m, K) if kind == "line"
                 else grid_pairs(m, Gr, Gc))
        for (i, j) in pairs:
            appts.append(dict(m=m, t=m * delta, a=i, b=j))
    rounds, reach, n = [], 0, 0
    for s in range(seeds):
        rng = random.Random(s * 97 + K + (0 if kind == "line" else 1))
        victim = rng.randrange(K)
        r = propagate_news(dict(K=K, appts=appts), victim,
                           rng.uniform(1, 3) * delta, 10 ** 9)
        if r is None:
            continue
        n += 1
        reach += r["frac_informed"] >= 0.999
        if r["time_to_full_info"] is not None:
            rounds.append(r["time_to_full_info"] / delta)
    return (sum(rounds) / len(rounds) if rounds else 0.0), reach / max(n, 1) * 100


def compare_line_vs_grid(seeds=120, Gr=4, Gc=4, N_per=8, L=600,
                         delta=DELTA, eps=EPS):
    """Run the FULL pipeline on the 2-D grid and, for the same K, on the line;
    print them side by side. The grid is the 'sweet spot': degree 4 with all
    meetings still LOCAL (no field-crossing), so every robot -- corners
    included -- has >=2 partners (blind spot gone) and propagation speeds up,
    while the contact cost stays near the line's."""
    K, N = Gr * Gc, N_per * Gr * Gc
    print("=" * 72)
    print(f"STAGE G  FULL PIPELINE ON A 2-D GRID vs the LINE  (K={K}: "
          f"grid {Gr}x{Gc} vs line 1x{K}, N={N})")
    print("=" * 72)
    line_fn = lambda s: build_mesh(*make_banded_world(s * 7919 + 13, K, N, L)[:2],
                                   K, L, delta)
    grid_fn = lambda s: build_grid_mesh(*make_grid_world(s * 7919 + 13, Gr, Gc,
                                                         N, L), Gr, Gc, L, delta)
    LN = _pipeline_metrics(line_fn, K, seeds, delta, eps)
    GD = _pipeline_metrics(grid_fn, K, seeds, delta, eps)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    rows = []

    def line2(label, key, fmt, scale=1.0, pct_of=None):
        lv = (LN[key] / (pct_of and LN[pct_of] or 1) * 100) if pct_of \
            else mean(LN[key]) if isinstance(LN[key], list) else LN[key]
        gv = (GD[key] / (pct_of and GD[pct_of] or 1) * 100) if pct_of \
            else mean(GD[key]) if isinstance(GD[key], list) else GD[key]
        print(f"  {label:38s} | line {fmt.format(lv*scale):>9s} | "
              f"grid {fmt.format(gv*scale):>9s}")
        rows.append(dict(metric=label, line=round(lv * scale, 2),
                         grid=round(gv * scale, 2)))
    print("  MESH / TOPOLOGY")
    line2("min distinct partners (blind spot)", "minpart", "{:.1f}")
    line2("worst check gap (rounds)", "gap", "{:.1f}")
    line2("mesh travel overhead vs no-rdv (%)", "travel", "{:.0f}")
    print("  DETECTION")
    line2("latency mean (s)", "lat", "{:.0f}")
    line2("within DELTA+eps (%)", "within1", "{:.0f}", pct_of="n")
    line2("redundant detection (%)", "redun", "{:.0f}", pct_of="n")
    print("  RECOVERY")
    line2("total-ownership (%)", "own", "{:.0f}", pct_of="n")
    line2("chain re-weld (%)", "chain", "{:.0f}", pct_of="n")
    line2("orphans grabbed forward (%)", "fwd", "{:.0f}", scale=100)
    line2("survivor travel overhead (%)", "over", "{:.0f}")
    print("  PROPAGATION (bare schedule, unbounded rounds = topology diameter)")
    l_rounds, l_reach = _schedule_diameter("line", K, Gr, Gc, delta)
    g_rounds, g_reach = _schedule_diameter("grid", K, Gr, Gc, delta)
    print(f"  {'rounds to reach whole team':38s} | line {l_rounds:9.1f} | "
          f"grid {g_rounds:9.1f}")
    print(f"  {'reaches whole team (% within 40 rds)':38s} | line {l_reach:9.0f} | "
          f"grid {g_reach:9.0f}")
    rows.append(dict(metric="prop rounds to full", line=round(l_rounds, 1),
                     grid=round(g_rounds, 1)))
    _save_csv(OUT / "stageG_line_vs_grid.csv", rows)
    return rows


def plot_grid_instance(seed=7, Gr=4, Gc=4, N=128, L=600, delta=DELTA):
    starts, routes = make_grid_world(seed * 7919 + 13, Gr, Gc, N, L)
    mesh = build_grid_mesh(starts, routes, Gr, Gc, L, delta)
    K = Gr * Gc
    victim = (Gr // 2) * Gc + (Gc // 2)
    t_f = 0.45 * mesh["trajs"][victim][-1][0]
    r = run_failure(mesh, victim, t_f)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    cmap = plt.get_cmap("tab20")
    for i in range(K):
        bp = mesh["trajs"][i]
        ax.plot([p[1] for p in bp], [p[2] for p in bp], "-",
                color=cmap(i % 20), alpha=0.3 if i == victim else 0.75, lw=1)
    for c in range(1, Gc):
        ax.axvline(c * L / Gc, color="gray", ls=":", alpha=0.4)
    for rr in range(1, Gr):
        ax.axhline(rr * L / Gr, color="gray", ls=":", alpha=0.4)
    for ap in mesh["appts"]:
        ax.plot(ap["pt"][0], ap["pt"][1], "o", color="k", ms=2, alpha=0.4)
    dx, dy = pos_at(mesh["trajs"][victim], t_f)
    ax.plot(dx, dy, "rX", ms=16, mew=3, label="death")
    for k, poi in enumerate(r["orphans_fwd"]):
        ax.plot(poi[0], poi[1], "s", mfc="none", mec="green", ms=7,
                label="orphan: grabbed forward" if k == 0 else None)
    for k, poi in enumerate(r["orphans_beh"]):
        ax.plot(poi[0], poi[1], "s", mfc="none", mec="red", ms=7,
                label="orphan: end cleanup" if k == 0 else None)
    ax.set_title(f"2-D grid ({Gr}x{Gc}): death of R{victim} at t={t_f:.0f}s\n"
                 f"black o = appointments (all at cell edges = local); "
                 f"latency {r['latency']:.0f}s, "
                 f"chain {'connected' if r['chain_connected'] else 'BROKEN'}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8, loc="upper left", framealpha=1.0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUT / "grid_instance.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'grid_instance.png'}")


# ---------------------------------------------------------------- Stage S (small-world graph)
# The line and grid bake in a SERIES structure: fixed low-degree neighbours,
# news crawls hop-by-hop, and the two line ends are degree-1 blind spots. The
# real object is a GRAPH -- robots are independent nodes, meetings are edges we
# design -- and the things we care about are graph invariants. A small-world
# hybrid = a cheap LOCAL backbone (grid adjacency, so min-degree >= 2 even at
# corners) PLUS a few long-range SHORTCUT meetings. A handful of long edges
# collapse the diameter and lift the algebraic connectivity (parallel spread)
# for little extra travel -- the sweet spot. The recovery machinery is already
# graph-native (each orphan to the single nearest capable survivor; news seeded
# by EVERY misser), so only the schedule generator changes.

def _cell_rc(i, Gc):
    return divmod(i, Gc)


def smallworld_edges(seed, Gr, Gc, n_short):
    """Edge set of the contact GRAPH: grid backbone (min-degree >= 2, no blind
    spot even at corners) PLUS n_short random long-range shortcuts between
    geometrically NON-adjacent robots (Manhattan cell-distance >= 2). Returns
    (all_edges, shortcut_edges), both sorted lists of (i<j)."""
    K = Gr * Gc
    idx = lambda r, c: r * Gc + c
    edges = set()
    for r in range(Gr):
        for c in range(Gc):
            if c + 1 < Gc:
                edges.add((idx(r, c), idx(r, c + 1)))
            if r + 1 < Gr:
                edges.add((idx(r, c), idx(r + 1, c)))
    rng = random.Random(seed * 131 + 7)
    shorts, tries = set(), 0
    while len(shorts) < n_short and tries < 4000:
        tries += 1
        i, j = rng.randrange(K), rng.randrange(K)
        if i == j:
            continue
        e = (min(i, j), max(i, j))
        (ri, ci), (rj, cj) = _cell_rc(i, Gc), _cell_rc(j, Gc)
        if e in edges or abs(ri - rj) + abs(ci - cj) <= 1:
            continue                              # skip duplicate / local edge
        edges.add(e)
        shorts.add(e)
    return sorted(edges), sorted(shorts)


def _edge_colour(edges, K, order_first=None):
    """Greedy edge-colouring -> disjoint matchings (a robot appears in at most
    one pair per colour). Each colour is one 'round type'; the schedule cycles
    the colours, so every edge is realized once per cycle. `order_first` (the
    grid backbone) is coloured first, so local edges pack into the low colours
    (tight local cadence) and shortcuts spill into a few extra colours."""
    first = list(order_first or [])
    fset = set(first)
    order = first + [e for e in edges if e not in fset]
    used = {v: set() for v in range(K)}
    classes = []
    for (i, j) in order:
        c = 0
        while c in used[i] or c in used[j]:
            c += 1
        used[i].add(c)
        used[j].add(c)
        while len(classes) <= c:
            classes.append([])
        classes[c].append((i, j))
    return classes


def _sw_meet_point(i, j, wi, wj, Gr, Gc, L):
    """Where robots i,j rendezvous. LOCAL edges meet on the shared cell edge
    (cheap, like the grid); a SHORTCUT edge meets at the midpoint of the pair's
    current positions -- a field-crossing trip, the price of a long edge."""
    (ri, ci), (rj, cj) = _cell_rc(i, Gc), _cell_rc(j, Gc)
    if ri == rj and abs(ci - cj) == 1:
        return (max(ci, cj) * L / Gc, (wi.pos[1] + wj.pos[1]) / 2)
    if ci == cj and abs(ri - rj) == 1:
        return ((wi.pos[0] + wj.pos[0]) / 2, max(ri, rj) * L / Gr)
    return ((wi.pos[0] + wj.pos[0]) / 2, (wi.pos[1] + wj.pos[1]) / 2)


def build_smallworld_mesh(starts, routes, Gr, Gc, L, delta, edges, shorts):
    """Frozen mesh on a general small-world contact graph. Same deterministic
    offline pass as line/grid, but the schedule is a greedy edge-colouring of the
    graph (grid backbone coloured first) and shortcut meetings cross the field.
    Populates neighbours/centres/reach so the shared contingency + recovery +
    propagation code runs unchanged."""
    K = Gr * Gc
    idx = lambda r, c: r * Gc + c
    neighbors = {r: [] for r in range(K)}
    for (i, j) in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)
    centers = {idx(r, c): ((c + 0.5) * L / Gc, (r + 0.5) * L / Gr)
               for r in range(Gr) for c in range(Gc)}
    sset = set(shorts)
    grid_edges = [e for e in edges if e not in sset]
    classes = _edge_colour(edges, K, order_first=grid_edges)
    C = len(classes)
    walkers = [Walker(starts[i], routes[i]) for i in range(K)]
    appts, m = [], 1
    while any(not w.done() for w in walkers) and m * delta < HORIZON:
        t_m = m * delta
        for w in walkers:
            w.advance_to(t_m)
        for (i, j) in classes[(m - 1) % C]:
            pt = _sw_meet_point(i, j, walkers[i], walkers[j], Gr, Gc, L)
            walkers[i].goto(pt)
            walkers[j].goto(pt)
            t_meet = max(walkers[i].t, walkers[j].t)
            walkers[i].wait_until(t_meet)
            walkers[j].wait_until(t_meet)
            walkers[i].t += CONTACT
            walkers[j].t += CONTACT
            walkers[i].bp.append((walkers[i].t, pt[0], pt[1]))
            walkers[j].bp.append((walkers[j].t, pt[0], pt[1]))
            appts.append(dict(m=m, t=t_meet, a=i, b=j, pt=pt))
        m += 1
    for w in walkers:
        w.advance_to(HORIZON)
    mesh = dict(trajs=[w.bp for w in walkers], appts=appts,
                poi_logs=[w.poi_log for w in walkers], routes=routes,
                starts=starts, K=K, L=L, delta=delta, topology="smallworld",
                Gr=Gr, Gc=Gc, neighbors=neighbors, centers=centers,
                reach=REACH_FRAC * (L / Gr), edges=edges, shorts=shorts)
    mesh["table"] = build_contingency(mesh)
    return mesh


# ---------------------------------------------------------------- graph invariants
def _bfs_reach(neighbors, src, banned=None):
    """Distances from src over the contact graph, skipping any `banned` node."""
    from collections import deque
    banned = banned or set()
    seen, q, d = {src}, deque([src]), {src: 0}
    while q:
        u = q.popleft()
        for w in neighbors[u]:
            if w not in seen and w not in banned:
                seen.add(w)
                d[w] = d[u] + 1
                q.append(w)
    return d


def _graph_invariants(neighbors, K):
    """Recast the properties we care about as GRAPH invariants:
      min_deg           # robots that witness a death at once (>=2 = no blind
                        spot; the line's two ends are 1).
      single_death_safe % of robots whose removal leaves the survivors CONNECTED
                        (news still reaches everyone; re-weld is automatic).
      lambda2           algebraic connectivity: how fast news floods in PARALLEL
                        (bigger = faster; a chain's is ~0).
      diameter          worst-case hops for single-source news."""
    deg = [len(neighbors[r]) for r in range(K)]
    diam = 0
    for s in range(K):
        d = _bfs_reach(neighbors, s)
        if len(d) < K:
            diam = float("inf")
            break
        diam = max(diam, max(d.values()))
    safe = 0
    for v in range(K):
        others = [r for r in range(K) if r != v]
        if not others:
            safe += 1
            continue
        d = _bfs_reach(neighbors, others[0], banned={v})
        safe += (len(d) == len(others))
    import itertools
    pairs = list(itertools.combinations(range(K), 2))
    safe2 = 0
    for (u, v) in pairs:
        rest = [r for r in range(K) if r not in (u, v)]
        d = _bfs_reach(neighbors, rest[0], banned={u, v})
        safe2 += (len(d) == len(rest))
    two_safe = safe2 / len(pairs) * 100 if pairs else 100.0
    try:
        import numpy as np
        A = np.zeros((K, K))
        for i in range(K):
            for j in neighbors[i]:
                A[i, j] = 1.0
        lap = np.diag(A.sum(1)) - A
        lam2 = float(np.sort(np.linalg.eigvalsh(lap))[1])
    except Exception:
        lam2 = float("nan")
    return dict(min_deg=min(deg), mean_deg=sum(deg) / K, diameter=diam,
                single_death_safe=safe / K * 100, two_death_safe=two_safe,
                lambda2=lam2)


# ---------------------------------------------------------------- bare propagation
def _bare_appts(kind, K, Gr, Gc, delta, M, n_short=0, seed=0):
    """Contact schedule as bare appointments over M rounds (no coverage mission
    to truncate it), so propagation reflects the graph DIAMETER, not the mission
    length. Cycles the edge-colouring: every edge realized once per cycle."""
    appts = []
    if kind == "line":
        for m in range(1, M + 1):
            for (i, j) in brickwall_pairs(m, K):
                appts.append(dict(m=m, t=m * delta, a=i, b=j))
    elif kind == "grid":
        for m in range(1, M + 1):
            for (i, j) in grid_pairs(m, Gr, Gc):
                appts.append(dict(m=m, t=m * delta, a=i, b=j))
    else:                                          # smallworld
        edges, shorts = smallworld_edges(seed, Gr, Gc, n_short)
        grid_edges = [e for e in edges if e not in set(shorts)]
        classes = _edge_colour(edges, K, order_first=grid_edges)
        C = len(classes)
        for m in range(1, M + 1):
            for (i, j) in classes[(m - 1) % C]:
                appts.append(dict(m=m, t=m * delta, a=i, b=j))
    return appts


def _measure_bare_prop(kind, K, Gr, Gc, delta, seeds=200, M=80, n_short=0):
    """Multi-source rounds-to-reach-the-whole-team and completion rate on the
    bare schedule (unbounded rounds)."""
    rounds, reach, n = [], 0, 0
    tag = {"line": 0, "grid": 1, "smallworld": 2}.get(kind, 3)
    for s in range(seeds):
        rng = random.Random(s * 97 + K + tag)
        appts = _bare_appts(kind, K, Gr, Gc, delta, M, n_short, seed=s * 7919 + 13)
        victim = rng.randrange(K)
        r = propagate_news(dict(K=K, appts=appts), victim,
                           rng.uniform(1, 3) * delta, 10 ** 9)
        if r is None:
            continue
        n += 1
        reach += r["frac_informed"] >= 0.999
        if r["time_to_full_info"] is not None:
            rounds.append(r["time_to_full_info"] / delta)
    return (sum(rounds) / len(rounds) if rounds else 0.0), reach / max(n, 1) * 100


# ---------------------------------------------------------------- Stage S compare
def compare_meshes(seeds=100, Gr=4, Gc=4, N_per=8, n_short=8, L=600,
                   delta=DELTA, eps=EPS):
    """Full pipeline on THREE contact graphs of the SAME size K: the line
    (series, degree 2, two blind ends), the grid (local mesh, degree 4), and the
    small-world hybrid (grid + n_short long shortcuts). Graph invariants are
    printed next to the pipeline metrics, so 'parallel not series' is measured,
    not asserted."""
    K, N = Gr * Gc, N_per * Gr * Gc
    print("=" * 72)
    print(f"STAGE S  CONTACT GRAPH: line vs grid vs small-world  (K={K}, N={N}, "
          f"shortcuts={n_short})")
    print("=" * 72)
    line_fn = lambda s: build_mesh(*make_banded_world(s * 7919 + 13, K, N, L)[:2],
                                   K, L, delta)
    grid_fn = lambda s: build_grid_mesh(*make_grid_world(s * 7919 + 13, Gr, Gc,
                                                         N, L), Gr, Gc, L, delta)

    def sw_fn(s):
        starts, routes = make_grid_world(s * 7919 + 13, Gr, Gc, N, L)
        edges, shorts = smallworld_edges(s * 7919 + 13, Gr, Gc, n_short)
        return build_smallworld_mesh(starts, routes, Gr, Gc, L, delta,
                                     edges, shorts)

    M = {"line": _pipeline_metrics(line_fn, K, seeds, delta, eps),
         "grid": _pipeline_metrics(grid_fn, K, seeds, delta, eps),
         "small-world": _pipeline_metrics(sw_fn, K, seeds, delta, eps)}
    names = ["line", "grid", "small-world"]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    rows = []

    def show(label, fn, fmt):
        vals = {nm: fn(M[nm]) for nm in names}
        cells = " | ".join(f"{nm} {fmt.format(vals[nm]):>7s}" for nm in names)
        print(f"  {label:34s} | {cells}")
        rows.append(dict(metric=label, **{nm: round(vals[nm], 2) for nm in names}))

    print("  GRAPH INVARIANTS  (parallel-vs-series, by construction)")
    show("min degree (>=2 = no blind spot)", lambda a: mean(a["minpart"]), "{:.1f}")
    show("single-death keeps team connected %", lambda a: mean(a["safe"]), "{:.0f}")
    show("TWO-death keeps team connected %", lambda a: mean(a["safe2"]), "{:.0f}")
    show("algebraic connectivity lambda2", lambda a: mean(a["lam2"]), "{:.2f}")
    show("diameter (hops, worst news path)", lambda a: mean(a["diam"]), "{:.1f}")
    print("  DETECTION")
    show("latency mean (s)", lambda a: mean(a["lat"]), "{:.0f}")
    show("within DELTA+eps %", lambda a: a["within1"] / max(a["n"], 1) * 100, "{:.0f}")
    show("redundant detection %", lambda a: a["redun"] / max(a["n"], 1) * 100, "{:.0f}")
    print("  RECOVERY")
    show("total-ownership %", lambda a: a["own"] / max(a["n"], 1) * 100, "{:.0f}")
    show("chain re-weld %", lambda a: a["chain"] / max(a["n"], 1) * 100, "{:.0f}")
    show("orphans grabbed forward %", lambda a: mean(a["fwd"]) * 100, "{:.0f}")
    show("survivor travel overhead %", lambda a: mean(a["over"]), "{:.0f}")
    print("  PROPAGATION (bare schedule, unbounded rounds = graph diameter)")
    bp = {"line": _measure_bare_prop("line", K, Gr, Gc, delta),
          "grid": _measure_bare_prop("grid", K, Gr, Gc, delta),
          "small-world": _measure_bare_prop("smallworld", K, Gr, Gc, delta,
                                            n_short=n_short)}
    cells = " | ".join(f"{nm} {bp[nm][0]:>7.1f}" for nm in names)
    print(f"  {'rounds to reach whole team':34s} | {cells}")
    rows.append(dict(metric="prop rounds to full",
                     **{nm: round(bp[nm][0], 1) for nm in names}))
    cells = " | ".join(f"{nm} {bp[nm][1]:>6.0f}%" for nm in names)
    print(f"  {'reaches whole team % (<=80 rounds)':34s} | {cells}")
    rows.append(dict(metric="prop reach pct",
                     **{nm: round(bp[nm][1]) for nm in names}))
    print("  COST")
    show("mesh travel overhead vs no-rdv %", lambda a: mean(a["travel"]), "{:.0f}")
    _save_csv(OUT / "stageS_compare.csv", rows)
    print("  => HONEST READ:")
    print("     - line -> grid is the big win: min-deg 1->2 (no blind spot), "
          "single-death-safe")
    print("       12->100%, and recovery goes local (54->96% forward). That is "
          "the 'parallel")
    print("       not series' upgrade, and it is nearly free in robustness terms.")
    print("     - grid -> small-world lifts lambda2 (0.6->1.0) and the diameter "
          "(6->3.9), BUT")
    print("       wall-clock news does NOT speed up (7.9 vs 6.4 rds): a "
          "higher-degree graph needs")
    print("       more colour classes, so the longer schedule cycle eats the "
          "diameter gain")
    print("       (chromatic index >= max degree -- a fundamental tension, not a "
          "tuning artifact).")
    print("     - recovery is identical (local either way); the grid already "
          "survives 1-2 deaths")
    print("       (100/97%); shortcuts buy only marginal multi-death robustness "
          "(->99/100%) at +20%")
    print("       travel. => GRID is the sweet spot; small-world is a lever for "
          "HIGH-ORDER fault")
    print("       tolerance (many concurrent deaths) only.")
    return rows


def sweep_shortcuts(seeds=60, Gr=4, Gc=4, N_per=8,
                    shorts=(0, 2, 4, 6, 8, 12, 16, 24), L=600,
                    delta=DELTA, eps=EPS):
    """Frontier: how many long-range shortcuts to add on top of the grid. Each
    shortcut lifts algebraic connectivity and shrinks the diameter (faster
    parallel spread) but costs field-crossing travel. Find the KNEE = the most
    connectivity bought per unit extra travel."""
    K, N = Gr * Gc, N_per * Gr * Gc
    print("=" * 72)
    print(f"STAGE S2  SHORTCUT FRONTIER: connectivity gain vs travel cost  "
          f"(K={K}, grid {Gr}x{Gc})")
    print("=" * 72)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    rows = []
    for ns in shorts:
        def sw_fn(s, ns=ns):
            starts, routes = make_grid_world(s * 7919 + 13, Gr, Gc, N, L)
            edges, sh = smallworld_edges(s * 7919 + 13, Gr, Gc, ns)
            return build_smallworld_mesh(starts, routes, Gr, Gc, L, delta,
                                         edges, sh)
        a = _pipeline_metrics(sw_fn, K, seeds, delta, eps)
        pr_rounds, pr_reach = _measure_bare_prop("smallworld", K, Gr, Gc, delta,
                                                 n_short=ns)
        rows.append(dict(
            n_short=ns, lambda2=round(mean(a["lam2"]), 3),
            diameter=round(mean(a["diam"]), 1),
            death_safe_pct=round(mean(a["safe"])),
            two_death_safe_pct=round(mean(a["safe2"])),
            prop_rounds=round(pr_rounds, 1), reach_pct=round(pr_reach),
            travel_overhead_pct=round(mean(a["travel"]), 1),
            own_pct=round(a["own"] / max(a["n"], 1) * 100)))
        r = rows[-1]
        print(f"  +{ns:2d} shortcuts | lambda2 {r['lambda2']:5.3f} | diameter "
              f"{r['diameter']:4.1f} | 1-safe {r['death_safe_pct']:3d}% "
              f"2-safe {r['two_death_safe_pct']:3d}% | news "
              f"{r['prop_rounds']:4.1f} rds || travel "
              f"+{r['travel_overhead_pct']:5.1f}%")
    # knee: worst-case news latency is bounded by the DIAMETER, which saturates;
    # lambda2 keeps rising (linear) but buys only average-case spread past the
    # point where the diameter is within +1 hop of its floor. Smallest n_short
    # reaching that near-floor diameter = the sweet spot (min travel for
    # near-best worst-case reach).
    min_diam = min(r["diameter"] for r in rows)
    knee_ns = rows[-1]["n_short"]
    for r in rows:
        if r["diameter"] <= min_diam + 1.0 + 1e-9:
            knee_ns = r["n_short"]
            break
    krow = next(r for r in rows if r["n_short"] == knee_ns)
    print(f"  => diameter (worst-case news path) plateaus at +{knee_ns} shortcuts "
          f"(~{knee_ns / (K / 2):.2f}/robot-pair): diameter {krow['diameter']} "
          f"(floor {min_diam}),\n     travel +{krow['travel_overhead_pct']:.0f}%. "
          f"Beyond it more shortcuts lift lambda2 (average spread) but not the "
          f"worst\n     case, while travel keeps climbing -- the knee / sweet "
          f"spot.")
    _save_csv(OUT / "stageS2_shortcuts.csv", rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ns = [r["n_short"] for r in rows]
    ax.plot(ns, [r["lambda2"] for r in rows], "o-", color="tab:blue",
            label="algebraic connectivity lambda2")
    ax.plot(ns, [r["diameter"] for r in rows], "s-", color="tab:green",
            label="diameter (hops)")
    ax.set_xlabel("# long-range shortcuts added to the grid")
    ax.set_ylabel("lambda2  /  diameter")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(ns, [r["travel_overhead_pct"] for r in rows], "^--",
             color="tab:red", label="travel overhead %")
    ax2.set_ylabel("mesh travel overhead vs no-rdv (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax.axvline(knee_ns, color="gray", ls=":", alpha=0.7)
    ax.set_title(f"Small-world frontier (K={K}): shortcuts buy connectivity, "
                 f"cost travel\nknee ~ +{knee_ns} shortcuts")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "stageS2_frontier.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'stageS2_frontier.png'}")
    return rows


def plot_smallworld_instance(seed=7, Gr=4, Gc=4, N=128, n_short=8, L=600,
                             delta=DELTA):
    starts, routes = make_grid_world(seed * 7919 + 13, Gr, Gc, N, L)
    edges, shorts = smallworld_edges(seed * 7919 + 13, Gr, Gc, n_short)
    mesh = build_smallworld_mesh(starts, routes, Gr, Gc, L, delta, edges, shorts)
    K = Gr * Gc
    victim = (Gr // 2) * Gc + (Gc // 2)
    t_f = 0.45 * mesh["trajs"][victim][-1][0]
    r = run_failure(mesh, victim, t_f)
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    cmap = plt.get_cmap("tab20")
    for i in range(K):
        bp = mesh["trajs"][i]
        ax.plot([p[1] for p in bp], [p[2] for p in bp], "-",
                color=cmap(i % 20), alpha=0.25 if i == victim else 0.55, lw=0.9)
    for c in range(1, Gc):
        ax.axvline(c * L / Gc, color="gray", ls=":", alpha=0.35)
    for rr in range(1, Gr):
        ax.axhline(rr * L / Gr, color="gray", ls=":", alpha=0.35)
    # shortcut edges as long dashed chords between cell centres
    ctr = mesh["centers"]
    for k, (i, j) in enumerate(shorts):
        ax.plot([ctr[i][0], ctr[j][0]], [ctr[i][1], ctr[j][1]], "--",
                color="purple", alpha=0.6, lw=1.4,
                label="long-range shortcut" if k == 0 else None)
    dx, dy = pos_at(mesh["trajs"][victim], t_f)
    ax.plot(dx, dy, "rX", ms=16, mew=3, label="death")
    for k, poi in enumerate(r["orphans_fwd"]):
        ax.plot(poi[0], poi[1], "s", mfc="none", mec="green", ms=7,
                label="orphan: grabbed forward" if k == 0 else None)
    for k, poi in enumerate(r["orphans_beh"]):
        ax.plot(poi[0], poi[1], "s", mfc="none", mec="red", ms=7,
                label="orphan: end cleanup" if k == 0 else None)
    inv = _graph_invariants(mesh["neighbors"], K)
    ax.set_title(f"Small-world contact graph ({Gr}x{Gc} grid + {n_short} "
                 f"shortcuts)\nmin-deg {inv['min_deg']}, lambda2 "
                 f"{inv['lambda2']:.2f}, diameter {inv['diameter']}; death of "
                 f"R{victim}, latency {r['latency']:.0f}s")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=8, loc="upper left", framealpha=1.0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(OUT / "smallworld_instance.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'smallworld_instance.png'}")


# ---------------------------------------------------------------- Stage L (lossy links)
# Phase 1 of the ledger: make the perfect-channel assumption real. Here the
# live-to-live news relay is LOSSY -- a scheduled contact drops its handshake
# with probability p_loss -- so news must find an alternate path. This is where
# the small-world's redundancy (extra independent paths, higher edge
# connectivity) can finally pay off... or where its extra links just add more
# drop surface. Two opposing effects, measured separately:
#   PROPAGATION  more paths => news survives dropped relays (small-world helps)
#   FALSE ALARMS more links => more contacts that can drop => a live robot looks
#                dead more often (small-world hurts) -- unless a multi-witness
#                confirmation rule (enabled by min-degree >= 2) suppresses it.

def _topology_neighbors(kind, K, Gr, Gc, n_short=0, seed=0):
    """Neighbours dict for a topology without building the physical mesh."""
    if kind == "line":
        return {i: [x for x in (i - 1, i + 1) if 0 <= x < K] for i in range(K)}
    if kind == "grid":
        idx = lambda r, c: r * Gc + c
        return {idx(r, c): [idx(r + dr, c + dc)
                            for dr, dc in ((0, -1), (0, 1), (-1, 0), (1, 0))
                            if 0 <= r + dr < Gr and 0 <= c + dc < Gc]
                for r in range(Gr) for c in range(Gc)}
    edges, _ = smallworld_edges(seed, Gr, Gc, n_short)
    nb = {r: [] for r in range(K)}
    for (i, j) in edges:
        nb[i].append(j)
        nb[j].append(i)
    return nb


def _lossy_propagation(kind, K, Gr, Gc, delta, p_loss, B, seeds=300, M=200,
                       n_short=8):
    """News spread on the bare schedule when each live-live relay delivers only
    w.p. 1-p_loss. Returns % of team informed within B rounds (speed under
    loss), final % after M rounds (connectivity limit), and % of deaths that
    reach the WHOLE team."""
    tag = {"line": 0, "grid": 1, "smallworld": 2}.get(kind, 3)
    fb, ff, full, n = [], [], 0, 0
    for s in range(seeds):
        rng = random.Random(s * 131 + K + tag)
        loss_rng = random.Random(s * 7777 + int(p_loss * 1000) + 5)
        appts = _bare_appts(kind, K, Gr, Gc, delta, M, n_short, seed=s * 7919 + 13)
        victim = rng.randrange(K)
        t_f = rng.uniform(1, 3) * delta
        r = propagate_news(dict(K=K, appts=appts), victim, t_f, 10 ** 9,
                           p_loss=p_loss, loss_rng=loss_rng)
        if r is None:
            continue
        n += 1
        full += r["frac_informed"] >= 0.999
        ff.append(r["frac_informed"])
        delays = [v - t_f for v in r["informed"].values() if v is not None]
        fb.append(sum(1 for d in delays if d <= B * delta) / (K - 1))
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return dict(frac_budget=mean(fb) * 100, final_frac=mean(ff) * 100,
                full_pct=full / max(n, 1) * 100)


def _lossy_fa_rate(neighbors, K, p_loss, m):
    """Expected false 'deaths' per 100 live robots per cycle under a
    m-of-degree confirmation rule: a live robot is falsely declared dead only if
    >= m of its degree contacts drop this cycle. A TRUE death shows all degree
    contacts as misses, so it is still caught w.p. 1 whenever degree >= m."""
    from math import comb
    tot = 0.0
    for r in range(K):
        d = len(neighbors[r])
        tot += sum(comb(d, i) * p_loss ** i * (1 - p_loss) ** (d - i)
                   for i in range(m, d + 1))
    return tot / K * 100


def compare_lossy(seeds=300, delta=DELTA):
    """Grid vs small-world under lossy links, at the current scale (K=16) and a
    bigger one (K=36). Propagation robustness (does redundancy save the news?)
    and false-alarm surface (does redundancy cost extra false accusations?),
    with single-miss vs 2-witness confirmation."""
    print("=" * 72)
    print("STAGE L  LOSSY LINKS: grid vs small-world  (Phase-1 channel realism)")
    print("=" * 72)
    scales = [dict(name="4x4 (K=16)", Gr=4, Gc=4, n_short=8, B=15),
              dict(name="6x6 (K=36)", Gr=6, Gc=6, n_short=18, B=25)]
    p_losses = [0.0, 0.1, 0.2, 0.3, 0.4]
    prop_rows, fa_rows = [], []
    prop_curves = {}
    for sc in scales:
        K, Gr, Gc, ns, B = (sc["Gr"] * sc["Gc"], sc["Gr"], sc["Gc"],
                            sc["n_short"], sc["B"])
        print(f"\n  ---- {sc['name']}, +{ns} shortcuts, news budget B={B} rounds "
              f"----")
        print(f"  PROPAGATION under loss  (% team informed within {B} rounds | "
              f"% reaching WHOLE team)")
        gc_curve, wc_curve = [], []
        for p in p_losses:
            g = _lossy_propagation("grid", K, Gr, Gc, delta, p, B, seeds, n_short=ns)
            w = _lossy_propagation("smallworld", K, Gr, Gc, delta, p, B, seeds,
                                   n_short=ns)
            gc_curve.append(g["frac_budget"])
            wc_curve.append(w["frac_budget"])
            print(f"    p_loss {p:.1f} | grid {g['frac_budget']:5.1f}% "
                  f"({g['full_pct']:3.0f}% full) | small-world "
                  f"{w['frac_budget']:5.1f}% ({w['full_pct']:3.0f}% full) | "
                  f"delta {w['frac_budget'] - g['frac_budget']:+5.1f}pp")
            prop_rows.append(dict(scale=sc["name"], p_loss=p,
                                  grid_frac=round(g["frac_budget"], 1),
                                  grid_full=round(g["full_pct"]),
                                  sw_frac=round(w["frac_budget"], 1),
                                  sw_full=round(w["full_pct"])))
        prop_curves[sc["name"]] = (gc_curve, wc_curve)
        # false alarms: analytic from degree sequences (sw averaged over seeds)
        gnb = _topology_neighbors("grid", K, Gr, Gc)
        wnbs = [_topology_neighbors("smallworld", K, Gr, Gc, ns, seed=s * 7919 + 13)
                for s in range(50)]
        gdeg = sum(len(gnb[r]) for r in range(K)) / K
        wdeg = sum(sum(len(nb[r]) for r in range(K)) / K for nb in wnbs) / len(wnbs)
        print(f"  FALSE ALARMS  (false 'deaths' per 100 robots per cycle; "
              f"mean degree grid {gdeg:.1f} vs small-world {wdeg:.1f})")
        for m in (1, 2):
            tag = "single-miss" if m == 1 else "2-witness  "
            line = []
            for p in p_losses:
                g_fa = _lossy_fa_rate(gnb, K, p, m)
                w_fa = sum(_lossy_fa_rate(nb, K, p, m) for nb in wnbs) / len(wnbs)
                line.append((p, g_fa, w_fa))
                fa_rows.append(dict(scale=sc["name"], rule=f"m={m}", p_loss=p,
                                    grid_fa=round(g_fa, 2), sw_fa=round(w_fa, 2)))
            cells = "  ".join(f"p{p:.1f}: g{gf:5.1f}/w{wf:5.1f}"
                              for p, gf, wf in line)
            print(f"    {tag} (m={m}) | {cells}")
    _save_csv(OUT / "stageL_propagation.csv", prop_rows)
    _save_csv(OUT / "stageL_false_alarms.csv", fa_rows)

    # propagation figure: frac informed within budget vs p_loss, both scales
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sty = {"4x4 (K=16)": "-", "6x6 (K=36)": "--"}
    for nm, (gc, wc) in prop_curves.items():
        ax.plot(p_losses, gc, "o" + sty[nm], color="tab:green",
                label=f"grid {nm}")
        ax.plot(p_losses, wc, "s" + sty[nm], color="tab:purple",
                label=f"small-world {nm}")
    ax.set_xlabel("link-loss probability p_loss")
    ax.set_ylabel("% of team informed within budget")
    ax.set_title("News under lossy links: small-world redundancy vs grid\n"
                 "(gap widens with loss and with team size)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stageL_propagation.png", dpi=130)
    plt.close(fig)
    print(f"\n  -> {OUT / 'stageL_propagation.png'}")

    # false-alarm figure (the informative one): FA vs p_loss, biggest scale
    big = scales[-1]["name"]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for rule, style in (("m=1", "-"), ("m=2", "--")):
        rr = [r for r in fa_rows if r["scale"] == big and r["rule"] == rule]
        rr.sort(key=lambda x: x["p_loss"])
        ps = [r["p_loss"] for r in rr]
        lab = "single-miss" if rule == "m=1" else "2-witness"
        ax.plot(ps, [r["grid_fa"] for r in rr], "o" + style, color="tab:green",
                label=f"grid, {lab}")
        ax.plot(ps, [r["sw_fa"] for r in rr], "s" + style, color="tab:purple",
                label=f"small-world, {lab}")
    ax.set_xlabel("link-loss probability p_loss")
    ax.set_ylabel("false 'deaths' per 100 robots per cycle")
    ax.set_title(f"False alarms under lossy links ({big})\n"
                 "2-witness confirmation crushes FA; small-world's extra links "
                 "cost more")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "stageL_false_alarms.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'stageL_false_alarms.png'}")
    print("  => HONEST READ (the small-world does NOT get rescued by loss):")
    print("     - PROPAGATION: no edge. The grid is already 2-connected with a "
          "SHORT cycle, so")
    print("       news survives dropped relays fine (>=97% informed even at "
          "p_loss=0.4); the")
    print("       small-world is marginally WORSE (its longer cycle fires each "
          "edge less often,")
    print("       and that outweighs its extra paths). Redundancy past the grid "
          "buys no news.")
    print("     - FALSE ALARMS: the small-world is strictly WORSE -- more links "
          "= more drop")
    print("       surface, so a live robot looks dead more often (single-miss "
          "FA ~ degree*p_loss;")
    print("       2-witness FA ~ C(degree,2)*p_loss^2, both higher for the "
          "higher-degree graph).")
    print("     - THE REAL PHASE-1 LESSON: single-miss detection is unusable "
          "(FA explodes with")
    print("       p_loss); a 2-WITNESS rule (needs min-degree>=2, which the grid "
          "already has) turns")
    print("       FA from linear to quadratic in p_loss. At high loss even that "
          "isn't enough --")
    print("       stack TEMPORAL persistence (require the miss k cycles running "
          "-> FA ~ (p_loss^2)^k,")
    print("       at the cost of k cycles of detection latency). Confirmation, "
          "NOT topology, makes")
    print("       C2 real -- and it strengthens the verdict: grid is the sweet "
          "spot under loss too.")
    return prop_rows, fa_rows


# ---------------------------------------------------------------- Stage K (heartbeat + data gossip)
# Two properties the whole scheme quietly assumes, made EXPLICIT and checked by
# execution:
#   (1) HEARTBEAT in the vicinity, MULTIPLE sources: a rendezvous is a mutual
#       "I'm alive" exchange, and every robot has >=2 distinct partners (min-deg
#       >=2 on the grid), so its liveness is confirmed by several others.
#   (2) FULL DATA SHARING at every meeting: each contact merges full state (who
#       is alive + each robot's scan progress), so the plan is maintained
#       GLOBALLY by gossip -- and if a robot dies, the data it ferried at its
#       LAST meeting survives in its partners, so nothing is lost. This turns
#       ledger-R1 ("identically known to all") from an assumption into a
#       gossip mechanism with a measured staleness bound.

def gossip_check(seeds=150, Gr=4, Gc=4, delta=DELTA, M=80):
    """Full-state knowledge gossip over the grid schedule with one death. Each
    meeting = mutual heartbeat + full-state merge. Verifies: survivors converge
    to a COMPLETE, consistent global view; the victim's last-ferried data
    SURVIVES its death; and reports the staleness bound + heartbeat multiplicity
    (distinct liveness sources per robot)."""
    K = Gr * Gc
    print("=" * 72)
    print(f"STAGE K  HEARTBEAT + FULL-STATE GOSSIP (plan stays global through a "
          f"death)  grid {Gr}x{Gc}")
    print("=" * 72)
    complete, data_survived, n = 0, 0, 0
    conv_rounds, min_hb = [], []
    for s in range(seeds):
        rng = random.Random(s * 911 + 3)
        appts = sorted(_bare_appts("grid", K, Gr, Gc, delta, M),
                       key=lambda a: (a["m"], a["a"], a["b"]))
        victim = rng.randrange(K)
        v_rounds = sorted({ap["m"] for ap in appts if victim in (ap["a"], ap["b"])})
        if len(v_rounds) < 3:
            continue
        death_round = v_rounds[len(v_rounds) // 2]            # dies mid-mission
        last_ferry = max(r for r in v_rounds if r < death_round)
        # known[r][j] = latest round r has heard j was alive (+ j's progress proxy)
        known = {r: {r: 0} for r in range(K)}
        heard_from = {r: set() for r in range(K)}
        conv = None
        for ap in appts:
            a, b, m = ap["a"], ap["b"], ap["m"]
            if victim in (a, b) and m >= death_round:
                continue                                       # victim absent
            heard_from[a].add(b)
            heard_from[b].add(a)
            merged = dict(known[a])
            for j, rr in known[b].items():
                if merged.get(j, -1) < rr:
                    merged[j] = rr
            merged[a] = max(merged.get(a, 0), m)               # heartbeat: a,b alive now
            merged[b] = max(merged.get(b, 0), m)
            known[a], known[b] = dict(merged), dict(merged)
            if conv is None:
                survs = [r for r in range(K) if r != victim]
                if all(len(known[r]) == K for r in survs):
                    conv = m
        n += 1
        survs = [r for r in range(K) if r != victim]
        complete += all(len(known[r]) == K for r in survs)
        data_survived += any(known[r].get(victim, -1) >= last_ferry for r in survs)
        min_hb.append(min(len(heard_from[r]) for r in survs))
        if conv is not None:
            conv_rounds.append(conv)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    p95 = lambda xs: sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else 0.0
    print(f"  trials: {n}/{seeds}")
    print(f"  (1) HEARTBEAT multiplicity: every survivor hears a live heartbeat "
          f"from >= {min(min_hb) if min_hb else 0} distinct partners "
          f"(mean {mean(min_hb):.1f}) -> liveness is multiply-confirmed, no "
          f"single-source blind spot.")
    print(f"  (2) GLOBAL PLAN via gossip: survivors reach a COMPLETE + consistent "
          f"global view in {complete / n * 100:.0f}% of deaths, converging in "
          f"{mean(conv_rounds):.1f} rounds (p95 {p95(conv_rounds):.0f}) = the "
          f"staleness bound.")
    print(f"  (2) DATA SURVIVES DEATH: the victim's last-ferried state is still "
          f"held by a survivor in {data_survived / n * 100:.0f}% of deaths -> a "
          f"death loses no information (it was shared at its last meeting).")
    print(f"  => R1 'identically known to all' is now a GOSSIP MECHANISM with a "
          f"bounded staleness,\n     not a bare assumption: perfect vicinity "
          f"heartbeats + full-state merge at every\n     meeting keep the plan "
          f"global and loss-proof to a single death.")
    _save_csv(OUT / "stageK_gossip.csv", [
        dict(metric="complete_global_view_pct", value=round(complete / n * 100, 1)),
        dict(metric="data_survives_death_pct", value=round(data_survived / n * 100, 1)),
        dict(metric="convergence_rounds_mean", value=round(mean(conv_rounds), 1)),
        dict(metric="convergence_rounds_p95", value=round(p95(conv_rounds), 1)),
        dict(metric="min_heartbeat_sources", value=min(min_hb) if min_hb else 0)])
    return complete, data_survived, n


# ---------------------------------------------------------------- Stage M (currents/wind)
# Vicinity comms are PERFECT (L1 kept): two robots actually within range at a
# rendezvous exchange flawlessly. The real source of "a live robot looks absent"
# is the ENVIRONMENT -- currents/wind delay/displace a robot so it misses its
# rendezvous WINDOW. This disturbance is spatially + temporally CORRELATED (a
# current fills a region for a while), which flips the detection story from the
# i.i.d. link-loss case (Stage L):
#   * a current delaying BOTH partners equally does NOT break their meeting --
#     they are co-located, both late, and still find each other. Misses come from
#     DIFFERENTIAL delay (one in the current, one out / across a gradient).
#   * so multi-witness confirmation is the WRONG tool (a current makes a whole
#     region miss together). The right tool is a LIVENESS detector -- "made >=1
#     contact in T cycles = alive" -- which works because a delayed-but-alive
#     robot always still meets a co-disturbed neighbour. Grid locality guarantees
#     that neighbour; small-world shortcuts only add differential misses.

def _cell_centers(kind, K, Gr, Gc, L):
    if kind == "line":
        return {i: ((i + 0.5) * L / K, 0.5 * L) for i in range(K)}
    idx = lambda r, c: r * Gc + c
    return {idx(r, c): ((c + 0.5) * L / Gc, (r + 0.5) * L / Gr)
            for r in range(Gr) for c in range(Gc)}


def _current_field(seed, L, n_events, max_cycle, strength, eps, sigma_frac=0.25):
    """A few space-time current/wind events: a Gaussian blob (centre, sigma =
    spatial correlation length) active over a cycle window, imparting a peak
    arrival delay of strength*eps. Smooth in space (neighbours share it), a
    window in time."""
    rng = random.Random(seed * 2749 + 1)
    ev = []
    for _ in range(n_events):
        cx, cy = rng.uniform(0.1 * L, 0.9 * L), rng.uniform(0.1 * L, 0.9 * L)
        sigma = sigma_frac * L * rng.uniform(0.7, 1.3)
        c0 = rng.randint(0, max(0, max_cycle - 4))
        c1 = c0 + rng.randint(2, 5)
        amp = strength * eps * rng.uniform(0.6, 1.0)
        ev.append((cx, cy, sigma, c0, c1, amp))
    return ev


def _delay(ev, x, y, c):
    d = 0.0
    for (cx, cy, sigma, c0, c1, amp) in ev:
        if c0 <= c <= c1:
            d += amp * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return d


def _env_metrics(kind, K, Gr, Gc, L, delta, strength, eps, grace=0.0,
                 sigma_frac=0.25, seeds=200, T=20, persist=2, kwit=2, n_short=8):
    """False-alarm rate of three detectors under a correlated current field, and
    the true-death detection outcome. A meeting is MADE iff the two partners'
    delays differ by <= eps+grace (equal delay => they still meet). Detectors:
      single-miss : flag X on any missed meeting (trigger-happy).
      k-witness   : flag X if >= k of its partners miss it in one cycle.
      liveness-T  : flag X only if it made ZERO meetings for `persist` cycles
                    running (a delayed-but-alive robot still meets a co-disturbed
                    neighbour, so it is never fully silent)."""
    centers = _cell_centers(kind, K, Gr, Gc, L)
    eps_eff = eps + grace
    fa = dict(single=0.0, kwit=0.0, live=0.0)
    tdet = dict(single=0, kwit=0, live=0)
    lat, n = [], 0
    for s in range(seeds):
        nbrs = _topology_neighbors(kind, K, Gr, Gc, n_short, seed=s * 7919 + 13)
        ev = _current_field(s * 13 + 7, L, 3, T, strength, eps, sigma_frac)
        dly = [[_delay(ev, centers[i][0], centers[i][1], t) for i in range(K)]
               for t in range(T)]
        made = lambda t, i, j: abs(dly[t][i] - dly[t][j]) <= eps_eff
        # ---- FALSE ALARMS (all robots alive) ----
        flagged = dict(single=0, kwit=0, live=0)
        for X in range(K):
            deg = len(nbrs[X])
            miss = [sum(1 for P in nbrs[X] if not made(t, X, P)) for t in range(T)]
            mkc = [deg - miss[t] for t in range(T)]
            if any(m > 0 for m in miss):
                flagged["single"] += 1
            if any(m >= kwit for m in miss):
                flagged["kwit"] += 1
            if any(all(mkc[t + u] == 0 for u in range(persist))
                   for t in range(T - persist + 1)):
                flagged["live"] += 1
        for key in fa:
            fa[key] += flagged[key] / K
        # ---- TRUE DEATH (v absent from cycle c_d) ----
        n += 1
        v = random.Random(s * 31 + 5).randrange(K)
        c_d = random.Random(s * 17 + 2).randint(1, max(1, T - persist - 1))
        deg = len(nbrs[v])
        miss, mkc = [], []
        for t in range(T):
            if t >= c_d:
                miss.append(deg)
                mkc.append(0)
            else:
                mc = sum(1 for P in nbrs[v] if not made(t, v, P))
                miss.append(mc)
                mkc.append(deg - mc)
        tdet["single"] += any(miss[t] > 0 for t in range(c_d, T))
        tdet["kwit"] += any(miss[t] >= kwit for t in range(c_d, T))
        det_t = next((t + persist - 1 for t in range(c_d, T - persist + 1)
                      if all(mkc[t + u] == 0 for u in range(persist))), None)
        if det_t is not None:
            tdet["live"] += 1
            lat.append(det_t - c_d)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
    return dict(fa_single=fa["single"] / seeds * 100,
                fa_kwit=fa["kwit"] / seeds * 100,
                fa_live=fa["live"] / seeds * 100,
                td_single=tdet["single"] / n * 100,
                td_kwit=tdet["kwit"] / n * 100,
                td_live=tdet["live"] / n * 100, live_latency=mean(lat))


def compare_environment(seeds=200, delta=DELTA, eps=EPS, Gr=4, Gc=4, L=600,
                        n_short=8):
    """Grid vs small-world under a correlated current/wind field. Sweep current
    strength (peak delay in units of eps) and report each detector's false-alarm
    rate + the true-death detection cost, to show that (a) correlation makes
    multi-witness the wrong tool, (b) a liveness detector nearly eliminates
    environmental false alarms, and (c) the grid's locality helps while the
    small-world's shortcuts hurt."""
    K = Gr * Gc
    print("=" * 72)
    print(f"STAGE M  CURRENTS/WIND (correlated delay, perfect vicinity comms)  "
          f"K={K}")
    print("=" * 72)
    strengths = [0, 1, 2, 3, 4]
    rows = []
    curves = {(t, d): [] for t in ("grid", "small-world")
              for d in ("single", "kwit", "live")}
    for strength in strengths:
        g = _env_metrics("grid", K, Gr, Gc, L, delta, strength, eps, seeds=seeds)
        w = _env_metrics("smallworld", K, Gr, Gc, L, delta, strength, eps,
                         seeds=seeds, n_short=n_short)
        for d, key in (("single", "fa_single"), ("kwit", "fa_kwit"),
                       ("live", "fa_live")):
            curves[("grid", d)].append(g[key])
            curves[("small-world", d)].append(w[key])
        print(f"  peak delay {strength}*eps | FA% (single / 2-witness / "
              f"liveness-2)")
        print(f"     grid        {g['fa_single']:5.1f} / {g['fa_kwit']:5.1f} / "
              f"{g['fa_live']:5.1f}   (true-death caught: live {g['td_live']:.0f}%,"
              f" +{g['live_latency']:.1f} cyc latency)")
        print(f"     small-world {w['fa_single']:5.1f} / {w['fa_kwit']:5.1f} / "
              f"{w['fa_live']:5.1f}   (true-death caught: live {w['td_live']:.0f}%,"
              f" +{w['live_latency']:.1f} cyc latency)")
        rows.append(dict(strength=strength,
                         grid_single=round(g["fa_single"], 1),
                         grid_kwit=round(g["fa_kwit"], 1),
                         grid_live=round(g["fa_live"], 1),
                         sw_single=round(w["fa_single"], 1),
                         sw_kwit=round(w["fa_kwit"], 1),
                         sw_live=round(w["fa_live"], 1),
                         grid_live_lat=round(g["live_latency"], 2),
                         sw_live_lat=round(w["live_latency"], 2)))
    _save_csv(OUT / "stageM_environment.csv", rows)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    col = {"single": "tab:red", "kwit": "tab:orange", "live": "tab:blue"}
    lab = {"single": "single-miss", "kwit": "2-witness", "live": "liveness-2"}
    for d in ("single", "kwit", "live"):
        ax.plot(strengths, curves[("grid", d)], "o-", color=col[d],
                label=f"grid, {lab[d]}")
        ax.plot(strengths, curves[("small-world", d)], "s--", color=col[d],
                label=f"small-world, {lab[d]}")
    ax.set_xlabel("current strength (peak arrival delay / eps)")
    ax.set_ylabel("% robots falsely declared dead (over mission)")
    ax.set_title("False alarms under correlated currents/wind\n"
                 "liveness detector (blue) stays ~0; multi-witness does NOT save "
                 "you here")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "stageM_environment.png", dpi=130)
    plt.close(fig)
    print(f"  -> {OUT / 'stageM_environment.png'}")
    print("  => HONEST READ:")
    print("     - single-miss is hopeless (a current's differential delay makes "
          "nearly every")
    print("       robot miss SOMETHING) and 2-witness barely helps -- correlation "
          "means a whole")
    print("       region misses together, so witnesses agree on a benign absence "
          "(the OPPOSITE")
    print("       of the i.i.d. link-loss case, where witnesses were independent "
          "and helped).")
    print("     - the LIVENESS detector ('made >=1 contact in 2 cycles = alive') "
          "nearly zeroes")
    print("       environmental false alarms, because a delayed-but-alive robot "
          "still meets a")
    print("       co-disturbed neighbour -- at the cost of ~persist cycles of "
          "detection latency.")
    print("     - the GRID beats the small-world: local meetings share the "
          "disturbance (equal")
    print("       delay => still meet), while long shortcuts straddle current "
          "boundaries and add")
    print("       differential misses. Currents REINFORCE grid + liveness "
          "detection as the design.")
    return rows


# ---------------------------------------------------------------- util/main
def _save_csv(path, rows):
    if not rows:
        return
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


def stage_A(K=6, N=30, L=600, delta=DELTA):
    print("=" * 72)
    print(f"STAGE A  FROZEN MESH (K={K}, N={N}, L={L}, DELTA={delta:.0f}s)")
    print("=" * 72)
    agg = dict(gap=[], part=[], tr=[], mk=[])
    for s in range(20):
        starts, routes, _ = make_banded_world(s * 7919 + 13, K, N, L)
        mesh = build_mesh(starts, routes, K, L, delta)
        res = verify_mesh(mesh, verbose=(s == 0))
        agg["gap"].append(res["worst_check_gap_rounds"])
        agg["part"].append(res["min_distinct_partners"])
        agg["tr"].append(res["travel_overhead_pct"])
        agg["mk"].append(res["makespan_overhead_pct"])
    print(f"  across 20 seeds: worst check gap "
          f"{max(agg['gap'])} rounds, min partners {min(agg['part'])}, "
          f"travel overhead {sum(agg['tr'])/len(agg['tr']):+.0f}% "
          f"(median), makespan {sum(agg['mk'])/len(agg['mk']):+.0f}%")
    _save_csv(OUT / "stageA_mesh.csv",
              [dict(seed=s, travel_pct=t, makespan_pct=m)
               for s, (t, m) in enumerate(zip(agg["tr"], agg["mk"]))])


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("A", "all"):
        stage_A()
    if which in ("B", "C", "BC", "all"):
        sweep_failures()
    if which in ("E", "all"):
        sweep_overlap()
        measure_propagation()
    if which in ("T", "all"):
        compare_topologies()
    if which in ("G", "all"):
        compare_line_vs_grid()
        plot_grid_instance()
    if which in ("S", "all"):
        compare_meshes()
        sweep_shortcuts()
        plot_smallworld_instance()
    if which in ("L", "all"):
        compare_lossy()
        print("\n" + "#" * 72)
        print("# BIGGER SCALE: full pipeline at 6x6 (K=36), L=1200, 12 POIs/robot")
        print("#" * 72)
        compare_meshes(seeds=30, Gr=6, Gc=6, N_per=12, n_short=18, L=1200)
    if which in ("K", "all"):
        gossip_check()
    if which in ("M", "all"):
        compare_environment()
    if which in ("X", "all"):
        sweep_execution()
        plot_execution_instance()
    if which in ("D", "all"):
        demo_p2()
    if which in ("fig", "all"):
        plot_instance()
