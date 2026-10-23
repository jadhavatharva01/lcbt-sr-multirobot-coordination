"""
rendezvous_byproduct_mesh.py

PROBE: can a planned physical rendezvous be a BYPRODUCT of path planning?

Instead of inserting an artificial meeting waypoint that both robots detour to,
we choose the meeting point AND each robot's per-segment POI allocation so that
the two robots ARRIVE at the same point at the same time purely by balancing
(POIs scanned) against (distance travelled) -- the user's "3 far POIs cost the
same time as 5 near POIs" idea.

THE REAL MATH
-------------
    t_A = travel_A / V + n_A * SCAN
    t_B = travel_B / V + n_B * SCAN
    a meeting exists   <=>   t_A == t_B   (within EPS)
Each POI is worth V*SCAN metres of travel  (= 120 m at V=2, SCAN=60):
    travel_A - travel_B = V * (n_B - n_A) * SCAN

ZERO-WAIT FEASIBILITY reduces to a clean geometric test. Pin meetings to the
global clock (round r meets at tau_r = (r+1)*DELTA). Robot X can reach a point m
by tau_r iff m is inside its reachability disk of radius V*(tau_r - t_X). Two
robots can meet with ZERO WAIT iff their disks overlap:
    d(A,B) <= V*(budget_A + budget_B)
If they overlap: put m in the overlap, spend each robot's leftover time slack on
nearby POIs (productive), pad any remainder with travel (the deferred cost).
If they don't: the pair cannot meet on the clock -> a residual WAIT is forced,
    T* = (d/V + t_A + t_B)/2      (earliest common arrival time)

The meeting MESH is a round-robin so every robot meets a (rotating) partner each
DELTA -- that is what gives detection-of-any-death within DELTA+EPS.

Constants mirror encounter_planning_probe.py / full_pipeline_sim.py. No
HoloOcean. `--dry_run` runs self-tests; otherwise it prints a report and renders
a 2D figure to results/byproduct_mesh/.
"""

import argparse
import copy
import math
import os
import random

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# --- kinematic constants (match the rest of the pipeline) --------------------
V = 2.0        # m/s cruise
SCAN = 60.0    # s per POI on-station scan
DELTA = 300.0  # s rendezvous cadence
EPS = 30.0     # s meeting grace window
NUM_TOL = 1e-6
UTURN_DEG = 120.0  # a turn this sharp or sharper counts as a U-turn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "byproduct_mesh")

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


# --- geometry helpers --------------------------------------------------------
def dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def polyline_len(pts):
    return sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def make_scenario(K, N, L, seed):
    """POIs spread over the field. Each robot gets a HOME anchor on a spread
    grid; it starts there and keeps a territory around it. Meetings happen at
    the boundary between two robots' homes (their shared edge), so the mesh
    disperses instead of collapsing to one point."""
    rng = np.random.default_rng(seed)
    pois = rng.uniform(0.06 * L, 0.94 * L, size=(N, 2))
    cols = math.ceil(math.sqrt(K))
    rows = math.ceil(K / cols)
    homes = np.array([[L * ((r % cols) + 0.5) / cols,
                       L * ((r // cols) + 0.5) / rows] for r in range(K)],
                     float)
    # small jitter so distinct pairs never share a boundary midpoint (e.g. in a
    # square grid both diagonals would otherwise meet at the exact centre).
    homes += rng.uniform(-0.06, 0.06, size=homes.shape) * (L / max(cols, rows))
    return homes, pois


def round_robin_rounds(K):
    """Standard circle method. Returns list of rounds; each round is a list of
    (a, b) pairs. If K is odd, one robot gets a bye (paired with None)."""
    players = list(range(K))
    if K % 2 == 1:
        players.append(None)  # bye slot
    n = len(players)
    rounds = []
    for _ in range(n - 1):
        pairs = [(players[i], players[n - 1 - i]) for i in range(n // 2)]
        rounds.append(pairs)
        players = [players[0]] + [players[-1]] + players[1:-1]  # rotate, fix 0
    return rounds


def assign_pois(start, end, slack, pois, assigned):
    """Greedy cheapest-insertion: pack unassigned POIs into the leg start->end
    while the added time (SCAN + detour/V) stays within `slack`. Mutates
    `assigned`. Returns (waypoints, poi_idx, productive_time, padding_time)."""
    waypoints = [np.asarray(start, float), np.asarray(end, float)]
    idx = []
    remaining = max(0.0, slack)
    while True:
        best_j = best_cost = best_pos = None
        for j in range(len(pois)):
            if assigned[j]:
                continue
            p = pois[j]
            for e in range(len(waypoints) - 1):
                a, b = waypoints[e], waypoints[e + 1]
                extra = dist(a, p) + dist(p, b) - dist(a, b)
                cost = SCAN + extra / V
                if cost <= remaining + NUM_TOL and (best_cost is None or cost < best_cost):
                    best_j, best_cost, best_pos = j, cost, e + 1
        if best_j is None:
            break
        waypoints.insert(best_pos, np.asarray(pois[best_j], float))
        assigned[best_j] = True
        idx.append(best_j)
        remaining -= best_cost
    productive = max(0.0, slack) - remaining
    return waypoints, idx, productive, remaining


def plan_pair(pa, pb, ta, tb, tau, m, pois, assigned):
    """Plan one scheduled meeting between two robots currently at (pa,ta),(pb,tb)
    at the FIXED boundary point `m`, targeting clock time `tau`. Zero-wait iff m
    lies inside both reachability disks (each robot can reach m by tau)."""
    pa = np.asarray(pa, float)
    pb = np.asarray(pb, float)
    m = np.asarray(m, float)
    # earliest each can reach the boundary point m
    tA_min = ta + dist(pa, m) / V
    tB_min = tb + dist(pb, m) / V
    T = max(tau, tA_min, tB_min)
    feasible = T <= tau + NUM_TOL
    wait = T - tau
    budA, budB = T - ta, T - tb
    rA, rB = V * budA, V * budB   # reachability-disk radii, centred at pa, pb
    slackA = budA - dist(pa, m) / V
    slackB = budB - dist(pb, m) / V
    wpA, idxA, prodA, padA = assign_pois(pa, m, slackA, pois, assigned)
    wpB, idxB, prodB, padB = assign_pois(pb, m, slackB, pois, assigned)
    travelA = polyline_len(wpA) + V * padA   # incl. pad-travel to land on time
    travelB = polyline_len(wpB) + V * padB
    return dict(
        m=m, T=T, tau=tau, wait=wait, feasible=feasible,
        preA=pa, preB=pb, budA=budA, budB=budB, rA=rA, rB=rB,
        wpA=wpA, wpB=wpB, idxA=idxA, idxB=idxB,
        nA=len(idxA), nB=len(idxB), travelA=travelA, travelB=travelB,
        prodA=prodA, prodB=prodB, padA=padA, padB=padB,
        # arrival time check (should equal T for both):
        tA=ta + len(idxA) * SCAN + travelA / V,
        tB=tb + len(idxB) * SCAN + travelB / V,
    )


def construct(homes, pois, K, num_rounds, delta=DELTA, assign="greedy"):
    base = round_robin_rounds(K)
    homes = np.asarray(homes, float)
    pos = [homes[r].copy() for r in range(K)]
    tclk = [0.0] * K
    tours = [[("start", pos[r].copy(), 0.0)] for r in range(K)]
    assigned = np.zeros(len(pois), bool)
    blocked = np.ones(len(pois), bool)   # for smart: suppress greedy in-loop assign
    meetings = []
    meet_times = [[] for _ in range(K)]
    byes = [0] * K
    for r in range(num_rounds):
        pairs = base[r % len(base)]
        tau = (r + 1) * delta
        for (a, b) in pairs:
            if a is None or b is None:
                present = a if a is not None else b
                if present is not None:
                    byes[present] += 1
                continue
            m = 0.5 * (homes[a] + homes[b])   # shared boundary of the two homes
            amask = assigned if assign == "greedy" else blocked
            M = plan_pair(pos[a], pos[b], tclk[a], tclk[b], tau, m, pois, amask)
            M.update(round=r, a=a, b=b)
            meetings.append(M)
            # commit robot a
            for k, wp in enumerate(M["wpA"][1:], 0):  # skip the start (already there)
                kind = "poi" if k < len(M["idxA"]) else "meet"
                tours[a].append((kind, np.asarray(wp, float), None))
            tours[a][-1] = ("meet", M["m"].copy(), M["T"])
            # commit robot b
            for k, wp in enumerate(M["wpB"][1:], 0):
                kind = "poi" if k < len(M["idxB"]) else "meet"
                tours[b].append((kind, np.asarray(wp, float), None))
            tours[b][-1] = ("meet", M["m"].copy(), M["T"])
            pos[a] = M["m"].copy(); pos[b] = M["m"].copy()
            tclk[a] = M["T"]; tclk[b] = M["T"]
            meet_times[a].append(M["T"]); meet_times[b].append(M["T"])
    res = dict(tours=tours, meetings=meetings, assigned=assigned,
               meet_times=meet_times, byes=byes, K=K, pois=pois,
               num_rounds=num_rounds, tclk=tclk, delta=delta)
    if assign == "smart":
        greedy_res = construct(homes, pois, K, num_rounds, delta, assign="greedy")
        init_map = _extract_seg_pois(greedy_res, pois)
        segs = segments_of(res)
        assign_global(segs, pois, random.Random(12345), init_map=init_map)
        rebuild_smart_tours(res, segs, pois)
        for seg in segs:
            for j in seg["pois"]:
                assigned[j] = True
    return res


# --- smart assignment: Team-Orienteering-style regret + ALNS ------------------
def segments_of(res):
    """Extract the fixed-endpoint, time-budgeted segments from a POI-free
    skeleton (each is one prize-collecting route in the Team Orienteering view)."""
    segs = []
    for r, tour in enumerate(res["tours"]):
        for i in range(len(tour) - 1):
            T0 = tour[i][2] if tour[i][2] is not None else 0.0
            segs.append(dict(robot=r, i=i,
                             S=np.asarray(tour[i][1], float),
                             E=np.asarray(tour[i + 1][1], float),
                             budget=tour[i + 1][2] - T0, pois=[]))
    return segs


def _best_insertion(seg, j, pois):
    """Cheapest feasible insertion of POI j into a segment; (added_time, pos) or None.
    Feasible = path length still fits the time budget after adding one more scan."""
    pts = [seg["S"]] + [np.asarray(pois[t], float) for t in seg["pois"]] + [seg["E"]]
    allow = V * (seg["budget"] - (len(seg["pois"]) + 1) * SCAN)
    if allow < 0:
        return None
    base, p = _path_len(pts), np.asarray(pois[j], float)
    best = None
    for pos in range(len(pts) - 1):
        L = _path_len(pts[:pos + 1] + [p] + pts[pos + 1:])
        if L <= allow + 1e-6:
            cost = SCAN + (L - base) / V
            if best is None or cost < best[0]:
                best = (cost, pos)
    return best


def _cheapest_fill(segs, pois, unassigned):
    """Global cheapest-insertion: repeatedly insert the POI/segment/position with
    the smallest added time, across ALL segments. For UNIT-prize coverage this
    packs the most POIs (cheap insertions leave the most budget for others) and
    beats per-segment greedy by removing the round-order bias."""
    while True:
        best = None                          # (cost, j, si, pos)
        for j in list(unassigned):
            for si, seg in enumerate(segs):
                b = _best_insertion(seg, j, pois)
                if b and (best is None or b[0] < best[0]):
                    best = (b[0], j, si, b[1])
        if best is None:
            break
        _, j, si, pos = best
        segs[si]["pois"].insert(pos, j)
        unassigned.discard(j)


def _poi_marginal_cost(seg, idx, pois):
    """Detour+scan time POI at position idx contributes to its segment."""
    pts = [seg["S"]] + [np.asarray(pois[t], float) for t in seg["pois"]] + [seg["E"]]
    p, a, b = pts[idx + 1], pts[idx], pts[idx + 2]
    return SCAN + (dist(a, p) + dist(p, b) - dist(a, b)) / V


def _extract_seg_pois(res, pois):
    """Map (robot, segment-index) -> ordered POI ids from a built (greedy) plan."""
    lut = {(round(float(p[0]), 2), round(float(p[1]), 2)): j
           for j, p in enumerate(pois)}
    out = {}
    for r, tour in enumerate(res["tours"]):
        anchors = [i for i, nd in enumerate(tour) if nd[0] in ("start", "meet")]
        for k in range(len(anchors) - 1):
            lst = []
            for t in range(anchors[k] + 1, anchors[k + 1]):
                key = (round(float(tour[t][1][0]), 2), round(float(tour[t][1][1]), 2))
                if key in lut:
                    lst.append(lut[key])
            out[(r, k)] = lst
    return out


def assign_global(segs, pois, rng, init_map=None, iters=40, destroy_frac=0.25):
    """ALNS-lite for max coverage. Incumbent = best of {the greedy plan (via
    init_map) and a from-scratch global-cheapest build}, so it can never do
    worse than greedy. Then destroy (worst/random) + cheapest repair, keeping
    only improvements — freeing budget from pricey POIs to fit more cheap ones."""
    cov = lambda: sum(len(s["pois"]) for s in segs)
    snap = lambda: [list(s["pois"]) for s in segs]

    def restore(snp):
        for s, pl in zip(segs, snp):
            s["pois"] = list(pl)

    seeds = []
    if init_map is not None:                 # greedy plan as a seed
        for s in segs:
            s["pois"] = list(init_map.get((s["robot"], s["i"]), []))
        seeds.append((cov(), snap()))
    for s in segs:                           # from-scratch cheapest build
        s["pois"] = []
    _cheapest_fill(segs, pois, set(range(len(pois))))
    seeds.append((cov(), snap()))
    seeds.sort(key=lambda x: -x[0])
    bestc, best = seeds[0]
    restore(best)
    for it in range(iters):
        pool = [(si, k, j) for si, s in enumerate(segs) for k, j in enumerate(s["pois"])]
        if not pool:
            break
        nrem = max(1, int(destroy_frac * len(pool)))
        if it % 2 == 0:                      # worst removal: drop priciest POIs
            scored = sorted(pool, key=lambda x: -_poi_marginal_cost(segs[x[0]], x[1], pois))
            rem = {j for _, _, j in scored[:nrem]}
        else:                                # random removal
            rem = {j for _, _, j in rng.sample(pool, nrem)}
        for s in segs:
            s["pois"] = [j for j in s["pois"] if j not in rem]
        un = set(range(len(pois))) - {j for s in segs for j in s["pois"]}
        _cheapest_fill(segs, pois, un)
        c = cov()
        if c > bestc:
            best, bestc = snap(), c
        else:
            restore(best)
    restore(best)
    return bestc


def rebuild_smart_tours(res, segs, pois):
    by_robot = {}
    for seg in segs:
        by_robot.setdefault(seg["robot"], []).append(seg)
    for r in by_robot:
        by_robot[r].sort(key=lambda s: s["i"])
    for r, tour in enumerate(res["tours"]):
        new = [tour[0]]
        for seg in by_robot.get(r, []):
            for j in seg["pois"]:
                new.append(("poi", np.asarray(pois[j], float), None))
            new.append(tour[seg["i"] + 1])
        res["tours"][r] = new


# --- metrics -----------------------------------------------------------------
def tour_budget_stats(res):
    """Coverage + productive/padding time computed straight from the tours, so
    the numbers are correct for ANY assignment method (greedy or smart)."""
    covered = 0
    prod = pad = scan_time = 0.0
    for tour in res["tours"]:
        anchors = [i for i, nd in enumerate(tour) if nd[0] in ("start", "meet")]
        for k in range(len(anchors) - 1):
            i0, i1 = anchors[k], anchors[k + 1]
            n = i1 - i0 - 1
            covered += n
            pts = [tour[t][1] for t in range(i0, i1 + 1)]
            T0 = tour[i0][2] if tour[i0][2] is not None else 0.0
            budget = tour[i1][2] - T0
            used = n * SCAN + _path_len(pts) / V
            prod += used
            pad += max(0.0, budget - used)
            scan_time += n * SCAN
    return covered, prod, pad, scan_time


def summarize(res):
    M = res["meetings"]
    total = len(M)
    zero = sum(1 for m in M if m["feasible"])
    wait_total = sum(m["wait"] for m in M)
    waited = [m for m in M if not m["feasible"]]
    covered, prod, pad, scan_time = tour_budget_stats(res)
    travel_prod = prod - scan_time if prod >= scan_time else 0.0
    N = len(res["pois"])
    # per-robot detection cadence: gap between consecutive meetings
    worst_gap = 0.0
    worst_robot = -1
    for r in range(res["K"]):
        ts = [0.0] + sorted(res["meet_times"][r])
        for i in range(1, len(ts)):
            g = ts[i] - ts[i - 1]
            if g > worst_gap:
                worst_gap, worst_robot = g, r
    return dict(total=total, zero=zero, wait_total=wait_total,
                n_waited=len(waited), prod=prod, pad=pad, scan_time=scan_time,
                travel_prod=travel_prod, covered=covered, N=N,
                worst_gap=worst_gap, worst_robot=worst_robot,
                byes=sum(res["byes"]), delta=res["delta"])


def report(res, s):
    print("=" * 68)
    print("BYPRODUCT-MEETING MESH  --  time-balance feasibility probe")
    print("=" * 68)
    print(f"robots={res['K']}  POIs={s['N']}  rounds={res['num_rounds']}  "
          f"cadence Δ={s['delta']:.0f}s  ε={EPS:.0f}s  V={V}  SCAN={SCAN:.0f}s")
    print("-" * 68)
    print(f"scheduled meetings          : {s['total']}")
    print(f"  ZERO-WAIT (disks overlap) : {s['zero']}/{s['total']} "
          f"({100.0*s['zero']/max(1,s['total']):.0f}%)")
    print(f"  wait-forced (disks apart) : {s['n_waited']}   "
          f"total residual wait = {s['wait_total']:.0f}s")
    if s['byes']:
        print(f"  byes (odd-K cadence gaps) : {s['byes']}")
    print("-" * 68)
    print(f"POI coverage                : {s['covered']}/{s['N']} "
          f"({100.0*s['covered']/max(1,s['N']):.0f}%)   "
          f"leftover={s['N']-s['covered']}")
    seg_time = s['prod'] + s['pad']
    print(f"segment time budget         : productive={s['prod']:.0f}s  "
          f"padding(travel-to-time)={s['pad']:.0f}s  "
          f"({100.0*s['pad']/max(1e-9,seg_time):.0f}% padding)")
    print(f"  of productive: scans={s['scan_time']:.0f}s  "
          f"useful-travel={s['travel_prod']:.0f}s")
    gap = s['worst_gap']
    ok = gap <= s['delta'] + EPS + NUM_TOL
    print("-" * 68)
    print(f"detection cadence           : worst inter-meeting gap = {gap:.0f}s "
          f"(robot U{s['worst_robot']})  "
          f"{'OK  ≤ Δ+ε' if ok else 'BROKEN  > Δ+ε'}")
    print("-" * 68)
    # a representative balance line (feasible meeting, prefer nA != nB)
    reps = [m for m in res["meetings"] if m["feasible"]]
    rep = next((m for m in reps if m["nA"] != m["nB"]), reps[0] if reps else None)
    if rep is not None:
        print("example time-balance (the real math), meeting "
              f"U{rep['a']}–U{rep['b']} @ round {rep['round']}:")
        print(f"  U{rep['a']}: {rep['nA']} POIs  travel={rep['travelA']:.0f} m "
              f"→ t = {rep['tA']:.1f} s")
        print(f"  U{rep['b']}: {rep['nB']} POIs  travel={rep['travelB']:.0f} m "
              f"→ t = {rep['tB']:.1f} s")
        print(f"  |Δt| = {abs(rep['tA']-rep['tB']):.3f} s   "
              f"(≤ ε ⇒ they meet)   "
              f"check: {rep['nB']-rep['nA']} extra POIs ↔ "
              f"{V*(rep['nB']-rep['nA'])*SCAN:.0f} m travel")
    print("=" * 68)
    verdict(res, s)


def verdict(res, s):
    print("VERDICT")
    if s['zero'] == s['total'] and s['byes'] == 0:
        print("  ✔ every scheduled meeting constructs at ZERO WAIT purely by")
        print("    balancing POI-count against distance. Meetings ARE a")
        print("    byproduct of the path plan here -- no artificial loiter.")
    elif s['zero'] >= 0.7 * s['total']:
        print(f"  ~ most meetings ({100.0*s['zero']/max(1,s['total']):.0f}%) "
              "construct at zero wait; the rest need the field/cadence")
        print("    relaxed -- rotating partners drift too far apart for one Δ.")
    else:
        print("  ✘ this regime is too spread for cadence Δ: many pairs can't")
        print("    reach a common point in time -> waiting dominates. Either")
        print("    shrink Δ-travel (slower rotation / nearer partners) or accept")
        print("    dedicated detour-meetings for the gaps.")
    print(f"  deferred cost (you said tune later): {100.0*s['pad']/max(1e-9,s['prod']+s['pad']):.0f}% "
          "of segment time is pad-travel just to land on the clock.")
    print("=" * 68)


# --- turn minimisation (route-swap at rendezvous) ---------------------------
def _ang(u, v):
    """Turn angle in degrees between motion vectors u (in) and v (out):
    0 = go straight, 180 = U-turn."""
    nu = math.hypot(u[0], u[1]); nv = math.hypot(v[0], v[1])
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = (u[0] * v[0] + u[1] * v[1]) / (nu * nv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def turning_stats(tours, uturn_deg=UTURN_DEG):
    total = 0.0; nu = 0; nv = 0
    for tour in tours:
        pts = [w for (_, w, _) in tour]
        for i in range(1, len(pts) - 1):
            u = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
            v = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            a = _ang(u, v)
            total += a; nv += 1
            if a >= uturn_deg:
                nu += 1
    return dict(total_deg=total, n_uturns=nu, nverts=nv,
                mean_deg=total / max(1, nv))


def _score_angles(angles):
    """Lexicographic turn score: (#sharp turns, sum of squared angles). Lower is
    better -- minimises U-turns first, then overall sharpness."""
    return (sum(1 for a in angles if a >= UTURN_DEG), sum(a * a for a in angles))


def _turn_score(pts):
    angs = []
    for i in range(1, len(pts) - 1):
        u = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        v = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        angs.append(_ang(u, v))
    return _score_angles(angs)


def _better(a, b):
    """True if score tuple a is meaningfully better (lower) than b."""
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[1] < b[1] - 1e-6


def _meet_holders(tours, T, m):
    """Robots (with the node index) whose tour currently holds meeting (T, m)."""
    out = []
    for r, tour in enumerate(tours):
        for idx, node in enumerate(tour):
            if node[0] == "meet" and abs(node[2] - T) < 1e-6 \
                    and abs(node[1][0] - m[0]) < 1e-3 \
                    and abs(node[1][1] - m[1]) < 1e-3:
                out.append((r, idx))
                break
    return out


def optimize_turns(res):
    """At each rendezvous (in time order) the two robots may SWAP their onward
    routes -- both are at the same point at the same time, so exchanging the
    tour suffixes is space- and time-consistent and every meeting downstream
    still has exactly two boats. Swap whenever it lowers total turning."""
    tours = res["tours"]
    meets, seen = [], set()
    for tour in tours:
        for node in tour:
            if node[0] == "meet":
                key = (round(node[2], 3), round(node[1][0], 2), round(node[1][1], 2))
                if key not in seen:
                    seen.add(key)
                    meets.append((node[2], np.asarray(node[1], float)))
    meets.sort(key=lambda x: x[0])
    swaps = 0
    for (T, m) in meets:
        h = _meet_holders(tours, T, m)
        if len(h) != 2:
            continue
        (a, ia), (b, ib) = h
        if ia == 0 or ib == 0 or ia + 1 >= len(tours[a]) or ib + 1 >= len(tours[b]):
            continue
        arr_a = (m[0] - tours[a][ia - 1][1][0], m[1] - tours[a][ia - 1][1][1])
        arr_b = (m[0] - tours[b][ib - 1][1][0], m[1] - tours[b][ib - 1][1][1])
        an = tours[a][ia + 1][1]; bn = tours[b][ib + 1][1]
        dep_a = (an[0] - m[0], an[1] - m[1])
        dep_b = (bn[0] - m[0], bn[1] - m[1])
        noswap = _score_angles([_ang(arr_a, dep_a), _ang(arr_b, dep_b)])
        swap = _score_angles([_ang(arr_a, dep_b), _ang(arr_b, dep_a)])
        if _better(swap, noswap):
            sa, sb = tours[a][ia + 1:], tours[b][ib + 1:]
            tours[a] = tours[a][:ia + 1] + sb
            tours[b] = tours[b][:ib + 1] + sa
            swaps += 1
    return swaps


# --- turn-aware within-segment ordering -------------------------------------
def _path_turn(pts):
    tot = 0.0
    for i in range(1, len(pts) - 1):
        u = (pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        v = (pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        tot += _ang(u, v)
    return tot


def _path_len(pts):
    return sum(dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def order_turn_aware(S, E, interior, max_len, p_prev=None, p_next=None):
    """Reorder the POIs of one segment (fixed endpoints S,E) to minimise turning
    while keeping path length <= max_len (so the robot still lands on the clock).
    p_prev/p_next are the fixed neighbouring points so the turns AT the anchors
    S and E are scored too. Starts from the incoming (cheapest) order + a
    sweep-by-projection candidate, then 2-opt/Or-opt. Never worse than input."""
    S = np.asarray(S, float); E = np.asarray(E, float)
    inp = [np.asarray(p, float) for p in interior]
    if len(inp) <= 1:
        return inp
    pre = [np.asarray(p_prev, float)] if p_prev is not None else []
    post = [np.asarray(p_next, float)] if p_next is not None else []

    def full(o):                       # includes anchor turns via pre/post
        return pre + [S] + list(o) + [E] + post

    def seg_len(o):                    # only the S..E span is time-budgeted
        return _path_len([S] + list(o) + [E])

    cands = [inp]
    axis = E - S; na = math.hypot(axis[0], axis[1])
    if na > 1e-9:
        ax = axis / na
        cands.append(sorted(inp, key=lambda p: (p[0] - S[0]) * ax[0]
                                                + (p[1] - S[1]) * ax[1]))
    feas = [c for c in cands if seg_len(c) <= max_len + 1e-6] or [inp]
    best = min(feas, key=lambda o: _turn_score(full(o)))
    bestc = _turn_score(full(best))
    improved, guard = True, 0
    while improved and guard < 300:
        improved, guard = False, guard + 1
        n = len(best)
        for i in range(n - 1):               # 2-opt: reverse best[i..j]
            for j in range(i + 1, n):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if seg_len(cand) <= max_len + 1e-6:
                    c = _turn_score(full(cand))
                    if _better(c, bestc):
                        best, bestc, improved = cand, c, True
        for i in range(len(best)):           # Or-opt: relocate one POI
            for j in range(len(best)):
                if i == j:
                    continue
                rest = best[:i] + best[i + 1:]
                cand = rest[:j] + [best[i]] + rest[j:]
                if seg_len(cand) <= max_len + 1e-6:
                    c = _turn_score(full(cand))
                    if _better(c, bestc):
                        best, bestc, improved = cand, c, True
    return best


def optimize_ordering(res):
    """Turn-aware reorder of every segment's interior POIs (endpoints & meeting
    times fixed → coverage and detection guarantee untouched). Per-robot safety
    guard: keep the reorder only if it does not raise that robot's U-turn count."""
    tours = res["tours"]
    for r, tour in enumerate(tours):
        anchors = [i for i, nd in enumerate(tour) if nd[0] in ("start", "meet")]
        rebuilt = [tour[anchors[0]]]
        for k in range(len(anchors) - 1):
            i0, i1 = anchors[k], anchors[k + 1]
            S, E = tour[i0][1], tour[i1][1]
            interior = [tour[t][1] for t in range(i0 + 1, i1)]
            if len(interior) >= 2:
                T0 = tour[i0][2] if tour[i0][2] is not None else 0.0
                max_len = V * ((tour[i1][2] - T0) - len(interior) * SCAN)
                p_prev = rebuilt[-2][1] if len(rebuilt) >= 2 else None
                p_next = tour[i1 + 1][1] if i1 + 1 < len(tour) else None
                interior = order_turn_aware(S, E, interior, max_len, p_prev, p_next)
            for p in interior:
                rebuilt.append(("poi", np.asarray(p, float), None))
            rebuilt.append(tour[i1])
        if turning_stats([rebuilt])["n_uturns"] <= turning_stats([tour])["n_uturns"]:
            tours[r] = rebuilt


def poi_owner(tours, pois):
    lut = {(round(float(p[0]), 2), round(float(p[1]), 2)): j
           for j, p in enumerate(pois)}
    owner = -np.ones(len(pois), int)
    for r, tour in enumerate(tours):
        for (k, w, _) in tour:
            if k == "poi":
                key = (round(float(w[0]), 2), round(float(w[1]), 2))
                if key in lut:
                    owner[lut[key]] = r
    return owner


# --- figure ------------------------------------------------------------------
def draw_mesh_panel(ax, tours, pois, meetings, K, L, title):
    owner = poi_owner(tours, pois)
    for r in range(K):
        col = PALETTE[r % len(PALETTE)]
        pts = np.array([w for (_, w, _) in tours[r]])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.6, alpha=0.5, zorder=2)
        st = tours[r][0][1]
        ax.scatter([st[0]], [st[1]], marker="s", s=95, color=col,
                   edgecolor="k", lw=0.7, zorder=5)
        ax.annotate(f"U{r}", (st[0], st[1]), textcoords="offset points",
                    xytext=(7, 7), fontsize=10, fontweight="bold", color=col)
    for j in range(len(pois)):
        r = owner[j]
        if r >= 0:
            ax.scatter([pois[j, 0]], [pois[j, 1]], s=42,
                       color=PALETTE[r % len(PALETTE)], marker="o",
                       edgecolor="k", lw=0.4, zorder=4)
        else:
            ax.scatter([pois[j, 0]], [pois[j, 1]], s=34, color="#9a9a9a",
                       marker="x", lw=1.3, zorder=3)
        ax.annotate(str(j), (pois[j, 0], pois[j, 1]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="#222", zorder=9,
                    fontweight="bold")
    for m in meetings:
        c = "#2ca02c" if m["feasible"] else "#d62728"
        ls = "-" if m["feasible"] else "--"
        ax.add_patch(Circle(m["m"], 0.018 * L, fill=False, ec=c, lw=2.0,
                            ls=ls, zorder=6))
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-0.03 * L, 1.03 * L)
    ax.set_ylim(-0.03 * L, 1.05 * L)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.15)


def render_turn_compare(tours_before, res, L, path, tb, ta, swaps):
    fig = plt.figure(figsize=(17.5, 8.6))
    gs = fig.add_gridspec(1, 2, wspace=0.14)
    axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])
    draw_mesh_panel(axL, tours_before, res["pois"], res["meetings"], res["K"], L,
                    f"CURRENT  (cheapest-insertion)\nturning {tb['total_deg']:.0f}°, "
                    f"{tb['n_uturns']} U-turns (>120°)")
    draw_mesh_panel(axR, res["tours"], res["pois"], res["meetings"], res["K"], L,
                    f"NEW  (turn-aware ordering + {swaps} rendezvous swaps)\n"
                    f"turning {ta['total_deg']:.0f}°, {ta['n_uturns']} U-turns "
                    f"(−{tb['n_uturns']-ta['n_uturns']})")
    fig.suptitle("Cutting U-turns: current method vs turn-aware "
                 "(number = POI id, colour = owning robot)",
                 fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def render_assign_compare(res_g, res_s, args, path):
    fig = plt.figure(figsize=(17.5, 8.2))
    gs = fig.add_gridspec(1, 2, wspace=0.14)
    axL = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1])
    N = len(res_g["pois"])
    cg = int((poi_owner(res_g["tours"], res_g["pois"]) >= 0).sum())
    cs = int((poi_owner(res_s["tours"], res_s["pois"]) >= 0).sum())
    draw_mesh_panel(axL, res_g["tours"], res_g["pois"], res_g["meetings"],
                    res_g["K"], args.L,
                    f"CURRENT  greedy assignment\ncoverage {cg}/{N} "
                    f"({100*cg/N:.0f}%),  {N-cg} uncovered (grey ×)")
    draw_mesh_panel(axR, res_s["tours"], res_s["pois"], res_s["meetings"],
                    res_s["K"], args.L,
                    f"SMART  assignment (Team-Orienteering + ALNS)\ncoverage "
                    f"{cs}/{N} ({100*cs/N:.0f}%),  {N-cs} uncovered  (+{cs-cg})")
    fig.suptitle("Smarter POI assignment: same mesh & guarantee, fewer "
                 "uncovered POIs\n(number = POI id, colour = owning robot)",
                 fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def render(res, s, L, path):
    # choose the most striking balance example (feasible, biggest POI mismatch);
    # if none exists (e.g. wait-dominated regime with ~0 coverage), single panel.
    reps = [m for m in res["meetings"] if m["feasible"] and m["nA"] != m["nB"]]
    rep = max(reps, key=lambda m: abs(m["nA"] - m["nB"])) if reps else None
    if rep is None:
        fig = plt.figure(figsize=(10.0, 9.6))
        ax = fig.add_subplot(1, 1, 1)
        ax2 = None
    else:
        fig = plt.figure(figsize=(17.5, 8.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.16)
        ax = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

    # ---- panel 1: full constructed mesh ----
    pois = res["pois"]
    owner = poi_owner(res["tours"], pois)   # from final tours (swap-robust)
    # tour lines + start berths
    for r in range(res["K"]):
        col = PALETTE[r % len(PALETTE)]
        pts = np.array([w for (_, w, _) in res["tours"][r]])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=col, lw=1.6, alpha=0.5, zorder=2)
        st = res["tours"][r][0][1]
        ax.scatter([st[0]], [st[1]], marker="s", s=95, color=col,
                   edgecolor="k", lw=0.7, zorder=5)
        ax.annotate(f"U{r}", (st[0], st[1]), textcoords="offset points",
                    xytext=(7, 7), fontsize=10, fontweight="bold", color=col)
    # every POI marked with its index (colour = owning robot; grey = uncovered)
    for j in range(len(pois)):
        r = owner[j]
        if r >= 0:
            col = PALETTE[r % len(PALETTE)]
            ax.scatter([pois[j, 0]], [pois[j, 1]], s=42, color=col, marker="o",
                       edgecolor="k", lw=0.4, zorder=4)
        else:
            col = "#9a9a9a"
            ax.scatter([pois[j, 0]], [pois[j, 1]], s=34, color=col, marker="x",
                       lw=1.3, zorder=3)
        ax.annotate(str(j), (pois[j, 0], pois[j, 1]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color="#222", zorder=9,
                    fontweight="bold")
    for i, m in enumerate(res["meetings"]):
        c = "#2ca02c" if m["feasible"] else "#d62728"
        style = "-" if m["feasible"] else "--"
        ring = Circle(m["m"], 0.018 * L, fill=False, ec=c, lw=2.0, ls=style, zorder=6)
        ax.add_patch(ring)
        dy = 11 if i % 2 == 0 else -17    # alternate to reduce overlap
        ax.annotate(f"U{m['a']}·U{m['b']}", (m["m"][0], m["m"][1]),
                    textcoords="offset points", xytext=(0, dy), ha="center",
                    fontsize=7, color=c, zorder=7)
    ax.set_title("Constructed byproduct-meeting mesh  "
                 "(number = POI id, colour = owning robot)\n"
                 "green ring = zero-wait meeting, red dashed = wait forced",
                 fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-0.03 * L, 1.03 * L)
    ax.set_ylim(-0.03 * L, 1.05 * L)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(alpha=0.15)
    handles = [
        Line2D([], [], marker="s", ls="", mfc="#666", mec="k", ms=9,
               label="robot start / home"),
        Line2D([], [], marker="o", ls="", mfc="#666", mec="k", ms=8,
               label="POI covered (coloured by owner)"),
        Line2D([], [], marker="x", ls="", mec="#9a9a9a", ms=8,
               label="POI uncovered"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#2ca02c", ms=11,
               label="zero-wait meeting"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#d62728", ms=11,
               label="wait-forced meeting"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9,
              borderpad=0.6)

    # ---- panel 2: zoom on one meeting showing the reachability disks ----
    if ax2 is not None:
        ca = PALETTE[rep["a"] % len(PALETTE)]
        cb = PALETTE[rep["b"] % len(PALETTE)]
        # approach paths (with POI stops) converging on the meeting point
        for wp, col, lab in ((rep["wpA"], ca, rep["a"]), (rep["wpB"], cb, rep["b"])):
            wp = np.array(wp)
            ax2.plot(wp[:, 0], wp[:, 1], "-o", color=col, lw=2.0, ms=5, zorder=4)
            ax2.scatter([wp[0, 0]], [wp[0, 1]], marker="s", s=110, color=col,
                        edgecolor="k", lw=0.8, zorder=5)
            ax2.annotate(f"U{lab} start", (wp[0, 0], wp[0, 1]),
                         textcoords="offset points", xytext=(6, 6),
                         fontsize=9, color=col, fontweight="bold")
        ax2.scatter([rep["m"][0]], [rep["m"][1]], marker="*", s=320,
                    color="#111", zorder=6)
        ax2.add_patch(Circle(rep["m"], 0.015 * L, fill=False, ec="#2ca02c",
                             lw=2.2, zorder=6))
        ax2.annotate("meeting point", (rep["m"][0], rep["m"][1]),
                     textcoords="offset points", xytext=(8, -22), fontsize=9,
                     color="#2ca02c")
        txt = (f"THE REAL MATH  (t_A = t_B)\n"
               f"U{rep['a']}: {rep['nA']} POIs, {rep['travelA']:.0f} m  →  "
               f"t = {rep['tA']:.0f} s\n"
               f"U{rep['b']}: {rep['nB']} POIs, {rep['travelB']:.0f} m  →  "
               f"t = {rep['tB']:.0f} s\n"
               f"|Δt| = {abs(rep['tA']-rep['tB']):.1f} s  (≤ ε = {EPS:.0f} s)\n"
               f"{rep['nB']-rep['nA']:+d} POIs balanced by "
               f"{V*(rep['nB']-rep['nA'])*SCAN:+.0f} m travel")
        ax2.text(0.02, 0.98, txt, transform=ax2.transAxes, va="top", ha="left",
                 fontsize=9.5, family="monospace",
                 bbox=dict(boxstyle="round", fc="#fbfbe8", ec="#999"))
        # frame tightly around the two robots + meeting point
        xs = [rep["preA"][0], rep["preB"][0], rep["m"][0]]
        ys = [rep["preA"][1], rep["preB"][1], rep["m"][1]]
        for wp in (rep["wpA"], rep["wpB"]):
            wp = np.array(wp)
            xs += list(wp[:, 0]); ys += list(wp[:, 1])
        cx = 0.5 * (max(xs) + min(xs)); cy = 0.5 * (max(ys) + min(ys))
        half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) + 45.0
        ax2.set_xlim(cx - half, cx + half)
        ax2.set_ylim(cy - half, cy + half)
        # reach-disks: shade only if they're informative (comparable to frame);
        # at generous Δ they cover everything, so just annotate that instead.
        frame = max(max(xs) - min(xs), max(ys) - min(ys)) + 1.0
        if min(rep["rA"], rep["rB"]) <= 1.6 * frame:
            ax2.add_patch(Circle(rep["preA"], rep["rA"], fill=True, fc=ca,
                                 alpha=0.10, ec=ca, lw=1.2, ls=":", zorder=1))
            ax2.add_patch(Circle(rep["preB"], rep["rB"], fill=True, fc=cb,
                                 alpha=0.10, ec=cb, lw=1.2, ls=":", zorder=1))
            sub = "shaded disks = reachable-by-t regions; m in overlap ⇒ zero wait"
        else:
            ax2.text(0.02, 0.02,
                     "reach-disks (r = V·budget) cover this whole area\n"
                     "⇒ meeting trivially feasible, zero wait",
                     transform=ax2.transAxes, va="bottom", ha="left",
                     fontsize=8.5, style="italic", color="#555")
            sub = "each robot balances POIs against distance to arrive together"
        ax2.set_title("How one meeting balances\n" + sub, fontsize=11)
        ax2.set_aspect("equal")
        ax2.grid(alpha=0.15)

    fig.suptitle("Meetings as a byproduct of path planning "
                 "— time-balance construction", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --- coverage vs Δ Pareto sweep ---------------------------------------------
def _sweep_deltas(K, N, L, horizon, deltas, seeds, method="new"):
    """Run the pipeline over (Δ × seeds) at fixed mission time H, averaging.
    Returns a dict of per-Δ metric lists."""
    R = {k: [] for k in ("delta", "det", "rounds", "cov_mean", "cov_std",
                         "pad_mean", "feas_mean", "gap_mean", "broken",
                         "uturn_mean")}
    for d in sorted(deltas):
        rounds = max(1, round(horizon / d))
        covs, pads, feas, gaps, uturns = [], [], [], [], []
        for sd in seeds:
            homes, pois = make_scenario(K, N, L, sd)
            res = construct(homes, pois, K, rounds, d)
            if method == "new":
                optimize_ordering(res)
                optimize_turns(res)
            s = summarize(res)
            covs.append(s["covered"] / max(1, N))
            pads.append(s["pad"] / max(1e-9, s["prod"] + s["pad"]))
            feas.append(s["zero"] / max(1, s["total"]))
            gaps.append(s["worst_gap"])
            uturns.append(turning_stats(res["tours"])["n_uturns"])
        R["delta"].append(d)
        R["rounds"].append(rounds)
        R["det"].append(d + EPS)
        R["cov_mean"].append(100 * float(np.mean(covs)))
        R["cov_std"].append(100 * float(np.std(covs)))
        R["pad_mean"].append(100 * float(np.mean(pads)))
        R["feas_mean"].append(100 * float(np.mean(feas)))
        R["gap_mean"].append(float(np.mean(gaps)))
        R["broken"].append(float(np.mean(gaps)) > d + EPS + 1.0)
        R["uturn_mean"].append(float(np.mean(uturns)))
    return R


def run_sweep(args):
    """For each Δ (at FIXED mission time H → rounds = H/Δ), average the full
    pipeline over several seeds. Returns {method: {metric: [...per Δ...]}}."""
    seeds = list(range(args.seed, args.seed + args.sweep_seeds))
    methods = ["current", "new"] if args.overlay else ["new"]
    out = {}
    print(f"[sweep] K={args.K} N={args.N} L={args.L} H={args.horizon:.0f}s "
          f"seeds={len(seeds)}  Δ={sorted(args.deltas)}")
    for mth in methods:
        out[mth] = _sweep_deltas(args.K, args.N, args.L, args.horizon,
                                 args.deltas, seeds, mth)
    # console table (primary = new)
    P = out.get("new", out[methods[0]])
    print("-" * 68)
    print(f"{'Δ+ε(s)':>8} {'rounds':>7} {'coverage%':>10} {'pad%':>6} "
          f"{'feas%':>6} {'Δ-guar':>7}")
    for i in range(len(P["delta"])):
        flag = "BROKEN" if P["broken"][i] else "ok"
        print(f"{P['det'][i]:>8.0f} {P['rounds'][i]:>7d} "
              f"{P['cov_mean'][i]:>7.0f}±{P['cov_std'][i]:<2.0f} "
              f"{P['pad_mean'][i]:>6.0f} {P['feas_mean'][i]:>6.0f} {flag:>7}")
    print("-" * 68)
    return out


def _knee(det, cov):
    det, cov = np.asarray(det, float), np.asarray(cov, float)
    dn = (det - det.min()) / (det.max() - det.min() + 1e-9)
    cn = (cov - cov.min()) / (cov.max() - cov.min() + 1e-9)
    return int(np.argmin(np.hypot(dn - 0.0, cn - 1.0)))   # closest to top-left


def render_sweep(out, args, path):
    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.34)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

    # ---- panel A: the Pareto frontier (coverage is method-independent) ----
    P = out.get("new", out[list(out)[0]])
    det = np.array(P["det"]); cov = np.array(P["cov_mean"]); sd = np.array(P["cov_std"])
    axA.fill_between(det, cov - sd, cov + sd, color="#1f77b4", alpha=0.15, zorder=1)
    axA.plot(det, cov, "-o", color="#1f77b4", lw=2.2, ms=6,
             label="area coverage (%)", zorder=3)
    for i, b in enumerate(P["broken"]):
        if b:
            axA.scatter([det[i]], [cov[i]], s=175, facecolors="none",
                        edgecolors="#d62728", linewidths=2.2, zorder=5)
    ki = _knee(P["det"], P["cov_mean"])
    axA.scatter([det[ki]], [cov[ki]], marker="*", s=360, color="#f59f00",
                edgecolor="k", lw=0.6, zorder=6)
    axA.annotate(f"knee: Δ={P['delta'][ki]:.0f}s\n"
                 f"detect ≤ {det[ki]:.0f}s → {cov[ki]:.0f}% cover",
                 (det[ki], cov[ki]), textcoords="offset points", xytext=(14, -36),
                 fontsize=9, bbox=dict(boxstyle="round", fc="#fff8e1", ec="#f59f00"))
    axA2 = axA.twinx()   # overhead that drives the bend, on a second axis
    axA2.plot(det, P["pad_mean"], "--D", color="#e8590c", lw=1.6, ms=5,
              label="pad-travel overhead (%)", zorder=2)
    axA2.set_ylabel("pad-travel overhead (% of segment time)", color="#e8590c")
    axA2.tick_params(axis="y", labelcolor="#e8590c")
    axA2.set_ylim(0, 105)
    handles = [Line2D([], [], color="#1f77b4", marker="o", label="area coverage (%)"),
               Line2D([], [], color="#e8590c", ls="--", marker="D",
                      label="pad-travel overhead (%)"),
               Line2D([], [], marker="*", ls="", color="#f59f00", mec="k", ms=13,
                      label="knee (best coverage / detection)")]
    if any(P["broken"]):
        handles.append(Line2D([], [], marker="o", ls="", mfc="none",
                              mec="#d62728", ms=11, label="Δ+ε broken (wait regime)"))
    axA.legend(handles=handles, loc="center right", fontsize=8.5)
    axA.set_xlabel("detection latency  Δ + ε  (s)   →  slower detection")
    axA.set_ylabel("area coverage (%)", color="#1f77b4")
    axA.tick_params(axis="y", labelcolor="#1f77b4")
    axA.set_title("Coverage vs detection-latency Pareto frontier\n"
                  f"K={args.K} robots, N={args.N} POIs, fixed mission H="
                  f"{args.horizon:.0f}s, {args.sweep_seeds} seeds (band = ±1σ)",
                  fontsize=11)
    axA.grid(alpha=0.18); axA.set_ylim(0, 105)

    # ---- panel B: where the new method wins -- U-turns vs Δ ----
    styles = {"current": ("#9aa0a6", "current (cheapest-insertion)"),
              "new": ("#1f77b4", "new (turn-aware + swaps)")}
    for mth in ("current", "new"):
        if mth in out:
            R = out[mth]
            col, lab = styles[mth]
            axB.plot(R["delta"], R["uturn_mean"], "-o", color=col, lw=2.0,
                     ms=5, label=lab)
    axB.set_xlabel("rendezvous cadence  Δ  (s)")
    axB.set_ylabel("mean U-turns per plan (turn > 120°)")
    axB.set_title("Where the new method wins:\nfewer U-turns at every cadence",
                  fontsize=11)
    axB.legend(fontsize=8.5, loc="upper right"); axB.grid(alpha=0.18)
    axB.set_ylim(bottom=0)

    fig.suptitle("Rendezvous mesh: the detection-speed vs coverage trade-off",
                 fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --- K-scaling sweep (how many robots for this map) --------------------------
def run_kscan(args):
    """Sweep team size K (fixed survey N, field L). For each K, compute the
    coverage-vs-detection frontier. Returns {K: R}."""
    seeds = list(range(args.seed, args.seed + args.sweep_seeds))
    print(f"[kscan] N={args.N} L={args.L} H={args.horizon:.0f}s "
          f"seeds={len(seeds)}  K={args.klist}  Δ={sorted(args.deltas)}")
    kdata = {}
    for K in args.klist:
        kdata[K] = _sweep_deltas(K, args.N, args.L, args.horizon,
                                 args.deltas, seeds, "new")
    # console: coverage at each detection budget, per K
    dets = kdata[args.klist[0]]["det"]
    print("-" * (14 + 8 * len(args.klist)))
    print("coverage(%) by team size K, per detection budget Δ+ε:")
    print(f"{'Δ+ε(s)':>8} " + " ".join(f"K={K:<4}" for K in args.klist))
    for i in range(len(dets)):
        row = " ".join(f"{kdata[K]['cov_mean'][i]:>5.0f}" for K in args.klist)
        print(f"{dets[i]:>8.0f} {row}")
    print("-" * (14 + 8 * len(args.klist)))
    return kdata


def render_kscan(kdata, args, path):
    Ks = sorted(kdata)
    cmap = plt.get_cmap("viridis")(np.linspace(0.12, 0.86, len(Ks)))
    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.24)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

    # ---- panel A: family of coverage-vs-detection frontiers, one per K ----
    for K, c in zip(Ks, cmap):
        R = kdata[K]
        axA.plot(R["det"], R["cov_mean"], "-o", color=c, lw=2.0, ms=5,
                 label=f"K={K} robots", zorder=3)
        for i, b in enumerate(R["broken"]):
            if b:
                axA.scatter([R["det"][i]], [R["cov_mean"][i]], s=150,
                            facecolors="none", edgecolors="#d62728",
                            linewidths=2.0, zorder=5)
    axA.set_xlabel("detection latency  Δ + ε  (s)   →  slower detection")
    axA.set_ylabel("area coverage (%)")
    axA.set_title("Family of frontiers: more robots push the curve outward\n"
                  f"fixed survey N={args.N} POIs, field {args.L:.0f} m, "
                  f"H={args.horizon:.0f}s, {args.sweep_seeds} seeds",
                  fontsize=11)
    axA.grid(alpha=0.18); axA.set_ylim(0, 105)
    axA.legend(loc="lower right", fontsize=8.5)

    # ---- panel B: coverage vs K at a few detection budgets (scaling curve) ----
    deltas = kdata[Ks[0]]["delta"]
    targets = [deltas[0], deltas[len(deltas) // 2], deltas[-1]]
    tstyles = ["-o", "-s", "-^"]
    tcols = ["#d6336c", "#1c7ed6", "#2f9e44"]
    for dt, st, tc in zip(targets, tstyles, tcols):
        di = deltas.index(dt)
        covs = [kdata[K]["cov_mean"][di] for K in Ks]
        axB.plot(Ks, covs, st, color=tc, lw=2.0, ms=6,
                 label=f"detect ≤ {dt + EPS:.0f}s  (Δ={dt:.0f})")
    axB.set_xlabel("number of robots  K")
    axB.set_ylabel("area coverage (%)")
    axB.set_title("Scaling curve: coverage vs team size\n"
                  "(gaps shrink → diminishing returns)", fontsize=11)
    axB.set_xticks(Ks)
    axB.grid(alpha=0.18); axB.set_ylim(0, 105)
    axB.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("K-scaling: how many robots does this survey need?",
                 fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --- self tests --------------------------------------------------------------
def run_dry():
    print("[dry_run] self-tests")
    for (K, N, seed, rounds) in [(4, 40, 7, 4), (4, 60, 3, 6),
                                 (6, 80, 11, 5), (3, 30, 1, 3)]:
        starts, pois = make_scenario(K, N, 600.0, seed)
        res = construct(starts, pois, K, rounds)
        s = summarize(res)
        # 1) feasible meetings really are simultaneous
        for m in res["meetings"]:
            if m["feasible"]:
                assert abs(m["tA"] - m["tB"]) <= EPS + 1e-3, \
                    f"feasible meeting not simultaneous: {abs(m['tA']-m['tB'])}"
                assert m["wait"] <= 1e-6, "feasible meeting has wait"
        # 2) POIs never double-assigned (assign_pois flips the shared flag)
        allocated = sum(m["nA"] + m["nB"] for m in res["meetings"])
        assert allocated == int(res["assigned"].sum()), "POI double count"
        assert allocated <= N, "assigned more POIs than exist"
        # 3) per-robot timelines monotone non-decreasing
        for r in range(K):
            ts = [t for (_, _, t) in res["tours"][r] if t is not None]
            assert all(ts[i] <= ts[i + 1] + 1e-6 for i in range(len(ts) - 1)), \
                "non-monotone timeline"
        # 4) arrival identity: t = t0 + n*SCAN + travel/V matches stored T
        for m in res["meetings"]:
            assert abs(m["tA"] - m["T"]) <= 1e-3, "arrival identity A broken"
            assert abs(m["tB"] - m["T"]) <= 1e-3, "arrival identity B broken"
        # 5) turn-aware ordering + route-swap: mesh-preserving & non-worsening
        cov0 = int((poi_owner(res["tours"], pois) >= 0).sum())
        t0 = turning_stats(res["tours"])
        optimize_ordering(res)
        t1 = turning_stats(res["tours"])
        assert t1["n_uturns"] <= t0["n_uturns"], "ordering added U-turns"
        assert int((poi_owner(res["tours"], pois) >= 0).sum()) == cov0, \
            "ordering changed coverage"
        # ordering keeps segments feasible: arrival at each meeting still <= T
        for tour in res["tours"]:
            anchors = [i for i, nd in enumerate(tour) if nd[0] in ("start", "meet")]
            for k in range(len(anchors) - 1):
                i0, i1 = anchors[k], anchors[k + 1]
                pts = [tour[t][1] for t in range(i0, i1 + 1)]
                n = i1 - i0 - 1
                T0 = tour[i0][2] if tour[i0][2] is not None else 0.0
                dur = _path_len(pts) / V + n * SCAN
                assert dur <= (tour[i1][2] - T0) + 1e-3, "ordering broke a deadline"
        tb = turning_stats(res["tours"])
        swaps = optimize_turns(res)
        ta = turning_stats(res["tours"])
        assert ta["n_uturns"] <= tb["n_uturns"], "swaps added U-turns"
        assert int((poi_owner(res["tours"], pois) >= 0).sum()) == cov0, \
            "swap changed coverage"
        keys = {(round(nd[2], 3), round(nd[1][0], 2), round(nd[1][1], 2))
                for tour in res["tours"] for nd in tour if nd[0] == "meet"}
        for (T3, mx, my) in keys:
            cnt = sum(1 for tour in res["tours"] for nd in tour
                      if nd[0] == "meet" and abs(nd[2] - T3) < 1e-3
                      and abs(nd[1][0] - mx) < 1e-2 and abs(nd[1][1] - my) < 1e-2)
            assert cnt == 2, f"meeting has {cnt} holders after swap"
        for r in range(K):
            ts = [t for (_, _, t) in res["tours"][r] if t is not None]
            assert all(ts[i] <= ts[i + 1] + 1e-6 for i in range(len(ts) - 1)), \
                "non-monotone timeline after swap"
        print(f"  ok  K={K} N={N} seed={seed} rounds={rounds}: "
              f"zero-wait {s['zero']}/{s['total']}, cover {s['covered']}/{N}, "
              f"U-turns {t0['n_uturns']}→{ta['n_uturns']} "
              f"(order+{swaps} swaps)")
    # smart assignment: never covers fewer than greedy, mesh preserved
    hs, ps = make_scenario(4, 50, 600.0, 9)
    g = construct(hs, ps, 4, 5, assign="greedy")
    sm = construct(hs, ps, 4, 5, assign="smart")
    cg = int((poi_owner(g["tours"], ps) >= 0).sum())
    cs = int((poi_owner(sm["tours"], ps) >= 0).sum())
    assert cs >= cg, f"smart ({cs}) covered fewer than greedy ({cg})"
    keys = {(round(nd[2], 3), round(nd[1][0], 2), round(nd[1][1], 2))
            for tour in sm["tours"] for nd in tour if nd[0] == "meet"}
    for (T3, mx, my) in keys:
        cnt = sum(1 for tour in sm["tours"] for nd in tour if nd[0] == "meet"
                  and abs(nd[2] - T3) < 1e-3 and abs(nd[1][0] - mx) < 1e-2
                  and abs(nd[1][1] - my) < 1e-2)
        assert cnt == 2, f"smart meeting has {cnt} holders"
    for r in range(4):
        ts = [t for (_, _, t) in sm["tours"][r] if t is not None]
        assert all(ts[i] <= ts[i + 1] + 1e-6 for i in range(len(ts) - 1)), \
            "smart timeline non-monotone"
    print(f"  ok  smart assignment: greedy {cg} → smart {cs} cover (≥, mesh intact)")
    print("[dry_run] all self-tests passed")


def build_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, default=4, help="number of robots")
    p.add_argument("--N", type=int, default=40, help="number of POIs")
    p.add_argument("--L", type=float, default=600.0, help="field size (m)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rounds", type=int, default=4, help="rendezvous rounds")
    p.add_argument("--delta", type=float, default=DELTA,
                   help="rendezvous cadence (s); shrink to force wait-regime")
    p.add_argument("--turn_aware", action=argparse.BooleanOptionalAction,
                   default=True, help="turn-aware within-segment POI ordering")
    p.add_argument("--optimize_turns", action=argparse.BooleanOptionalAction,
                   default=True, help="swap onward routes at meetings to cut U-turns")
    p.add_argument("--turn_compare", action=argparse.BooleanOptionalAction,
                   default=True, help="also write a before/after turning figure")
    p.add_argument("--out", default=os.path.join(OUT, "byproduct_mesh.png"))
    p.add_argument("--dry_run", action="store_true")
    # coverage-vs-Δ Pareto sweep
    p.add_argument("--sweep", action="store_true",
                   help="run the coverage-vs-detection-latency Pareto sweep")
    p.add_argument("--deltas", type=float, nargs="+",
                   default=[150, 200, 250, 300, 400, 500, 700, 900, 1200],
                   help="Δ values to sweep (s)")
    p.add_argument("--horizon", type=float, default=2400.0,
                   help="fixed total mission time H (s); rounds = H/Δ")
    p.add_argument("--sweep_seeds", type=int, default=5)
    p.add_argument("--overlay", action=argparse.BooleanOptionalAction, default=True,
                   help="overlay current vs new method on the frontier")
    p.add_argument("--sweep_out", default=os.path.join(OUT, "pareto_sweep.png"))
    # K-scaling sweep
    p.add_argument("--kscan", action="store_true",
                   help="sweep team size K (fixed survey N) — frontiers + scaling")
    p.add_argument("--klist", type=int, nargs="+", default=[2, 4, 6, 8],
                   help="robot counts to sweep")
    p.add_argument("--kscan_out", default=os.path.join(OUT, "kscaling.png"))
    # smart assignment
    p.add_argument("--assign", choices=["greedy", "smart"], default="greedy",
                   help="POI→segment assignment: greedy or smart (TOP + ALNS)")
    p.add_argument("--assign_compare", action="store_true",
                   help="render greedy vs smart assignment side by side")
    p.add_argument("--assign_out", default=os.path.join(OUT, "assign_compare.png"))
    return p


def main():
    args = build_args().parse_args()
    if args.dry_run:
        run_dry()
        return
    if args.sweep:
        out = run_sweep(args)
        path = render_sweep(out, args, os.path.abspath(args.sweep_out))
        print(f"[figure] {path}")
        return
    if args.kscan:
        kdata = run_kscan(args)
        path = render_kscan(kdata, args, os.path.abspath(args.kscan_out))
        print(f"[figure] {path}")
        return
    if args.assign_compare:
        homes, pois = make_scenario(args.K, args.N, args.L, args.seed)
        rg = construct(homes, pois, args.K, args.rounds, args.delta, "greedy")
        rs = construct(homes, pois, args.K, args.rounds, args.delta, "smart")
        for r in (rg, rs):
            optimize_ordering(r); optimize_turns(r)
        path = render_assign_compare(rg, rs, args, os.path.abspath(args.assign_out))
        cg = int((poi_owner(rg["tours"], pois) >= 0).sum())
        cs = int((poi_owner(rs["tours"], pois) >= 0).sum())
        print(f"[assign] greedy {cg}/{args.N}  ->  smart {cs}/{args.N}  "
              f"(+{cs - cg} POIs)")
        print(f"[figure] {path}")
        return
    homes, pois = make_scenario(args.K, args.N, args.L, args.seed)
    res = construct(homes, pois, args.K, args.rounds, args.delta, args.assign)
    tours_current = copy.deepcopy(res["tours"])   # current method = cheapest-insertion
    t0 = turning_stats(res["tours"])
    if args.turn_aware:
        optimize_ordering(res)
    t1 = turning_stats(res["tours"])
    swaps = optimize_turns(res) if args.optimize_turns else 0
    t2 = turning_stats(res["tours"])
    s = summarize(res)
    report(res, s)
    print("turning (U-turn = turn > 120°):")
    print(f"  current (cheapest-insertion) : total {t0['total_deg']:.0f}°, "
          f"mean {t0['mean_deg']:.0f}°/vtx, U-turns {t0['n_uturns']}/{t0['nverts']}")
    if args.turn_aware:
        print(f"  + turn-aware ordering        : total {t1['total_deg']:.0f}°, "
              f"mean {t1['mean_deg']:.0f}°/vtx, U-turns {t1['n_uturns']}/{t1['nverts']}")
    if args.optimize_turns:
        print(f"  + rendezvous route-swap ({swaps:2d})   : total {t2['total_deg']:.0f}°, "
              f"mean {t2['mean_deg']:.0f}°/vtx, U-turns {t2['n_uturns']}/{t2['nverts']}")
    red = 100.0 * (t0['total_deg'] - t2['total_deg']) / max(1e-9, t0['total_deg'])
    print(f"  NEW vs CURRENT               : −{red:.0f}% total turning, "
          f"{t0['n_uturns'] - t2['n_uturns']} fewer U-turns")
    print("=" * 68)
    path = render(res, s, args.L, os.path.abspath(args.out))
    print(f"[figure] {path}")
    if args.turn_compare:
        cmp_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                "byproduct_turns.png")
        render_turn_compare(tours_current, res, args.L, cmp_path, t0, t2, swaps)
        print(f"[figure] {cmp_path}")


if __name__ == "__main__":
    main()
