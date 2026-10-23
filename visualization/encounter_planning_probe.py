#!/usr/bin/env python3
"""
FEASIBILITY PROBE for "path planning for asynchronous comms" (rendezvous idea,
docs/rendezvous_allocation_idea.md). Tests the make-or-break questions BEFORE
any method is claimed:

  Q1 NATURAL DENSITY   Do coverage-optimal routes already produce close-range
                       (<R) encounters "for free"? (Answer: NO - efficient
                       allocation partitions space; robots never re-meet.)
  Q2 COST OF A BOUND   Plan pairwise rendezvous INTO the schedule so every
                       robot gets a close-range liveness check at least every
                       DELTA seconds. Compare two planners:
                         A) unconstrained cheapest-insertion allocation, then
                            scheduled midpoint rendezvous with the nearest
                            buddy (allocation ignores comms);
                         B) buddy-constrained allocation (adjacent territory
                            for paired robots), then the same rendezvous.
                       Metric: extra travel % / extra makespan % vs DELTA.
                       Hypothesis: encounter-aware ALLOCATION (B) makes the
                       bound cheap; comms-blind allocation (A) makes it dear.
  Q3a DOES IT PAY      Detection latency of a real failure on rendezvous-
                       planned routes: channel-only belief (beta = rho*eta,
                       paper formula) vs belief + missed-rendezvous evidence,
                       under short bursts (paper GE) and long outages.
  Q3b WHERE IT BREAKS  False rendezvous-misses on ALIVE robots from timing
                       drift (speed error accrued since the last sync) vs the
                       wait tolerance eps. (A false miss = a targeted-relay
                       probe / verification, NOT an instant takeover.)

World: USVs from a dock line, POIs uniform in an L x L field, constant speed,
fixed scan time per POI. Close range R = reliable link; long range = shared
Gilbert-Elliott burst channel (params consistent with unified_comparison_sim).

Run:  python3 visualization/encounter_planning_probe.py [q1|q2|q3|all]
Outputs -> results/encounter_probe/
"""

import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "results" / "encounter_probe"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- parameters
V = 2.0          # m/s USV speed
SCAN = 60.0      # s per POI
R = 20.0         # m: close range where the link is treated as reliable
HB = 1.0         # s heartbeat period
W_ETA = 5.0      # s channel-health window (paper)
DT = 1.0         # s sim step
CONTACT = 2.0    # s dwell at a rendezvous once both are present

# Gilbert-Elliott shared channel (long range), consistent with the paper sims
GE_SHORT = dict(drop_good=0.2, drop_bad=0.95, p_gb=0.15, p_bg=0.30)   # ~3 s bursts
GE_LONG = dict(drop_good=0.2, drop_bad=0.95, p_gb=0.02, p_bg=0.02)    # ~50 s outages


# ---------------------------------------------------------------- geometry
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def make_world(seed, k_robots, n_pois, L):
    rng = random.Random(seed)
    x0 = L / 2 - 5.0 * (k_robots - 1)
    starts = [(x0 + 10.0 * i, 0.0) for i in range(k_robots)]
    pois = [(rng.uniform(0.05 * L, 0.95 * L), rng.uniform(0.1 * L, 0.95 * L))
            for _ in range(n_pois)]
    return starts, pois


def route_length(start, route):
    p, tot = start, 0.0
    for q in route:
        tot += dist(p, q)
        p = q
    return tot


def cheapest_insertion_routes(starts, pois):
    """Multi-robot cheapest-insertion over the given robots/POIs."""
    routes = [[] for _ in starts]
    remaining = list(pois)
    while remaining:
        best = None
        for q in remaining:
            for r in range(len(starts)):
                seq = routes[r]
                base = route_length(starts[r], seq)
                for pos in range(len(seq) + 1):
                    cand = seq[:pos] + [q] + seq[pos:]
                    inc = route_length(starts[r], cand) - base
                    if best is None or inc < best[0]:
                        best = (inc, q, r, pos)
        _, q, r, pos = best
        routes[r].insert(pos, q)
        remaining.remove(q)
    return routes


def buddy_constrained_routes(starts, pois, k):
    """Planner B allocation: pair robots (adjacent on the dock), split the
    field into K/2 vertical bands with equal POI counts, run cheapest
    insertion for each pair inside its own band -> buddies hold adjacent
    territory, so a midpoint rendezvous is short."""
    pairs = [(2 * i, 2 * i + 1) for i in range(k // 2)]
    by_x = sorted(pois, key=lambda q: q[0])
    per = math.ceil(len(pois) / len(pairs))
    routes = [None] * k
    for gi, (a, b) in enumerate(pairs):
        band = by_x[gi * per:(gi + 1) * per]
        sub = cheapest_insertion_routes([starts[a], starts[b]], band)
        routes[a], routes[b] = sub[0], sub[1]
    return routes, pairs


def interleaved_routes(starts, pois, k):
    """Planner C allocation: pair robots, band the field as in B, but give
    each pair ONE joint cheapest-insertion path over its band and assign
    alternating POIs to the two buddies. The pair leapfrogs through shared
    territory, so a liveness check is a short pass-by of the partner's
    current scan location, not a dedicated cross-cluster trip."""
    pairs = [(2 * i, 2 * i + 1) for i in range(k // 2)]
    by_x = sorted(pois, key=lambda q: q[0])
    per = math.ceil(len(pois) / len(pairs))
    routes = [None] * k
    for gi, (a, b) in enumerate(pairs):
        band = by_x[gi * per:(gi + 1) * per]
        mid_dock = ((starts[a][0] + starts[b][0]) / 2,
                    (starts[a][1] + starts[b][1]) / 2)
        joint = cheapest_insertion_routes([mid_dock], band)[0]
        routes[a] = joint[0::2]
        routes[b] = joint[1::2]
    return routes, pairs


def nearest_centroid_pairs(starts, routes):
    """Planner A buddies: greedily pair robots whose route centroids are
    nearest (the best you can do AFTER a comms-blind allocation)."""
    cent = []
    for r, route in enumerate(routes):
        pts = route if route else [starts[r]]
        cent.append((sum(p[0] for p in pts) / len(pts),
                     sum(p[1] for p in pts) / len(pts)))
    unpaired, pairs = set(range(len(routes))), []
    while len(unpaired) > 1:
        best = None
        for a in unpaired:
            for b in unpaired:
                if a < b:
                    d = dist(cent[a], cent[b])
                    if best is None or d < best[0]:
                        best = (d, a, b)
        _, a, b = best
        pairs.append((a, b))
        unpaired -= {a, b}
    return pairs


# ------------------------------------------------- timelines with rendezvous
class Walker:
    def __init__(self, start, route):
        self.bp = [(0.0, start[0], start[1])]
        self.queue = list(route)
        self.t = 0.0
        self.pos = start

    def advance_to(self, t_target):
        """Run route work until t_target; stop mid-leg if needed."""
        while self.queue and self.t < t_target:
            q = self.queue[0]
            d = dist(self.pos, q)
            t_arr = self.t + d / V
            if t_arr > t_target:  # stop mid-leg
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


def build_pair_timelines(starts, routes, pair, delta, horizon_cap=20000.0):
    """Jointly build the two buddies' space-time timelines, inserting a
    synchronized midpoint rendezvous every DELTA seconds until both robots
    have finished their routes. The early arriver waits (sync), so the pair's
    clocks re-zero at every rendezvous. Returns (tl_a, tl_b, meetings) where
    meetings = [(t_meet, x, y)]."""
    a, b = pair
    wa, wb = Walker(starts[a], routes[a]), Walker(starts[b], routes[b])
    meetings = []
    m = 1
    while not (wa.done() and wb.done()) and m * delta < horizon_cap:
        t_m = m * delta
        wa.advance_to(t_m)
        wb.advance_to(t_m)
        if wa.done() and wb.done():
            break
        mid = ((wa.pos[0] + wb.pos[0]) / 2, (wa.pos[1] + wb.pos[1]) / 2)
        # robots only need to come within the contact range R, not co-locate:
        # each stops 0.45*R short of the midpoint (separation <= 0.9*R).
        for w in (wa, wb):
            d = dist(w.pos, mid)
            if d > 1e-9:
                back = min(0.45 * R, d)
                w.goto((mid[0] - back * (mid[0] - w.pos[0]) / d,
                        mid[1] - back * (mid[1] - w.pos[1]) / d))
        t_meet = max(wa.t, wb.t)
        wa.wait_until(t_meet)
        wb.wait_until(t_meet)
        wa.t += CONTACT
        wb.t += CONTACT
        wa.bp.append((wa.t, mid[0], mid[1]))
        wb.bp.append((wb.t, mid[0], mid[1]))
        meetings.append((t_meet, mid[0], mid[1]))
        m = int(t_meet // delta) + 1
    # drain remaining work
    wa.advance_to(horizon_cap)
    wb.advance_to(horizon_cap)
    return wa.bp, wb.bp, meetings


def build_pair_timelines_passby(starts, routes, pair, delta,
                                horizon_cap=20000.0):
    """Planner C rendezvous primitive: every DELTA seconds the designated
    visitor (alternating) diverts THROUGH its partner's current location and
    keeps going -- no stop-and-wait. The partner is usually mid-scan (a 60 s
    sitting target), so a couple of chase steps suffice; a sync wait is the
    rare fallback. Contact = passing within R."""
    a, b = pair
    wa, wb = Walker(starts[a], routes[a]), Walker(starts[b], routes[b])
    meetings = []
    m, visitor_is_a = 1, True
    while not (wa.done() and wb.done()) and m * delta < horizon_cap:
        t_m = m * delta
        wa.advance_to(t_m)
        wb.advance_to(t_m)
        if wa.done() and wb.done():
            break
        vis, tgt = (wa, wb) if visitor_is_a else (wb, wa)
        for _ in range(3):  # chase the partner's (usually static) position
            aim = tgt.pos
            vis.goto(aim)
            tgt.advance_to(vis.t)
            if dist(vis.pos, tgt.pos) <= R:
                break
        else:  # rare: partner kept moving -- force a short sync
            vis.goto(tgt.pos)
            if vis.t < tgt.t:
                vis.wait_until(tgt.t)
            else:
                tgt.wait_until(vis.t)
        t_meet = max(vis.t, tgt.t)
        meetings.append((t_meet, vis.pos[0], vis.pos[1]))
        visitor_is_a = not visitor_is_a
        m = int(t_meet // delta) + 1
    wa.advance_to(horizon_cap)
    wb.advance_to(horizon_cap)
    return wa.bp, wb.bp, meetings


def plan_mission(seed, k, n, L, planner, delta):
    """Returns (timelines, pairs, meetings_per_pair, stats). delta=None ->
    no rendezvous (baseline)."""
    starts, pois = make_world(seed, k, n, L)
    if planner == "B":
        routes, pairs = buddy_constrained_routes(starts, pois, k)
    elif planner == "C":
        routes, pairs = interleaved_routes(starts, pois, k)
    else:
        routes = cheapest_insertion_routes(starts, pois)
        pairs = nearest_centroid_pairs(starts, routes)
    tls = [None] * k
    meetings = {}
    if delta is None:
        for r in range(k):
            w_bp = [(0.0, starts[r][0], starts[r][1])]
            t, p = 0.0, starts[r]
            for q in routes[r]:
                t += dist(p, q) / V
                w_bp.append((t, q[0], q[1]))
                t += SCAN
                w_bp.append((t, q[0], q[1]))
                p = q
            tls[r] = w_bp
        for pr in pairs:
            meetings[pr] = []
    else:
        builder = (build_pair_timelines_passby if planner == "C"
                   else build_pair_timelines)
        for pr in pairs:
            ta, tb, mts = builder(starts, routes, pr, delta)
            tls[pr[0]], tls[pr[1]] = ta, tb
            meetings[pr] = mts
    total_dist = sum(_bp_dist(bp) for bp in tls)
    makespan = max(bp[-1][0] for bp in tls)
    return tls, pairs, meetings, dict(total_dist=total_dist, makespan=makespan)


def _bp_dist(bp):
    return sum(dist((bp[i][1], bp[i][2]), (bp[i + 1][1], bp[i + 1][2]))
               for i in range(len(bp) - 1))


def pos_at(bp, t):
    if t <= bp[0][0]:
        return bp[0][1], bp[0][2]
    for i in range(1, len(bp)):
        if t <= bp[i][0]:
            t0, x0, y0 = bp[i - 1]
            t1, x1, y1 = bp[i]
            if t1 == t0:
                return x1, y1
            a = (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return bp[-1][1], bp[-1][2]


def finish_time(bp):
    return bp[-1][0]


# ---------------------------------------------------------------- Q1
def extract_windows(timelines, dt=DT, r_range=R):
    T = max(finish_time(bp) for bp in timelines)
    n = len(timelines)
    windows, open_w = [], {}
    t = 0.0
    while t <= T + dt:
        pos = [pos_at(bp, t) for bp in timelines]
        for i in range(n):
            for j in range(i + 1, n):
                close = dist(pos[i], pos[j]) <= r_range
                key = (i, j)
                if close and key not in open_w:
                    open_w[key] = t
                elif not close and key in open_w:
                    windows.append((i, j, open_w.pop(key), t - dt))
        t += dt
    for (i, j), t0 in open_w.items():
        windows.append((i, j, t0, T))
    return windows


def q1_natural_density(seeds=30):
    print("=" * 72)
    print("Q1  NATURAL ENCOUNTER DENSITY (no rendezvous planned)")
    print("=" * 72)
    rows = []
    for k, n in [(4, 16), (8, 32)]:
        for L in [150, 300, 600, 1200]:
            mg, un, tot = [], 0, 0
            for s in range(seeds):
                tls, _, _, _ = plan_mission(s * 7919 + 13, k, n, L, "A", None)
                wins = extract_windows(tls)
                for r in range(k):
                    fin = finish_time(tls[r])
                    ts = sorted({0.0, fin} |
                                {t for i, j, t0, t1 in wins if r in (i, j)
                                 for t in (t0, t1) if t <= fin})
                    gaps = [b - a for a, b in zip(ts, ts[1:])]
                    mg.append(max(gaps) / fin if fin else 1.0)
                    tot += 1
                    if not any(t0 > 60.0 for i, j, t0, t1 in wins if r in (i, j)):
                        un += 1
            rows.append(dict(K=k, N=n, L=L,
                             mean_maxgap_frac=sum(mg) / len(mg),
                             frac_unchecked=un / tot))
            print(f"  K={k:2d} N={n:2d} L={L:5d} m | mean max-gap "
                  f"{rows[-1]['mean_maxgap_frac'] * 100:5.1f}% of mission | "
                  f"no mid-mission check: {rows[-1]['frac_unchecked'] * 100:5.1f}%")
    _save_csv(OUT / "q1_natural_density.csv", rows)
    return rows


# ---------------------------------------------------------------- Q2
def q2_bound_cost(seeds=20, L=600, k=4, n=16):
    print("=" * 72)
    print(f"Q2  PRICE OF A GUARANTEED CHECK EVERY DELTA (K={k}, N={n}, L={L} m)")
    print("    A = comms-blind allocation + rendezvous; B = buddy-constrained")
    print("    allocation + rendezvous. Baseline = A with no rendezvous.")
    print("=" * 72)
    deltas = [500, 300, 200, 120]
    rows, curves = [], {"A": ([], []), "B": ([], []), "C": ([], [])}
    # baselines
    base_d, base_m = [], []
    for s in range(seeds):
        _, _, _, st = plan_mission(s * 7919 + 13, k, n, L, "A", None)
        base_d.append(st["total_dist"])
        base_m.append(st["makespan"])
    for p in ["B", "C"]:
        pd = [plan_mission(s * 7919 + 13, k, n, L, p, None)[3]["total_dist"]
              for s in range(seeds)]
        print(f"  allocation-constraint penalty of {p} alone (no rendezvous): "
              f"{(sum(pd) / sum(base_d) - 1) * 100:+.1f}% travel")
    for planner in ["A", "B", "C"]:
        for delta in deltas:
            ed, em = [], []
            for s in range(seeds):
                _, _, _, st = plan_mission(s * 7919 + 13, k, n, L, planner, delta)
                ed.append((st["total_dist"] / base_d[s] - 1) * 100)
                em.append((st["makespan"] / base_m[s] - 1) * 100)
            curves[planner][0].append(sum(ed) / len(ed))
            curves[planner][1].append(sum(em) / len(em))
            rows.append(dict(planner=planner, delta=delta,
                             extra_travel_pct=curves[planner][0][-1],
                             extra_makespan_pct=curves[planner][1][-1]))
            print(f"  planner {planner} DELTA={delta:4d} s | extra travel "
                  f"{curves[planner][0][-1]:6.1f}% | extra makespan "
                  f"{curves[planner][1][-1]:6.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = {"A": "A comms-blind alloc + midpoint meets",
              "B": "B buddy bands + midpoint meets",
              "C": "C interleaved buddies + pass-by"}
    for planner, style in [("A", "o--"), ("B", "s-"), ("C", "^-")]:
        axes[0].plot(deltas, curves[planner][0], style, label=labels[planner])
        axes[1].plot(deltas, curves[planner][1], style, label=labels[planner])
    axes[0].set_ylabel("extra travel (%)")
    axes[1].set_ylabel("extra makespan (%)")
    for ax in axes:
        ax.set_xlabel("guaranteed check interval DELTA (s)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("Price of a deterministic detection bound")
    axes[1].set_title(f"(K={k}, N={n}, L={L} m, R={R:.0f} m)")
    fig.tight_layout()
    fig.savefig(OUT / "q2_bound_cost.png", dpi=130)
    plt.close(fig)
    _save_csv(OUT / "q2_bound_cost.csv", rows)
    return rows


# ---------------------------------------------------------------- Q3a
def _ge_series(rng, T, p):
    bad, out = False, []
    for _ in range(int(T) + 2):
        bad = (rng.random() < p["p_gb"]) if not bad else (rng.random() >= p["p_bg"])
        out.append(bad)
    return out


def q3_detection(seeds=200, L=600, k=6, n=24, delta=300, eps=30.0):
    """Fair two-sided comparison on rendezvous-planned (B) routes:
    channel-only belief at tau in {0.5, 0.75} -- measuring BOTH its false
    takeovers on alive robots AND its real-failure latency -- vs the
    missed-rendezvous detector (0 false alarms absent drift, latency bounded
    by DELTA+eps), vs the combination."""
    print("=" * 72)
    print("Q3a FALSE ALARMS *AND* LATENCY on rendezvous-planned routes")
    print(f"    (planner B, K={k}, N={n}, DELTA={delta}s, eps={eps}s)")
    print("=" * 72)
    # channel-only sensitivity sweep: rho timescale multiplier s (silence must
    # exceed 2sh before evidence accrues, certain at (2+4)sh) x threshold tau.
    # This is the paper's ROC methodology, re-run at LONG mission duration.
    sens = [1, 4, 16, 48]
    taus = [0.5, 0.9]
    configs = [(s, tau) for s in sens for tau in taus]
    rows = []
    for chan_name, ge, rlim in [
            ("short bursts (paper)", GE_SHORT, None),
            ("long outages (~50 s)", GE_LONG, None),
            ("range-limited (150 m)", GE_SHORT, 150.0)]:
        cstats = {c: dict(lat=[], fa=0, undet=0) for c in configs}
        rdv_lat, rdv_undet = [], 0
        seeds_used = seeds
        for s_i in range(seeds_used):
            rng = random.Random(s_i * 104729 + 7)
            tls, pairs, meetings, _ = plan_mission(s_i * 7919 + 13, k, n, L,
                                                   "B", delta)
            T = max(finish_time(bp) for bp in tls)
            victim = rng.randrange(k)
            t_f = rng.uniform(0.2, 0.6) * finish_time(tls[victim])
            vic_meets = sorted(m[0] for pr, ms in meetings.items()
                               for m in ms if victim in pr)
            nxt = [tm for tm in vic_meets if tm > t_f]
            det_rdv = (nxt[0] + eps) if nxt else None
            rdv_lat.append((det_rdv - t_f) if det_rdv is not None else T - t_f)
            if det_rdv is None:
                rdv_undet += 1
            ge_bad = _ge_series(rng, T + delta, ge)
            last_heard = [[0.0] * k for _ in range(k)]
            det = {c: None for c in configs}
            accused = {c: set() for c in configs}
            min_gate = 2 * sens[0] * HB  # below this silence nothing fires
            t = HB
            while t <= T:
                drop = ge["drop_bad"] if ge_bad[int(t)] else ge["drop_good"]
                pos = [pos_at(bp, t) for bp in tls]
                for m in range(k):
                    if m == victim and t >= t_f:
                        continue
                    heard_cnt = 0
                    for x in range(k):
                        if x == m:
                            continue
                        alive = not (x == victim and t >= t_f)
                        d = dist(pos[m], pos[x])
                        in_range = rlim is None or d <= rlim
                        if alive and (d <= R or
                                      (in_range and rng.random() >= drop)):
                            last_heard[m][x] = t
                        if t - last_heard[m][x] <= W_ETA:
                            heard_cnt += 1
                    for x in range(k):
                        if x == m:
                            continue
                        sig = t - last_heard[m][x]
                        if sig <= min_gate:
                            continue
                        x_recent = 1 if sig <= W_ETA else 0
                        eta = (heard_cnt - x_recent) / (k - 2)
                        if eta <= 0.0:
                            continue
                        for c in configs:
                            sc, tau = c
                            rho = max(0.0, min(1.0,
                                               (sig - 2 * sc * HB) / (4 * sc * HB)))
                            if rho * eta >= tau:
                                if x == victim and t > t_f:
                                    if det[c] is None:
                                        det[c] = t
                                elif x != victim:
                                    accused[c].add(x)
                t += HB
            for c in configs:
                cstats[c]["fa"] += len(accused[c])
                if det[c] is None:
                    cstats[c]["undet"] += 1
                    cstats[c]["lat"].append(T - t_f)
                else:
                    cstats[c]["lat"].append(det[c] - t_f)
        p95 = lambda xs: sorted(xs)[int(0.95 * (len(xs) - 1))]
        print(f"  --- {chan_name} (mission ~20-30 min) ---")
        for c in configs:
            st = cstats[c]
            rows.append(dict(
                channel=chan_name, detector=f"channel s={c[0]} tau={c[1]}",
                false_tk_per_trial=st["fa"] / seeds_used,
                mean_lat=sum(st["lat"]) / len(st["lat"]),
                p95_lat=p95(st["lat"]), undetected=st["undet"] / seeds_used))
            r = rows[-1]
            print(f"  channel s={c[0]:2d} tau={c[1]:.1f} | FA/trial "
                  f"{r['false_tk_per_trial']:5.2f} | latency mean "
                  f"{r['mean_lat']:6.1f}s p95 {r['p95_lat']:6.1f}s | "
                  f"undetected {r['undetected'] * 100:5.1f}%")
        rows.append(dict(
            channel=chan_name, detector="rendezvous-miss only",
            false_tk_per_trial=0.0, mean_lat=sum(rdv_lat) / len(rdv_lat),
            p95_lat=p95(rdv_lat), undetected=rdv_undet / seeds_used))
        r = rows[-1]
        print(f"  rendezvous-miss only  | FA/trial  0.00 | latency mean "
              f"{r['mean_lat']:6.1f}s p95 {r['p95_lat']:6.1f}s | "
              f"undetected {r['undetected'] * 100:5.1f}%  "
              f"(bound Δ+ε = {delta + eps:.0f}s)")

    _save_csv(OUT / "q3a_latency.csv", rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for chan_name, marker in [("short bursts (paper)", "o"),
                              ("long outages (~50 s)", "s"),
                              ("range-limited (150 m)", "d")]:
        pts = [(r["false_tk_per_trial"], r["p95_lat"]) for r in rows
               if r["channel"] == chan_name and "channel" in r["detector"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker, ls="",
                alpha=0.7, label=f"channel-only sweep, {chan_name}")
    ax.axhline(delta + eps, color="k", ls="--", lw=1)
    ax.plot([0], [next(r["p95_lat"] for r in rows
                       if r["detector"] == "rendezvous-miss only")],
            "r*", ms=14, label="rendezvous-miss (FA=0, bounded)")
    ax.set_xlabel("false takeovers / trial (alive robots accused)")
    ax.set_ylabel("p95 detection latency (s)")
    ax.set_title("Long missions: statistical silence detectors vs planned contacts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "q3a_latency.png", dpi=130)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- Q3b
def q3_drift(seeds=250, L=600, k=4, n=16, delta=300):
    print("=" * 72)
    print("Q3b FALSE RENDEZVOUS-MISSES ON ALIVE ROBOTS vs TIMING DRIFT")
    print("    speed error accrues since the LAST sync (rendezvous re-zeroes")
    print("    the pair's clocks); miss iff a robot is > eps late.")
    print("=" * 72)
    sigmas = [0.02, 0.05, 0.10]
    epss = [10.0, 30.0, 60.0]
    rows, table = [], {}
    # planned meeting spacing comes from the actual plans
    for sg in sigmas:
        for eps in epss:
            miss, nm = 0, 0
            for s in range(seeds):
                rng = random.Random(s * 6151 + 3)
                _, pairs, meetings, _ = plan_mission(s * 7919 + 13, k, n, L,
                                                     "B", delta)
                for pr, ms in meetings.items():
                    fa = max(0.6, rng.gauss(1.0, sg))
                    fb = max(0.6, rng.gauss(1.0, sg))
                    t_prev = 0.0
                    for (tm, _, _) in ms:
                        elapsed = tm - t_prev
                        ea = elapsed * (1.0 / fa - 1.0)  # >0 = late
                        eb = elapsed * (1.0 / fb - 1.0)
                        nm += 1
                        if max(ea, eb) > eps:
                            miss += 1
                        t_prev = tm  # sync re-zeroes the pair
            rate = miss / max(nm, 1)
            table[(sg, eps)] = rate
            rows.append(dict(sigma_speed=sg, eps=eps, false_miss_rate=rate,
                             meetings_per_trial=nm / seeds))
            print(f"  speed-error sigma={sg:.2f} eps={eps:4.0f}s | "
                  f"false-miss rate {rate * 100:5.1f}%  "
                  f"({nm / seeds:.1f} meetings/trial)")

    fig, ax = plt.subplots(figsize=(6, 4))
    for eps in epss:
        ax.plot(sigmas, [table[(sg, eps)] * 100 for sg in sigmas], "o-",
                label=f"tolerance eps={eps:.0f} s")
    ax.set_xlabel("per-robot speed error sigma (fraction)")
    ax.set_ylabel("false rendezvous-miss rate (%)")
    ax.set_title(f"Drift vs tolerance (DELTA={delta} s; sync re-zeroes drift)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "q3b_drift.png", dpi=130)
    plt.close(fig)
    _save_csv(OUT / "q3b_drift.csv", rows)
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


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("q1", "all"):
        q1_natural_density()
    if which in ("q2", "all"):
        q2_bound_cost()
    if which in ("q3", "all"):
        q3_detection()
        q3_drift()
