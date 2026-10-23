"""
taskalloc_coverage_first.py

A coverage-FIRST, mathematically-grounded task allocation for the rendezvous
multi-robot problem. The whole problem is task allocation; here 100% coverage is
a HARD constraint and we optimise makespan on top, then layer failure-recovery
and (next) the co-optimised rendezvous mesh.

FORMULATION (fault-tolerant min-makespan multi-robot allocation w/ rendezvous):
  decide x[i,r]∈{0,1} (POI i → robot r), the per-robot route, and a meeting plan.
  C1 (coverage, HARD):  ∀i  Σ_r x[i,r] = 1                    (every POI covered)
  C2 (recovery, HARD):  ∀ victim v, its POIs reassign to survivors with a
                        feasible completion (bounded post-failure makespan)
  C3 (detection):       ∀ robot, ∀ Δ-window a rendezvous exists → detect ≤ Δ+ε
  objective (lexicographic): coverage=100% ▸ min makespan ▸ min worst-case
                        post-failure makespan ▸ min meeting overhead

This module builds LAYER 0 (coverage+min-makespan), LAYER 1 (recovery
feasibility) and LAYER 2 (the rendezvous mesh, with the CAR-FCD correlation-
alignment constraint C4 folded in -- see the LAYER 2 banner further down).

Method (principled decomposition, NP-hard problem):
  L0: balanced compact partition via constrained k-means — Hungarian BALANCED
      assignment (exact for given centroids) inside Lloyd iterations → 100%
      coverage by construction + balanced load; per-robot open-TSP (nearest-
      neighbour + 2-opt) from a common dock; makespan-rebalancing local search.
  L1: Delaunay adjacency of regions; for each single failure reassign the
      victim's POIs to survivors and recompute makespan → worst-case resilience.
  L2: rotating-buddy meeting mesh overlaid on the L0 routes (every robot meets
      someone every Δ, so C3 holds by construction), executed forward with the
      meetings as hard time+place constraints; the C4 corridor constraint
      ρ(d_ij) ≥ ρ_min is imposed on the approach geometry and priced in makespan.

Constants mirror the rest of the pipeline.
  --dry_run  self-tests (L0/L1 and L2)
  --l2       weave the mesh on one instance and render the Layer-2 figure
  --sweep    the strength / alpha sweeps behind that figure (writes CSVs)
otherwise renders the Layer-0/1 figure.
"""

import argparse
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay

V = 2.0        # m/s cruise
SCAN = 60.0    # s per POI scan
DELTA = 300.0  # s rendezvous cadence
EPS = 30.0     # s meeting grace

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "taskalloc")
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]


# --- geometry ----------------------------------------------------------------
def dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def make_scenario(N, L, seed):
    rng = np.random.default_rng(seed)
    pois = rng.uniform(0.05 * L, 0.95 * L, size=(N, 2))
    dock = np.array([0.5 * L, 0.02 * L])       # common launch point
    return pois, dock


# --- Layer 0: balanced compact partition (100% coverage) ---------------------
def _kpp(pois, K, rng):
    idx = [int(rng.integers(len(pois)))]
    for _ in range(K - 1):
        d2 = np.min([np.sum((pois - pois[c]) ** 2, axis=1) for c in idx], axis=0)
        p = d2 / d2.sum()
        idx.append(int(rng.choice(len(pois), p=p)))
    return pois[idx].copy()


def _targets(N, K):
    base, rem = divmod(N, K)
    return [base + 1] * rem + [base] * (K - rem)


def _balanced_assign(pois, centroids, targets):
    """Exact balanced assignment for fixed centroids: each robot r gets exactly
    targets[r] POIs, minimising total squared distance (Hungarian)."""
    slots = []
    for r, t in enumerate(targets):
        slots += [r] * t
    C = np.empty((len(pois), len(slots)))
    for s, r in enumerate(slots):
        C[:, s] = np.sum((pois - centroids[r]) ** 2, axis=1)
    rows, cols = linear_sum_assignment(C)
    labels = np.empty(len(pois), int)
    for i, s in zip(rows, cols):
        labels[i] = slots[s]
    return labels


def balanced_partition(pois, K, seed=0, iters=15):
    """Constrained k-means: balanced Hungarian assignment inside Lloyd. Returns
    labels (every POI assigned → 100% coverage) and centroids."""
    rng = np.random.default_rng(seed)
    centroids = _kpp(pois, K, rng)
    targets = _targets(len(pois), K)
    labels = _balanced_assign(pois, centroids, targets)
    for _ in range(iters):
        newc = np.array([pois[labels == r].mean(0) if np.any(labels == r)
                         else centroids[r] for r in range(K)])
        lab2 = _balanced_assign(pois, newc, targets)
        if np.array_equal(lab2, labels) and np.allclose(newc, centroids):
            centroids = newc
            break
        centroids, labels = newc, lab2
    return labels, centroids


def open_tsp(dock, pts):
    """Open path from dock through pts: nearest-neighbour then 2-opt. Returns the
    visiting order (indices into pts)."""
    n = len(pts)
    if n == 0:
        return []
    unvisited = set(range(n))
    order = []
    cur = np.asarray(dock, float)
    while unvisited:
        j = min(unvisited, key=lambda k: dist(cur, pts[k]))
        order.append(j); unvisited.discard(j); cur = pts[j]
    # 2-opt on [dock, pts[order]...] with dock fixed at the front
    route = [dock] + [pts[j] for j in order]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for k in range(i + 1, len(route)):
                a, b = route[i - 1], route[i]
                c = route[k]
                d = route[k + 1] if k + 1 < len(route) else None
                old = dist(a, b) + (dist(c, d) if d is not None else 0.0)
                new = dist(a, c) + (dist(b, d) if d is not None else 0.0)
                if new < old - 1e-9:
                    route[i:k + 1] = route[i:k + 1][::-1]
                    order[i - 1:k] = order[i - 1:k][::-1]
                    improved = True
    return order


def route_len(dock, pts, order):
    if not len(order):
        return 0.0
    seq = [dock] + [pts[j] for j in order]
    return sum(dist(seq[i], seq[i + 1]) for i in range(len(seq) - 1))


def route_time(dock, pts, order, speed=None):
    return route_len(dock, pts, order) / (V if speed is None else speed) \
        + len(order) * SCAN


def baseline_makespan(alloc, cruise=None):
    """The no-rendezvous plan executed at cruise speed. Comparing a woven mesh
    against a MAX-speed baseline would charge the mesh for the speed reserve as
    well; this isolates the cost of the meetings themselves."""
    v = v_cruise(cruise)
    return max(route_time(alloc["dock"], alloc["pois"][alloc["labels"] == r],
                          alloc["orders"][r], speed=v)
               for r in range(alloc["K"]))


def _robot_times(pois, labels, dock, K):
    times, orders = [], []
    for r in range(K):
        pts = pois[labels == r]
        o = open_tsp(dock, pts)
        orders.append(o)
        times.append(route_time(dock, pts, o))
    return np.array(times), orders


def rebalance(pois, labels, dock, K, centroids, rounds=40):
    """Makespan local search: move a POI from the bottleneck robot to the nearest
    other robot when it lowers the makespan. Keeps 100% coverage (just relabels)."""
    labels = labels.copy()
    for _ in range(rounds):
        times, _ = _robot_times(pois, labels, dock, K)
        b = int(np.argmax(times))
        idxs = np.where(labels == b)[0]
        if len(idxs) <= 1:
            break
        best_gain, best = 0.0, None
        for i in idxs:
            # candidate target = nearest other centroid
            d = [dist(pois[i], centroids[r]) if r != b else 1e18 for r in range(K)]
            r2 = int(np.argmin(d))
            trial = labels.copy(); trial[i] = r2
            tt, _ = _robot_times(pois, trial, dock, K)
            gain = times.max() - tt.max()
            if gain > best_gain + 1e-6:
                best_gain, best = gain, trial
        if best is None:
            break
        labels = best
    return labels


def nominal_times(pois, labels, dock, orders, K):
    """When each POI is visited on the untouched Layer-0 tour. This is what makes
    a NATURAL crossing identifiable: a pair of POIs close in space whose nominal
    visit times already agree is a rendezvous that costs nothing."""
    t_nom = {}
    for r in range(K):
        idx = np.where(labels == r)[0]
        pts = pois[idx]
        t, cur = 0.0, np.asarray(dock, float)
        for k in orders[r]:
            t += dist(cur, pts[k]) / V + SCAN
            t_nom[int(idx[k])] = t
            cur = pts[k]
    return t_nom


def allocate(pois, K, dock, seed=0):
    labels, centroids = balanced_partition(pois, K, seed)
    labels = rebalance(pois, labels, dock, K, centroids)
    centroids = np.array([pois[labels == r].mean(0) if np.any(labels == r)
                          else centroids[r] for r in range(K)])
    times, orders = _robot_times(pois, labels, dock, K)
    return dict(labels=labels, centroids=centroids, orders=orders,
                times=times, makespan=float(times.max()), K=K, dock=dock,
                pois=pois, t_nom=nominal_times(pois, labels, dock, orders, K))


# --- Layer 1: single-failure recovery feasibility ----------------------------
def adjacency(centroids):
    K = len(centroids)
    adj = {r: set() for r in range(K)}
    if K <= 3:
        for a in range(K):
            for b in range(a + 1, K):
                adj[a].add(b); adj[b].add(a)
        return adj
    tri = Delaunay(centroids)
    for s in tri.simplices:
        for a in s:
            for b in s:
                if a != b:
                    adj[int(a)].add(int(b))
    return adj


def recovery(alloc, rule="least_load"):
    """For each single failure, reassign the victim's POIs to the survivors and
    recompute the team makespan. Recovery is always feasible; what we report is
    the worst-case post-failure makespan, i.e. the resilience cost.

    Two assignment rules:
      "nearest"    -- give each orphan to the nearest surviving centroid. Simple,
                      but it ignores how loaded that survivor already is, so it
                      can pile a whole dead robot's work onto one neighbour that
                      was already the bottleneck.
      "least_load" -- give each orphan to whichever survivor it costs least to
                      add, hardest orphan first. Measured strictly better on
                      every victim of every instance tested, and the margin
                      GROWS with team size (38% of the recovery penalty removed
                      at K=4, 67% at K=6, 75% at K=8), because more survivors
                      means more freedom to balance and `nearest` discards all
                      of it. Costs more planning time -- it re-solves a route per
                      candidate survivor per orphan -- but it is all offline.
    """
    pois, labels, dock, K = alloc["pois"], alloc["labels"], alloc["dock"], alloc["K"]
    cen = alloc["centroids"]
    per_victim = {}
    for v in range(K):
        surv = [r for r in range(K) if r != v]
        if not surv:
            continue
        nl = labels.copy()
        orphans = list(np.where(labels == v)[0])
        if rule == "nearest":
            for i in orphans:
                nl[i] = min(surv, key=lambda s: dist(pois[i], cen[s]))
        else:
            # hardest first: the orphan with no nearby survivor is the one whose
            # placement constrains the others, so it should choose first
            orphans.sort(key=lambda i: -min(dist(pois[i], cen[s]) for s in surv))
            for i in orphans:
                best, best_t = None, None
                for sv in surv:
                    trial = nl.copy()
                    trial[i] = sv
                    t = route_time(dock, pois[trial == sv],
                                   open_tsp(dock, pois[trial == sv]))
                    if best_t is None or t < best_t:
                        best, best_t = sv, t
                nl[i] = best
        per_victim[v] = max(route_time(dock, pois[nl == sv],
                                       open_tsp(dock, pois[nl == sv]))
                            for sv in surv)
    worst = max(per_victim.values()) if per_victim else 0.0
    return per_victim, worst


# --- figure ------------------------------------------------------------------
def render(alloc, per_victim, worst, L, path):
    pois, labels, dock, K = alloc["pois"], alloc["labels"], alloc["dock"], alloc["K"]
    fig = plt.figure(figsize=(15.5, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.22)
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])

    # panel A: the allocation (100% coverage, colour = robot region)
    for r in range(K):
        col = PALETTE[r % len(PALETTE)]
        pts = pois[labels == r]
        axA.scatter(pts[:, 0], pts[:, 1], s=34, color=col, edgecolor="k",
                    lw=0.3, zorder=3)
        seq = [dock] + [pts[j] for j in alloc["orders"][r]]
        seq = np.array(seq)
        axA.plot(seq[:, 0], seq[:, 1], "-", color=col, lw=1.5, alpha=0.6, zorder=2,
                 label=f"U{r}: {int((labels==r).sum())} POIs, "
                       f"{alloc['times'][r]/60:.1f} min")
    axA.scatter([dock[0]], [dock[1]], marker="*", s=360, color="#111",
                edgecolor="w", lw=0.6, zorder=6)
    axA.annotate("dock", (dock[0], dock[1]), textcoords="offset points",
                 xytext=(8, -4), fontsize=9)
    axA.set_title(f"LAYER 0 — coverage-first allocation: 100% of {len(pois)} POIs "
                  f"covered\nbalanced min-makespan = {alloc['makespan']/60:.1f} min "
                  f"(K={K} robots)", fontsize=11)
    axA.set_aspect("equal"); axA.set_xlim(-0.03 * L, 1.03 * L)
    axA.set_ylim(-0.03 * L, 1.05 * L); axA.grid(alpha=0.15)
    axA.set_xlabel("x (m)"); axA.set_ylabel("y (m)")
    axA.legend(loc="upper right", fontsize=8)

    # panel B: recovery — nominal vs worst-case post-failure makespan
    nom = alloc["makespan"] / 60.0
    vics = list(range(K))
    vals = [per_victim[v] / 60.0 for v in vics]
    cols = [PALETTE[v % len(PALETTE)] for v in vics]
    axB.bar([f"U{v}†" for v in vics], vals, color=cols, alpha=0.85,
            edgecolor="k", lw=0.4)
    axB.axhline(nom, color="#333", ls="--", lw=1.5,
                label=f"nominal makespan = {nom:.1f} min")
    axB.axhline(worst / 60.0, color="#d62728", ls=":", lw=1.5,
                label=f"worst-case = {worst/60:.1f} min "
                      f"(+{100*(worst/alloc['makespan']-1):.0f}%)")
    axB.set_ylabel("makespan after failure (min)")
    axB.set_title("LAYER 1 — single-failure recovery (least-added-load "
                  "reassignment)\nbar = team makespan if that robot dies "
                  "(survivors take over)", fontsize=11)
    axB.legend(fontsize=8.5, loc="lower right"); axB.grid(alpha=0.18, axis="y")

    fig.suptitle("Coverage-first task allocation: 100% coverage guaranteed, "
                 "single-failure recoverable", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --- self tests --------------------------------------------------------------
def run_dry():
    print("[dry_run] self-tests")
    for (N, K, seed) in [(40, 4, 7), (90, 4, 5), (80, 6, 2), (30, 3, 1), (50, 2, 9)]:
        pois, dock = make_scenario(N, 600.0, seed)
        alloc = allocate(pois, K, dock, seed)
        # 1) 100% coverage: every POI assigned exactly once, all labels valid
        assert len(alloc["labels"]) == N, "missing POIs"
        assert set(np.unique(alloc["labels"])).issubset(set(range(K))), "bad label"
        counts = np.bincount(alloc["labels"], minlength=K)
        assert counts.sum() == N, "coverage != 100%"
        # 2) balance: counts differ by <= 1 before rebalance-driven drift, <= a few after
        assert counts.max() - counts.min() <= max(2, N // K), "wildly unbalanced"
        # 3) recovery feasible for every single failure
        per_v, worst = recovery(alloc)
        assert len(per_v) == K, "recovery missing a victim"
        assert worst >= alloc["makespan"] - 1e-6, "post-failure < nominal?!"
        assert all(np.isfinite(list(per_v.values()))), "recovery infeasible"
        print(f"  ok  N={N} K={K} seed={seed}: cover {N}/{N} (100%), "
              f"counts {counts.tolist()}, makespan {alloc['makespan']/60:.1f}min, "
              f"worst-recovery {worst/60:.1f}min "
              f"(+{100*(worst/alloc['makespan']-1):.0f}%)")
    print("[dry_run] all self-tests passed")


def build_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--N", type=int, default=90)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--L", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--out", default=os.path.join(OUT, "coverage_first.png"))
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--l2", action="store_true",
                   help="weave the rendezvous mesh and render the Layer-2 figure")
    p.add_argument("--sweep", action="store_true",
                   help="run the strength / alpha sweeps (writes CSVs)")
    p.add_argument("--delta", type=float, default=600.0,
                   help="rendezvous cadence (s); must clear cadence_feasible")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="C4 target per-meeting benign false-alarm rate")
    p.add_argument("--sigma_x", type=float, default=4.0,
                   help="current strength, in units of EPS")
    p.add_argument("--ell", type=float, default=ELL,
                   help="disturbance correlation length (m)")
    p.add_argument("--schedule", default="elastic", choices=sorted(SCHEDULE),
                   help="how appointments are timed: grid (clock) or elastic "
                        "(packed windows + floating appointment + rolling "
                        "deadline); see SCHEDULE")
    return p


def main():
    args = build_args().parse_args()
    if args.dry_run:
        run_dry()
        print()
        run_dry_l2()
        return
    if args.l2 or args.sweep:
        main_l2(args)
        return
    pois, dock = make_scenario(args.N, args.L, args.seed)
    alloc = allocate(pois, args.K, dock, args.seed)
    per_v, worst = recovery(alloc)
    print(f"[alloc] K={args.K} N={args.N}: 100% coverage, "
          f"makespan {alloc['makespan']/60:.1f} min, "
          f"counts {np.bincount(alloc['labels']).tolist()}")
    print(f"[recovery] worst-case post-failure makespan {worst/60:.1f} min "
          f"(+{100*(worst/alloc['makespan']-1):.0f}% vs nominal)")
    path = render(alloc, per_v, worst, args.L, os.path.abspath(args.out))
    print(f"[figure] {path}")



# =============================================================================
# LAYER 2 — co-optimised rendezvous mesh with C4 correlation alignment
# =============================================================================
# Folds the CAR-FCD C4 constraint (car_fcd_sim.py) onto the 100%-coverage base
# built above.  The standalone prototype models alignment as an abstract scalar
# `align` in [0,1] that interpolates an approach point from the territory centre
# to the meeting midpoint, and prices it with the stylised proxy
# travel_overhead = 100*align ("% of one meeting leg").  On a REAL allocation the
# same constraint is a ROUTING constraint and its price is whatever the router
# actually pays in makespan.  Layer 2 measures that price.
#
# The disturbance physics is imported from car_fcd_sim (one source of truth):
#   rho(d) = exp(-d/ell)
#   p(d)   = 2*Phi(-eps / sqrt(2 sigma^2 (1-rho(d))))         benign miss prob
#   C4     : d_ij <= d* = -ell*ln(rho*),  1-rho* = EPS^2/(2 sigma^2 z_{1-a/2}^2)
#
# ACCRUAL MODEL (assumption A-L2, the bridge from abstract `align` to real
# routes).  A robot's arrival offset at a meeting is sigma*u(a_i), where a_i is
# the time-weighted mean of its trajectory over the last t_acc seconds before the
# appointment -- the LOCALE its approach took place in.  With no alignment that
# mean sits in the robot's own territory (recovers car_fcd align=0); with both
# partners confined to a corridor of radius d*/2 around the meeting point it
# collapses onto the meeting (align=1).  So car_fcd's `align` is exactly the
# fraction of the approach window spent in the shared corridor, and its real cost
# is the coverage work that window can no longer do.  Station-keeping inside the
# corridor counts as being in that locale.
# The meeting criterion is car_fcd's:  made iff |offset_i - offset_j| <= eps
# (each partner waits eps past the appointment, so a COMMON delay still meets and
# only the DIFFERENTIAL misses).  Note this makes symmetric early arrival useless
# -- the only slack that helps is a wider eps, which is the `slack` baseline.
#
# FOUR schedules on the same coverage base:
#   plain  : mesh only, meetings at the region boundary, no alignment (C3 only)
#   loiter : C4 bought with DEAD TIME -- dead-head into the corridor and hold
#   coopt  : C4 bought with WORK -- co-optimised placement puts the meeting ON a
#            task: a cross-boundary POI pair within d* (both partners scan) or,
#            always available, one partner's boundary POI (owner scans at the
#            meeting point, guest holds).  This is the "co-optimise meeting
#            placement WITH assignment" half of the Layer-2 brief.
#   slack  : no alignment; buy the same false-alarm reduction by widening the
#            meeting window instead.  The skeptic's baseline -- waiting is the
#            obvious way to miss fewer meetings and C4 has to beat it on the
#            makespan-vs-false-alarm plane to be worth anything.

import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import car_fcd_sim as cf                                    # noqa: E402
from scipy.stats import norm                                # noqa: E402

ELL = 150.0      # m   disturbance correlation length (one cell in car_fcd)
SIGMA_X = 2.0    # disturbance amplitude in units of EPS (car_fcd "strength")
T_ACC = 120.0    # s   accrual horizon: the approach that sets the offset


# --- C4 math (thin wrappers over the validated car_fcd formulas) -------------
def d_star(sigma, ell, alpha):
    """Largest approach separation whose benign miss probability is <= alpha.
    1 - rho* = EPS^2/(2 sigma^2 z^2);  d* = -ell*ln(rho*).  inf = unconstrained."""
    if sigma <= 0:
        return float("inf")
    z = norm.ppf(1 - alpha / 2.0)
    one_minus_rho = EPS * EPS / (2.0 * sigma * sigma * z * z)
    if one_minus_rho >= 1.0:
        return float("inf")
    return -ell * math.log(1.0 - one_minus_rho)


def kappa_of(sigma_x, delta):
    """Fractional speed perturbation of the current, calibrated so that a
    reference approach of delta/2 accrues car_fcd's sigma = sigma_x*EPS.
    sigma_x=2, delta=600 -> kappa=0.2, i.e. a current worth +-20% of cruise."""
    return sigma_x * EPS / (delta / 2.0)


def p_miss_pair(kappa, t_i, t_j, d, ell, eps=EPS):
    """Benign per-meeting miss probability for partners whose final approaches
    lasted t_i, t_j seconds through locales d apart:
        offset_k             = kappa * t_k * u(a_k)
        Var(offset_i - off_j)= kappa^2 (t_i^2 + t_j^2 - 2 t_i t_j rho(d))
    Equal-length approaches through the same locale cancel (that is C4); a long
    lone approach does not.  t_i=t_j=delta/2 recovers car_fcd's 2 sigma^2 (1-rho).
    Note the asymmetry penalty: one partner parked and the other still running a
    long approach is WORSE than both approaching together."""
    rho = math.exp(-d / ell) if d > 0 else 1.0
    var = kappa * kappa * (t_i * t_i + t_j * t_j - 2.0 * t_i * t_j * rho)
    if var <= 1e-12:
        return 0.0
    return float(2.0 * norm.cdf(-eps / math.sqrt(var)))


def p_miss(sigma, d, ell, eps=EPS):
    """car_fcd's fully-mobile reference case (both approaches of the reference
    length): reproduces car_fcd.analytic_p exactly -- asserted in the self-tests.
    This is the worst case that C4's d* is sized for."""
    if sigma <= 0 or d <= 0:
        return 0.0
    var = 2.0 * sigma * sigma * (1.0 - math.exp(-d / ell))
    if var <= 0:
        return 0.0
    return float(2.0 * norm.cdf(-eps / math.sqrt(var)))


# =============================================================================
# DETERMINISTIC-EXECUTION MODEL (v2)
# =============================================================================
# Scope decision: wind and waves are OUT OF SCOPE here. Execution is
# deterministic -- robots arrive when the plan says. Consequently the benign
# false-alarm rate is identically zero BY CONSTRUCTION and is therefore not
# reported: with no disturbance, nothing but a death can make a robot miss an
# appointment, so quoting 0% would be circular. What remains non-trivial, and
# is what this layer now measures, is the SCHEDULING problem: can a mesh be
# built that keeps 100% coverage, meets every Delta, leaves enough capacity to
# absorb a takeover, and what does it cost?
#
# Two mechanisms are added:
#
#  SPEED RESERVE. Robots cruise at CRUISE_FRAC of their maximum speed and keep
#  the rest in hand. Holding the reserve is cheap -- drag goes as v^2, so slow
#  cruising is more efficient per metre -- and it is spent only when a robot
#  must catch up, which in a deterministic world means only after a takeover.
#  It is not free in schedule terms: it directly raises the cadence bound,
#  because a slower cruise cannot reach as far between appointments.
#
#  TIME BALANCING. A robot that would reach an appointment early used to stand
#  and wait. Instead it packs the window with work, choosing the POI that
#  leaves the SMALLEST residual rather than the nearest one, so the leftover
#  idle shrinks toward the sub-task quantum that cannot be removed (a partial
#  POI will not fit, so some residual is structural).
# MEASURED (2026-10-18): holding a reserve costs 16-38 points of makespan and,
# in this architecture, buys nothing -- the scheduler protects appointments by
# DEFERRING work, never by running late, so the sprint never fires. Default is
# therefore no reserve. Set < 1.0 to hold one; see takeover_test(). The reserve
# only has something to absorb once execution is non-deterministic (wind, out
# of scope here) or a hard mission deadline removes the option to defer.
CRUISE_FRAC = 1.0        # cruise speed as a fraction of V; rest is reserve


def v_cruise(frac=None):
    return V * (CRUISE_FRAC if frac is None else frac)


# --- the MEASURED disturbance response (from HoloOcean, not from our model) --
# holoocean_eq3_validation.py showed Eq. (3)'s proportionality to exposure time
# and its inheritance of the field's correlation both hold, but that its
# LINEARITY in the disturbance does not: the true response is convex and
# asymmetric, so a Gaussian offset understates the late tail. These helpers let
# every false-alarm number be recomputed through the measured response instead,
# which is what `model="measured"` selects in c4_metrics.
_RESP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "results", "holoocean_eq3", "eq3_response.csv")
_T_REF = 63.42          # s, the nominal transit the response was measured on


def _load_response():
    """Fractional delay phi(c) = theta(c)/t_ref against along-track current."""
    if not os.path.exists(_RESP):
        return None
    c, th = [], []
    with open(_RESP) as fh:
        for row in csv.DictReader(fh):
            c.append(float(row["current"])); th.append(float(row["theta_s"]))
    c, th = np.array(c), np.array(th) / _T_REF
    return c, th


_RESP_CACHE = None


def phi_measured(c):
    """Measured fractional delay, linearly extrapolated beyond the swept range
    using the endpoint slopes (the true curve is convex, so this UNDERstates the
    head-current tail -- the conservative direction for our claims)."""
    global _RESP_CACHE
    if _RESP_CACHE is None:
        _RESP_CACHE = _load_response()
    if _RESP_CACHE is None:
        raise SystemExit("measured response not found; run "
                         "holoocean_eq3_validation.py --exp probe first")
    cs, ph = _RESP_CACHE
    y = np.interp(c, cs, ph)
    lo = (ph[1] - ph[0]) / (cs[1] - cs[0])
    hi = (ph[-1] - ph[-2]) / (cs[-1] - cs[-2])
    y = np.where(c < cs[0], ph[0] + lo * (c - cs[0]), y)
    y = np.where(c > cs[-1], ph[-1] + hi * (c - cs[-1]), y)
    return y


def calibrate_amp(kappa, n=200000, seed=0):
    """Field amplitude A such that the MEASURED response has the same spread as
    the Gaussian model's kappa, so the two are compared at matched dispersion."""
    from scipy.optimize import brentq
    u = np.random.default_rng(seed).standard_normal(n)
    f = lambda A: float(np.std(phi_measured(A * u))) - kappa
    try:
        return float(brentq(f, 1e-4, 50.0))
    except ValueError:
        return float("nan")


def _safe_chol(points, ell):
    """Cholesky of exp(-d/ell), robust to the near-coincident approach points
    that C4 deliberately creates."""
    D = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    R = np.exp(-D / ell)
    for jit in (1e-9, 1e-7, 1e-5, 1e-3):
        try:
            return np.linalg.cholesky(R + jit * np.eye(len(points)))
        except np.linalg.LinAlgError:
            continue
    w, U = np.linalg.eigh(R)
    return U @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))


# --- the rendezvous mesh (rotating buddies) ----------------------------------
def _max_matching(K, adj, stale):
    """Maximum matching (exact; K is small), tie-broken toward the STALEST pairs.
    Greedy maximal matching is not enough: on a 4-region graph it can strand two
    non-adjacent robots and open a 2*Delta detection gap."""
    memo = {}

    def rec(used):
        if used in memo:
            return memo[used]
        a = next((x for x in range(K) if not (used >> x) & 1), None)
        if a is None:
            return (0, 0.0, ())
        best = rec(used | (1 << a))                      # leave `a` unmatched
        for b in adj[a]:
            if b == a or (used >> b) & 1:
                continue
            n, w, pl = rec(used | (1 << a) | (1 << b))
            cand = (n + 1, w + stale.get((min(a, b), max(a, b)), 99),
                    pl + ((min(a, b), max(a, b)),))
            if cand[:2] > best[:2]:
                best = cand
        memo[used] = best
        return best

    return rec(0)[2]


def meeting_schedule(alloc, rounds, adj=None):
    """Rotating-buddy mesh over region adjacency: each round is a MAXIMUM
    matching that prefers the stalest admissible partner, so every robot that
    can be matched meets someone every Delta (detection gap ~ Delta) while
    cycling through all its neighbours -- no fixed buddy, hence no single
    monitoring blind spot."""
    K = alloc["K"]
    adj = adjacency(alloc["centroids"]) if adj is None else adj
    last, sched = {}, []
    for r in range(rounds):
        stale = {(a, b): r - last.get((a, b), -99)
                 for a in range(K) for b in adj[a] if b > a}
        pairs = list(_max_matching(K, adj, stale))
        for ab in pairs:
            last[ab] = r
        sched.append(pairs)
    return sched


def cadence_feasible(alloc, delta=DELTA, t_acc=0.0, cruise=None):
    """Necessary condition for a Delta-cadence mesh: from its worst meeting site
    a robot must reach its farthest POI and get back inside one window, AT
    CRUISE SPEED (the reserve is held for takeover, not spent on routine
    travel). Returns (ok, required_delta). Holding a reserve tightens this
    bound -- that is the price of the reserve, and it is paid in cadence."""
    cen, labels, pois = alloc["centroids"], alloc["labels"], alloc["pois"]
    adj = adjacency(cen)
    worst = 0.0
    for r in range(alloc["K"]):
        sites = [0.5 * (cen[r] + cen[b]) for b in adj[r]]
        idx = np.where(labels == r)[0]
        if not len(idx) or not sites:
            continue
        rr = max(max(dist(pois[i], s) for s in sites) for i in idx)
        worst = max(worst, 2 * rr / v_cruise(cruise) + SCAN + t_acc)
    return (worst <= delta), worst


def _cross_candidates(alloc, a, b, limit=60):
    """Candidate NATURAL CROSSINGS for the pair (a,b): cross-boundary POI pairs,
    kept by spatial separation.  Selection then rescores them in time against the
    round's slot, so the chosen crossing is one both robots were already going to
    work at about that moment.  Unlike the C4 corridor, nothing here is a detour
    to an artificial point: both partners end the approach on a task they owed."""
    pois, labels = alloc["pois"], alloc["labels"]
    A, B = np.where(labels == a)[0], np.where(labels == b)[0]
    cand = [(dist(pois[i], pois[j]), int(i), int(j)) for i in A for j in B]
    cand.sort()
    return cand[:limit]


def _cross_pairs(pois, labels, a, b, limit=16):
    """Closest cross-boundary POI pairs (one owned by each partner) -- candidate
    co-optimised meeting sites where BOTH partners satisfy C4 while scanning."""
    A, B = np.where(labels == a)[0], np.where(labels == b)[0]
    cand = [(dist(pois[i], pois[j]), int(i), int(j)) for i in A for j in B]
    cand.sort()
    return cand[:limit]


# --- executable rover --------------------------------------------------------
class _Rover:
    """Piecewise-linear trajectory with explicit travel / scan / idle books."""

    def __init__(self, rid, dock, todo):
        self.rid = rid
        self.pos = np.asarray(dock, float).copy()
        self.t = 0.0
        self.todo = list(todo)
        self.done = []
        self.track = []                 # (t0, t1, p0, p1), linear motion
        self.travel = 0.0
        self.idle = 0.0
        self.scan_t = 0.0
        self.sprint_m = 0.0
        self.last_scan = 0.0
        self.meet_t = []

    def goto(self, p, speed=None):
        """Travel at cruise unless `speed` says otherwise (a sprint spends the
        reserve). Sprint distance is tracked so its cost can be reported."""
        p = np.asarray(p, float)
        d = dist(self.pos, p)
        sp = v_cruise() if speed is None else speed
        if d > 1e-9:
            self.track.append((self.t, self.t + d / sp, self.pos.copy(), p.copy()))
            self.t += d / sp
            self.travel += d
            if speed is not None and sp > v_cruise() + 1e-9:
                self.sprint_m += d
            self.pos = p.copy()

    def hold(self, until, kind="idle"):
        if until > self.t + 1e-9:
            self.track.append((self.t, until, self.pos.copy(), self.pos.copy()))
            if kind == "idle":
                self.idle += until - self.t
            else:
                self.scan_t += until - self.t
            self.t = until

    def scan(self, i, pois):
        self.goto(pois[i])
        self.hold(self.t + SCAN, kind="scan")
        self.todo.remove(i)
        self.done.append(i)
        self.last_scan = self.t


def _window_legs(track, t0, t1):
    """Every MOVING segment in [t0,t1] as (locale, duration).

    HoloOcean refutes assumption A-L2 (see holoocean_eq3_validation stop test):
    a fixed-duration stop neither resets nor absorbs accumulated delay -- the
    offset measured at the end of A->B->dwell->C equals the delay of BOTH legs,
    independent of the dwell. So the exposure carried into an appointment is the
    whole window's travel, and only CLOCK-based waiting (arriving early and
    holding until the appointment) absorbs any of it."""
    out = []
    for (ta, tb, pa, pb) in track:
        s0, e0 = max(ta, t0), min(tb, t1)
        if e0 <= s0 or dist(pa, pb) < 1e-9:
            continue
        f = ((s0 + e0) / 2.0 - ta) / (tb - ta) if tb > ta else 0.0
        out.append((pa + f * (pb - pa), e0 - s0))
    return out


def _final_leg(track, fallback):
    """The robot's final continuous travel before the meeting: (locale, duration).

    ACCRUAL MODEL (assumption A-L2).  A work stop is a resynchronisation point --
    the router replans from the robot's actual clock every time it finishes a
    scan -- so the delay a robot carries INTO an appointment is the disturbance
    on its last uninterrupted approach: offset = kappa * t_leg * u(locale), with
    kappa the fractional speed perturbation of the current and u the correlated
    field.  A long approach from deep inside its own territory is both large and
    uncorrelated with its partner's; a short hop from a shared corridor is small
    and correlated.  This is the real-routing form of car_fcd's approach point."""
    segs = []
    for (ta, tb, pa, pb) in reversed(track):
        if dist(pa, pb) < 1e-9:              # a scan or a hold ends the leg
            break
        segs.append((ta, tb, pa, pb))
    if not segs:
        return np.asarray(fallback, float).copy(), 0.0
    dur = sum(tb - ta for ta, tb, _p, _q in segs)
    acc = np.zeros(2)
    for (ta, tb, pa, pb) in segs:
        acc += (tb - ta) * 0.5 * (pa + pb)
    return acc / dur, dur


def _stopped(rv):
    """True if the rover's last trajectory segment is stationary -- i.e. it has a
    genuine stop from which the final approach leg starts."""
    return bool(rv.track) and dist(rv.track[-1][2], rv.track[-1][3]) < 1e-9


def _corridor_entry(p, m, R):
    v = np.asarray(p, float) - m
    n = float(np.hypot(v[0], v[1]))
    if n <= R or n < 1e-9:
        return np.asarray(p, float).copy()
    return m + v * (R / n)


def _work(rv, pois, deadline, after, pool=None, exclude=None, balance=False,
          pull=0.0, endpoint=None):
    """Scan as many of the rover's own remaining POIs as fit before `deadline`,
    always preserving the ability to pay `after(p)` seconds afterwards.

    `balance=True` is the time-balancing rule: among the POIs that fit, take the
    one that leaves the SMALLEST slack at the deadline, so the window is packed
    rather than merely served nearest-first. What is left over is then only the
    sub-task quantum -- a partial POI cannot be done, so some residual idle is
    structural, not waste."""
    V_c = v_cruise()
    while True:
        feas = []
        for i in rv.todo:
            if pool is not None and i not in pool:
                continue
            if exclude is not None and i in exclude:
                continue
            fin = rv.t + dist(rv.pos, pois[i]) / V_c + SCAN
            if fin + after(pois[i]) <= deadline + 1e-9:
                feas.append((i, fin + after(pois[i])))
        if not feas:
            return
        if balance:
            pick = max(feas, key=lambda kv: kv[1])[0]        # least slack left
        elif pull > 0.0 and endpoint is not None:
            # SCHEDULE-AWARE SELECTION: among the POIs that fit, prefer those
            # that also sit toward the upcoming meeting point, so the robot
            # drifts into the shared boundary as the appointment approaches
            # instead of being hauled there at the end. `pull` is how strongly
            # proximity-to-the-meeting competes with proximity-to-me.
            pick = min(feas, key=lambda kv: dist(rv.pos, pois[kv[0]])
                       + pull * dist(pois[kv[0]], endpoint))[0]
        else:
            pick = min(feas, key=lambda kv: dist(rv.pos, pois[kv[0]]))[0]
        rv.scan(pick, pois)


def _pack_window(rv, pois, deadline, endpoint, cand_k=9, exclude=None):
    """Choose WHICH OF ITS OWN remaining POIs a robot does before an
    appointment, and in what order, so the window is filled instead of leaving
    unusable leftover time.

    Greedy-nearest takes one POI at a time and stops when the next will not
    fit, which routinely strands 40-90 s -- less than a single scan, so it can
    never be used. The robot is not short of work; it has simply chosen a
    combination that does not land on the deadline. Two of its own nearby tasks
    often fit where one distant one did.

    Exact over the `cand_k` nearest candidates by Held-Karp over subsets:
    dp[mask][last] = earliest time having scanned `mask` and standing at
    `last`. A subset is admissible if the robot can still reach `endpoint` by
    the deadline. Maximise POIs scanned, tie-break on earliest finish.
    2^9 x 9 states is nothing, and it is all offline."""
    todo = [i for i in rv.todo if not (exclude and i in exclude)]
    if not todo:
        return []
    V_c = v_cruise()
    cand = sorted(todo, key=lambda i: dist(rv.pos, pois[i]))[:cand_k]
    n = len(cand)
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    par = [[-1] * n for _ in range(1 << n)]
    for j in range(n):
        t = rv.t + dist(rv.pos, pois[cand[j]]) / V_c + SCAN
        if t + dist(pois[cand[j]], endpoint) / V_c <= deadline + 1e-9:
            dp[1 << j][j] = t
    best_mask, best_last, best = 0, -1, (0, 0.0)
    for mask in range(1 << n):
        for last in range(n):
            t = dp[mask][last]
            if t == INF:
                continue
            cnt = bin(mask).count("1")
            if (cnt, -t) > (best[0], -best[1]):
                best, best_mask, best_last = (cnt, t), mask, last
            for j in range(n):
                if mask >> j & 1:
                    continue
                t2 = t + dist(pois[cand[last]], pois[cand[j]]) / V_c + SCAN
                if t2 + dist(pois[cand[j]], endpoint) / V_c > deadline + 1e-9:
                    continue
                if t2 < dp[mask | 1 << j][j]:
                    dp[mask | 1 << j][j] = t2
                    par[mask | 1 << j][j] = last
    seq, mask, last = [], best_mask, best_last
    while last >= 0:
        seq.append(cand[last])
        plast = par[mask][last]
        mask ^= 1 << last
        last = plast
    return seq[::-1]


def _pack_frontier(rv, pois, deadline, endpoint, cand_k=9, exclude=None):
    """The whole count-vs-arrival trade-off for one robot's approach window.

    `_pack_window` answers "how many of my own POIs fit?".  That is the wrong
    question for a PAIR: the appointment happens when the LATER partner shows
    up, so a robot that squeezes in a third task and lands on the deadline has
    spent its partner's time as well as its own.  The frontier answers the
    question that composes -- for every achievable count c, the earliest the
    robot can be standing at `endpoint` having scanned c of its own tasks --
    and lets the pair pick the combination.

    Returns {count: (arrival_at_endpoint, [poi ids in order])}, always
    including count 0 (go straight there).  Same Held-Karp recursion as
    `_pack_window`; the only change is that every reachable state is scored at
    the ENDPOINT, not at the last task, and the best is kept per count rather
    than overall."""
    V_c = v_cruise()
    out = {0: (rv.t + dist(rv.pos, endpoint) / V_c, [])}
    todo = [i for i in rv.todo if not (exclude and i in exclude)]
    if not todo:
        return out
    cand = sorted(todo, key=lambda i: dist(rv.pos, pois[i]))[:cand_k]
    n = len(cand)
    INF = float("inf")
    dp = [[INF] * n for _ in range(1 << n)]
    par = [[-1] * n for _ in range(1 << n)]
    for j in range(n):
        t = rv.t + dist(rv.pos, pois[cand[j]]) / V_c + SCAN
        if t + dist(pois[cand[j]], endpoint) / V_c <= deadline + 1e-9:
            dp[1 << j][j] = t
    best = {}
    for mask in range(1, 1 << n):
        cnt = bin(mask).count("1")
        for last in range(n):
            t = dp[mask][last]
            if t == INF:
                continue
            fin = t + dist(pois[cand[last]], endpoint) / V_c
            if fin < best.get(cnt, (INF,))[0]:
                best[cnt] = (fin, mask, last)
            for j in range(n):
                if mask >> j & 1:
                    continue
                t2 = t + dist(pois[cand[last]], pois[cand[j]]) / V_c + SCAN
                if t2 + dist(pois[cand[j]], endpoint) / V_c > deadline + 1e-9:
                    continue
                if t2 < dp[mask | 1 << j][j]:
                    dp[mask | 1 << j][j] = t2
                    par[mask | 1 << j][j] = last
    for cnt, (fin, mask, last) in best.items():
        seq = []
        while last >= 0:
            seq.append(cand[last])
            plast = par[mask][last]
            mask ^= 1 << last
            last = plast
        out[cnt] = (fin, seq[::-1])
    return out


def _pack_pair(rva, rvb, pois, deadline, m, cand_k=9, exclude=None,
               credit=SCAN):
    """Co-plan the two approach windows of one appointment (argmin form).

    With a floating appointment the pair's cost for the round is the time the
    LATER partner arrives, and its benefit is the tasks both of them retire on
    the way.  Writing F_x for robot x's count-vs-arrival frontier,

        (c_a*, c_b*) = argmin_{c_a, c_b}  max(F_a(c_a), F_b(c_b))
                                          - credit * (c_a + c_b)

    with `credit` the wall-clock a task is worth (one SCAN: the work has to be
    done by someone eventually, so retiring it now buys back exactly that).
    This is what makes "A takes two near tasks while B takes one far one" come
    out as the answer instead of having to be asked for: the two frontiers are
    searched against each other, so a robot stops adding tasks at the point
    where the delay it imposes on its partner outweighs the work it retires.
    Independent per-robot packing cannot see that term -- it maximises its own
    count and silently bills the difference to the other robot."""
    fa = _pack_frontier(rva, pois, deadline, m, cand_k, exclude)
    fb = _pack_frontier(rvb, pois, deadline, m, cand_k, exclude)
    # NO-STARVATION FLOOR.  Left alone the trade has a degenerate optimum: a
    # robot can always make the appointment earlier by doing nothing, so with a
    # low enough `credit` both partners run empty windows for ever and the
    # mission never terminates. The floor forbids only that -- a robot may give
    # up tasks to its partner's clock, but never the last one it could reach --
    # which bounds deferral without pinning the trade-off.
    la = min(1, max(fa)) if fa else 0
    lb = min(1, max(fb)) if fb else 0
    best = None
    for ca, (ta_, sa) in fa.items():
        if ca < la:
            continue
        for cb, (tb_, sb) in fb.items():
            if cb < lb:
                continue
            sc = max(ta_, tb_) - credit * (ca + cb)
            if best is None or sc < best[0] - 1e-9:
                best = (sc, sa, sb)
    return best[1], best[2]


def _work_packed(rv, pois, deadline, endpoint, exclude=None, cand_k=9):
    for i in _pack_window(rv, pois, deadline, endpoint, cand_k=cand_k,
                          exclude=exclude):
        rv.scan(i, pois)


def _work_free(rv, pois, until, exclude=None):  # noqa: D401
    """Unpartnered window: work continuously (a POI started before the round
    boundary may finish after it) so round boundaries cost nothing."""
    while rv.t < until - 1e-9:
        pool = [i for i in rv.todo if not (exclude and i in exclude)]
        if not pool:
            return
        rv.scan(min(pool, key=lambda k: dist(rv.pos, pois[k])), pois)


def plan_crossings(alloc, sched, rovers, r0, lookahead, cache):
    """Reserve the cross-boundary POI PAIRS that will host the next few meetings
    -- closest available pair first, one per scheduled contact -- and return
    (plan, protected), protected being the anchors owed at a LATER round.

    The reservation is what makes a natural crossing actually happen.  Left to
    the router, ordinary coverage work eats the few close boundary pairs before
    the round that needed them (measured: anchors consumed, every later meeting
    degenerating into a long uncorrelated run to an arbitrary midpoint).  It is
    kept to a short ROLLING look-ahead and recomputed from what is still
    unscanned: reserving the whole mission up front locks ~half the POIs away
    and starves the robots (measured: +249% makespan, idle explodes).  This is
    the co-optimisation of meeting placement WITH the assignment -- anchors are
    common knowledge and each robot simply owes its half of the pair."""
    pois = alloc["pois"]
    plan, used = {}, set()
    for rr in range(r0, min(r0 + 1 + lookahead, len(sched))):
        for (a, b) in sched[rr]:
            if (a, b) not in cache:
                cache[(a, b)] = _cross_candidates(alloc, a, b)
            for (sep, i, j) in cache[(a, b)]:
                if i in used or j in used:
                    continue
                if i not in rovers[a].todo or j not in rovers[b].todo:
                    continue
                plan[(rr, a, b)] = (i, j, 0.5 * (pois[i] + pois[j]), sep)
                used.add(i); used.add(j)
                break
    protected = {k for (rr, _a, _b), ent in plan.items() if rr > r0
                 for k in ent[:2]}
    return plan, protected


def _site(placement, a, b, alloc, rovers, ds, xpairs, rnd, aligned, tau=0.0,
          w_time=0.3, plan=None):
    """Meeting placement.  Returns (point, kind, {robot: reserved POI or None}).
    'boundary' -> midpoint of the two region centroids (the natural mesh site).
    'task'     -> put the meeting ON a task so the approach can be productive:
                  coopt2 = a cross-boundary POI pair within d* (BOTH partners
                           scan inside the corridor), or
                  coopt1 = one partner's boundary POI (owner scans at the meeting
                           point, guest holds) -- always available.
    This is the "co-optimise meeting placement WITH assignment" half of L2."""
    pois = alloc["pois"]
    mid = 0.5 * (alloc["centroids"][a] + alloc["centroids"][b])
    if placement == "cross":
        ent = (plan or {}).get((rnd, a, b))
        if ent is not None:
            i, j, mm, _sep = ent
            # a reserved crossing is only real if both partners can still make it:
            # reach their own half of the pair, scan it, and close the last hop by
            # tau.  An unreachable "crossing" is not a crossing -- it silently
            # degrades into a long uncorrelated run at an arbitrary midpoint.
            ok = all(k in rovers[w].todo
                     and rovers[w].t + dist(rovers[w].pos, pois[k]) / v_cruise()
                     + SCAN + dist(pois[k], mm) / v_cruise() <= tau + 1e-9
                     for w, k in ((a, i), (b, j)))
            if ok:
                return mm, "cross", {a: i, b: j}
        return mid, "loiter", {a: None, b: None}
    if placement != "task":
        return mid, ("loiter" if aligned else "plain"), {a: None, b: None}
    if np.isfinite(ds):
        for (sep, i, j) in xpairs[(a, b)]:
            if sep <= ds and i in rovers[a].todo and j in rovers[b].todo:
                return 0.5 * (pois[i] + pois[j]), "coopt2", {a: i, b: j}
    owner, guest = (a, b) if (rnd % 2 == 0) else (b, a)
    for who, oth in ((owner, guest), (guest, owner)):
        if not rovers[who].todo:
            continue
        i = min(rovers[who].todo, key=lambda k: dist(pois[k], mid))
        m = pois[i].copy()
        gp = [k for k in rovers[oth].todo if dist(pois[k], m) <= ds]
        j = min(gp, key=lambda k: dist(pois[k], m)) if gp else None
        return m, "coopt1", {who: i, oth: j}
    return mid, "loiter", {a: None, b: None}


# --- the weave: execute the coverage routes with the mesh overlaid -----------
# HOW THE APPOINTMENT IS TIMED, independent of where it is placed.
#   grid     every meeting sits on the absolute clock grid tau_r = r*Delta, and
#            a robot that is ready early stands at the site until the clock
#            catches up.  The wait is pure loss: it is time the robot had, with
#            work left to do, and could not spend.
#   elastic  three changes that remove it without touching the guarantee:
#            (i)  pack    -- choose WHICH of its own tasks fill the window by
#                           exact subset search, not nearest-first (so a robot
#                           can take two near tasks where its partner takes one
#                           far one, and both still make the appointment);
#            (ii) float   -- hold the meeting when the later partner actually
#                           arrives, not when the clock says; the pair pays
#                           |t_a - t_b| instead of 2*(tau - arrival);
#            (iii)rolling -- anchor the next deadline to the LAST meeting rather
#                           than to the clock.  This is what makes (ii) free:
#                           consecutive meetings are then <= Delta apart BY
#                           CONSTRUCTION, so time saved is spent earlier instead
#                           of widening the detection gap.  Floating on an
#                           absolute grid looks better still (26.8% vs 29.1%)
#                           but only by borrowing cadence -- its worst gap is
#                           660 s at Delta=600 -- and the borrowed amount is
#                           almost exactly the apparent gain.
# ABSORB_STAR -- the absorption margin the elastic schedule ships with.  Not a
# tuned constant: `sweep_absorb` puts the knee at 120-180 s in EVERY tested
# (N, K, L, Delta), including Delta from 500 to 800 s.  It does not scale with
# Delta, because what has to be absorbed is the partners' differential arrival
# spread -- a property of the routing geometry and the scan quantum, not of the
# cadence.  Below it false alarms climb steeply; above it mission cost is
# already at its floor and each further minute of holding buys under 0.2 pp.
ABSORB_STAR = 3 * SCAN            # 180 s

SCHEDULE = dict(grid=dict(pack=False, float_meet=False, rolling=False,
                          absorb=0.0),
                elastic=dict(pack=True, float_meet=True, rolling=True,
                             absorb=ABSORB_STAR))

# variant := (where the meeting is placed, how the robot approaches it)
PLACEMENT = dict(plain="boundary", slack="boundary", loiter="boundary",
                 coopt="task", cross="cross")
APPROACH = dict(plain="none", slack="early", loiter="corridor",
                coopt="corridor", cross="natural")


def weave(alloc, variant="plain", alpha=0.1, sigma_x=SIGMA_X, ell=ELL,
          delta=DELTA, early=0.0, hold_s=SCAN, w_time=0.3, lookahead=1,
          sprint=True, balance=False, pack=False, pull=0.0, sched=None,
          max_rounds=200, stall_cap=3, float_meet=False, slide=None,
          joint=False, credit=SCAN, cand_k=12, rolling=False,
          schedule="grid", absorb=None):
    """Run the 100%-coverage allocation with the rendezvous mesh overlaid.

    variant: plain | slack | loiter | coopt   (see the header comment)
      plain  work to the last moment, then run straight to the boundary site
      slack  same, but leave `early` seconds sooner and absorb the delay waiting
      loiter C4: stop in the shared corridor (radius d*/2) and dwell `hold_s`
             there, so the final approach is a short correlated hop
      coopt  C4 at no dwell cost where possible: the meeting is sited ON a task,
             so the scan that ends the approach IS the corridor stop
      cross  NATURAL TOUR CROSSING: no corridor and no dwell at all.  The meeting
             is a cross-boundary POI pair both partners were already going to
             work at about this time; each ends its approach by scanning its own
             half of the pair, so the final legs are short and co-located as a
             by-product of the route order rather than as a paid stop
    schedule: grid | elastic  (see SCHEDULE above -- how the appointment is
      TIMED, orthogonal to where it is placed).

    Returns coverage, makespan, per-meeting approach geometry, and the time
    budget (scan / travel / idle).  `feasible` is False if the cadence strands
    work; such a run is not a result."""
    pack, float_meet, rolling = (SCHEDULE[schedule]["pack"] or pack,
                                 SCHEDULE[schedule]["float_meet"] or float_meet,
                                 SCHEDULE[schedule]["rolling"] or rolling)
    absorb = SCHEDULE[schedule]["absorb"] if absorb is None else absorb
    pois, labels, dock, K = alloc["pois"], alloc["labels"], alloc["dock"], alloc["K"]
    sigma = sigma_x * EPS
    ds = d_star(sigma, ell, alpha)
    placement, approach = PLACEMENT[variant], APPROACH[variant]
    aligned = approach == "corridor"
    R_c = (ds / 2.0) if (aligned and np.isfinite(ds)) else float("inf")

    rovers = {r: _Rover(r, dock, np.where(labels == r)[0].tolist()) for r in range(K)}
    sched = meeting_schedule(alloc, max_rounds) if sched is None else sched
    xpairs, meetings = {}, []
    late, stall, feasible, rounds_used = 0, {r: 0 for r in range(K)}, True, 0
    last_meet = {r: 0.0 for r in range(K)}

    for r in range(max_rounds):
        if all(not rv.todo for rv in rovers.values()):
            break
        rounds_used = r + 1
        tau = (r + 1) * delta
        # ROLLING DEADLINE.  An absolute grid is what forces the idle: a robot
        # that is ready early must still stand at the site until the clock says
        # tau.  Letting the appointment float forward (float_meet) removes that
        # wait but borrows from the guarantee -- an early meeting followed by a
        # late one opens a gap of delta + slide.  Anchoring the next deadline to
        # the LAST meeting instead of to the clock gives the float back for
        # free: every robot's consecutive meetings are then at most delta apart
        # BY CONSTRUCTION, and time saved early is spent earlier, not idled
        # away.  The pair takes the tighter of its two anchors.
        tau_p = {rid: tau for rid in rovers}
        if rolling:
            tau_p = {rid: last_meet[rid] + delta for rid in rovers}
            for (a, b) in sched[r]:
                t = min(tau_p[a], tau_p[b])
                tau_p[a] = tau_p[b] = t
            tau = max(tau_p.values())
        before = {rid: len(rv.done) for rid, rv in rovers.items()}
        if placement == "cross":
            plan, future = plan_crossings(alloc, sched, rovers, r, lookahead,
                                          xpairs)
        else:
            plan, future = {}, set()
        partner = {}
        for (a, b) in sched[r]:
            if (a, b) not in xpairs:
                if placement == "task":
                    xpairs[(a, b)] = _cross_pairs(pois, labels, a, b)
                elif placement == "cross":
                    xpairs[(a, b)] = _cross_candidates(alloc, a, b)
            m, kind, res = _site(placement, a, b, alloc, rovers, ds, xpairs, r,
                                 aligned, tau=tau_p[a], w_time=w_time, plan=plan)
            partner[a] = dict(other=b, m=m, kind=kind, poi=res[a])
            partner[b] = dict(other=a, m=m, kind=kind, poi=res[b])

        # CO-PLAN EACH APPOINTMENT'S TWO APPROACH WINDOWS (see _pack_pair).
        # Both partners' states are known at the top of the round, and their
        # candidate tasks are disjoint by construction (the regions partition
        # the field), so the pair problem can be solved offline here, before
        # either robot moves.
        jseq = {}
        if joint and approach == "none":
            for (a, b) in sched[r]:
                if a not in partner or b not in partner:
                    continue
                jseq[a], jseq[b] = _pack_pair(rovers[a], rovers[b], pois,
                                              tau_p[a], partner[a]["m"],
                                              cand_k=cand_k, credit=credit)

        legs, arr = {}, {}
        for rid, rv in rovers.items():
            if rid not in partner:
                _work_free(rv, pois, tau_p[rid], exclude=future)
                last_meet[rid] = max(last_meet[rid], rv.t)
                continue
            tau = tau_p[rid]
            m, res_poi = partner[rid]["m"], partner[rid]["poi"]
            if approach == "corridor":
                # the same stop budget is reserved either way, so the coopt gain
                # is purely "that stop was a scan instead of an idle dwell"
                hop = min(R_c, dist(rv.pos, m)) / V
                _work(rv, pois, tau - hold_s - hop,
                      after=lambda p, m=m: max(0.0, dist(p, m) - R_c) / v_cruise(),
                      balance=balance)
                rv.goto(_corridor_entry(rv.pos, m, R_c))
                if res_poi is not None:
                    # the reserved scan IS the corridor stop: alignment for free
                    _work(rv, pois, tau, after=lambda p, m=m: dist(p, m) / v_cruise(),
                          pool={res_poi}, balance=balance)
                if not _stopped(rv):
                    rv.hold(min(rv.t + hold_s, tau))
            elif approach == "natural":
                # no dwell budget at all: the reserved scan IS the stop, and the
                # short hop off it is the whole disturbance-accruing approach
                if res_poi is not None:
                    pr = pois[res_poi]
                    _work(rv, pois, tau,
                          after=lambda p, pr=pr, m=m: (dist(p, pr) / v_cruise()
                                                       + SCAN
                                                       + dist(pr, m) / v_cruise()),
                          exclude=future | {res_poi}, balance=balance)
                    _work(rv, pois, tau, after=lambda p, m=m: dist(p, m) / v_cruise(),
                          pool={res_poi}, balance=balance)
                else:
                    _work(rv, pois, tau - hold_s,
                          after=lambda p, m=m: dist(p, m) / v_cruise(), exclude=future,
                          balance=balance)
                if not _stopped(rv):
                    rv.hold(min(rv.t + hold_s, tau))
            else:
                dl = tau - (early if approach == "early" else 0.0)
                if rid in jseq:
                    for i in jseq[rid]:
                        rv.scan(i, pois)
                elif pack:
                    _work_packed(rv, pois, dl, m, cand_k=cand_k)
                else:
                    _work(rv, pois, dl,
                          after=lambda p, m=m: dist(p, m) / v_cruise(),
                          balance=balance, pull=pull, endpoint=m)
            # SPEND THE RESERVE ONLY TO SAVE AN APPOINTMENT: if cruising
            # would arrive late, sprint the last leg at full speed. In a
            # deterministic mission this never fires on the nominal plan -- it
            # fires only when a robot is carrying work it did not plan for,
            # i.e. after a takeover. That is what the reserve is for.
            need = dist(rv.pos, m)
            if sprint and rv.t + need / v_cruise() > tau + 1e-9:
                rv.goto(m, speed=V)
            else:
                rv.goto(m)
            arr[rid] = rv.t
            if rv.t > tau + 1e-6:
                late += 1

        # THE APPOINTMENT TIME.  Nothing forces the meeting onto the clock grid.
        # The plan is deterministic (wind and waves are out of scope, Sec. II),
        # so each robot can compute its partner's arrival offline and the two
        # agree on a time without talking.  On the grid both partners pay
        # (tau - arrival) of idle; floating the appointment to the LATER of the
        # two arrivals costs the pair only |t_a - t_b| and hands the rest of the
        # window back to the next round.  The price is cadence -- a meeting that
        # slides `slide` seconds early stretches the worst inter-meeting gap to
        # delta + slide -- so an equal-guarantee comparison runs the grid at
        # delta - slide.
        t_meet = {}
        for (a, b) in sched[r]:
            if a not in arr or b not in arr:
                continue
            tau_ab = tau_p[a]
            tm = max(arr[a], arr[b])
            if not float_meet:
                tm = tau_ab
            else:
                # ABSORPTION MARGIN.  The idle a grid schedule imposes is not
                # pure waste: holding at the site until a time fixed in advance
                # is the only mechanism in this system that absorbs accumulated
                # delay (HoloOcean kills the alternative -- a stop does not
                # reset it, see _window_legs).  Floating the appointment to the
                # later arrival therefore removes the absorption along with the
                # waste.  `absorb` buys it back explicitly and at a known price:
                # every partner holds at least this long, so the pair pays
                # 2*absorb per meeting in makespan and gets `absorb` seconds of
                # tolerated differential delay. absorb=0 is pure elastic;
                # absorb -> infinity recovers the grid. The grid was simply this
                # knob stuck at whatever value the clock happened to leave over.
                tm += absorb
                if slide is not None:
                    tm = max(tm, tau_ab - slide)
                tm = min(tm, max(tau_ab, max(arr[a], arr[b])))
            t_meet[a] = t_meet[b] = min(max(tm, arr[a], arr[b]),
                                        max(tau_ab, arr[a], arr[b]))

        for rid, rv in rovers.items():
            if rid not in arr:
                continue
            tm = t_meet.get(rid, tau_p[rid])
            legs[rid] = (_final_leg(rv.track, partner[rid]["m"])
                         + (max(0.0, tm - rv.t),
                            _window_legs(rv.track, tm - delta, tm)))
            rv.hold(tm)
            rv.meet_t.append(tm)
            last_meet[rid] = tm

        for (a, b) in sched[r]:
            if a not in legs or b not in legs:
                continue
            (aa, ta_, sa, wa), (ab, tb_, sb, wb) = legs[a], legs[b]
            meetings.append(dict(round=r, tau=tau, i=a, j=b, m=partner[a]["m"],
                                 a_i=aa, a_j=ab, t_i=ta_, t_j=tb_, s_i=sa, s_j=sb,
                                 w_i=wa, w_j=wb,
                                 d=float(np.hypot(*(aa - ab))),
                                 kind=partner[a]["kind"]))

        for rid, rv in rovers.items():
            stall[rid] = 0 if len(rv.done) > before[rid] else stall[rid] + 1
            if rv.todo and stall[rid] >= stall_cap:
                feasible = False
        if not feasible:
            break

    done = sorted(i for rv in rovers.values() for i in rv.done)
    makespan = max(rv.last_scan for rv in rovers.values())
    gaps = {}
    for rid, rv in rovers.items():
        ts = [0.0] + [t for t in rv.meet_t if t <= makespan + delta]
        gaps[rid] = max((ts[k + 1] - ts[k] for k in range(len(ts) - 1)), default=0.0)
    return dict(variant=variant, alpha=alpha, early=early, d_star=ds, R_c=R_c,
                delta=delta, sigma_x=sigma_x, ell=ell, feasible=feasible,
                coverage=len(done), n_pois=len(pois), done=done,
                makespan=makespan, meetings=meetings, late=late, gaps=gaps,
                rounds=rounds_used,
                travel=sum(rv.travel for rv in rovers.values()),
                idle=sum(rv.idle for rv in rovers.values()),
                scan=sum(rv.scan_t for rv in rovers.values()),
                sprint_m=sum(rv.sprint_m for rv in rovers.values()),
                tracks={r: rv.track for r, rv in rovers.items()},
                n_coopt=sum(1 for mt in meetings
                            if mt["kind"].startswith("coopt")
                            or mt["kind"] == "cross"),
                n_coopt2=sum(1 for mt in meetings if mt["kind"] == "coopt2"),
                n_fallback=sum(1 for mt in meetings if mt["kind"] == "loiter"
                               and PLACEMENT[variant] == "cross"))


# --- evaluating the C4 guarantee on the realised geometry --------------------
def c4_metrics(res, sigma_x=None, ell=None, eps=EPS, mc_seeds=400, seed0=0,
               model="gaussian"):
    """False-alarm rate of the schedule the router actually produced.

    fa_analytic : Pr[|offset_i - offset_j| > eps] from the closed form -- the
                  quantity C4's d* is sized against (no slack credited).
    fa_mc       : the realised rate, Monte-Carlo over the correlated field, with
                  lateness clipped by each partner's slack:
                      lateness_k = max(0, offset_k - slack_k)
                  i.e. arriving early genuinely absorbs a delay.  This is the
                  honest headline: it credits absorption AND correlation, so the
                  two mechanisms compete on the same axis."""
    sigma_x = res["sigma_x"] if sigma_x is None else sigma_x
    ell = res["ell"] if ell is None else ell
    kappa = kappa_of(sigma_x, res["delta"])
    ms = res["meetings"]
    nan = float("nan")
    if not ms:
        return dict(fa_analytic=nan, fa_nocorr=nan, fa_mc=nan, d_mean=nan,
                    d_max=nan, leg_mean=nan, slack_mean=nan, sat=nan, n=0,
                    eps=eps)
    ps = [p_miss_pair(kappa, mt["t_i"], mt["t_j"], mt["d"], ell, eps) for mt in ms]
    # same schedule, correlation switched OFF: separates the two levers a short
    # aligned approach pulls at once -- small offsets vs offsets that cancel
    pn = [p_miss_pair(kappa, mt["t_i"], mt["t_j"], 1e9, ell, eps) for mt in ms]
    rounds = {}
    for mt in ms:
        rounds.setdefault(mt["round"], []).append(mt)
    rng = np.random.default_rng(seed0)
    amp = calibrate_amp(kappa) if model in ("measured", "carry") else None
    amp_c = amp
    miss = tot = 0
    for _r, grp in rounds.items():
        pts = np.array([p for mt in grp for p in (mt["a_i"], mt["a_j"])])
        ch = _safe_chol(pts, ell)
        legs = np.array([t for mt in grp for t in (mt["t_i"], mt["t_j"])])
        slack = np.array([sv for mt in grp for sv in (mt["s_i"], mt["s_j"])])
        if model == "carry":
            # one field sample covering every travel segment of both partners
            seg_pts, idx = [], []
            for mt in grp:
                for key in ("w_i", "w_j"):
                    k0 = len(seg_pts)
                    seg_pts += [p for p, _d in mt[key]]
                    idx.append((k0, len(seg_pts),
                                np.array([d for _p, d in mt[key]])))
            if not seg_pts:
                continue
            sch = _safe_chol(np.array(seg_pts), ell)
            for _s in range(mc_seeds):
                uu = sch @ rng.standard_normal(len(seg_pts))
                off = np.array([float(np.sum(dur * (kappa * uu[a0:a1] if
                                model == "carry" and False else
                                phi_measured(amp_c * uu[a0:a1]))))
                                for (a0, a1, dur) in idx])
                lateness = np.maximum(0.0, off - slack)
                for e in range(len(grp)):
                    miss += int(abs(lateness[2 * e] - lateness[2 * e + 1]) > eps)
                    tot += 1
            continue
        for _s in range(mc_seeds):
            u = ch @ rng.standard_normal(len(pts))
            # gaussian  : offset = kappa * t * u          (Eq. 3 as written)
            # measured  : offset = t * phi(A u)           (HoloOcean's response)
            off = (kappa * legs * u if model == "gaussian"
                   else legs * phi_measured(amp * u))
            lateness = np.maximum(0.0, off - slack)
            for e in range(len(grp)):
                miss += int(abs(lateness[2 * e] - lateness[2 * e + 1]) > eps)
                tot += 1
    ds = res["d_star"]
    return dict(fa_analytic=100.0 * float(np.mean(ps)),
                fa_nocorr=100.0 * float(np.mean(pn)),
                fa_mc=100.0 * miss / max(tot, 1),
                d_mean=float(np.mean([mt["d"] for mt in ms])),
                d_max=float(np.max([mt["d"] for mt in ms])),
                leg_mean=float(np.mean([mt["t_i"] for mt in ms]
                                       + [mt["t_j"] for mt in ms])),
                slack_mean=float(np.mean([mt["s_i"] for mt in ms]
                                         + [mt["s_j"] for mt in ms])),
                sat=100.0 * sum(1 for mt in ms if mt["d"] <= ds + 1e-6) / len(ms),
                n=len(ms), eps=eps)


# --- experiments -------------------------------------------------------------
POLICIES = [("plain", "plain", {}),
            ("slack e60", "slack", dict(early=60.0)),
            ("slack e120", "slack", dict(early=120.0)),
            ("slack e240", "slack", dict(early=240.0)),
            ("loiter", "loiter", {}),
            ("coopt", "coopt", {}),
            ("cross la1", "cross", dict(lookahead=1)),
            ("cross la4", "cross", dict(lookahead=4))]


def _run_policy(alloc, variant, kw, alpha, sigma_x, ell, delta, mc_seeds,
                schedule="grid"):
    res = weave(alloc, variant, alpha=alpha, sigma_x=sigma_x, ell=ell,
                delta=delta, schedule=schedule, **kw)
    met = c4_metrics(res, mc_seeds=mc_seeds)
    return res, met


def sweep_strength(N=90, K=4, L=600.0, seeds=(5, 11, 23, 37), alpha=0.2,
                   ell=ELL, delta=600.0, strengths=(1.0, 2.0, 3.0, 4.0),
                   mc_seeds=150, schedule="grid"):
    """car_fcd's headline axis (benign false alarms vs current strength), but on
    the real coverage allocation and with the makespan price attached."""
    rows = []
    for sx in strengths:
        for tag, variant, kw in POLICIES:
            ov, fa, fm, sat = [], [], [], []
            for sd in seeds:
                pois, dock = make_scenario(N, L, sd)
                alloc = allocate(pois, K, dock, sd)
                res, met = _run_policy(alloc, variant, kw, alpha, sx, ell,
                                       delta, mc_seeds, schedule)
                assert res["coverage"] == res["n_pois"] and res["feasible"]
                ov.append(100.0 * (res["makespan"] / alloc["makespan"] - 1.0))
                fa.append(met["fa_analytic"]); fm.append(met["fa_mc"])
                sat.append(met["sat"])
            rows.append(dict(policy=tag, sigma_x=sx,
                             ovh=float(np.mean(ov)), ovh_sd=float(np.std(ov)),
                             fa_analytic=float(np.mean(fa)),
                             fa_mc=float(np.mean(fm)), fa_mc_sd=float(np.std(fm)),
                             c4_sat=float(np.mean(sat))))
    return rows


def sweep_alpha(N=90, K=4, L=600.0, seeds=(5, 11, 23, 37), sigma_x=4.0,
                ell=ELL, delta=600.0, alphas=(0.4, 0.2, 0.1, 0.05, 0.02),
                mc_seeds=150, schedule="grid"):
    """The C4 design knob: tightening alpha shrinks d* and so the corridor.
    Shows what buying a stricter guarantee costs on real routes."""
    rows = []
    for a in alphas:
        for tag, variant, kw in (("loiter", "loiter", {}), ("coopt", "coopt", {})):
            ov, fm, sat, ds = [], [], [], None
            for sd in seeds:
                pois, dock = make_scenario(N, L, sd)
                alloc = allocate(pois, K, dock, sd)
                res, met = _run_policy(alloc, variant, kw, a, sigma_x, ell,
                                       delta, mc_seeds, schedule)
                ds = res["d_star"]
                ov.append(100.0 * (res["makespan"] / alloc["makespan"] - 1.0))
                fm.append(met["fa_mc"]); sat.append(met["sat"])
            rows.append(dict(policy=tag, alpha=a, d_star=ds,
                             ovh=float(np.mean(ov)), ovh_sd=float(np.std(ov)),
                             fa_mc=float(np.mean(fm)), c4_sat=float(np.mean(sat))))
    return rows


def sweep_lookahead(N=90, K=4, L=600.0, seeds=(5, 11, 23, 37), alpha=0.2,
                    sigma_x=4.0, ell=ELL, delta=600.0,
                    lookaheads=(0, 1, 2, 3, 4, 6), mc_seeds=150,
                    schedule="grid"):
    """How far ahead the crossing anchors are reserved is the natural-crossing
    design's only real knob, and it traces the whole cost-vs-guarantee frontier:
    reserve nothing and the anchors get eaten by ordinary work; reserve a lot and
    the locked POIs starve the routes."""
    rows = []
    for la in lookaheads:
        ov, fm, fa, lg, fb = [], [], [], [], []
        for sd in seeds:
            pois, dock = make_scenario(N, L, sd)
            alloc = allocate(pois, K, dock, sd)
            res = weave(alloc, "cross", alpha=alpha, sigma_x=sigma_x, ell=ell,
                        delta=delta, lookahead=la, schedule=schedule)
            met = c4_metrics(res, mc_seeds=mc_seeds)
            assert res["coverage"] == res["n_pois"] and res["feasible"]
            ov.append(100.0 * (res["makespan"] / alloc["makespan"] - 1.0))
            fm.append(met["fa_mc"]); fa.append(met["fa_analytic"])
            lg.append(met["leg_mean"])
            fb.append(100.0 * res["n_fallback"] / max(len(res["meetings"]), 1))
        rows.append(dict(lookahead=la, ovh=float(np.mean(ov)),
                         ovh_sd=float(np.std(ov)), fa_mc=float(np.mean(fm)),
                         fa_analytic=float(np.mean(fa)), leg=float(np.mean(lg)),
                         fallback=float(np.mean(fb))))
    return rows


def sweep_absorb(N=90, K=4, L=600.0, seeds=(5, 11, 23, 37, 41, 53), alpha=0.2,
                 sigma_x=4.0, ell=ELL, delta=600.0,
                 absorbs=(0.0, 20.0, 40.0, 60.0, 90.0, 120.0, 180.0),
                 mc_seeds=300):
    """The schedule frontier: what the grid's idle was actually buying.

    A grid schedule leaves every robot standing at the site until the clock
    reaches tau.  That looks like pure waste -- and most of it is -- but the
    hold is also the only delay-ABSORBING mechanism the system has left once
    HoloOcean rules out a stop resetting accrued delay.  The elastic schedule
    removes the wait, so it removes the absorption too, and benign false alarms
    rise.  `absorb` puts the absorption back as an explicit, priced quantity
    rather than an accident of the clock: every partner holds at least that
    long past the later arrival.

    The row at absorb=0 is the aggressive end; the row where false alarms
    rejoin the grid's is the honest headline, because at that point the elastic
    schedule is strictly better -- same cadence, same false-alarm rate, cheaper
    mission.  Everything between is a choice."""
    rows = []
    base = dict(schedule="grid")
    for tag, kw in [("grid", base)] + [("elastic", dict(schedule="elastic",
                                                        absorb=ab))
                                       for ab in absorbs]:
        ov, fc, fg, idl, gp = [], [], [], [], []
        for sd in seeds:
            pois, dock = make_scenario(N, L, sd)
            alloc = allocate(pois, K, dock, sd)
            res = weave(alloc, "plain", alpha=alpha, sigma_x=sigma_x, ell=ell,
                        delta=delta, **kw)
            assert res["coverage"] == res["n_pois"] and res["feasible"]
            assert res["late"] == 0 and max(res["gaps"].values()) <= delta + 1e-6
            ov.append(100.0 * (res["makespan"] / baseline_makespan(alloc) - 1.0))
            fc.append(c4_metrics(res, mc_seeds=mc_seeds, model="carry")["fa_mc"])
            fg.append(c4_metrics(res, mc_seeds=mc_seeds)["fa_mc"])
            idl.append(res["idle"] / 60.0)
            gp.append(max(res["gaps"].values()))
        rows.append(dict(schedule=tag, absorb=kw.get("absorb", float("nan")),
                         ovh=float(np.mean(ov)), ovh_sd=float(np.std(ov)),
                         fa_carry=float(np.mean(fc)), fa_gauss=float(np.mean(fg)),
                         idle_min=float(np.mean(idl)), max_gap=float(max(gp))))
    return rows


def _save_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# --- Layer 2 figure ----------------------------------------------------------
def render_l2(alloc, runs, strength_rows, la_rows, L, path):
    """A: the woven mesh on the coverage base.  B: where the time goes.
    C: the C4 guarantee (and its price) as the current strengthens."""
    fig = plt.figure(figsize=(17.0, 12.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.0],
                          height_ratios=[1.0, 1.0, 0.85], wspace=0.22, hspace=0.42)
    axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[2, 1])

    res = runs["cross la1"][0]
    pois, labels, dock, K = alloc["pois"], alloc["labels"], alloc["dock"], alloc["K"]
    for r in range(K):
        col = PALETTE[r % len(PALETTE)]
        pts = pois[labels == r]
        axA.scatter(pts[:, 0], pts[:, 1], s=26, color=col, edgecolor="k", lw=0.3,
                    zorder=3)
        seg = [(p0, p1) for (_a, _b, p0, p1) in res["tracks"][r]
               if dist(p0, p1) > 1e-9]
        for p0, p1 in seg:
            axA.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", color=col, lw=1.1,
                     alpha=0.55, zorder=2)
    ds = res["d_star"]
    for mt in res["meetings"]:
        m = mt["m"]
        axA.add_patch(plt.Circle((m[0], m[1]), max(ds / 2.0, 7.0), color="#d62728",
                                 alpha=0.16, zorder=4))
        axA.plot([m[0]], [m[1]], "x", color="#d62728", ms=7, mew=1.6, zorder=5)
    axA.scatter([dock[0]], [dock[1]], marker="*", s=320, color="#111",
                edgecolor="w", lw=0.6, zorder=6)
    axA.set_aspect("equal"); axA.set_xlim(-0.03 * L, 1.03 * L)
    axA.set_ylim(-0.03 * L, 1.06 * L); axA.grid(alpha=0.15)
    axA.set_xlabel("x (m)"); axA.set_ylabel("y (m)")
    axA.set_title("LAYER 2 — rendezvous mesh woven onto the 100%%-coverage base\n"
                  "natural crossings: each x is a reserved cross-boundary POI "
                  "pair both partners owed;\n%d meetings, detection gap = %.0f s, "
                  "coverage %d/%d"
                  % (len(res["meetings"]), max(res["gaps"].values()),
                     res["coverage"], res["n_pois"]), fontsize=11)
    axA.legend(handles=[
        Line2D([], [], marker="x", ls="", color="#d62728", label="rendezvous"),
        Line2D([], [], marker="*", ls="", color="#111", label="dock")],
        loc="upper right", fontsize=8.5)

    # B: where the time goes
    names = [k for k in runs]
    base = alloc["makespan"] * alloc["K"]
    sc = np.array([runs[k][0]["scan"] for k in names]) / 60.0
    tr = np.array([runs[k][0]["travel"] / V for k in names]) / 60.0
    idl = np.array([runs[k][0]["idle"] for k in names]) / 60.0
    x = np.arange(len(names))
    axB.bar(x, sc, color="#2ca02c", label="scanning (the mission)")
    axB.bar(x, tr, bottom=sc, color="#1f77b4", label="travel")
    axB.bar(x, idl, bottom=sc + tr, color="#d62728", alpha=0.85,
            label="idle (what alignment costs)")
    axB.set_xticks(x); axB.set_xticklabels(names, fontsize=8.5, rotation=12)
    axB.set_ylabel("team time (min)")
    axB.set_title("B — where the time goes (team totals)", fontsize=10.5)
    axB.legend(fontsize=8); axB.grid(alpha=0.18, axis="y")

    # C: the guarantee, and its price
    pol = sorted({r["policy"] for r in strength_rows})
    style = {"plain": ("#7f7f7f", "o", "-"), "slack e60": ("#1f77b4", "s", "--"),
             "slack e120": ("#17becf", "^", "--"), "slack e240": ("#9467bd", "v", "--"),
             "loiter": ("#d62728", "D", "-"), "coopt": ("#e377c2", "P", "-"),
             "cross la1": ("#2ca02c", "*", "-"), "cross la4": ("#8c564b", "X", "-")}
    for pname in pol:
        rs = [r for r in strength_rows if r["policy"] == pname]
        rs.sort(key=lambda r: r["sigma_x"])
        c, mk, ls = style.get(pname, ("#333", "o", "-"))
        xs = [r["sigma_x"] for r in rs]
        axC.plot(xs, [max(r["fa_mc"], 1e-2) for r in rs], ls, marker=mk, color=c,
                 lw=1.8, ms=5, label="%s  (+%.0f%% makespan)" % (pname, rs[0]["ovh"]))
        axC.plot(xs, [max(r["fa_analytic"], 1e-2) for r in rs], ":", color=c,
                 lw=1.1, alpha=0.75)
    axC.set_yscale("log")
    axC.set_xlabel("current strength  (sigma / EPS)")
    axC.set_ylabel("benign false alarms  (% of meetings)")
    axC.set_title("C — the C4 guarantee is flat in the current; nothing else is\n"
                  "solid = realised (slack credited), dotted = no-slack bound",
                  fontsize=10.5)
    axC.legend(fontsize=7.4, loc="lower right"); axC.grid(alpha=0.2, which="both")

    # D: the cost-vs-guarantee frontier, and where natural crossings sit on it
    for pname, (res_, met_) in runs.items():
        c, mk, _ls = style.get(pname, ("#333", "o", "-"))
        axD.scatter([100 * (res_["makespan"] / alloc["makespan"] - 1)],
                    [max(met_["fa_analytic"], 2e-2)], s=110, color=c, marker=mk,
                    edgecolor="k", lw=0.4, zorder=4, label=pname)
    la_rows = sorted(la_rows, key=lambda r: r["lookahead"])
    axD.plot([r["ovh"] for r in la_rows],
             [max(r["fa_analytic"], 2e-2) for r in la_rows], "-", color="#2ca02c",
             lw=1.6, alpha=0.75, zorder=3,
             label="crossings, reserving 0..%d rounds ahead" % la_rows[-1]["lookahead"])
    for r in la_rows:
        axD.annotate("%d" % r["lookahead"],
                     (r["ovh"], max(r["fa_analytic"], 2e-2)),
                     textcoords="offset points", xytext=(4, 4), fontsize=7.5,
                     color="#2ca02c")
    axD.set_yscale("log")
    axD.set_xlabel("makespan over L0  (%)   <- cheaper")
    axD.set_ylabel("no-slack false-alarm bound  (%)")
    axD.set_title("D — cost vs guarantee.  Natural crossings move the frontier\n"
                  "LEFT: the same protection without buying a dedicated stop",
                  fontsize=10.5)
    axD.legend(fontsize=7.2, loc="upper right"); axD.grid(alpha=0.2, which="both")

    fig.suptitle("Layer 2: paying for rendezvous alignment — a dedicated C4 corridor "
                 "vs natural tour crossings", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --- the schedule figure -----------------------------------------------------
def render_schedule(alloc, rows, seed_runs, path):
    """Three panels: where the mission time goes, the cost/false-alarm frontier
    the absorption margin traces, and the per-window task counts that make the
    asymmetry concrete."""
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16.2, 4.9))
    g, e = seed_runs["grid"], seed_runs["elastic"]

    # A: time budget -- three bars, because the shipped point (absorb*) is a
    # deliberate retreat from the fastest one and the figure should say so.
    z = seed_runs.get("aggressive")
    labels = ["scan", "travel", "idle"]
    series = [("grid (clock)", "#9aa5b1", g),
              ("elastic, absorb*=%.0fs" % ABSORB_STAR, "#1f77b4", e)]
    if z is not None:
        series.append(("elastic, absorb=0", "#8fc7ea", z))
    x = np.arange(3); w = 0.8 / len(series)
    for k, (nm, col, rr) in enumerate(series):
        vals = [rr["scan"] / 60, rr["travel"] / V / 60, rr["idle"] / 60]
        off = (k - (len(series) - 1) / 2) * w
        axA.bar(x + off, vals, w, label=nm, color=col, edgecolor="k",
                linewidth=.6)
        for xi, v in zip(x, vals):
            axA.text(xi + off, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    axA.set_xticks(x); axA.set_xticklabels(labels)
    axA.set_ylabel("robot-minutes, summed over the fleet")
    axA.set_title("A  where the mission time goes\n"
                  "scan is fixed; the whole story is idle", fontsize=10.5)
    axA.legend(fontsize=8); axA.grid(alpha=.2, axis="y")

    # B: the frontier, under BOTH disturbance models.  Which one is right is
    # exactly what the HoloOcean stop test left open, so the figure shows both
    # rather than picking the flattering one.
    gr = [r for r in rows if r["schedule"] == "grid"][0]
    el = sorted([r for r in rows if r["schedule"] == "elastic"],
                key=lambda r: r["absorb"])
    for key, col, nm in (("fa_gauss", "#1f77b4", "final-leg (A-L2) model"),
                         ("fa_carry", "#2ca02c", "carry-forward model")):
        axB.plot([r[key] for r in el], [r["ovh"] for r in el], "o-", color=col,
                 lw=2, ms=5.5,
                 label="elastic, absorb %.0f\u2192%.0f s  [%s]"
                       % (el[0]["absorb"], el[-1]["absorb"], nm))
        axB.plot([gr[key]], [gr["ovh"]], "s", color=col, ms=11,
                 markeredgecolor="k", markeredgewidth=.8)
        axB.annotate("grid", (gr[key], gr["ovh"]), textcoords="offset points",
                     xytext=(9, -3), fontsize=8.5, color=col, fontweight="bold")
    for r in (el[0], el[-1]):
        for key, col in (("fa_gauss", "#1f77b4"), ("fa_carry", "#2ca02c")):
            axB.annotate(f"{r['absorb']:.0f}s", (r[key], r["ovh"]),
                         textcoords="offset points", xytext=(-4, -13),
                         fontsize=7.8, color=col, ha="center")
    star = min(el, key=lambda r: abs(r["absorb"] - ABSORB_STAR))
    for key in ("fa_gauss", "fa_carry"):
        axB.plot([star[key]], [star["ovh"]], "*", ms=19, color="#ffb000",
                 markeredgecolor="k", markeredgewidth=.7, zorder=5,
                 label="shipped: absorb* = %.0f s" % ABSORB_STAR
                 if key == "fa_gauss" else None)
    axB.axhline(gr["ovh"], color="k", ls=":", lw=1, alpha=.55)
    lo = min(r["ovh"] for r in el)
    axB.set_ylim(lo - 1.2, gr["ovh"] + 2.6)
    axB.annotate("", xy=(gr["fa_gauss"], star["ovh"]),
                 xytext=(gr["fa_gauss"], gr["ovh"]),
                 arrowprops=dict(arrowstyle="->", color="#444", lw=1.4))
    axB.text(gr["fa_gauss"] + .5, 0.5 * (gr["ovh"] + star["ovh"]),
             "%.0f pp cheaper at the\nsame false-alarm rate"
             % (gr["ovh"] - star["ovh"]), fontsize=8, color="#333",
             va="center")
    axB.set_xlabel("benign false alarms per meeting (%)")
    axB.set_ylabel("mission cost over coverage-only (%)")
    axB.set_title("B  the idle was buying delay absorption\n"  # noqa: E501
                  "at absorb*: cost %.0f%%\u2192%.0f%%,  FA %+.1f pp "
                  "(carry) / %+.1f pp (A-L2)"
                  % (gr["ovh"], star["ovh"],
                     star["fa_carry"] - gr["fa_carry"],
                     star["fa_gauss"] - gr["fa_gauss"]), fontsize=10.5)
    axB.legend(fontsize=7.6, loc="lower left"); axB.grid(alpha=.2)

    # C: tasks retired per approach window, per partner
    def counts(res):
        out = []
        prev = {r: 0.0 for r in range(alloc["K"])}
        for mm in sorted(res["meetings"], key=lambda z: z["round"]):
            out.append((mm["round"], mm["i"], mm["j"], mm["s_i"], mm["s_j"]))
        return out
    for nm, col, rr in series:
        wv = sorted([w for mm in rr["meetings"] for w in (mm["s_i"], mm["s_j"])])
        axC.plot(np.linspace(0, 100, len(wv)), wv, lw=2.2, color=col,
                 label=f"{nm} (mean {np.mean(wv):.0f} s)")
    axC.axhline(SCAN, color="k", ls="--", lw=1)
    axC.text(2, SCAN + 8, "one scan = 60 s: a wait shorter than this\n"
             "cannot be filled with work, however it is packed",
             fontsize=7.8)
    axC.set_xlabel("percentile of appointments")
    axC.set_ylabel("time spent waiting at the meeting site (s)")
    axC.set_title("C  the wait distribution\n"
                  "the grid\u2019s long tail is the clock, not the meeting",
                  fontsize=10.5)
    axC.legend(fontsize=8.5, loc="upper left"); axC.grid(alpha=.2)

    fig.suptitle("Scheduling the appointment: packed windows, a floating "
                 "meeting time, and a rolling deadline — at an unchanged "
                 "detection cadence", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Layer 2 self-tests ------------------------------------------------------
def run_dry_l2():
    print("[dry_run] layer 2 self-tests")
    # 1) the physics agrees with the validated prototype, exactly
    for sx in (1.0, 2.0, 3.0):
        for d in (5.0, 30.0, 150.0):
            assert abs(p_miss(sx * EPS, d, ELL) - cf.analytic_p(sx * EPS, d, ELL)) \
                < 1e-12, "p_miss diverges from car_fcd.analytic_p"
    # 2) d* inverts car_fcd's required_align on the same geometry
    for sx in (1.0, 2.0, 4.0):
        for a in (0.05, 0.2, 0.4):
            d0 = 150.0
            al_cf = cf.required_align(sx * EPS, ELL, d0, a)
            ds = d_star(sx * EPS, ELL, a)
            expect = 0.0 if ds >= d0 else min(0.99, 1 - ds / d0)
            assert abs(al_cf - expect) < 1e-9, "d* inconsistent with car_fcd"
    print("  ok  physics matches car_fcd (analytic_p, required_align)")

    # 3) the cadence feasibility bound is real: Delta=300 strands work here
    pois, dock = make_scenario(90, 600.0, 5)
    alloc = allocate(pois, 4, dock, 5)
    ok300, need = cadence_feasible(alloc, 300.0, 0.0)
    assert not ok300 and need > 300.0, "expected Delta=300 to be infeasible"
    tight = weave(alloc, "plain", delta=300.0)
    assert not tight["feasible"], "infeasible cadence should be flagged"
    assert cadence_feasible(alloc, 600.0, 0.0)[0], "Delta=600 should be feasible"
    print("  ok  cadence bound: needs Delta >= %.0f s; Delta=300 flagged "
          "infeasible, Delta=600 feasible" % need)

    # 4) every variant keeps coverage HARD and meets the detection cadence
    runs = {}
    for tag, variant, kw in POLICIES:
        res = weave(alloc, variant, alpha=0.2, sigma_x=4.0, delta=600.0, **kw)
        met = c4_metrics(res, mc_seeds=80)
        runs[tag] = (res, met)
        assert res["feasible"], "%s: cadence stranded work" % tag
        assert res["coverage"] == res["n_pois"], "%s: coverage != 100%%" % tag
        assert len(set(res["done"])) == len(res["done"]), "%s: POI scanned twice" % tag
        assert max(res["gaps"].values()) <= 600.0 + 1e-6, \
            "%s: detection gap exceeds Delta" % tag
        assert res["late"] == 0, "%s: missed an appointment" % tag
        print("    %-11s cover %d/%d  makespan %5.1f min  gap %3.0f s  "
              "FA %4.1f%% (analytic %4.1f%%)  C4 %3.0f%%"
              % (tag, res["coverage"], res["n_pois"], res["makespan"] / 60,
                 max(res["gaps"].values()), met["fa_mc"], met["fa_analytic"],
                 met["sat"]))
    print("  ok  100% coverage + detection gap <= Delta for every variant")

    # 4b) natural crossings: cheaper than the bare mesh, far better protected
    cr, cm = runs["cross la1"]
    pl, pm = runs["plain"]
    assert cr["makespan"] <= pl["makespan"] * 1.05, \
        "natural crossings should not cost more than the bare mesh"
    assert cm["fa_mc"] < 0.25 * pm["fa_mc"] + 0.05, \
        "natural crossings should cut realised false alarms"
    assert cm["leg_mean"] < 0.6 * pm["leg_mean"], \
        "crossings should shorten the final approach"
    assert runs["cross la4"][1]["fa_analytic"] < 0.1 * cm["fa_analytic"] + 0.5, \
        "a deeper reservation should tighten the no-slack bound"
    print("  ok  crossings: %.1f%% vs plain %.1f%% makespan, realised FA "
          "%.2f%% vs %.2f%%, leg %.0f s vs %.0f s"
          % (100 * (cr["makespan"] / alloc["makespan"] - 1),
             100 * (pl["makespan"] / alloc["makespan"] - 1),
             cm["fa_mc"], pm["fa_mc"], cm["leg_mean"], pm["leg_mean"]))

    # 5) C4 is satisfied by the corridor variants and violated by the others
    for tag in ("loiter", "coopt"):
        assert runs[tag][1]["sat"] > 99.0, "%s does not satisfy C4" % tag
        assert runs[tag][1]["fa_analytic"] < 1.0, "%s: C4 bound not delivered" % tag
    assert runs["plain"][1]["sat"] < 50.0, "plain should violate C4"
    assert runs["plain"][1]["d_mean"] > 10 * runs["loiter"][1]["d_mean"], \
        "alignment did not shrink the approach separation"
    print("  ok  C4 satisfied by corridor variants (sat>99%), violated by plain")

    # 6) alignment is not free: it costs makespan
    assert runs["loiter"][0]["makespan"] > runs["plain"][0]["makespan"], \
        "alignment came for free -- the cost model is broken"
    assert runs["plain"][0]["makespan"] > alloc["makespan"], "mesh came for free"
    print("  ok  cost ordering: L0 < plain mesh < aligned mesh")

    # 7) the guarantee is realization-invariant; the unaligned schedule is not
    fa_plain, fa_align = [], []
    for sx in (1.0, 2.0, 4.0):
        fa_plain.append(c4_metrics(weave(alloc, "plain", alpha=0.2, sigma_x=sx,
                                         delta=600.0), mc_seeds=80)["fa_analytic"])
        fa_align.append(c4_metrics(weave(alloc, "loiter", alpha=0.2, sigma_x=sx,
                                         delta=600.0), mc_seeds=80)["fa_analytic"])
    assert fa_plain[0] < fa_plain[1] < fa_plain[2], "unaligned FA should diverge"
    assert max(fa_align) < 1.0, "aligned FA should stay flat"
    print("  ok  FA vs strength: unaligned %.1f->%.1f%%, aligned %.1f->%.1f%%"
          % (fa_plain[0], fa_plain[-1], fa_align[0], fa_align[-1]))

    # 8) the elastic schedule is a free lunch: strictly cheaper mission at an
    #    unchanged guarantee.  Both halves of that claim are asserted, because
    #    the tempting version of this change (float on an absolute grid) buys
    #    its speed by widening the detection gap and would pass the first
    #    assertion alone.
    seeds8 = (5, 11, 23, 37, 41, 53, 67, 71, 83, 97)
    gain, waits, win = [], [], 0
    for seed in seeds8:
        ps, dk = make_scenario(90, 600.0, seed)
        al = allocate(ps, 4, dk, seed)
        g = weave(al, "plain", delta=600.0, schedule="grid")
        e = weave(al, "plain", delta=600.0, schedule="elastic")
        z = weave(al, "plain", delta=600.0, schedule="elastic", absorb=0.0)
        for r in (g, e, z):
            assert r["feasible"] and r["coverage"] == r["n_pois"], "elastic: coverage"
            assert len(set(r["done"])) == len(r["done"]), "elastic: POI scanned twice"
            assert r["late"] == 0, "elastic: missed an appointment"
            assert max(r["gaps"].values()) <= max(g["gaps"].values()) + 1e-6, \
                "elastic widened the detection gap -- the gain would be borrowed"
        # absorb=0 is the unconditional claim: it beats the grid on every
        # instance. At absorb* the schedule is holding 180 s back for
        # absorption, so it wins on the mean and on most -- not all -- seeds,
        # and the test says so rather than quietly dropping the losers.
        assert z["makespan"] < g["makespan"], \
            "absorb=0 should beat the grid on every seed (%d)" % seed
        win += e["makespan"] < g["makespan"]
        bl = baseline_makespan(al)
        gain.append((g["makespan"] / bl - 1, e["makespan"] / bl - 1))
        waits.append((g["idle"] / 60, e["idle"] / 60))
    og = 100 * float(np.mean([x[0] for x in gain]))
    ng = 100 * float(np.mean([x[1] for x in gain]))
    assert ng < og - 5.0, "elastic gain should be worth several points"
    assert win >= 0.75 * len(seeds8), "elastic should win the large majority"
    print("  ok  elastic schedule (absorb*=%.0f s): mesh cost %.1f%% -> %.1f%% "
          "of L0, idle %.1f -> %.1f min, detection gap never widened, wins "
          "%d/%d seeds (absorb=0 wins all %d)"
          % (ABSORB_STAR, og, ng, float(np.mean([x[0] for x in waits])),
             float(np.mean([x[1] for x in waits])), win, len(seeds8),
             len(seeds8)))

    # 9) ... but the idle it removes was also doing work. Holding to a time
    #    fixed in advance is the only delay-ABSORBING mechanism left once a
    #    stop is ruled out (HoloOcean, see _window_legs), so absorb=0 trades
    #    false alarms for speed. The claim that survives is the `absorb`
    #    frontier's far end: put the absorption back explicitly and the
    #    elastic schedule still wins on cost at no cost in false alarms.
    cg, ce, fg_, fe = [], [], [], []
    for seed in (5, 11, 23):
        ps, dk = make_scenario(90, 600.0, seed)
        al = allocate(ps, 4, dk, seed)
        g = weave(al, "plain", alpha=0.2, sigma_x=4.0, delta=600.0,
                  schedule="grid")
        e = weave(al, "plain", alpha=0.2, sigma_x=4.0, delta=600.0,
                  schedule="elastic", absorb=180.0)
        z = weave(al, "plain", alpha=0.2, sigma_x=4.0, delta=600.0,
                  schedule="elastic", absorb=0.0)
        assert max(e["gaps"].values()) <= 600.0 + 1e-6 and e["late"] == 0
        assert e["coverage"] == e["n_pois"]
        cg.append(g["makespan"]); ce.append(e["makespan"])
        fg_.append(c4_metrics(g, mc_seeds=120)["fa_mc"])
        fe.append(c4_metrics(e, mc_seeds=120)["fa_mc"])
        assert c4_metrics(z, mc_seeds=120)["fa_mc"] > fg_[-1], \
            "absorb=0 should cost false alarms -- that is the trade being priced"
    assert float(np.mean(ce)) < float(np.mean(cg)), "absorb=180 should still be cheaper"
    assert float(np.mean(fe)) <= float(np.mean(fg_)) + 0.5, \
        "absorb=180 should not cost false alarms"
    print("  ok  absorption is priced, not lost: absorb=180 is %.1f min cheaper "
          "than the grid at FA %.1f%% vs %.1f%%; absorb=0 is faster still but "
          "trades false alarms"
          % ((float(np.mean(cg)) - float(np.mean(ce))) / 60,
             float(np.mean(fe)), float(np.mean(fg_))))
    print("[dry_run] layer 2 self-tests passed")


def main_l2(args):
    pois, dock = make_scenario(args.N, args.L, args.seed)
    alloc = allocate(pois, args.K, dock, args.seed)
    ok, need = cadence_feasible(alloc, args.delta, 0.0)
    print("[L0] K=%d N=%d: 100%% coverage, makespan %.1f min"
          % (args.K, args.N, alloc["makespan"] / 60))
    print("[L2] cadence Delta=%.0f s %s (this instance needs >= %.0f s)"
          % (args.delta, "OK" if ok else "INFEASIBLE", need))
    if not ok:
        print("     -> raise --delta or --K; a stranded route is not a result.")
    print("[C4] sigma=%.0fx EPS, ell=%.0f m, alpha=%.2f  ->  d* = %.1f m"
          % (args.sigma_x, args.ell, args.alpha,
             d_star(args.sigma_x * EPS, args.ell, args.alpha)))

    runs = {}
    print("\n[sched] appointments timed: %s" % args.schedule)
    print("\n%-11s %8s %10s %9s %9s %8s %8s"
          % ("policy", "cover", "makespan", "vs L0", "FA real", "FA anal", "C4 sat"))
    for tag, variant, kw in POLICIES:
        res, met = _run_policy(alloc, variant, kw, args.alpha, args.sigma_x,
                               args.ell, args.delta, 300, args.schedule)
        runs[tag] = (res, met)
        print("%-11s %5d/%-3d %7.1f min %+8.0f%% %8.1f%% %8.1f%% %7.0f%%"
              % (tag, res["coverage"], res["n_pois"], res["makespan"] / 60,
                 100 * (res["makespan"] / alloc["makespan"] - 1),
                 met["fa_mc"], met["fa_analytic"], met["sat"]))

    srows = sweep_strength(N=args.N, K=args.K, L=args.L, alpha=args.alpha,
                           ell=args.ell, delta=args.delta,
                           schedule=args.schedule)
    print("\n[sweep] %s" % _save_rows(os.path.join(OUT, "l2_strength.csv"), srows))
    lrows = sweep_lookahead(N=args.N, K=args.K, L=args.L, alpha=args.alpha,
                            sigma_x=args.sigma_x, ell=args.ell,
                            delta=args.delta, schedule=args.schedule)
    print("[sweep] %s" % _save_rows(os.path.join(OUT, "l2_lookahead.csv"), lrows))
    for r in lrows:
        print("  reserve %d rounds ahead: +%5.1f%% makespan  FA %5.2f%%  "
              "no-slack bound %5.1f%%  leg %4.0f s  fallback %3.0f%%"
              % (r["lookahead"], r["ovh"], r["fa_mc"], r["fa_analytic"],
                 r["leg"], r["fallback"]))
    if args.sweep:
        arows = sweep_alpha(N=args.N, K=args.K, L=args.L, sigma_x=args.sigma_x,
                            ell=args.ell, delta=args.delta,
                            schedule=args.schedule)
        print("[sweep] %s" % _save_rows(os.path.join(OUT, "l2_alpha.csv"), arows))
        for r in arows:
            print("  alpha %.2f  d* %5.1f m  %-7s  +%5.1f%% makespan  "
                  "FA %4.1f%%  C4 sat %3.0f%%"
                  % (r["alpha"], r["d_star"], r["policy"], r["ovh"],
                     r["fa_mc"], r["c4_sat"]))

    path = render_l2(alloc, runs, srows, lrows, args.L,
                     os.path.abspath(os.path.join(OUT, "layer2_c4.png")))
    print("[figure] %s" % path)


if __name__ == "__main__":
    main()


# --- the admissible-mission envelope -----------------------------------------
# The question a mission planner actually asks is not "how cheap is alignment"
# but "can I fly this mission at all". A mission is ADMISSIBLE for a policy when
# that policy simultaneously covers every task, meets the detection cadence,
# finishes inside the deadline, and holds benign false alarms at or below the
# target. Absorption (arrive early) buys false-alarm reduction with time, so it
# trades directly against the deadline -- and it cannot be pushed arbitrarily
# far, because past a point the reserved slack strands work (Prop. 3). Alignment
# buys the same reduction out of route ORDER, which the deadline does not price.
# The envelope is where that difference becomes a capability rather than a saving.
ENVELOPE_POLICIES = [
    ("unaligned mesh", "plain", {}),
    ("absorb +60 s", "slack", dict(early=60.0)),
    ("absorb +120 s", "slack", dict(early=120.0)),
    ("absorb +240 s", "slack", dict(early=240.0)),
    ("C4 corridor", "loiter", {}),
    ("crossings h=1", "cross", dict(lookahead=1)),
    ("crossings h=4", "cross", dict(lookahead=4)),
]


def sweep_envelope(N=90, K=4, L=600.0, seeds=(5, 11, 23, 37), alpha=0.2,
                   ell=ELL, delta=600.0, strengths=(1.0, 2.0, 3.0, 4.0),
                   deadlines=(1.3, 1.5, 1.75, 2.0, 2.5, 3.5), fa_target=1.0,
                   mc_seeds=300, model="measured"):
    """For each policy: its makespan (which the disturbance does not affect, so
    the route is woven once per seed) and its false-alarm rate at each strength.
    Returns per-policy rows plus the admissibility grid over (deadline, sigma)."""
    rows = []
    for tag, variant, kw in ENVELOPE_POLICIES:
        ov, fa, feas = [], {sx: [] for sx in strengths}, True
        for sd in seeds:
            pois, dock = make_scenario(N, L, sd)
            alloc = allocate(pois, K, dock, sd)
            res = weave(alloc, variant, alpha=alpha, sigma_x=1.0, ell=ell,
                        delta=delta, **kw)
            if not (res["feasible"] and res["coverage"] == res["n_pois"]):
                feas = False
                break
            ov.append(100.0 * (res["makespan"] / alloc["makespan"] - 1.0))
            for sx in strengths:
                res["sigma_x"] = sx
                fa[sx].append(c4_metrics(res, sigma_x=sx, mc_seeds=mc_seeds,
                                         model=model)["fa_mc"])
        if not feas:
            rows.append(dict(policy=tag, feasible=False, ovh=float("nan")))
            continue
        row = dict(policy=tag, feasible=True, ovh=float(np.mean(ov)))
        for sx in strengths:
            row["fa_%g" % sx] = float(np.mean(fa[sx]))
        rows.append(row)

    grid = []
    for D in deadlines:
        for sx in strengths:
            adm = [r["policy"] for r in rows
                   if r["feasible"] and (1 + r["ovh"] / 100.0) <= D
                   and r["fa_%g" % sx] <= fa_target]
            grid.append(dict(deadline=D, sigma_x=sx, n_admissible=len(adm),
                             admissible=" | ".join(adm) if adm else "NONE"))
    return rows, grid


# --- does the reserve actually buy anything? ---------------------------------
def takeover_test(alloc, victim=0, delta=500.0, sprint=True, balance=True,
                  seed=0):
    """Kill one robot, hand its tasks to the nearest survivors, and re-execute
    the mesh. The survivors are now carrying work they never planned for, and
    still owe every appointment they agreed to. This is the only situation in a
    deterministic mission where the speed reserve has anything to do, so it is
    the test that decides whether holding the reserve was worth its cost in
    cadence."""
    pois, labels, dock, K = (alloc["pois"], alloc["labels"].copy(),
                             alloc["dock"], alloc["K"])
    cen = alloc["centroids"]
    surv = [r for r in range(K) if r != victim]
    for i in np.where(labels == victim)[0]:
        labels[i] = min(surv, key=lambda t: dist(pois[i], cen[t]))
    remap = {r: k for k, r in enumerate(surv)}
    sub = dict(alloc)
    sub["labels"] = np.array([remap[r] for r in labels])
    sub["K"] = K - 1
    sub["centroids"] = np.array([pois[sub["labels"] == k].mean(0)
                                 if np.any(sub["labels"] == k) else cen[surv[k]]
                                 for k in range(K - 1)])
    times, orders = _robot_times(pois, sub["labels"], dock, K - 1)
    sub["orders"], sub["times"] = orders, times
    sub["makespan"] = float(times.max())
    sub["t_nom"] = nominal_times(pois, sub["labels"], dock, orders, K - 1)
    res = weave(sub, "plain", delta=delta, sprint=sprint, balance=balance)
    return dict(feasible=res["feasible"], coverage=res["coverage"],
                n_pois=res["n_pois"], late=res["late"],
                n_meetings=len(res["meetings"]),
                makespan=res["makespan"], sprint_m=res["sprint_m"],
                idle=res["idle"])
