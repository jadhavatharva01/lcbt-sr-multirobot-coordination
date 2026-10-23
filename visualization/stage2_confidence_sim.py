"""
STAGE 2 (E3): channel-gated commit-confidence vs margin-only, under blackout.

The allocation decision (NOT failure recovery). A robot commits to a task when
it believes it is the best bidder for it. Distributed + lossy comms means a robot
only knows the bids it has HEARD. The failure mode of a margin-only rule:

    During a blackout a robot hears few/no competing bids, so its apparent margin
    over the "best heard competitor" looks huge -> FAKE confidence -> it commits
    fast -> but a stronger bidder it never heard also commits -> COLLISION (two
    robots on one task) and a suboptimal assignment.

THE IDEA (same principle as the detector): the absence of competing bids is only
meaningful relative to channel health. Confidence = bid-margin x channel-health.
A big margin during a blackout (low channel-health) yields LOW confidence, so the
robot does not commit on information it never actually received.

THE HONEST TEST (the skeptic's null hypothesis): maybe channel-gating is just
"wait longer" in disguise -- you could get the same safety by raising every
robot's wait cap uniformly. So we DON'T compare at a single operating point; we
sweep the wait cap for BOTH policies and compare the collisions-vs-latency
Pareto curve. Channel-gating wins ONLY if it gives fewer collisions at the SAME
mean commit latency -- i.e. it waits SELECTIVELY (only when the channel is bad)
rather than everyone waiting always.
"""

import argparse
import random
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from unified_comparison_sim import (  # noqa: E402
    Comms, make_channel, dist, average, save_csv, frange, line_plot, print_table,
)


def make_costs(seed, n, k):
    """Spatial cost matrix: cost[r][t] = distance(robot r, task t)."""
    rng = random.Random(seed * 2657 + 19)
    rpos = [(rng.uniform(0.0, 50.0), rng.uniform(0.0, 45.0)) for _ in range(n)]
    tpos = [(rng.uniform(5.0, 50.0), rng.uniform(5.0, 45.0)) for _ in range(k)]
    cost = [[dist(rpos[r], tpos[t]) for t in range(k)] for r in range(n)]
    return cost


def optimal_cost(cost, n, k):
    """Min-cost assignment of k distinct tasks to k distinct robots (brute force,
    k small)."""
    best = float("inf")
    for combo in permutations(range(n), k):
        c = sum(cost[combo[t]][t] for t in range(k))
        best = min(best, c)
    return best


def simulate_allocation(seed, comms, policy, cfg):
    n, k = cfg["n_robots"], cfg["n_tasks"]
    thr, w_max, scale = cfg["thr"], cfg["w_max"], cfg["scale"]
    cost = make_costs(seed, n, k)
    rng = random.Random(seed * 7919 + 5)
    drop_prob = make_channel(comms, rng)
    window = cfg["window"]

    # per-robot knowledge
    heard_round = [[-1] * n for _ in range(n)]      # last round r heard from j
    heard_bid = [[float("inf")] * k for _ in range(n)]  # best OTHERS' bid r heard for task t
    known_taken = [set() for _ in range(n)]         # tasks r has heard are claimed
    committed = [None] * n                            # task each robot committed to
    commit_round = [None] * n

    def target_of(r):
        opts = [t for t in range(k) if t not in known_taken[r]]
        if not opts:
            return None
        return min(opts, key=lambda t: cost[r][t])

    def channel_health(r, t_now):
        heard = sum(1 for j in range(n)
                    if j != r and heard_round[r][j] >= t_now - window)
        return heard / (n - 1) if n > 1 else 1.0

    for t_now in range(w_max + 1):
        # ---- broadcast + reception ----
        msgs = []
        for j in range(n):
            tj = committed[j] if committed[j] is not None else target_of(j)
            if tj is not None:
                msgs.append((j, tj, cost[j][tj], committed[j] is not None))
        for r in range(n):
            for (j, tj, bid, is_commit) in msgs:
                if j == r:
                    continue
                if rng.random() >= drop_prob(t_now):
                    heard_round[r][j] = t_now
                    if bid < heard_bid[r][tj]:
                        heard_bid[r][tj] = bid
                    if is_commit:
                        known_taken[r].add(tj)

        # ---- commit decisions ----
        for r in range(n):
            if committed[r] is not None:
                continue
            tgt = target_of(r)
            if tgt is None:
                continue
            own = cost[r][tgt]
            second = heard_bid[r][tgt]
            forced = (t_now >= w_max)
            # if r heard a strictly better bidder for tgt, it is NOT the winner
            if own > second:
                if forced:
                    committed[r] = tgt; commit_round[r] = t_now
                continue
            margin = max(0.0, min(1.0, (second - own) / scale)) if second < float("inf") else 1.0
            if policy == "margin_only":
                conf = margin
            else:  # channel_gated
                conf = margin * channel_health(r, t_now)
            if conf >= thr or forced:
                committed[r] = tgt; commit_round[r] = t_now

    # ---- outcomes ----
    by_task = {t: [] for t in range(k)}
    for r in range(n):
        if committed[r] is not None:
            by_task[committed[r]].append(r)
    collisions = sum(1 for t in range(k) if len(by_task[t]) >= 2)
    coverage = sum(1 for t in range(k) if len(by_task[t]) >= 1) / k
    latencies = [commit_round[r] for r in range(n) if commit_round[r] is not None]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    # assignment quality: best committer per covered task vs optimal
    realized = sum(min(cost[r][t] for r in by_task[t]) for t in range(k) if by_task[t])
    wasted = sum(cost[r][t] for t in range(k) for r in by_task[t][1:])  # duplicate travel
    opt = optimal_cost(cost, n, k)
    regret = (realized - opt) / opt if opt > 0 else 0.0

    return {
        "mode": policy, "policy": policy, "w_max": w_max,
        "collisions": collisions, "coverage": coverage,
        "mean_commit_latency": mean_latency,
        "regret": regret, "wasted_travel": wasted,
    }


METRICS = ["collisions", "coverage", "mean_commit_latency", "regret", "wasted_travel"]


def cfg_of(args, w_max):
    return {"n_robots": args.n_robots, "n_tasks": args.n_tasks, "thr": args.thr,
            "w_max": w_max, "scale": args.scale, "window": args.window}


def cmd_stage2(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    regimes = [("iid", False), ("burst", True)]
    w_maxes = list(range(0, args.max_wait + 1, args.wait_step))
    policies = ["margin_only", "channel_gated"]
    all_rows = []

    for tag, burst in regimes:
        comms = Comms(delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
                      drop_rate=args.drop_rate, burst=burst, burst_drop=args.burst_drop)
        # sweep the wait cap -> a (latency, collisions) curve per policy
        curve = {p: {"lat": [], "coll": [], "reg": []} for p in policies}
        for w in w_maxes:
            for p in policies:
                tr = [simulate_allocation(8000 + i, comms, p, cfg_of(args, w))
                      for i in range(args.trials)]
                avg = average(tr); avg["w_max"] = w; avg["policy"] = p
                all_rows.append({"regime": tag, **avg})
                curve[p]["lat"].append(avg["mean_commit_latency"])
                curve[p]["coll"].append(avg["collisions"])
                curve[p]["reg"].append(avg["regret"])

        # collisions vs wait cap (both policies)
        line_plot(w_maxes, {p: curve[p]["coll"] for p in policies},
                  f"stage2_{tag}_collisions_vs_wait", "wait cap (rounds)", out)
        line_plot(w_maxes, {p: curve[p]["lat"] for p in policies},
                  f"stage2_{tag}_latency_vs_wait", "wait cap (rounds)", out)

        print(f"\n===== STAGE 2 ({tag}, drop={args.drop_rate}) : "
              f"collisions & latency vs wait cap =====")
        print(f"  {'w_max':>6}"
              + f"{'margin coll':>14}{'gated coll':>14}"
              + f"{'margin lat':>14}{'gated lat':>14}")
        for i, w in enumerate(w_maxes):
            print(f"  {w:>6}"
                  f"{curve['margin_only']['coll'][i]:>14.3f}"
                  f"{curve['channel_gated']['coll'][i]:>14.3f}"
                  f"{curve['margin_only']['lat'][i]:>14.3f}"
                  f"{curve['channel_gated']['lat'][i]:>14.3f}")

        # ---- the honest Pareto test: at matched latency, who has fewer collisions? ----
        print(f"  --- Pareto (matched latency) [{tag}] ---")
        for i, w in enumerate(w_maxes):
            gl = curve["channel_gated"]["lat"][i]
            gc = curve["channel_gated"]["coll"][i]
            # find margin_only operating point with latency closest to gl
            j = min(range(len(w_maxes)),
                    key=lambda jj: abs(curve["margin_only"]["lat"][jj] - gl))
            mc = curve["margin_only"]["coll"][j]
            verdict = "gated better" if gc < mc - 1e-6 else (
                      "margin better" if mc < gc - 1e-6 else "tie")
            print(f"  gated(w={w}, lat={gl:.2f}, coll={gc:.3f})  vs  "
                  f"margin@matched(lat={curve['margin_only']['lat'][j]:.2f}, "
                  f"coll={mc:.3f})  -> {verdict}")

    save_csv(all_rows, out / "stage2_all.csv")
    print(f"\nSaved CSV + plots -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=400)
    p.add_argument("--n_robots", type=int, default=6)
    p.add_argument("--n_tasks", type=int, default=4)
    p.add_argument("--delay_mean", type=float, default=1.0)
    p.add_argument("--delay_jitter", type=float, default=0.5)
    p.add_argument("--drop_rate", type=float, default=0.2)
    p.add_argument("--burst_drop", type=float, default=0.97)
    p.add_argument("--thr", type=float, default=0.5, help="confidence threshold to commit")
    p.add_argument("--scale", type=float, default=15.0, help="bid-margin normalization")
    p.add_argument("--window", type=int, default=3, help="rounds for channel-health")
    p.add_argument("--max_wait", type=int, default=10)
    p.add_argument("--wait_step", type=int, default=2)
    p.add_argument("--output_dir", type=str,
                   default=str(Path.home() / "phase1a_ws/results/stage2"))
    args = p.parse_args()
    cmd_stage2(args)


if __name__ == "__main__":
    main()
