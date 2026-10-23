"""
Multi-task / multi-zone extension of the unified honest comparison.

Single-task version asked: "who retrieves the one object?" -> a contention
problem. This asks the harder question: "K objects appear at K different
zones at once; the fleet must SPREAD OUT to cover them." That is a
decentralized multi-robot task-allocation (MRTA) problem on top of the same
asynchronous comms model.

Why multi-task makes the disagreement-tolerance story sharper
-------------------------------------------------------------
With one task, bad comms cause duplicate activation on that task. With K
tasks, bad comms cause two coupled failures at once:
  * over-allocation: several robots pile onto the same "obvious best" task,
  * under-coverage: other tasks are left with nobody assigned.
A method that merely reaches consensus slowly (CBBA) pays a lot of messages.
Our layered method instead tolerates inconsistent local assignments and
repairs them at execution time: per-zone intent/heartbeat/grasp guards bump
losers off a contested zone, and bumped robots RE-ASSIGN to an uncovered
task. Headline metric becomes task COVERAGE and per-zone collisions, not just
who-wins-one-object.

Everything reuses the comms model, policy zoo, and physics from
unified_comparison_sim, so methods are still only distinguished by policy.

Subcommands: compare | sweep   (ablate handled by single-task file)
"""

import argparse
import itertools
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from unified_comparison_sim import (  # noqa: E402
    POI_MAP, BASE_ROBOTS, Comms, dist, effective_speed, compute_bid,
    build_policies, average, save_csv, frange, bar_plot, line_plot, print_table,
    simulate_consensus, make_channel,
)


# ----------------------------------------------------------------------------
# World: K tasks at K distinct zones, N robots (one body each)
# ----------------------------------------------------------------------------
def make_world(seed, num_tasks, jitter, nominal_speed, crit_duration,
               search_visit_time):
    rng = random.Random(seed)
    task_pois = rng.sample(list(POI_MAP.keys()), num_tasks)

    robots = {}
    for rid, base in BASE_ROBOTS.items():
        prof = {
            "pos": (base["pos"][0] + rng.uniform(-jitter, jitter),
                    base["pos"][1] + rng.uniform(-jitter, jitter)),
            "battery": base["battery"], "health": base["health"],
            "obstacle_density": base["obstacle_density"],
        }
        v = max(effective_speed(prof, nominal_speed), 0.01)
        cost, travel = {}, {}
        for k in task_pois:
            poi = POI_MAP[k]
            cost[k] = compute_bid(prof, poi)
            travel[k] = dist(prof["pos"], poi) / v
        robots[rid] = {"pos": prof["pos"], "cost": cost, "travel": travel}

    # the cluster of victims is revealed by one sweep
    detection_time = (rng.randint(1, 10)) * search_visit_time
    return robots, task_pois, detection_time, crit_duration


# greedy sequential assignment from a (possibly partial) cost view
def greedy_assign(cost_view, tasks):
    """cost_view: {rid: {task: cost}} -> {rid: task} (each task to <=1 robot)."""
    pairs = sorted(
        (c, rid, k)
        for rid, row in cost_view.items()
        for k, c in row.items() if k in tasks)
    taken_r, taken_k, result = set(), set(), {}
    for c, rid, k in pairs:
        if rid not in taken_r and k not in taken_k:
            result[rid] = k
            taken_r.add(rid)
            taken_k.add(k)
    return result


def optimal_assignment(robots, tasks):
    """Brute-force min-total-cost assignment for the oracle reference."""
    rids = list(robots.keys())
    best, best_cost = None, math.inf
    for combo in itertools.permutations(rids, len(tasks)):
        c = sum(robots[combo[i]]["cost"][tasks[i]] for i in range(len(tasks)))
        if c < best_cost:
            best_cost, best = c, {combo[i]: tasks[i] for i in range(len(tasks))}
    return best or {}


# ----------------------------------------------------------------------------
# Trial
# ----------------------------------------------------------------------------
def run_trial(seed, comms: Comms, policy, cfg):
    robots, tasks, detection_time, crit_duration = make_world(
        seed, cfg["num_tasks"], cfg["position_jitter"], cfg["nominal_speed"],
        cfg["crit_duration"], cfg["search_visit_time"])
    n = len(robots)
    K = len(tasks)
    mutex = policy.use_grasp_zone or cfg["universal_mutex"]
    # re-assignment requires awareness that a task got taken (intent/heartbeat)
    use_reassign = policy.use_intent or policy.use_heartbeat

    # ---- Oracle: optimal assignment, no comms, no conflict ----
    if policy.name == "oracle":
        assign = optimal_assignment(robots, tasks)
        done_times = []
        energy = 0.0
        for rid, k in assign.items():
            r = robots[rid]
            done_times.append(detection_time + r["travel"][k] + crit_duration)
            energy += dist(r["pos"], POI_MAP[k])
        return metrics(
            policy, comms, K, covered=len(assign), collisions=0,
            makespan=max(done_times, default=0.0), energy=energy,
            messages=n, activated=len(assign), wasted_robots=n - len(assign),
            reassigns=0, success_all=1 if len(assign) == min(n, K) else 0)

    rng = random.Random(seed * 7919 + 13)
    hb_period = cfg["hb_period"]
    drop_prob = make_channel(comms, rng)

    # deliver() only SAMPLES the channel; message accounting is explicit below.
    def deliver(send_time):
        if rng.random() < drop_prob(send_time):
            return None
        return send_time + max(0.0, rng.gauss(comms.delay_mean, comms.delay_jitter))

    rounds = max(1, policy.consensus_rounds)
    window = policy.window_factor * (comms.delay_mean + comms.delay_jitter)
    if rounds > 1:
        window = rounds * comms.delay_mean
    decision_time = detection_time + window

    # ---- CBBA consensus path: agree on a conflict-free assignment, THEN act ----
    if policy.consensus:
        # Anytime/deadline behaviour: a mission deadline caps how long CBBA may
        # deliberate. It runs consensus rounds until convergence OR the deadline,
        # then acts on whatever (possibly inconsistent) assignment it has. If it
        # had to truncate, agents disagree -> the same execution code resolves
        # the conflicts (collisions, coverage loss), with NO guards.
        max_rounds = policy.consensus_max_rounds
        deadline = cfg.get("deadline")
        if deadline:
            max_rounds = max(1, int(deadline / comms.delay_mean))
        rounds_used, c_msgs, converged, known = simulate_consensus(
            list(robots), lambda i: dict(robots[i]["cost"]), comms, rng,
            max_rounds, tasks=tasks, assign_fn=greedy_assign,
            drop_prob=drop_prob, base_time=detection_time)
        start_t = detection_time + rounds_used * comms.delay_mean
        # each agent executes its own converged assignment (no guards, no retarget)
        completed, collisions, energy, activated = {}, 0, 0.0, set()
        contenders = {}
        for rid in robots:
            k = greedy_assign(known[rid], tasks).get(rid)
            if k is None:
                continue
            contenders.setdefault(k, []).append((rid, start_t + robots[rid]["travel"][k]))
        for k, group in contenders.items():
            comp, coll, _lost, moved, e = resolve_zone(
                k, group, robots, {r: start_t for r in robots}, policy, deliver,
                crit_duration, mutex, hb_period)
            if comp is not None:
                completed[k] = comp
            collisions += coll
            activated.update(moved)
            energy += e
        return metrics(
            policy, comms, K, covered=len(completed), collisions=collisions,
            makespan=max(completed.values(), default=0.0), energy=energy,
            messages=c_msgs, activated=len(activated),
            wasted_robots=sum(1 for r in activated if greedy_assign(
                known[r], tasks).get(r) not in completed),
            reassigns=0, success_all=1 if len(completed) == min(n, K) else 0)

    # broadcast medium: one transmission per sender per round reaches all.
    # bidding cost = rounds * N (this is where CBBA's overhead lives).
    msg = [rounds * n]

    # ---- Phase 1: broadcast cost vectors (one msg/pair, re-sent each round) ----
    views = {rid: {rid: dict(robots[rid]["cost"])} for rid in robots}
    for s in robots:
        for d in robots:
            if s == d:
                continue
            for rnd in range(rounds):
                arr = deliver(detection_time + rnd * comms.delay_mean)
                if arr is not None and arr <= decision_time:
                    views[d][s] = dict(robots[s]["cost"])
                    break

    # ---- Local assignment per robot (views differ -> disagreement) ----
    primary = {}     # rid -> task it intends to pursue (or None)
    confidence = {}
    for rid in robots:
        confidence[rid] = len(views[rid]) / n
        if policy.abstain_confidence > 0 and confidence[rid] < policy.abstain_confidence:
            primary[rid] = None
            continue
        assign = greedy_assign(views[rid], tasks)
        primary[rid] = assign.get(rid)  # None if it loses every task in its view

    # ---- Execution with per-zone guards + re-assignment ----
    # Each robot maintains a ranked task preference (own ascending cost).
    pref = {rid: sorted(tasks, key=lambda k: robots[rid]["cost"][k]) for rid in robots}
    target = dict(primary)
    start = {}
    for rid in robots:
        start[rid] = decision_time + compute_wait(robots[rid], target[rid], views[rid],
                                                  confidence[rid], policy)

    completed = {}       # task -> done_time (first successful grasp)
    collisions = 0
    energy = 0.0         # only counts distance robots actually travelled
    activated = set()    # robots that physically moved toward any zone
    reassign_count = 0
    blocked_perm = set()

    max_rounds = (K + 1) if use_reassign else 1
    for _ in range(max_rounds):
        # group current targets by zone
        contenders = {}
        for rid in robots:
            k = target[rid]
            if k is None or rid in blocked_perm or k in completed:
                continue
            reaches = start[rid] + robots[rid]["travel"][k]
            contenders.setdefault(k, []).append((rid, reaches))

        losers = []
        for k, group in contenders.items():
            # intent: every robot that commits to this zone broadcasts once
            if policy.use_intent:
                msg[0] += len(group)
            comp, coll, lost, moved, e = resolve_zone(
                k, group, robots, start, policy, deliver, crit_duration, mutex, hb_period)
            if comp is not None:
                completed[k] = comp
            collisions += coll
            losers.extend(lost)
            activated.update(moved)
            energy += e
            # heartbeat: robots that actually drove broadcast periodically over
            # their travel leg (the recurring cost of the liveness layer)
            if policy.use_heartbeat:
                for rid in moved:
                    msg[0] += max(1, round(robots[rid]["travel"][k] / hb_period))

        if not use_reassign or not losers:
            for rid in losers:
                blocked_perm.add(rid)
            break

        # re-assignment: each loser tries its next-best still-open task
        progressed = False
        for rid in losers:
            nxt = next((k for k in pref[rid]
                        if k not in completed and k != target[rid]), None)
            if nxt is None:
                blocked_perm.add(rid)
                continue
            target[rid] = nxt
            start[rid] = start[rid] + comms.delay_mean  # retarget penalty
            reassign_count += 1
            progressed = True
        if not progressed:
            break

    covered = len(completed)
    makespan = max(completed.values(), default=0.0)
    # a robot is "wasted" if it moved but its final target was not completed by it
    wasted = sum(1 for rid in activated if target[rid] not in completed)
    success_all = 1 if covered == min(n, K) else 0

    return metrics(
        policy, comms, K, covered=covered, collisions=collisions,
        makespan=makespan, energy=energy, messages=msg[0],
        activated=len(activated), wasted_robots=wasted,
        reassigns=reassign_count, success_all=success_all)


def compute_wait(robot, task, view, confidence, policy):
    if task is None or policy.commit_wait_mode == "none":
        return 0.0
    if policy.commit_wait_mode in ("fixed", "conservative"):
        return policy.fixed_wait
    if policy.commit_wait_mode == "risk":
        # margin = how clearly THIS robot is the best for its chosen task
        others = sorted(v[task] for rid, v in view.items() if task in v)
        margin = (others[1] - others[0]) if len(others) >= 2 else 0.0
        wait = (policy.min_wait
                + (1.0 - confidence) * policy.conf_scale
                + policy.margin_scale / (1.0 + margin))
        return min(policy.max_wait, wait)
    return 0.0


def resolve_zone(task, group, robots, start, policy, deliver, crit_duration, mutex,
                 hb_period=0.75):
    """group: [(rid, reaches)] all targeting `task`.
    Returns (done_time|None, collisions, losers, moved_set, energy).
    Energy is honest about HOW FAR each robot actually drove:
      intent-bumped -> 0 (caught before driving),
      heartbeat-bumped -> half the leg (recalled mid-route),
      reached (winner or grasp-denied) -> full leg.
    """
    group = sorted(group, key=lambda g: robots[g[0]]["cost"][task])  # best first
    rids = [g[0] for g in group]
    reaches = {rid: r for rid, r in group}
    full = {rid: dist(robots[rid]["pos"], POI_MAP[task]) for rid in rids}
    losers, alive = [], list(rids)
    moved, energy = set(), 0.0

    # intent: a worse robot whose better rival's intent arrives before it drives
    if policy.use_intent:
        for rid in list(alive):
            for better in rids:
                if better == rid or robots[better]["cost"][task] >= robots[rid]["cost"][task]:
                    continue
                arr = deliver(start[better])
                if arr is not None and arr <= start[rid]:
                    alive.remove(rid)
                    losers.append(rid)  # 0 energy: never left
                    break

    # heartbeat: en-route backoff if a better robot is already moving. The
    # better robot emits heartbeats every hb_period from its start; rid backs
    # off if ANY is delivered before it reaches. Slower hb_period -> fewer
    # emissions -> lower catch probability under loss.
    if policy.use_heartbeat:
        for rid in list(alive):
            for better in alive:
                if better == rid or robots[better]["cost"][task] >= robots[rid]["cost"][task]:
                    continue
                if start[better] > reaches[rid]:
                    continue
                t = start[better]
                caught = False
                while t < reaches[rid]:
                    arr = deliver(t)
                    if arr is not None and arr < reaches[rid]:
                        caught = True
                        break
                    t += hb_period
                if caught:
                    alive.remove(rid)
                    losers.append(rid)
                    moved.add(rid)
                    energy += 0.5 * full[rid]  # recalled mid-route
                    break

    if not alive:
        return None, 0, losers, moved, energy

    alive.sort(key=lambda rid: reaches[rid])
    for rid in alive:
        moved.add(rid)
        energy += full[rid]

    if mutex:
        winner = alive[0]
        losers.extend(alive[1:])  # drove the full leg, then denied at the zone
        return reaches[winner] + crit_duration, 0, losers, moved, energy

    # no mutex: everyone who arrives grabs -> overlaps are collisions
    collisions = 0
    for i in range(1, len(alive)):
        if reaches[alive[i]] < reaches[alive[i - 1]] + crit_duration:
            collisions += 1
    return reaches[alive[0]] + crit_duration, collisions, [], moved, energy


def metrics(policy, comms, K, covered, collisions, makespan, energy, messages,
            activated, wasted_robots, reassigns, success_all):
    return {
        "mode": policy.name,
        "drop_rate": comms.drop_rate,
        "delay_mean": comms.delay_mean,
        "num_tasks": K,
        "coverage_frac": covered / K if K else 0.0,
        "tasks_covered": covered,
        "collisions": collisions,
        "makespan": makespan,
        "energy_travel": energy,
        "messages": messages,
        "robots_activated": activated,
        "wasted_robots": wasted_robots,
        "reassignments": reassigns,
        "full_coverage": success_all,
    }


# ----------------------------------------------------------------------------
def make_comms(args, drop_rate=None):
    return Comms(
        delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
        drop_rate=args.drop_rate if drop_rate is None else drop_rate,
        burst=args.burst, burst_drop=args.burst_drop)


def trial_cfg(args):
    return {
        "num_tasks": args.num_tasks,
        "position_jitter": args.position_jitter,
        "nominal_speed": 5.0,
        "crit_duration": 2.0,
        "search_visit_time": 1.0,
        "hb_period": args.hb_period,
        "universal_mutex": args.universal_mutex,
        "deadline": getattr(args, "deadline", None),
    }


def cmd_compare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comms = make_comms(args)
    cfg = trial_cfg(args)
    avg_rows, all_rows = [], []
    for name, pol in build_policies().items():
        rows = [run_trial(4000 + i, comms, pol, cfg) for i in range(args.trials)]
        all_rows.extend(rows)
        avg_rows.append(average(rows))
    save_csv(all_rows, out / "mt_compare_raw.csv")
    save_csv(avg_rows, out / "mt_compare_avg.csv")
    metrics_list = ["coverage_frac", "collisions", "makespan", "energy_travel",
                    "messages", "wasted_robots", "reassignments", "full_coverage"]
    for m in metrics_list:
        bar_plot(avg_rows, m, out, prefix="mt_")
    print_table(avg_rows, metrics_list)
    print(f"\nSaved -> {out}")


def cmd_sweep(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = trial_cfg(args)
    policies = build_policies()
    drops = [round(x, 2) for x in frange(0.0, args.max_drop, args.drop_step)]
    sweep_metrics = ["coverage_frac", "collisions", "energy_travel",
                     "makespan", "messages", "full_coverage"]
    curves = {m: {name: [] for name in policies} for m in sweep_metrics}
    rows = []
    for dr in drops:
        comms = make_comms(args, drop_rate=dr)
        for name, pol in policies.items():
            trials = [run_trial(5000 + i, comms, pol, cfg) for i in range(args.trials)]
            avg = average(trials)
            avg["drop_rate"] = dr
            rows.append(avg)
            for m in sweep_metrics:
                curves[m][name].append(avg[m])
    save_csv(rows, out / "mt_sweep_drop.csv")
    for m in sweep_metrics:
        line_plot(drops, curves[m], "mt_" + m, "drop_rate", out)
    print(f"Swept drop_rate over {drops}\nSaved -> {out}")


def cmd_deadline(args):
    """Fair time-critical fight: sweep a mission deadline. CBBA is truncated at
    the deadline and acts on its partial (possibly inconsistent) assignment;
    ours acts immediately with guards. Reveals whether guard-based reactivity
    degrades more gracefully than deadline-truncated consensus."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policies = build_policies()
    sweep_pols = {k: policies[k] for k in
                  ["cbba_consensus", "ours_layered", "auction_only"]}
    deadlines = [round(x, 1) for x in frange(args.min_deadline, args.max_deadline,
                                             args.deadline_step)]
    sweep_metrics = ["coverage_frac", "collisions", "makespan", "full_coverage"]
    curves = {m: {name: [] for name in sweep_pols} for m in sweep_metrics}
    rows = []
    for dl in deadlines:
        comms = make_comms(args)
        cfg = dict(trial_cfg(args))
        cfg["deadline"] = dl
        for name, pol in sweep_pols.items():
            trials = [run_trial(8000 + i, comms, pol, cfg) for i in range(args.trials)]
            avg = average(trials)
            avg["deadline"] = dl
            avg["mode"] = name
            rows.append(avg)
            for m in sweep_metrics:
                curves[m][name].append(avg[m])
    save_csv(rows, out / "mt_deadline.csv")
    for m in sweep_metrics:
        line_plot(deadlines, curves[m], "mt_dl_" + m, "deadline (s)", out)
    print(f"Swept deadline over {deadlines} at drop={args.drop_rate}, "
          f"burst={args.burst}\nSaved -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--trials", type=int, default=300)
        sp.add_argument("--num_tasks", type=int, default=3)
        sp.add_argument("--delay_mean", type=float, default=2.0)
        sp.add_argument("--delay_jitter", type=float, default=1.0)
        sp.add_argument("--drop_rate", type=float, default=0.25)
        sp.add_argument("--position_jitter", type=float, default=6.0)
        sp.add_argument("--hb_period", type=float, default=0.75,
                        help="heartbeat broadcast period (s); lower = more msgs")
        sp.add_argument("--universal_mutex", action="store_true",
                        help="OPTIONAL: grant grasp-zone mutex to ALL methods "
                             "(off by default; excluded from evaluation)")
        sp.add_argument("--burst", action="store_true",
                        help="use Gilbert-Elliott correlated burst loss instead of i.i.d.")
        sp.add_argument("--burst_drop", type=float, default=0.95)
        sp.add_argument("--output_dir", type=str,
                        default=str(Path.home() / "phase1a_ws/results/unified_multitask"))

    c = sub.add_parser("compare"); common(c)
    s = sub.add_parser("sweep"); common(s)
    s.add_argument("--max_drop", type=float, default=0.6)
    s.add_argument("--drop_step", type=float, default=0.1)
    d = sub.add_parser("deadline"); common(d)
    d.add_argument("--min_deadline", type=float, default=2.0)
    d.add_argument("--max_deadline", type=float, default=24.0)
    d.add_argument("--deadline_step", type=float, default=2.0)

    args = p.parse_args()
    if args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "deadline":
        cmd_deadline(args)


if __name__ == "__main__":
    main()
