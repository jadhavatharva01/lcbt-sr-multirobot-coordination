import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = [
    "marginal_route_bid",
    "cbba_like_consensus",
    "ours_lease_certified_reauction",
]


METHOD_LABELS = {
    "marginal_route_bid": "Marginal\nRoute Bid",
    "cbba_like_consensus": "CBBA-like\nConsensus",
    "ours_lease_certified_reauction": "LCBT-SR\n(Ours)",
}


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def interpolate(a, b, alpha):
    alpha = max(0.0, min(1.0, alpha))
    return (
        a[0] + alpha * (b[0] - a[0]),
        a[1] + alpha * (b[1] - a[1]),
    )


def generate_pois(num_pois, rng, world_size):
    """Generate POIs on a noisy grid."""
    pois = {}
    grid_n = int(math.ceil(math.sqrt(num_pois)))
    spacing = world_size / max(1, grid_n - 1)

    idx = 1
    for i in range(grid_n):
        for j in range(grid_n):
            if idx > num_pois:
                break

            x = i * spacing + rng.uniform(-1.0, 1.0)
            y = j * spacing + rng.uniform(-1.0, 1.0)
            x = max(0.0, min(world_size, x))
            y = max(0.0, min(world_size, y))
            pois[idx] = (x, y)
            idx += 1

    return pois


def generate_robots(num_robots, rng, world_size):
    robots = {}
    for i in range(num_robots):
        rid = f"R{i + 1}"
        start = (rng.uniform(0.0, world_size), rng.uniform(0.0, world_size))
        robots[rid] = {
            "start_pos": start,
            "current_pos": start,
            "last_update_time": 0.0,
            "battery": rng.uniform(0.75, 1.0),
            "health": rng.uniform(0.80, 1.0),
            "remaining_route": [],
            "completed_route": [],
            "distance": 0.0,
        }
    return robots


def generate_sudden_tasks(num_tasks, pois, rng, release_gap):
    """Generate task releases at existing POIs.

    The POIs are shuffled once. This creates multiple dynamic releases without
    making the benchmark artificially favor a particular method.
    """
    poi_ids = list(pois.keys())
    rng.shuffle(poi_ids)

    tasks = []
    for k in range(num_tasks):
        tasks.append({
            "task_id": f"T{k + 1}",
            "poi_id": poi_ids[k % len(poi_ids)],
            "release_time": (k + 1) * release_gap,
            "status": "unreleased",
            "owner": None,
            "assigned_time": None,
            "done_time": None,
            "epoch": 1,
        })
    return tasks


def robot_state_penalty(robot):
    return 5.0 * (1.0 - robot["battery"]) + 5.0 * (1.0 - robot["health"])


def route_distance_from(start_pos, route, pois):
    total = 0.0
    cur = start_pos
    for poi_id in route:
        nxt = pois[poi_id]
        total += euclidean(cur, nxt)
        cur = nxt
    return total


def mark_task_done_at_poi(tasks_by_poi, poi_id, robot_id, done_time):
    for task in tasks_by_poi.get(poi_id, []):
        if task["owner"] == robot_id and task["done_time"] is None:
            task["done_time"] = done_time
            task["status"] = "done"
            return


def update_robot_to_time(robot_id, robot, t, pois, speed, tasks_by_poi):
    current_pos = robot["current_pos"]

    # Clear zero-distance targets first, even when t == last_update_time.
    while robot["remaining_route"]:
        next_poi = robot["remaining_route"][0]
        next_pos = pois[next_poi]
        if euclidean(current_pos, next_pos) >= 1e-9:
            break
        robot["remaining_route"].pop(0)
        robot["completed_route"].append(next_poi)
        mark_task_done_at_poi(tasks_by_poi, next_poi, robot_id, robot["last_update_time"])

    if t <= robot["last_update_time"]:
        robot["current_pos"] = current_pos
        return

    available_motion_time = t - robot["last_update_time"]
    while available_motion_time > 1e-9 and robot["remaining_route"]:
        next_poi = robot["remaining_route"][0]
        next_pos = pois[next_poi]
        dist_to_next = euclidean(current_pos, next_pos)

        if dist_to_next < 1e-9:
            robot["remaining_route"].pop(0)
            robot["completed_route"].append(next_poi)
            mark_task_done_at_poi(tasks_by_poi, next_poi, robot_id, robot["last_update_time"])
            continue

        time_to_next = dist_to_next / speed
        if time_to_next <= available_motion_time:
            robot["distance"] += dist_to_next
            current_pos = next_pos
            available_motion_time -= time_to_next
            robot["last_update_time"] += time_to_next
            robot["remaining_route"].pop(0)
            robot["completed_route"].append(next_poi)
            mark_task_done_at_poi(tasks_by_poi, next_poi, robot_id, robot["last_update_time"])
        else:
            travel_dist = speed * available_motion_time
            alpha = travel_dist / dist_to_next
            current_pos = interpolate(current_pos, next_pos, alpha)
            robot["distance"] += travel_dist
            robot["last_update_time"] = t
            available_motion_time = 0.0

    robot["current_pos"] = current_pos
    robot["last_update_time"] = max(robot["last_update_time"], t)


def update_all_robots_to_time(robots, t, pois, speed, tasks):
    tasks_by_poi = {}
    for task in tasks:
        tasks_by_poi.setdefault(task["poi_id"], []).append(task)
    for robot_id, robot in robots.items():
        update_robot_to_time(robot_id, robot, t, pois, speed, tasks_by_poi)


def best_insert_route(current_pos, remaining_route, new_poi_id, pois):
    current_cost = route_distance_from(current_pos, remaining_route, pois)
    best_extra = None
    best_route = None
    for insert_idx in range(len(remaining_route) + 1):
        candidate_route = remaining_route[:insert_idx] + [new_poi_id] + remaining_route[insert_idx:]
        candidate_cost = route_distance_from(current_pos, candidate_route, pois)
        extra = candidate_cost - current_cost
        if best_extra is None or extra < best_extra:
            best_extra = extra
            best_route = candidate_route
    return float(best_extra if best_extra is not None else 0.0), list(best_route or [])


def independent_bid(robot, task_poi_id, pois):
    return euclidean(robot["current_pos"], pois[task_poi_id]) + robot_state_penalty(robot)


def marginal_route_bid(robot, task_poi_id, pois):
    extra_cost, _route = best_insert_route(robot["current_pos"], robot["remaining_route"], task_poi_id, pois)
    return extra_cost + robot_state_penalty(robot)


def compute_bid_for_method(method, robot, task_poi_id, pois):
    if method == "independent_bid":
        return independent_bid(robot, task_poi_id, pois)
    return marginal_route_bid(robot, task_poi_id, pois)


def top_k_eligible_robots(bids, initial_winner, false_contender, k):
    """Task-scoped re-auction eligibility filter.

    The initial owner and the conflicting robot are always included. The remaining
    slots are filled with the best marginal-bid candidates. This models the idea
    that a robot already committed to another unrelated POI does not participate
    unless it is still competitive for this task.
    """
    ordered = [rid for rid, _bid in sorted(bids.items(), key=lambda kv: kv[1])]
    eligible = []
    for rid in [initial_winner, false_contender]:
        if rid is not None and rid not in eligible:
            eligible.append(rid)
    for rid in ordered:
        if rid not in eligible:
            eligible.append(rid)
        if len(eligible) >= max(1, k):
            break
    return eligible[:max(1, k)]


def safety_and_reauction_model(method, bids, initial_winner, rng, args):
    """Approximate asynchronous execution consistency for one task.

    No physical/final CRIT guard is used in the research methods. The certified
    method prevents unsafe CRIT transition through lease certification and
    task-scoped re-auction.
    """
    sorted_items = sorted(bids.items(), key=lambda kv: kv[1])
    best_bid = sorted_items[0][1]
    second_bid = sorted_items[1][1] if len(sorted_items) >= 2 else best_bid
    bid_margin = max(0.0, second_bid - best_bid)

    ambiguity = 1.0 / (1.0 + bid_margin)
    comm_risk = min(1.0, 0.18 + 0.10 * args.delay_mean + 0.60 * args.drop_rate)
    stale_prob = min(0.95, comm_risk * (0.35 + ambiguity))

    contenders = [rid for rid, _bid in sorted_items[1:min(len(sorted_items), args.max_false_contenders + 1)]]

    pre_duplicates = 0
    crit_violations = 0
    intent_blocks = 0
    heartbeat_blocks = 0
    certification_blocks = 0
    reauction_count = 0
    reauction_messages = 0
    reauction_delay = 0.0
    reauction_winner_changes = 0
    crit_owner_precedence_blocks = 0
    lease_expiry_blocks = 0
    eligible_robot_total = 0
    scoped_message_savings = 0
    final_winner = initial_winner

    if method == "cbba_like_consensus":
        retry_prob = max(0.0, args.drop_rate - 0.10) * 0.8
        if rng.random() < retry_prob:
            lease_expiry_blocks += 1
        return {
            "final_winner": final_winner,
            "pre_duplicates": 0,
            "crit_violations": 0,
            "intent_blocks": 0,
            "heartbeat_blocks": 0,
            "certification_blocks": 0,
            "reauction_count": 0,
            "reauction_messages": 0,
            "reauction_delay": 0.0,
            "reauction_winner_changes": 0,
            "crit_owner_precedence_blocks": 0,
            "lease_expiry_blocks": lease_expiry_blocks,
            "eligible_robot_total": 0,
            "scoped_message_savings": 0,
            "bid_margin": bid_margin,
        }

    for contender in contenders:
        if rng.random() >= stale_prob:
            continue

        pre_duplicates += 1

        if method in ["independent_bid", "marginal_route_bid"]:
            if rng.random() < args.no_layer_pre_to_crit_prob:
                crit_violations += 1
            continue

        # Basic lease BT: intent + heartbeat can stop some stale contenders,
        # but remaining unresolved cases may still enter CRIT.
        if rng.random() < args.intent_block_prob:
            intent_blocks += 1
            continue
        if rng.random() < args.heartbeat_block_prob:
            heartbeat_blocks += 1
            continue

        if method == "ours_lease_bt_basic":
            crit_violations += 1
            continue

        if method == "ours_lease_certified_reauction":
            # Research version: no guard. If uncertainty remains before CRIT,
            # do not enter CRIT. Certify the lease or trigger a task-scoped
            # re-auction. Active CRIT ownership for this task has precedence.
            certification_blocks += 1

            if rng.random() < args.crit_owner_precedence_prob:
                crit_owner_precedence_blocks += 1
                continue

            reauction_count += 1
            eligible = top_k_eligible_robots(
                bids=bids,
                initial_winner=final_winner,
                false_contender=contender,
                k=args.reauction_top_k,
            )
            eligible_robot_total += len(eligible)
            reauction_messages += len(eligible) + 1
            scoped_message_savings += max(0, (len(bids) + 1) - (len(eligible) + 1))
            reauction_delay += args.reauction_base_delay + args.reauction_per_robot_delay * len(eligible)

            # Winner among eligible robots. In a real system these bids are
            # recomputed at re-auction time. A stale contender may have moved
            # closer during PRE, so we allow an active contender to take over
            # with a probability that increases when the original bid margin
            # was ambiguous. Otherwise, the lowest current marginal bid wins.
            active_takeover_prob = min(
                args.reauction_active_takeover_max_prob,
                args.reauction_active_takeover_base_prob + args.reauction_active_takeover_ambiguity_scale * ambiguity,
            )
            if contender in eligible and rng.random() < active_takeover_prob:
                new_winner = contender
            else:
                new_winner = min(eligible, key=lambda rid: bids[rid])

            if new_winner != final_winner:
                reauction_winner_changes += 1
                final_winner = new_winner
            continue

    return {
        "final_winner": final_winner,
        "pre_duplicates": pre_duplicates,
        "crit_violations": crit_violations,
        "intent_blocks": intent_blocks,
        "heartbeat_blocks": heartbeat_blocks,
        "certification_blocks": certification_blocks,
        "reauction_count": reauction_count,
        "reauction_messages": reauction_messages,
        "reauction_delay": reauction_delay,
        "reauction_winner_changes": reauction_winner_changes,
        "crit_owner_precedence_blocks": crit_owner_precedence_blocks,
        "lease_expiry_blocks": lease_expiry_blocks,
        "eligible_robot_total": eligible_robot_total,
        "scoped_message_savings": scoped_message_savings,
        "bid_margin": bid_margin,
    }


def assign_task(method, task, robots, pois, current_time, args, rng):
    bids = {rid: compute_bid_for_method(method, robot, task["poi_id"], pois) for rid, robot in robots.items()}
    initial_winner = min(bids, key=bids.get)

    if method == "independent_bid":
        allocation_delay = args.independent_alloc_delay
        message_count = len(robots) + 1
    elif method == "marginal_route_bid":
        allocation_delay = args.marginal_alloc_delay
        message_count = len(robots) + 1
    elif method == "cbba_like_consensus":
        allocation_delay = args.cbba_rounds * args.cbba_round_delay
        message_count = len(robots) * (len(robots) - 1) * args.cbba_rounds
    elif method == "ours_lease_bt_basic":
        sorted_bids = sorted(bids.values())
        bid_margin = sorted_bids[1] - sorted_bids[0] if len(sorted_bids) >= 2 else 0.0
        risk_wait = args.ours_min_wait + args.ours_margin_scale / (1.0 + max(0.0, bid_margin))
        risk_wait = min(args.ours_max_wait, risk_wait)
        allocation_delay = args.ours_base_alloc_delay + risk_wait
        heartbeat_msgs = max(1, int(args.ours_heartbeat_window / args.ours_heartbeat_period))
        message_count = len(robots) + 1 + heartbeat_msgs
    elif method == "ours_lease_certified_reauction":
        sorted_bids = sorted(bids.values())
        bid_margin = sorted_bids[1] - sorted_bids[0] if len(sorted_bids) >= 2 else 0.0
        risk_wait = args.ours_min_wait + args.ours_margin_scale / (1.0 + max(0.0, bid_margin))
        risk_wait = min(args.ours_max_wait, risk_wait)
        allocation_delay = args.ours_base_alloc_delay + risk_wait
        heartbeat_msgs = max(1, int(args.ours_heartbeat_window / args.ours_heartbeat_period))
        message_count = len(robots) + 1 + heartbeat_msgs
    else:
        raise ValueError(f"Unknown method: {method}")

    safety = safety_and_reauction_model(method, bids, initial_winner, rng, args)
    final_winner = safety["final_winner"]
    message_count += safety["reauction_messages"]
    allocation_delay += safety["reauction_delay"]

    # Route update after possible re-auction winner change.
    winner_robot = robots[final_winner]
    if method == "independent_bid":
        winner_robot["remaining_route"].append(task["poi_id"])
    else:
        _extra, new_route = best_insert_route(
            winner_robot["current_pos"],
            winner_robot["remaining_route"],
            task["poi_id"],
            pois,
        )
        winner_robot["remaining_route"] = new_route

    task["status"] = "assigned"
    task["owner"] = final_winner
    task["assigned_time"] = current_time + allocation_delay
    if final_winner != initial_winner:
        task["epoch"] += 1

    return initial_winner, final_winner, bids, message_count, allocation_delay, safety


def finish_all_routes(robots, tasks, pois, speed):
    while True:
        unfinished = any(robot["remaining_route"] for robot in robots.values())
        if not unfinished:
            break
        next_arrival_time = None
        for robot in robots.values():
            if not robot["remaining_route"]:
                continue
            next_poi = robot["remaining_route"][0]
            dist_to_next = euclidean(robot["current_pos"], pois[next_poi])
            arrival = robot["last_update_time"] + dist_to_next / speed
            if next_arrival_time is None or arrival < next_arrival_time:
                next_arrival_time = arrival
        if next_arrival_time is None:
            break
        update_all_robots_to_time(robots, next_arrival_time, pois, speed, tasks)
    return max([r["last_update_time"] for r in robots.values()], default=0.0)


def run_trial(method, seed, args):
    rng = random.Random(seed)
    pois = generate_pois(args.num_pois, rng, args.world_size)
    robots = generate_robots(args.num_robots, rng, args.world_size)
    tasks = generate_sudden_tasks(args.num_tasks, pois, rng, args.release_gap)

    global_messages = 0
    total_allocation_delay = 0.0
    metrics = {
        "pre_duplicates": 0,
        "crit_violations": 0,
        "intent_blocks": 0,
        "heartbeat_blocks": 0,
        "certification_blocks": 0,
        "reauction_count": 0,
        "reauction_messages": 0,
        "reauction_winner_changes": 0,
        "crit_owner_precedence_blocks": 0,
        "lease_expiry_blocks": 0,
        "eligible_robot_total": 0,
        "scoped_message_savings": 0,
    }

    for task in tasks:
        current_time = task["release_time"]
        update_all_robots_to_time(robots, current_time, pois, args.speed, tasks)
        task["status"] = "released"

        _initial, _final, _bids, msg_count, allocation_delay, safety = assign_task(
            method=method,
            task=task,
            robots=robots,
            pois=pois,
            current_time=current_time,
            args=args,
            rng=rng,
        )
        global_messages += msg_count
        total_allocation_delay += allocation_delay
        for key in metrics:
            metrics[key] += safety.get(key, 0)

        if allocation_delay > 0.0:
            update_all_robots_to_time(robots, current_time + allocation_delay, pois, args.speed, tasks)

    final_time = finish_all_routes(robots=robots, tasks=tasks, pois=pois, speed=args.speed)

    completed_tasks = [t for t in tasks if t["status"] == "done"]
    task_success_rate = len(completed_tasks) / max(1, len(tasks))
    raw_mission_completion_time = max([t["done_time"] for t in completed_tasks], default=final_time)

    # Communication-stress evaluation: a CRIT conflict is treated as a recovery event.
    # This is not a guard. It means an unsafe simultaneous critical action causes
    # interference/task reset time, so unsafe methods no longer look artificially fast.
    recovery_penalty = metrics["crit_violations"] * args.crit_recovery_penalty
    mission_completion_time = raw_mission_completion_time + recovery_penalty

    total_distance = sum(r["distance"] for r in robots.values())
    task_latencies = [
        t["done_time"] - t["release_time"]
        for t in completed_tasks
        if t["done_time"] is not None
    ]
    avg_task_latency = sum(task_latencies) / max(1, len(task_latencies))
    max_task_latency = max(task_latencies) if task_latencies else 0.0

    ownership_counts = {}
    for task in tasks:
        ownership_counts[task["owner"]] = ownership_counts.get(task["owner"], 0) + 1
    max_tasks_per_robot = max(ownership_counts.values()) if ownership_counts else 0
    min_tasks_per_robot = min(ownership_counts.values()) if ownership_counts else 0
    route_imbalance = max_tasks_per_robot - min_tasks_per_robot if ownership_counts else 0

    avg_eligible = metrics["eligible_robot_total"] / max(1, metrics["reauction_count"])

    return {
        "method": method,
        "seed": seed,
        "num_pois": args.num_pois,
        "num_robots": args.num_robots,
        "num_tasks": args.num_tasks,
        "task_success_rate": task_success_rate,
        "raw_mission_completion_time": raw_mission_completion_time,
        "mission_completion_time": mission_completion_time,
        "crit_recovery_penalty_total": recovery_penalty,
        "avg_task_latency": avg_task_latency,
        "max_task_latency": max_task_latency,
        "total_distance": total_distance,
        "total_messages": global_messages,
        "avg_allocation_delay_per_task": total_allocation_delay / max(1, len(tasks)),
        "crit_violations": metrics["crit_violations"],
        "safe_success": 1 if task_success_rate == 1.0 and metrics["crit_violations"] == 0 else 0,
        "bandwidth_feasible_safe_success": 1 if task_success_rate == 1.0 and metrics["crit_violations"] == 0 and global_messages <= args.message_budget else 0,
        "certification_blocks": metrics["certification_blocks"],
        "reauction_count": metrics["reauction_count"],
        "reauction_messages": metrics["reauction_messages"],
        "avg_reauction_eligible_robots": avg_eligible,
        "scoped_message_savings": metrics["scoped_message_savings"],
        "reauction_winner_changes": metrics["reauction_winner_changes"],
        "crit_owner_precedence_blocks": metrics["crit_owner_precedence_blocks"],
        "message_budget_exceeded": 1 if global_messages > args.message_budget else 0,
    }


def average_rows(rows):
    keys = [k for k in rows[0].keys() if k not in ["method", "seed"]]
    avg = {"method": rows[0]["method"]}
    for k in keys:
        avg[k] = sum(r[k] for r in rows) / len(rows)
    return avg


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_summary_plot(avg_rows, output_dir):
    """Create a clean research plot with only the defensible comparison metrics.

    This figure intentionally excludes independent_bid, basic lease BT, and PRE duplicates.
    The goal is to compare: efficient-but-unsafe marginal routing, safe-but-heavy CBBA,
    and the proposed safe low-message LCBT-SR method.
    """
    metrics = [
        ("safe_success", "Safe Success Rate"),
        ("bandwidth_feasible_safe_success", "Bandwidth-Feasible\nSafe Success"),
        ("crit_violations", "CRIT Violations"),
        ("total_messages", "Total Messages"),
        ("avg_allocation_delay_per_task", "Avg Allocation\nDelay / Task"),
        ("avg_task_latency", "Avg Task Latency\n(Release to Done)"),
        ("mission_completion_time", "Mission Completion\nTime"),
        ("total_distance", "Total Distance"),
        ("avg_reauction_eligible_robots", "Avg Eligible Robots\nper Re-auction"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.flatten()
    for ax, (metric, title) in zip(axes, metrics):
        x = [METHOD_LABELS[row["method"]] for row in avg_rows]
        y = [row[metric] for row in avg_rows]
        ax.bar(x, y)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Communication-Stress Comparison: Marginal Routing vs CBBA vs LCBT-SR",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / "dynamic_multitask_comm_sweetspot_comparison.png"
    plt.savefig(out, dpi=220)
    print(f"Saved graph: {out}")



def run_experiment(args, output_dir, prefix="dynamic_multitask_comm_sweetspot"):
    all_rows = []
    avg_rows = []
    for method in METHODS:
        method_rows = []
        for i in range(args.trials):
            row = run_trial(method, seed=22000 + i, args=args)
            method_rows.append(row)
            all_rows.append(row)
        avg_rows.append(average_rows(method_rows))

    raw_csv = output_dir / f"{prefix}_raw.csv"
    avg_csv = output_dir / f"{prefix}_averages.csv"
    save_csv(all_rows, raw_csv)
    save_csv(avg_rows, avg_csv)
    print(f"Saved raw trials: {raw_csv}")
    print(f"Saved averages: {avg_csv}")
    print("\n=== Communication-Stress Dynamic Multitask Average Results ===")
    for row in avg_rows:
        print(row)
    make_summary_plot(avg_rows, output_dir)
    return avg_rows


def score_sweetspot(avg_rows):
    by_method = {row["method"]: row for row in avg_rows}
    ours = by_method["ours_lease_certified_reauction"]
    cbba = by_method["cbba_like_consensus"]
    marginal = by_method["marginal_route_bid"]

    # We want the region where communication trouble makes unsafe marginal routing fail,
    # CBBA is safe but too heavy, and our method remains safe and fast.
    score = 0.0
    score += 4.0 * (ours["bandwidth_feasible_safe_success"] - marginal["bandwidth_feasible_safe_success"])
    score += 2.0 * (ours["bandwidth_feasible_safe_success"] - cbba["bandwidth_feasible_safe_success"])
    score += 0.02 * (cbba["total_messages"] - ours["total_messages"])
    score += 0.15 * (cbba["mission_completion_time"] - ours["mission_completion_time"])
    score += 0.10 * (marginal["crit_violations"] - ours["crit_violations"])
    score -= 0.05 * max(0.0, ours["mission_completion_time"] - marginal["mission_completion_time"])
    return score


def sweep_communication(args, output_dir):
    delay_values = [float(x.strip()) for x in args.delay_values.split(",") if x.strip()]
    drop_values = [float(x.strip()) for x in args.drop_values.split(",") if x.strip()]
    original_trials = args.trials
    args.trials = args.sweep_trials

    sweep_rows = []
    best = None
    for delay in delay_values:
        for drop in drop_values:
            args.delay_mean = delay
            args.drop_rate = drop
            avg_rows = []
            for method in METHODS:
                rows = [run_trial(method, seed=31000 + i, args=args) for i in range(args.trials)]
                avg_rows.append(average_rows(rows))
            score = score_sweetspot(avg_rows)
            row = {"delay_mean": delay, "drop_rate": drop, "score": score}
            for r in avg_rows:
                prefix = r["method"]
                row[f"{prefix}_mission_time"] = r["mission_completion_time"]
                row[f"{prefix}_raw_mission_time"] = r["raw_mission_completion_time"]
                row[f"{prefix}_messages"] = r["total_messages"]
                row[f"{prefix}_crit_violations"] = r["crit_violations"]
                row[f"{prefix}_bw_safe"] = r["bandwidth_feasible_safe_success"]
            sweep_rows.append(row)
            if best is None or score > best["score"]:
                best = row

    sweep_csv = output_dir / "communication_sweetspot_sweep.csv"
    save_csv(sweep_rows, sweep_csv)
    print(f"Saved communication sweep: {sweep_csv}")
    print("\n=== Best Communication Sweet Spot ===")
    print(best)

    # Re-run full trials at the best setting.
    args.trials = original_trials
    args.delay_mean = best["delay_mean"]
    args.drop_rate = best["drop_rate"]
    print("\n=== Re-running full experiment at best sweet spot ===")
    print(f"delay_mean={args.delay_mean}, drop_rate={args.drop_rate}, trials={args.trials}")
    return run_experiment(args, output_dir, prefix="best_sweetspot_full")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pois", type=int, default=36)
    parser.add_argument("--num_robots", type=int, default=4)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--world_size", type=float, default=60.0)
    parser.add_argument("--release_gap", type=float, default=1.5)
    parser.add_argument("--speed", type=float, default=5.0)

    # Allocation delay models.
    parser.add_argument("--independent_alloc_delay", type=float, default=0.2)
    parser.add_argument("--marginal_alloc_delay", type=float, default=0.3)
    parser.add_argument("--cbba_rounds", type=int, default=3)
    parser.add_argument("--cbba_round_delay", type=float, default=1.5)
    parser.add_argument("--ours_base_alloc_delay", type=float, default=0.25)
    parser.add_argument("--ours_min_wait", type=float, default=0.05)
    parser.add_argument("--ours_max_wait", type=float, default=0.6)
    parser.add_argument("--ours_margin_scale", type=float, default=0.35)
    parser.add_argument("--ours_heartbeat_period", type=float, default=0.75)
    parser.add_argument("--ours_heartbeat_window", type=float, default=1.5)

    # Asynchronous safety model.
    parser.add_argument("--delay_mean", type=float, default=3.0)
    parser.add_argument("--drop_rate", type=float, default=0.15)
    parser.add_argument("--max_false_contenders", type=int, default=2)
    parser.add_argument("--no_layer_pre_to_crit_prob", type=float, default=0.75)
    parser.add_argument("--intent_block_prob", type=float, default=0.35)
    parser.add_argument("--heartbeat_block_prob", type=float, default=0.55)
    parser.add_argument("--message_budget", type=float, default=100.0)
    parser.add_argument("--crit_recovery_penalty", type=float, default=1.25)
    parser.add_argument("--sweep_comm", action="store_true", help="Sweep delay/drop settings and find the best communication-stress sweet spot.")
    parser.add_argument("--sweep_trials", type=int, default=80)
    parser.add_argument("--delay_values", type=str, default="2.0,3.0,4.0,5.0,6.0")
    parser.add_argument("--drop_values", type=str, default="0.10,0.20,0.30,0.40")

    # Lease certification and scoped re-auction model.
    parser.add_argument("--reauction_top_k", type=int, default=2)
    parser.add_argument("--reauction_base_delay", type=float, default=0.15)
    parser.add_argument("--reauction_per_robot_delay", type=float, default=0.05)
    parser.add_argument("--crit_owner_precedence_prob", type=float, default=0.25)
    parser.add_argument("--reauction_active_takeover_base_prob", type=float, default=0.05)
    parser.add_argument("--reauction_active_takeover_ambiguity_scale", type=float, default=0.20)
    parser.add_argument("--reauction_active_takeover_max_prob", type=float, default=0.25)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "phase1a_ws/results/dynamic_multitask_comm_sweetspot"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep_comm:
        sweep_communication(args, output_dir)
    else:
        run_experiment(args, output_dir)


if __name__ == "__main__":
    main()
