import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = [
    "independent_bid",
    "marginal_route_bid",
    "cbba_like_consensus",
    "ours_marginal_lease_bt",
]


METHOD_LABELS = {
    "independent_bid": "Independent\nBid",
    "marginal_route_bid": "Marginal\nRoute Bid",
    "cbba_like_consensus": "CBBA-like\nConsensus",
    "ours_marginal_lease_bt": "Our Marginal\nLease BT",
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
        start = (
            rng.uniform(0.0, world_size),
            rng.uniform(0.0, world_size),
        )

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
    """Generate task releases at existing POIs."""
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
        })

    return tasks


def robot_state_penalty(robot):
    """Small penalty for weaker robot state."""
    return 5.0 * (1.0 - robot["battery"]) + 5.0 * (1.0 - robot["health"])


def route_distance_from(start_pos, route, pois):
    if not route:
        return 0.0

    total = 0.0
    current = start_pos

    for poi_id in route:
        nxt = pois[poi_id]
        total += euclidean(current, nxt)
        current = nxt

    return total


def mark_task_done_at_poi(tasks_by_poi, poi_id, robot_id, done_time):
    for task in tasks_by_poi.get(poi_id, []):
        if task["owner"] == robot_id and task["done_time"] is None:
            task["done_time"] = done_time
            task["status"] = "done"
            return


def update_robot_to_time(robot_id, robot, t, pois, speed, tasks_by_poi):
    """
    Move robot along its remaining route until time t.
    Handles zero-distance waypoints safely.
    """
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
    """Find cheapest insertion of new_poi_id into current remaining route."""
    current_cost = route_distance_from(current_pos, remaining_route, pois)

    best_extra = None
    best_route = None

    for insert_idx in range(len(remaining_route) + 1):
        candidate_route = (
            remaining_route[:insert_idx]
            + [new_poi_id]
            + remaining_route[insert_idx:]
        )
        candidate_cost = route_distance_from(current_pos, candidate_route, pois)
        extra = candidate_cost - current_cost

        if best_extra is None or extra < best_extra:
            best_extra = extra
            best_route = candidate_route

    return best_extra, best_route


def independent_bid(robot, task_poi_id, pois):
    """
    Independent bid: cost from current position directly to the new task.
    It does not reason about inserting into existing route.
    """
    return euclidean(robot["current_pos"], pois[task_poi_id]) + robot_state_penalty(robot)


def marginal_route_bid(robot, task_poi_id, pois):
    """
    Marginal route bid: extra route cost caused by inserting this task into
    the robot's current committed remaining route.
    """
    extra_cost, _route = best_insert_route(
        robot["current_pos"],
        robot["remaining_route"],
        task_poi_id,
        pois,
    )
    return extra_cost + robot_state_penalty(robot)


def compute_bid_for_method(method, robot, task_poi_id, pois):
    if method == "independent_bid":
        return independent_bid(robot, task_poi_id, pois)

    if method in [
        "marginal_route_bid",
        "cbba_like_consensus",
        "ours_marginal_lease_bt",
    ]:
        return marginal_route_bid(robot, task_poi_id, pois)

    raise ValueError(f"Unknown method: {method}")


def assign_task(method, task, robots, pois, current_time, args):
    bids = {}

    for rid, robot in robots.items():
        bids[rid] = compute_bid_for_method(
            method=method,
            robot=robot,
            task_poi_id=task["poi_id"],
            pois=pois,
        )

    winner = min(bids, key=bids.get)
    winner_robot = robots[winner]

    # Method-specific communication and allocation delay model.
    if method == "independent_bid":
        allocation_delay = args.independent_alloc_delay
        message_count = len(robots) + 1

    elif method == "marginal_route_bid":
        allocation_delay = args.marginal_alloc_delay
        message_count = len(robots) + 1

    elif method == "cbba_like_consensus":
        # Multi-round all-to-all style bid/state exchange.
        allocation_delay = args.cbba_rounds * args.cbba_round_delay
        message_count = len(robots) * (len(robots) - 1) * args.cbba_rounds

    elif method == "ours_marginal_lease_bt":
        sorted_bids = sorted(bids.values())
        if len(sorted_bids) >= 2:
            bid_margin = sorted_bids[1] - sorted_bids[0]
        else:
            bid_margin = 0.0

        risk_wait = args.ours_min_wait + args.ours_margin_scale / (1.0 + bid_margin)
        risk_wait = min(args.ours_max_wait, risk_wait)
        allocation_delay = args.ours_base_alloc_delay + risk_wait

        heartbeat_msgs = max(1, int(args.ours_heartbeat_window / args.ours_heartbeat_period))
        message_count = len(robots) + 1 + heartbeat_msgs

    else:
        raise ValueError(f"Unknown method: {method}")

    # Route update.
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
    task["owner"] = winner
    task["assigned_time"] = current_time + allocation_delay

    return winner, bids, message_count, allocation_delay


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

    for task in tasks:
        current_time = task["release_time"]

        # Robots move until task release.
        update_all_robots_to_time(robots, current_time, pois, args.speed, tasks)
        task["status"] = "released"

        winner, bids, msg_count, allocation_delay = assign_task(
            method=method,
            task=task,
            robots=robots,
            pois=pois,
            current_time=current_time,
            args=args,
        )
        global_messages += msg_count
        total_allocation_delay += allocation_delay

        # Robots keep moving while allocation/consensus happens.
        if allocation_delay > 0.0:
            update_all_robots_to_time(
                robots,
                current_time + allocation_delay,
                pois,
                args.speed,
                tasks,
            )

    final_time = finish_all_routes(
        robots=robots,
        tasks=tasks,
        pois=pois,
        speed=args.speed,
    )

    completed_tasks = [t for t in tasks if t["status"] == "done"]
    task_success_rate = len(completed_tasks) / max(1, len(tasks))

    mission_completion_time = max(
        [t["done_time"] for t in completed_tasks],
        default=final_time,
    )

    total_distance = sum(r["distance"] for r in robots.values())

    ownership_counts = {}
    for task in tasks:
        ownership_counts[task["owner"]] = ownership_counts.get(task["owner"], 0) + 1

    max_tasks_per_robot = max(ownership_counts.values()) if ownership_counts else 0
    min_tasks_per_robot = min(ownership_counts.values()) if ownership_counts else 0
    route_imbalance = max_tasks_per_robot - min_tasks_per_robot if ownership_counts else 0

    return {
        "method": method,
        "seed": seed,
        "num_pois": args.num_pois,
        "num_robots": args.num_robots,
        "num_tasks": args.num_tasks,
        "task_success_rate": task_success_rate,
        "mission_completion_time": mission_completion_time,
        "total_distance": total_distance,
        "total_messages": global_messages,
        "avg_allocation_delay_per_task": total_allocation_delay / max(1, len(tasks)),
        "route_imbalance": route_imbalance,
        "max_tasks_per_robot": max_tasks_per_robot,
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
    metrics = [
        ("mission_completion_time", "Mission Completion Time"),
        ("total_distance", "Total Distance"),
        ("total_messages", "Total Messages"),
        ("avg_allocation_delay_per_task", "Avg Allocation Delay / Task"),
        ("route_imbalance", "Route Imbalance"),
        ("max_tasks_per_robot", "Max Tasks Assigned to One Robot"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        x = [METHOD_LABELS[row["method"]] for row in avg_rows]
        y = [row[metric] for row in avg_rows]

        ax.bar(x, y)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Event-Driven Dynamic Multi-Task Coordination Comparison",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = output_dir / "dynamic_multitask_event_driven_cbba_ours_comparison.png"
    plt.savefig(out, dpi=220)
    print(f"Saved graph: {out}")


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
    parser.add_argument("--ours_base_alloc_delay", type=float, default=0.5)
    parser.add_argument("--ours_min_wait", type=float, default=0.3)
    parser.add_argument("--ours_max_wait", type=float, default=1.5)
    parser.add_argument("--ours_margin_scale", type=float, default=1.0)
    parser.add_argument("--ours_heartbeat_period", type=float, default=0.75)
    parser.add_argument("--ours_heartbeat_window", type=float, default=3.0)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "phase1a_ws/results/dynamic_multitask_event_driven"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    avg_rows = []

    for method in METHODS:
        method_rows = []
        for i in range(args.trials):
            row = run_trial(method, seed=18000 + i, args=args)
            method_rows.append(row)
            all_rows.append(row)
        avg_rows.append(average_rows(method_rows))

    raw_csv = output_dir / "dynamic_multitask_event_driven_cbba_ours_raw.csv"
    avg_csv = output_dir / "dynamic_multitask_event_driven_cbba_ours_averages.csv"

    save_csv(all_rows, raw_csv)
    save_csv(avg_rows, avg_csv)

    print(f"Saved raw trials: {raw_csv}")
    print(f"Saved averages: {avg_csv}")

    print("\n=== Event-Driven Dynamic Multitask Average Results ===")
    for row in avg_rows:
        print(row)

    make_summary_plot(avg_rows, output_dir)


if __name__ == "__main__":
    main()
