import argparse
import csv
import math
import random
from pathlib import Path
import matplotlib.pyplot as plt


METHODS = [
    "independent_bid",
    "marginal_route_bid",
]


def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def generate_pois(num_pois, rng, world_size):
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
        rid = f"R{i+1}"

        robots[rid] = {
            "start_pos": (
                rng.uniform(0.0, world_size),
                rng.uniform(0.0, world_size),
            ),
            "battery": rng.uniform(0.75, 1.0),
            "health": rng.uniform(0.80, 1.0),
            "route": [],
            "available_time": 0.0,
            "distance": 0.0,
            "messages": 0,
        }

    return robots


def generate_sudden_tasks(num_tasks, pois, rng, release_gap):
    poi_ids = list(pois.keys())
    rng.shuffle(poi_ids)

    tasks = []

    for k in range(num_tasks):
        tasks.append({
            "task_id": f"T{k+1}",
            "poi_id": poi_ids[k],
            "release_time": (k + 1) * release_gap,
            "status": "unassigned",
            "owner": None,
            "start_time": None,
            "done_time": None,
        })

    return tasks


def robot_state_penalty(robot):
    # Small cost penalty for weak robot state.
    return 5.0 * (1.0 - robot["battery"]) + 5.0 * (1.0 - robot["health"])


def route_distance(start_pos, route, pois):
    if not route:
        return 0.0

    total = 0.0
    current = start_pos

    for poi_id in route:
        nxt = pois[poi_id]
        total += euclidean(current, nxt)
        current = nxt

    return total


def best_insert_marginal_cost(robot, task_poi_id, pois):
    """
    Computes the minimum extra route distance caused by inserting a new task
    into the robot's current committed route.
    """
    current_route = list(robot["route"])
    current_cost = route_distance(robot["start_pos"], current_route, pois)

    best_extra = None
    best_route = None

    for insert_idx in range(len(current_route) + 1):
        candidate_route = (
            current_route[:insert_idx]
            + [task_poi_id]
            + current_route[insert_idx:]
        )

        candidate_cost = route_distance(robot["start_pos"], candidate_route, pois)
        extra = candidate_cost - current_cost

        if best_extra is None or extra < best_extra:
            best_extra = extra
            best_route = candidate_route

    return best_extra, best_route


def independent_bid(robot, task_poi_id, pois, current_time):
    """
    Independent bid ignores current committed route.
    It only considers the robot's expected current/final position and task distance.
    """
    if robot["route"]:
        # Approximate robot as committed to its last assigned task.
        reference_pos = pois[robot["route"][-1]]
    else:
        reference_pos = robot["start_pos"]

    task_pos = pois[task_poi_id]
    return euclidean(reference_pos, task_pos) + robot_state_penalty(robot)


def marginal_route_bid(robot, task_poi_id, pois, current_time):
    """
    Marginal route-dependent bid.
    Measures how much extra route cost the new task adds to the existing route.
    """
    extra_cost, _best_route = best_insert_marginal_cost(robot, task_poi_id, pois)
    return extra_cost + robot_state_penalty(robot)


def assign_task(method, task, robots, pois, current_time):
    bids = {}

    for rid, robot in robots.items():
        if method == "independent_bid":
            bids[rid] = independent_bid(robot, task["poi_id"], pois, current_time)
        elif method == "marginal_route_bid":
            bids[rid] = marginal_route_bid(robot, task["poi_id"], pois, current_time)
        else:
            raise ValueError(f"Unknown method: {method}")

    winner = min(bids, key=bids.get)

    # Message model: each robot sends one bid, winner broadcast once.
    message_count = len(robots) + 1

    task["owner"] = winner
    task["status"] = "assigned"

    winner_robot = robots[winner]

    if method == "marginal_route_bid":
        _extra, new_route = best_insert_marginal_cost(winner_robot, task["poi_id"], pois)
        winner_robot["route"] = new_route
    else:
        winner_robot["route"].append(task["poi_id"])

    for robot in robots.values():
        robot["messages"] += message_count

    return winner, bids, message_count


def simulate_execution(robots, tasks, pois, speed):
    """
    After all tasks are assigned, execute each robot's route.
    This keeps the first version simple and makes route-dependency visible.
    """
    for rid, robot in robots.items():
        current_pos = robot["start_pos"]
        current_time = 0.0

        for poi_id in robot["route"]:
            travel = euclidean(current_pos, pois[poi_id])
            robot["distance"] += travel
            current_time += travel / speed

            for task in tasks:
                if task["poi_id"] == poi_id and task["owner"] == rid and task["done_time"] is None:
                    task["start_time"] = max(task["release_time"], current_time - travel / speed)
                    task["done_time"] = max(task["release_time"], current_time)
                    task["status"] = "done"
                    break

            current_pos = pois[poi_id]

        robot["available_time"] = current_time


def run_trial(method, seed, args):
    rng = random.Random(seed)

    pois = generate_pois(args.num_pois, rng, args.world_size)
    robots = generate_robots(args.num_robots, rng, args.world_size)
    tasks = generate_sudden_tasks(args.num_tasks, pois, rng, args.release_gap)

    event_log = []

    for task in tasks:
        current_time = task["release_time"]

        winner, bids, msg_count = assign_task(
            method=method,
            task=task,
            robots=robots,
            pois=pois,
            current_time=current_time,
        )

        event_log.append({
            "time": current_time,
            "event": "TASK_RELEASE_AND_ASSIGN",
            "task_id": task["task_id"],
            "poi_id": task["poi_id"],
            "winner": winner,
            "messages": msg_count,
            "bids": bids,
        })

    simulate_execution(robots, tasks, pois, args.speed)

    completed_tasks = [t for t in tasks if t["status"] == "done"]
    task_success_rate = len(completed_tasks) / max(1, len(tasks))

    mission_completion_time = max(
        [t["done_time"] for t in completed_tasks],
        default=0.0,
    )

    total_distance = sum(r["distance"] for r in robots.values())
    total_messages = sum(r["messages"] for r in robots.values())

    # Route imbalance: how unevenly work is distributed.
    route_lengths = [len(r["route"]) for r in robots.values()]
    max_route_len = max(route_lengths) if route_lengths else 0
    min_route_len = min(route_lengths) if route_lengths else 0
    route_imbalance = max_route_len - min_route_len

    return {
        "method": method,
        "seed": seed,
        "num_pois": args.num_pois,
        "num_robots": args.num_robots,
        "num_tasks": args.num_tasks,
        "task_success_rate": task_success_rate,
        "mission_completion_time": mission_completion_time,
        "total_distance": total_distance,
        "total_messages": total_messages,
        "route_imbalance": route_imbalance,
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
        ("route_imbalance", "Route Imbalance"),
    ]

    labels = {
        "independent_bid": "Independent\nBid",
        "marginal_route_bid": "Marginal\nRoute Bid",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric, title) in zip(axes, metrics):
        x = [labels[row["method"]] for row in avg_rows]
        y = [row[metric] for row in avg_rows]

        ax.bar(x, y)
        ax.set_title(title)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for i, v in enumerate(y):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Dynamic Multi-Task Bidding Comparison", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out = output_dir / "dynamic_multitask_comparison.png"
    plt.savefig(out, dpi=220)
    print(f"Saved graph: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pois", type=int, default=25)
    parser.add_argument("--num_robots", type=int, default=5)
    parser.add_argument("--num_tasks", type=int, default=6)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--world_size", type=float, default=60.0)
    parser.add_argument("--release_gap", type=float, default=4.0)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.home() / "phase1a_ws/results/dynamic_multitask"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    avg_rows = []

    for method in METHODS:
        method_rows = []

        for i in range(args.trials):
            row = run_trial(method, seed=12000 + i, args=args)
            method_rows.append(row)
            all_rows.append(row)

        avg_rows.append(average_rows(method_rows))

    raw_csv = output_dir / "dynamic_multitask_raw.csv"
    avg_csv = output_dir / "dynamic_multitask_averages.csv"

    save_csv(all_rows, raw_csv)
    save_csv(avg_rows, avg_csv)

    print(f"Saved raw trials: {raw_csv}")
    print(f"Saved averages: {avg_csv}")

    print("\n=== Dynamic Multitask Average Results ===")
    for row in avg_rows:
        print(row)

    make_summary_plot(avg_rows, output_dir)


if __name__ == "__main__":
    main()
