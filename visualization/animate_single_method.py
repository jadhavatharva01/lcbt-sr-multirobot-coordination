import math
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


POI_MAP = {
    1: (10.0, 10.0),
    2: (20.0, 5.0),
    3: (5.0, 25.0),
    4: (30.0, 30.0),
    5: (15.0, 18.0),
    6: (40.0, 12.0),
    7: (8.0, 40.0),
    8: (35.0, 42.0),
    9: (50.0, 25.0),
    10: (22.0, 38.0),
}

RETRIEVERS = {
    "R1": {"pos": (0.0, 0.0), "battery": 0.90, "health": 0.95, "obstacle_density": 0.20},
    "R2": {"pos": (25.0, 10.0), "battery": 0.80, "health": 0.90, "obstacle_density": 0.40},
    "R3": {"pos": (45.0, 5.0), "battery": 0.85, "health": 0.90, "obstacle_density": 0.30},
    "R4": {"pos": (5.0, 45.0), "battery": 0.75, "health": 0.85, "obstacle_density": 0.50},
    "R5": {"pos": (50.0, 50.0), "battery": 0.95, "health": 0.80, "obstacle_density": 0.35},
}

TRUE_POI = 5
OBJECT_POS = POI_MAP[TRUE_POI]
GRASP_RADIUS = 2.0


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def effective_speed(robot, nominal_speed=5.0):
    b = robot["battery"]
    h = robot["health"]
    rho = robot["obstacle_density"]

    return (
        nominal_speed
        * (0.5 + 0.5 * b)
        * (0.5 + 0.5 * h)
        * (1.0 / (1.0 + rho))
    )


def bid(robot, poi_pos):
    d = dist(robot["pos"], poi_pos)
    return d + 10.0 * (1.0 - robot["battery"]) + 10.0 * (1.0 - robot["health"])


def lerp(a, b, alpha):
    alpha = max(0.0, min(1.0, alpha))
    return (
        a[0] + alpha * (b[0] - a[0]),
        a[1] + alpha * (b[1] - a[1]),
    )


def build_scenario(mode):
    """
    Hand-crafted representative timelines based on our current simulator results.

    Times are relative seconds after detection.
    """
    starts = {r: data["pos"] for r, data in RETRIEVERS.items()}
    bids = {r: bid(data, OBJECT_POS) for r, data in RETRIEVERS.items()}
    speeds = {r: effective_speed(data) for r, data in RETRIEVERS.items()}
    travel = {r: dist(starts[r], OBJECT_POS) / speeds[r] for r in starts}

    events = []
    trajectories = {}

    for r in starts:
        trajectories[r] = {
            "start": starts[r],
            "goal": OBJECT_POS,
            "move_start": None,
            "move_end": None,
            "status": "IDLE",
            "blocked_time": None,
            "blocked_reason": "",
            "crit_start": None,
            "done_time": None,
        }

    if mode == "auction_only":
        # Representative failure/waste case:
        # R1 and R2 both believe they won and both move.
        for r in ["R1", "R2"]:
            move_start = 2.0
            move_end = move_start + travel[r]
            trajectories[r]["move_start"] = move_start
            trajectories[r]["move_end"] = move_end
            trajectories[r]["crit_start"] = move_end
            trajectories[r]["done_time"] = move_end + 2.0
            events.append((move_start, f"{r} enters PRE"))
            events.append((move_end, f"{r} enters CRIT"))

        events.append((max(trajectories["R1"]["done_time"], trajectories["R2"]["done_time"]), "Mission done, but duplicate CRIT risk occurred"))

    elif mode == "fixed_intent":
        # R1 gets blocked sometimes, but too late/weak in many runs.
        for r in ["R1", "R2"]:
            move_start = 3.0
            move_end = move_start + travel[r]
            trajectories[r]["move_start"] = move_start
            trajectories[r]["move_end"] = move_end
            trajectories[r]["crit_start"] = move_end
            trajectories[r]["done_time"] = move_end + 2.0
            events.append((2.0, f"{r} broadcasts fixed intent"))
            events.append((move_start, f"{r} enters PRE"))

        events.append((max(trajectories["R1"]["done_time"], trajectories["R2"]["done_time"]), "Mission done, fixed intent still allowed duplicate approach"))

    elif mode == "cbba_like_consensus":
        # Consensus waits longer but sends only one robot.
        r = "R2"
        move_start = 6.0
        move_end = move_start + travel[r]
        trajectories[r]["move_start"] = move_start
        trajectories[r]["move_end"] = move_end
        trajectories[r]["crit_start"] = move_end
        trajectories[r]["done_time"] = move_end + 2.0
        events.append((0.0, "Detection"))
        events.append((1.0, "Consensus round 1"))
        events.append((3.0, "Consensus round 2"))
        events.append((5.0, "Consensus round 3"))
        events.append((move_start, "R2 selected after consensus, enters PRE"))
        events.append((move_end, "R2 enters CRIT"))
        events.append((move_end + 2.0, "R2 DONE"))

    elif mode == "event_triggered_consensus":
        r = "R2"
        move_start = 3.9
        move_end = move_start + travel[r]
        trajectories[r]["move_start"] = move_start
        trajectories[r]["move_end"] = move_end
        trajectories[r]["crit_start"] = move_end
        trajectories[r]["done_time"] = move_end + 2.0
        events.append((0.0, "Detection"))
        events.append((2.0, "Initial local allocation"))
        events.append((2.5, "Conflict detected, event-triggered consensus"))
        events.append((move_start, "R2 selected, enters PRE"))
        events.append((move_end, "R2 enters CRIT"))
        events.append((move_end + 2.0, "R2 DONE"))

    elif mode == "ours_layered":
        r = "R2"
        move_start = 3.68
        move_end = move_start + travel[r]

        trajectories[r]["move_start"] = move_start
        trajectories[r]["move_end"] = move_end
        trajectories[r]["crit_start"] = move_end
        trajectories[r]["done_time"] = move_end + 2.0

        trajectories["R1"]["blocked_time"] = 3.0
        trajectories["R1"]["blocked_reason"] = "Blocked by R2 intent/heartbeat"

        events.append((0.0, "Detection"))
        events.append((2.0, "Local auction closes"))
        events.append((2.7, "R2 computes low-risk wait"))
        events.append((3.0, "R1 blocked by better intent/heartbeat"))
        events.append((move_start, "R2 enters PRE"))
        events.append((move_end, "R2 enters CRIT"))
        events.append((move_end + 2.0, "R2 DONE"))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_time = max(
        [t for t, _ in events] + [0.0]
    ) + 1.0

    return trajectories, events, total_time


def robot_position_at_time(robot_id, traj, t):
    start = traj["start"]
    goal = traj["goal"]

    if traj["move_start"] is None:
        return start

    if t < traj["move_start"]:
        return start

    if t >= traj["move_end"]:
        return goal

    alpha = (t - traj["move_start"]) / (traj["move_end"] - traj["move_start"])
    return lerp(start, goal, alpha)


def robot_status_at_time(traj, t):
    if traj["blocked_time"] is not None and t >= traj["blocked_time"]:
        return "BLOCKED"

    if traj["move_start"] is None:
        return "IDLE"

    if t < traj["move_start"]:
        return "WAIT"

    if traj["crit_start"] is not None and t >= traj["crit_start"]:
        if traj["done_time"] is not None and t >= traj["done_time"]:
            return "DONE"
        return "CRIT"

    if traj["move_end"] is not None and t < traj["move_end"]:
        return "PRE"

    return "IDLE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=[
        "auction_only",
        "fixed_intent",
        "cbba_like_consensus",
        "event_triggered_consensus",
        "ours_layered",
    ])
    parser.add_argument("--output_dir", default=str(Path.home() / "phase1a_ws/results/animations"))
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories, events, total_time = build_scenario(args.mode)

    dt = 1.0 / args.fps
    frames = int(total_time / dt)

    fig, ax = plt.subplots(figsize=(8, 8))

    event_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    time_text = ax.text(
        0.02, 0.04, "",
        transform=ax.transAxes,
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    robot_artists = {}
    robot_labels = {}

    def init():
        ax.clear()

        ax.set_xlim(-5, 55)
        ax.set_ylim(-5, 55)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(f"2D Mission Animation: {args.mode}")

        for poi_id, (x, y) in POI_MAP.items():
            ax.scatter(x, y, marker="x", s=80)
            ax.text(x + 0.4, y + 0.4, f"P{poi_id}", fontsize=8)

        ox, oy = OBJECT_POS
        ax.scatter(ox, oy, marker="*", s=250)
        circle = plt.Circle((ox, oy), GRASP_RADIUS, fill=False, linestyle="--")
        ax.add_patch(circle)
        ax.text(ox + 1, oy - 2.5, "grasp zone", fontsize=8)

        robot_artists.clear()
        robot_labels.clear()

        for rid, data in RETRIEVERS.items():
            x, y = data["pos"]
            artist = ax.scatter(x, y, s=140)
            label = ax.text(x + 0.6, y + 0.6, rid, fontsize=9)
            robot_artists[rid] = artist
            robot_labels[rid] = label

        return list(robot_artists.values()) + list(robot_labels.values())

    def update(frame):
        t = frame * dt
        ax.set_title(f"2D Mission Animation: {args.mode}")

        for rid, traj in trajectories.items():
            x, y = robot_position_at_time(rid, traj, t)
            status = robot_status_at_time(traj, t)

            robot_artists[rid].set_offsets([[x, y]])
            robot_labels[rid].set_position((x + 0.6, y + 0.6))
            robot_labels[rid].set_text(f"{rid}\n{status}")

        recent_events = [txt for et, txt in events if 0 <= t - et <= 2.0]
        if recent_events:
            event_text.set_text("\n".join(recent_events[-4:]))
        else:
            event_text.set_text("")

        time_text.set_text(f"t = {t:.1f} s")

        return list(robot_artists.values()) + list(robot_labels.values()) + [event_text, time_text]

    init()
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, blit=False)

    output_path = output_dir / f"{args.mode}_animation.gif"
    anim.save(output_path, writer=PillowWriter(fps=args.fps))

    print(f"Saved animation to: {output_path}")


if __name__ == "__main__":
    main()
