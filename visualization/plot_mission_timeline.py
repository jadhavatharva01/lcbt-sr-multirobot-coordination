import csv
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


STATE_EVENTS = {
    "PRE": 1,
    "CRIT": 2,
    "DONE": 3,
    "INTENT_CONFLICT_BLOCKED": -1,
    "GRASP_DENIED": -2,
    "LEASE_EXPIRED_BEFORE_CRIT": -3,
}


def load_events(csv_path):
    events = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                row["event_time"] = float(row["event_time"])
            except Exception:
                continue

            try:
                row["poi"] = int(row["poi"])
            except Exception:
                row["poi"] = -1

            events.append(row)

    return events


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 plot_mission_timeline.py <events_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    events = load_events(csv_path)

    if not events:
        print("No events found.")
        sys.exit(1)

    t0 = min(e["event_time"] for e in events)
    robot_events = defaultdict(list)

    for e in events:
        event = e["event"]
        robot = e["robot"]

        if robot == "":
            continue

        if event in STATE_EVENTS:
            robot_events[robot].append((e["event_time"] - t0, event, STATE_EVENTS[event]))

    robots = sorted(robot_events.keys())

    if not robots:
        print("No robot state events found.")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(12, 5))

    y_positions = {robot: i for i, robot in enumerate(robots)}

    for robot in robots:
        y = y_positions[robot]
        for t, event, state_value in robot_events[robot]:
            ax.scatter(t, y, s=120)
            ax.text(t + 0.05, y + 0.05, event, fontsize=8, rotation=25)

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(robots)
    ax.set_xlabel("Time since first event (s)")
    ax.set_title("Mission Event Timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    output_path = csv_path.replace("_events.csv", "_timeline.png")
    plt.savefig(output_path, dpi=200)

    print(f"Saved timeline plot to: {output_path}")


if __name__ == "__main__":
    main()
