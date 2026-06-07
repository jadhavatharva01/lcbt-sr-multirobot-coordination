import csv
import sys
import math
from collections import defaultdict
import matplotlib.pyplot as plt


ROBOT_POSITIONS = {
    "R1": (0.0, 0.0),
    "R2": (25.0, 10.0),
    "R3": (45.0, 5.0),
    "R4": (5.0, 45.0),
    "R5": (50.0, 50.0),
}

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

GRASP_RADIUS = 2.0


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
        print("  python3 plot_2d_mission.py <events_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    events = load_events(csv_path)

    if not events:
        print("No events found.")
        sys.exit(1)

    detection_events = [e for e in events if e["event"] == "DETECTION"]
    true_poi = detection_events[0]["poi"] if detection_events else None

    pre_events = [e for e in events if e["event"] == "PRE"]
    crit_events = [e for e in events if e["event"] == "CRIT"]
    done_events = [e for e in events if e["event"] == "DONE"]
    blocked_events = [
        e for e in events
        if e["event"] in ["INTENT_CONFLICT_BLOCKED", "HEARTBEAT_CONFLICT_BLOCKED", "GRASP_DENIED"]
    ]

    fig, ax = plt.subplots(figsize=(9, 8))

    # Plot POIs
    for poi_id, (x, y) in POI_MAP.items():
        ax.scatter(x, y, s=90, marker="x")
        ax.text(x + 0.5, y + 0.5, f"POI {poi_id}", fontsize=8)

    # Highlight true object POI and grasp zone
    if true_poi in POI_MAP:
        ox, oy = POI_MAP[true_poi]
        ax.scatter(ox, oy, s=220, marker="*", label=f"Object at POI {true_poi}")
        circle = plt.Circle((ox, oy), GRASP_RADIUS, fill=False, linestyle="--")
        ax.add_patch(circle)
        ax.text(ox + 1.0, oy - 2.5, "grasp zone", fontsize=9)

    # Plot robot starts
    for robot, (x, y) in ROBOT_POSITIONS.items():
        ax.scatter(x, y, s=120, marker="o")
        ax.text(x + 0.5, y + 0.5, robot, fontsize=10)

    # Draw movement arrows for robots that entered PRE
    for e in pre_events:
        robot = e["robot"]
        poi = e["poi"]

        if robot not in ROBOT_POSITIONS or poi not in POI_MAP:
            continue

        sx, sy = ROBOT_POSITIONS[robot]
        gx, gy = POI_MAP[poi]

        ax.annotate(
            "",
            xy=(gx, gy),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->", linewidth=2),
        )
        ax.text(
            (sx + gx) / 2,
            (sy + gy) / 2,
            f"{robot} PRE",
            fontsize=9,
        )

    # Mark CRIT robots
    for e in crit_events:
        robot = e["robot"]
        poi = e["poi"]

        if poi in POI_MAP:
            x, y = POI_MAP[poi]
            ax.text(x + 1.5, y + 1.5, f"{robot} CRIT", fontsize=10)

    # Mark DONE robots
    for e in done_events:
        robot = e["robot"]
        poi = e["poi"]

        if poi in POI_MAP:
            x, y = POI_MAP[poi]
            ax.text(x + 1.5, y + 3.0, f"{robot} DONE", fontsize=10)

    # Mark blocked robots near their start point
    for e in blocked_events:
        robot = e["robot"]
        event = e["event"]

        if robot in ROBOT_POSITIONS:
            x, y = ROBOT_POSITIONS[robot]
            ax.text(x + 0.5, y - 1.5, f"{event}", fontsize=8)

    ax.set_title("2D Mission Visualization")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    plt.tight_layout()

    output_path = csv_path.replace("_events.csv", "_2d_mission.png")
    plt.savefig(output_path, dpi=200)

    print(f"Saved 2D mission plot to: {output_path}")


if __name__ == "__main__":
    main()
