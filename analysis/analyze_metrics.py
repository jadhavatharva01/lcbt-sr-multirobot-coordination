import csv
import sys
import math
from collections import defaultdict


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


def parse_value(value_text):
    result = {}
    if not value_text:
        return result

    for part in value_text.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def distance_robot_to_poi(robot, poi):
    if robot not in ROBOT_POSITIONS or poi not in POI_MAP:
        return 0.0

    rx, ry = ROBOT_POSITIONS[robot]
    px, py = POI_MAP[poi]

    return math.sqrt((px - rx) ** 2 + (py - ry) ** 2)


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

            try:
                row["epoch"] = int(row["epoch"])
            except Exception:
                row["epoch"] = 0

            row["value_dict"] = parse_value(row.get("value", ""))

            events.append(row)

    return events


def compute_metrics(events):
    mission_start_times = [e["event_time"] for e in events if e["event"] == "MISSION_START"]
    detection_times = [e["event_time"] for e in events if e["event"] == "DETECTION"]
    done_times = [e["event_time"] for e in events if e["event"] == "DONE"]

    mission_start = min(mission_start_times) if mission_start_times else None
    first_detection = min(detection_times) if detection_times else None
    final_done = max(done_times) if done_times else None

    detections = [e for e in events if e["event"] == "DETECTION"]
    done_events = [e for e in events if e["event"] == "DONE"]

    pre_events = [e for e in events if e["event"] == "PRE"]
    crit_events = [e for e in events if e["event"] == "CRIT"]
    auction_events = [e for e in events if e["event"] == "LOCAL_AUCTION_RESULT"]

    grasp_denials = [e for e in events if e["event"] == "GRASP_DENIED"]
    stale_ignored = [e for e in events if e["event"] == "STALE_DETECTION_IGNORED"]
    lease_expired = [e for e in events if e["event"] == "LEASE_EXPIRED_BEFORE_CRIT"]

    # Standard metrics
    mission_completion_time = None
    if mission_start is not None and final_done is not None:
        mission_completion_time = final_done - mission_start

    search_time_to_detection = None
    if mission_start is not None and first_detection is not None:
        search_time_to_detection = first_detection - mission_start

    tasks_detected = len(detections)
    tasks_completed = len(done_events)

    retrieval_success_rate = 0.0
    if tasks_detected > 0:
        retrieval_success_rate = tasks_completed / tasks_detected

    robots_activated = sorted(set(e["robot"] for e in pre_events))

    # Energy proxy: sum distance to POI for robots that entered PRE
    total_path_length_energy_proxy = 0.0
    for e in pre_events:
        total_path_length_energy_proxy += distance_robot_to_poi(e["robot"], e["poi"])

    # Message count approximation from logged communication events
    message_events = {
        "DETECTION",
        "BID",
        "GRASP_REQUEST",
        "GRASP_GRANTED",
        "GRASP_DENIED",
        "GRASP_RELEASE",
    }
    number_of_messages = sum(1 for e in events if e["event"] in message_events)

    # Allocation time = first local auction result time - detection time
    planning_allocation_time = None
    if detection_times and auction_events:
        planning_allocation_time = min(e["event_time"] for e in auction_events) - first_detection

    # Winner disagreement
    auction_by_task = defaultdict(list)
    for e in auction_events:
        key = (e["poi"], e["epoch"])
        winner = e["value_dict"].get("winner", "")
        auction_by_task[key].append((e["robot"], winner, e["event_time"]))

    winner_disagreement_count = 0
    inconsistency_duration = 0.0

    for key, records in auction_by_task.items():
        winners = set(r[1] for r in records if r[1])
        if len(winners) > 1:
            winner_disagreement_count += 1
            times = [r[2] for r in records]
            inconsistency_duration += max(times) - min(times)

    # PRE duplicates
    pre_by_task = defaultdict(set)
    for e in pre_events:
        pre_by_task[(e["poi"], e["epoch"])].add(e["robot"])

    number_of_pre_duplicates = 0
    for robots in pre_by_task.values():
        number_of_pre_duplicates += max(0, len(robots) - 1)

    # CRIT violations
    crit_by_task = defaultdict(set)
    for e in crit_events:
        crit_by_task[(e["poi"], e["epoch"])].add(e["robot"])

    crit_violations = 0
    for robots in crit_by_task.values():
        if len(robots) > 1:
            crit_violations += 1

    # CRIT overlap duration using CRIT to DONE intervals
    crit_start = {}
    done_time = {}

    for e in crit_events:
        crit_start[(e["robot"], e["poi"], e["epoch"])] = e["event_time"]

    for e in done_events:
        done_time[(e["robot"], e["poi"], e["epoch"])] = e["event_time"]

    intervals_by_task = defaultdict(list)
    for key, start in crit_start.items():
        robot, poi, epoch = key
        end = done_time.get(key)
        if end is None:
            continue
        intervals_by_task[(poi, epoch)].append((robot, start, end))

    crit_overlap_duration = 0.0
    for task, intervals in intervals_by_task.items():
        for i in range(len(intervals)):
            for j in range(i + 1, len(intervals)):
                _, s1, e1 = intervals[i]
                _, s2, e2 = intervals[j]
                overlap = max(0.0, min(e1, e2) - max(s1, s2))
                crit_overlap_duration += overlap

    metrics = {
        "Mission completion time": mission_completion_time,
        "Retrieval success rate": retrieval_success_rate,
        "Total path length / energy proxy": total_path_length_energy_proxy,
        "Number of messages": number_of_messages,
        "Planning / allocation time": planning_allocation_time,
        "Number of robots activated": len(robots_activated),
        "Robots activated": ",".join(robots_activated),
        "Search time to detection": search_time_to_detection,
        "Tasks completed": tasks_completed,
        "Winner disagreement count": winner_disagreement_count,
        "Inconsistency duration": inconsistency_duration,
        "Number of PRE duplicates": number_of_pre_duplicates,
        "Number of CRIT violations": crit_violations,
        "CRIT overlap duration": crit_overlap_duration,
        "Grasp-zone denials": len(grasp_denials),
        "Stale detections ignored": len(stale_ignored),
        "Lease expiries before CRIT": len(lease_expired),
    }

    return metrics


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python3 analyze_metrics.py <events_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    events = load_events(csv_path)
    metrics = compute_metrics(events)

    print("\n=== Metrics Summary ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
