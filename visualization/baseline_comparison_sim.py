import math
import random
import csv
import argparse
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt


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


@dataclass
class SimConfig:
    mode: str
    true_poi: int = 5
    num_pois: int = 10
    seed: int = 42

    search_visit_time: float = 1.0
    nominal_speed: float = 5.0
    crit_duration: float = 2.0

    # Communication model
    delay_mean: float = 2.0
    delay_jitter: float = 1.0
    drop_rate: float = 0.10

    # Coordination timing
    fixed_intent_wait: float = 1.0
    conservative_wait: float = 4.0
    min_intent_wait: float = 0.5
    max_intent_wait: float = 3.0
    confidence_wait_scale: float = 2.0
    margin_wait_scale: float = 2.0

    # Risk/safety settings
    use_grasp_zone: bool = True
    allow_double_crit_without_guard: bool = True


@dataclass
class RobotResult:
    robot_id: str
    bid: float
    distance: float
    effective_speed: float
    travel_time: float
    received_bids: dict = field(default_factory=dict)
    local_winner: str = ""
    intent_wait: float = 0.0
    enters_pre: bool = False
    pre_start: float = 0.0
    reaches_object: float = 0.0
    enters_crit: bool = False
    crit_start: float = 0.0
    done_time: float = 0.0
    blocked_reason: str = ""


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def effective_speed(robot, nominal_speed):
    b = robot["battery"]
    h = robot["health"]
    rho = robot["obstacle_density"]

    return (
        nominal_speed
        * (0.5 + 0.5 * b)
        * (0.5 + 0.5 * h)
        * (1.0 / (1.0 + rho))
    )


def compute_bid(robot, poi_pos, nominal_speed):
    d = distance(robot["pos"], poi_pos)
    return d + 10.0 * (1.0 - robot["battery"]) + 10.0 * (1.0 - robot["health"])


def simulate_message_receive(rng, cfg):
    if rng.random() < cfg.drop_rate:
        return None

    delay = max(0.0, rng.gauss(cfg.delay_mean, cfg.delay_jitter))
    return delay


def compute_risk_wait(num_bids, expected_retrievers, bid_margin, cfg):
    confidence = min(1.0, num_bids / max(1, expected_retrievers))

    wait = (
        cfg.min_intent_wait
        + (1.0 - confidence) * cfg.confidence_wait_scale
        + cfg.margin_wait_scale / (1.0 + bid_margin)
    )

    return min(cfg.max_intent_wait, wait)


def get_search_time_to_detection(cfg, rng):
    pois = list(range(1, cfg.num_pois + 1))
    visited = []
    time = 0.0

    while pois:
        p = rng.choice(pois)
        pois.remove(p)
        visited.append(p)
        time += cfg.search_visit_time

        if p == cfg.true_poi:
            return time, len(visited)

    return time, len(visited)


def run_single_trial(cfg: SimConfig):
    rng = random.Random(cfg.seed)
    poi_pos = POI_MAP[cfg.true_poi]

    search_time, pois_visited = get_search_time_to_detection(cfg, rng)
    detection_time = search_time

    results = {}

    # Compute true bids and travel estimates
    for rid, robot in RETRIEVERS.items():
        d = distance(robot["pos"], poi_pos)
        v_eff = effective_speed(robot, cfg.nominal_speed)
        travel = d / max(v_eff, 0.01)
        bid = compute_bid(robot, poi_pos, cfg.nominal_speed)

        results[rid] = RobotResult(
            robot_id=rid,
            bid=bid,
            distance=d,
            effective_speed=v_eff,
            travel_time=travel,
        )

    true_winner = min(results, key=lambda rid: results[rid].bid)
    true_best_bid = results[true_winner].bid

    messages = 0

    # Ideal centralized assignment
    if cfg.mode == "ideal_assignment":
        r = results[true_winner]
        r.local_winner = true_winner
        r.enters_pre = True
        r.pre_start = detection_time
        r.reaches_object = r.pre_start + r.travel_time
        r.enters_crit = True
        r.crit_start = r.reaches_object
        r.done_time = r.crit_start + cfg.crit_duration

        messages = len(RETRIEVERS)  # simplified centralized state messages

    # Nearest robot baseline
    elif cfg.mode == "nearest_robot":
        nearest = min(results, key=lambda rid: results[rid].distance)
        r = results[nearest]

        r.local_winner = nearest
        r.enters_pre = True
        r.pre_start = detection_time
        r.reaches_object = r.pre_start + r.travel_time
        r.enters_crit = True
        r.crit_start = r.reaches_object
        r.done_time = r.crit_start + cfg.crit_duration

        # Simplified broadcast of detection + assignment
        messages = 2

    # Centralized cost-based assignment baseline
    elif cfg.mode == "cost_based_assignment":
        def assignment_cost(rid):
            r = results[rid]
            robot = RETRIEVERS[rid]
            return (
                r.distance
                + 10.0 * (1.0 - robot["battery"])
                + 10.0 * (1.0 - robot["health"])
                + 5.0 * robot["obstacle_density"]
            )

        winner = min(results, key=assignment_cost)
        r = results[winner]

        r.local_winner = winner
        r.enters_pre = True
        r.pre_start = detection_time + 0.2
        r.reaches_object = r.pre_start + r.travel_time
        r.enters_crit = True
        r.crit_start = r.reaches_object
        r.done_time = r.crit_start + cfg.crit_duration

        # detection + global state + assignment
        messages = len(RETRIEVERS) + 2

    # Auction-only BT
    elif cfg.mode == "auction_only":
        auction_close_time = detection_time + cfg.delay_mean

        for rid, r in results.items():
            r.received_bids[rid] = r.bid
            messages += 1

            for other_id, other in results.items():
                if other_id == rid:
                    continue

                msg_delay = simulate_message_receive(rng, cfg)
                if msg_delay is None:
                    continue

                if detection_time + msg_delay <= auction_close_time:
                    r.received_bids[other_id] = other.bid
                    messages += 1

            r.local_winner = min(r.received_bids, key=lambda x: r.received_bids[x])

            if r.local_winner == rid:
                r.enters_pre = True
                r.pre_start = auction_close_time
                r.reaches_object = r.pre_start + r.travel_time
                r.enters_crit = True
                r.crit_start = r.reaches_object
                r.done_time = r.crit_start + cfg.crit_duration

    # Fixed intent sharing
    elif cfg.mode == "fixed_intent":
        auction_close_time = detection_time + cfg.delay_mean

        intents = []

        for rid, r in results.items():
            r.received_bids[rid] = r.bid
            messages += 1

            for other_id, other in results.items():
                if other_id == rid:
                    continue

                msg_delay = simulate_message_receive(rng, cfg)
                if msg_delay is None:
                    continue

                if detection_time + msg_delay <= auction_close_time:
                    r.received_bids[other_id] = other.bid
                    messages += 1

            r.local_winner = min(r.received_bids, key=lambda x: r.received_bids[x])

            if r.local_winner == rid:
                r.intent_wait = cfg.fixed_intent_wait
                intent_time = auction_close_time
                intents.append((rid, r.bid, intent_time))
                messages += 1

        for rid, r in results.items():
            if r.local_winner != rid:
                continue

            blocked = False
            my_pre_time = auction_close_time + r.intent_wait

            for other_id, other_bid, intent_time in intents:
                if other_id == rid:
                    continue

                msg_delay = simulate_message_receive(rng, cfg)
                if msg_delay is None:
                    continue

                arrival = intent_time + msg_delay
                messages += 1

                if arrival <= my_pre_time and other_bid < r.bid:
                    blocked = True
                    r.blocked_reason = "fixed_intent_blocked"
                    break

            if not blocked:
                r.enters_pre = True
                r.pre_start = my_pre_time
                r.reaches_object = r.pre_start + r.travel_time
                r.enters_crit = True
                r.crit_start = r.reaches_object
                r.done_time = r.crit_start + cfg.crit_duration

    # Shared intention / reliable intention-sharing BT baseline
    elif cfg.mode == "shared_intention_perfect":
        # Assumes robots eventually share intention state correctly.
        # Similar to ideal shared-intention planning, but includes a small sharing delay.
        winner = true_winner
        r = results[winner]

        shared_intention_delay = max(0.5, cfg.delay_mean * 0.5)

        r.local_winner = winner
        r.enters_pre = True
        r.pre_start = detection_time + shared_intention_delay
        r.reaches_object = r.pre_start + r.travel_time
        r.enters_crit = True
        r.crit_start = r.reaches_object
        r.done_time = r.crit_start + cfg.crit_duration

        messages = len(RETRIEVERS) * 2

    # CBBA-like multi-round consensus baseline
    elif cfg.mode == "cbba_like_consensus":
        # Several communication rounds are used to converge on the best bid.
        consensus_rounds = 3
        round_duration = cfg.delay_mean
        consensus_time = consensus_rounds * round_duration

        messages = len(RETRIEVERS) * (len(RETRIEVERS) - 1) * consensus_rounds

        # Packet loss can prevent full convergence. If too many messages drop,
        # fallback to best among known subset.
        received_global = {}

        for rid, r in results.items():
            received_global[rid] = r.bid

        # Approximate consensus reliability
        successful_fraction = max(0.0, 1.0 - cfg.drop_rate)
        if successful_fraction >= 0.75:
            winner = true_winner
        else:
            # degraded consensus: choose among robots whose bids survived
            surviving = []
            for rid in results:
                if rng.random() > cfg.drop_rate:
                    surviving.append(rid)

            if not surviving:
                surviving = [rng.choice(list(results.keys()))]

            winner = min(surviving, key=lambda rid: results[rid].bid)

        for rid, r in results.items():
            r.local_winner = winner

        r = results[winner]
        r.enters_pre = True
        r.pre_start = detection_time + consensus_time
        r.reaches_object = r.pre_start + r.travel_time
        r.enters_crit = True
        r.crit_start = r.reaches_object
        r.done_time = r.crit_start + cfg.crit_duration

    # Event-triggered consensus baseline
    elif cfg.mode == "event_triggered_consensus":
        # Communicate only when the local decision is uncertain.
        # Uses bid margin as an uncertainty trigger.
        initial_wait = cfg.delay_mean

        # First collect a partial auction view like auction_only
        candidate_winners = {}

        for rid, r in results.items():
            r.received_bids[rid] = r.bid
            messages += 1

            for other_id, other in results.items():
                if other_id == rid:
                    continue

                msg_delay = simulate_message_receive(rng, cfg)
                if msg_delay is None:
                    continue

                if detection_time + msg_delay <= detection_time + initial_wait:
                    r.received_bids[other_id] = other.bid
                    messages += 1

            sorted_known = sorted(r.received_bids.items(), key=lambda x: x[1])
            r.local_winner = sorted_known[0][0]
            candidate_winners[rid] = r.local_winner

        # Trigger extra consensus only if disagreement exists or margin is small
        disagreement = len(set(candidate_winners.values())) > 1

        known_best_values = []
        for r in results.values():
            sorted_bids = sorted(r.received_bids.values())
            if len(sorted_bids) >= 2:
                margin = sorted_bids[1] - sorted_bids[0]
            else:
                margin = 0.0
            known_best_values.append(margin)

        small_margin = min(known_best_values) < 2.0

        extra_delay = 0.0

        if disagreement or small_margin:
            extra_delay = cfg.delay_mean
            messages += len(RETRIEVERS) * 2

            # after event-triggered update, assume better convergence unless messages dropped badly
            if cfg.drop_rate < 0.25:
                winner = true_winner
            else:
                winner = min(candidate_winners.values(), key=lambda rid: results[rid].bid)
        else:
            winner = min(candidate_winners.values(), key=lambda rid: results[rid].bid)

        for rid, r in results.items():
            r.local_winner = winner

        r = results[winner]
        r.enters_pre = True
        r.pre_start = detection_time + initial_wait + extra_delay
        r.reaches_object = r.pre_start + r.travel_time
        r.enters_crit = True
        r.crit_start = r.reaches_object
        r.done_time = r.crit_start + cfg.crit_duration

    # Conservative consistency waiting
    elif cfg.mode == "conservative_wait":
        auction_close_time = detection_time + cfg.conservative_wait

        for rid, r in results.items():
            r.received_bids[rid] = r.bid
            messages += 1

            for other_id, other in results.items():
                if other_id == rid:
                    continue

                msg_delay = simulate_message_receive(rng, cfg)
                if msg_delay is None:
                    continue

                if detection_time + msg_delay <= auction_close_time:
                    r.received_bids[other_id] = other.bid
                    messages += 1

            confidence = len(r.received_bids) / len(RETRIEVERS)

            if confidence < 0.6:
                r.blocked_reason = "low_confidence"
                continue

            r.local_winner = min(r.received_bids, key=lambda x: r.received_bids[x])

            if r.local_winner == rid:
                r.enters_pre = True
                r.pre_start = auction_close_time
                r.reaches_object = r.pre_start + r.travel_time
                r.enters_crit = True
                r.crit_start = r.reaches_object
                r.done_time = r.crit_start + cfg.crit_duration

    # Our layered method
    elif cfg.mode == "ours_layered":
        # Short auction, then risk-aware wait + intent + heartbeat + grasp-zone backup.
        auction_close_time = detection_time + cfg.delay_mean

        intents = []

        for rid, r in results.items():
            r.received_bids[rid] = r.bid
            messages += 1

            for other_id, other in results.items():
                if other_id == rid:
                    continue

                msg_delay = simulate_message_receive(rng, cfg)
                if msg_delay is None:
                    continue

                if detection_time + msg_delay <= auction_close_time:
                    r.received_bids[other_id] = other.bid
                    messages += 1

            r.local_winner = min(r.received_bids, key=lambda x: r.received_bids[x])

            if r.local_winner == rid:
                sorted_bids = sorted(r.received_bids.values())
                if len(sorted_bids) >= 2:
                    bid_margin = sorted_bids[1] - sorted_bids[0]
                else:
                    bid_margin = 0.0

                r.intent_wait = compute_risk_wait(
                    num_bids=len(r.received_bids),
                    expected_retrievers=len(RETRIEVERS),
                    bid_margin=bid_margin,
                    cfg=cfg,
                )

                intent_time = auction_close_time
                intents.append((rid, r.bid, intent_time))
                messages += 1

        # Intent conflict filtering
        for rid, r in results.items():
            if r.local_winner != rid:
                continue

            blocked = False
            my_pre_time = auction_close_time + r.intent_wait

            for other_id, other_bid, intent_time in intents:
                if other_id == rid:
                    continue

                msg_delay = simulate_message_receive(rng, cfg)
                if msg_delay is None:
                    continue

                arrival = intent_time + msg_delay
                messages += 1

                if arrival <= my_pre_time and other_bid < r.bid:
                    blocked = True
                    r.blocked_reason = "risk_intent_blocked"
                    break

            if not blocked:
                r.enters_pre = True
                r.pre_start = my_pre_time
                r.reaches_object = r.pre_start + r.travel_time

        # Heartbeat filtering: if a better robot is already in PRE and heartbeat arrives,
        # later/worse robots back off before reaching object.
        pre_robots = [r for r in results.values() if r.enters_pre]
        pre_robots.sort(key=lambda x: x.pre_start)

        for r in pre_robots:
            for other in pre_robots:
                if other.robot_id == r.robot_id:
                    continue

                if other.bid < r.bid and other.pre_start <= r.reaches_object:
                    hb_delay = simulate_message_receive(rng, cfg)
                    if hb_delay is None:
                        continue

                    hb_arrival = other.pre_start + hb_delay
                    messages += 1

                    if hb_arrival < r.reaches_object:
                        r.enters_pre = False
                        r.blocked_reason = "heartbeat_blocked"
                        break

        # Final CRIT/grasp-zone
        active = [r for r in results.values() if r.enters_pre]
        active.sort(key=lambda x: x.reaches_object)

        grasp_owner = None

        for r in active:
            if cfg.use_grasp_zone:
                if grasp_owner is None:
                    grasp_owner = r.robot_id
                    r.enters_crit = True
                    r.crit_start = r.reaches_object
                    r.done_time = r.crit_start + cfg.crit_duration
                else:
                    r.blocked_reason = "grasp_denied"
            else:
                r.enters_crit = True
                r.crit_start = r.reaches_object
                r.done_time = r.crit_start + cfg.crit_duration

    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # If grasp-zone is enabled, enforce at most first CRIT for non-ours modes too
    if cfg.use_grasp_zone and cfg.mode in [
        "auction_only",
        "fixed_intent",
        "conservative_wait",
        "nearest_robot",
        "cost_based_assignment",
        "shared_intention_perfect",
        "cbba_like_consensus",
        "event_triggered_consensus",
    ]:
        crit_robots = [r for r in results.values() if r.enters_crit]
        crit_robots.sort(key=lambda x: x.crit_start)

        if len(crit_robots) > 1:
            owner = crit_robots[0].robot_id
            for r in crit_robots[1:]:
                r.enters_crit = False
                r.done_time = 0.0
                r.blocked_reason = "grasp_denied"

    # Metrics
    pre_robots = [r for r in results.values() if r.enters_pre]
    crit_robots = [r for r in results.values() if r.enters_crit]
    done_robots = [r for r in results.values() if r.done_time > 0.0]

    mission_success = 1 if len(done_robots) >= 1 else 0
    mission_completion_time = max([r.done_time for r in done_robots], default=0.0)

    total_distance = sum(r.distance for r in pre_robots)
    robots_activated = len(pre_robots)
    pre_duplicates = max(0, len(pre_robots) - 1)
    crit_violations = 1 if len(crit_robots) > 1 else 0
    grasp_denials = sum(1 for r in results.values() if r.blocked_reason == "grasp_denied")
    intent_blocks = sum(1 for r in results.values() if "intent_blocked" in r.blocked_reason)
    heartbeat_blocks = sum(1 for r in results.values() if r.blocked_reason == "heartbeat_blocked")
    low_conf_blocks = sum(1 for r in results.values() if r.blocked_reason == "low_confidence")

    # Winner disagreement count
    local_winners = set(r.local_winner for r in results.values() if r.local_winner)
    winner_disagreement = 1 if len(local_winners) > 1 else 0

    # Planning/allocation time approximated as first PRE time minus detection time
    first_pre = min([r.pre_start for r in pre_robots], default=0.0)
    planning_allocation_time = first_pre - detection_time if first_pre > 0 else 0.0

    return {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "mission_completion_time": mission_completion_time,
        "mission_success": mission_success,
        "total_distance_energy_proxy": total_distance,
        "messages": messages,
        "planning_allocation_time": planning_allocation_time,
        "robots_activated": robots_activated,
        "search_time_to_detection": search_time,
        "pois_visited_before_detection": pois_visited,
        "tasks_completed": mission_success,
        "winner_disagreement": winner_disagreement,
        "pre_duplicates": pre_duplicates,
        "crit_violations": crit_violations,
        "grasp_denials": grasp_denials,
        "intent_blocks": intent_blocks,
        "heartbeat_blocks": heartbeat_blocks,
        "low_confidence_blocks": low_conf_blocks,
    }


def average_results(results):
    if not results:
        return {}

    keys = [k for k in results[0].keys() if k not in ["mode", "seed"]]
    avg = {"mode": results[0]["mode"]}

    for k in keys:
        avg[k] = sum(r[k] for r in results) / len(results)

    return avg


def save_csv(rows, path):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_bar_plot(avg_rows, metric, output_dir):
    modes = [r["mode"] for r in avg_rows]
    values = [r[metric] for r in avg_rows]

    plt.figure(figsize=(10, 5))
    plt.bar(modes, values)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(metric.replace("_", " ").title())
    plt.tight_layout()

    out = output_dir / f"{metric}.png"
    plt.savefig(out, dpi=200)
    plt.close()

    print(f"Saved graph: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default=str(Path.home() / "phase1a_ws/results/baseline_comparison"))
    parser.add_argument("--delay_mean", type=float, default=2.0)
    parser.add_argument("--delay_jitter", type=float, default=1.0)
    parser.add_argument("--drop_rate", type=float, default=0.10)
    parser.add_argument("--grasp_zone", action="store_true", help="Enable final grasp-zone guard")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = [
        "ideal_assignment",
        "nearest_robot",
        "cost_based_assignment",
        "auction_only",
        "fixed_intent",
        "shared_intention_perfect",
        "conservative_wait",
        "cbba_like_consensus",
        "event_triggered_consensus",
        "ours_layered",
    ]

    all_rows = []
    avg_rows = []

    for mode in modes:
        mode_rows = []

        for i in range(args.trials):
            cfg = SimConfig(
                mode=mode,
                seed=42 + i,
                delay_mean=args.delay_mean,
                delay_jitter=args.delay_jitter,
                drop_rate=args.drop_rate,
                use_grasp_zone=args.grasp_zone,
            )

            row = run_single_trial(cfg)
            mode_rows.append(row)
            all_rows.append(row)

        avg_rows.append(average_results(mode_rows))

    raw_csv = output_dir / "comparison_raw_trials.csv"
    avg_csv = output_dir / "comparison_averages.csv"

    save_csv(all_rows, raw_csv)
    save_csv(avg_rows, avg_csv)

    print(f"Saved raw trials: {raw_csv}")
    print(f"Saved averages: {avg_csv}")

    metrics_to_plot = [
        "mission_completion_time",
        "mission_success",
        "total_distance_energy_proxy",
        "messages",
        "planning_allocation_time",
        "robots_activated",
        "winner_disagreement",
        "pre_duplicates",
        "crit_violations",
        "grasp_denials",
        "intent_blocks",
        "heartbeat_blocks",
    ]

    for metric in metrics_to_plot:
        make_bar_plot(avg_rows, metric, output_dir)

    print("\n=== Average Results ===")
    for row in avg_rows:
        print(row)


if __name__ == "__main__":
    main()
