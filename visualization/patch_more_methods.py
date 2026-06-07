from pathlib import Path
import re

path = Path.home() / "phase1a_ws/visualization/baseline_comparison_sim.py"
text = path.read_text()

# ------------------------------------------------------------
# 1. Replace the mode list
# ------------------------------------------------------------
old_modes = '''    modes = [
        "ideal_assignment",
        "auction_only",
        "fixed_intent",
        "conservative_wait",
        "ours_layered",
    ]'''

new_modes = '''    modes = [
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
    ]'''

text = text.replace(old_modes, new_modes)

# ------------------------------------------------------------
# 2. Insert new baseline branches before Auction-only BT
# ------------------------------------------------------------
insert_before = '''    # Auction-only BT
    elif cfg.mode == "auction_only":'''

new_branches = r'''    # Nearest robot baseline
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

''' + insert_before

text = text.replace(insert_before, new_branches)

# ------------------------------------------------------------
# 3. Insert shared-intention, CBBA-like, and event-triggered
#    before Conservative consistency waiting
# ------------------------------------------------------------
insert_before2 = '''    # Conservative consistency waiting
    elif cfg.mode == "conservative_wait":'''

new_branches2 = r'''    # Shared intention / reliable intention-sharing BT baseline
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

''' + insert_before2

text = text.replace(insert_before2, new_branches2)

# ------------------------------------------------------------
# 4. Update grasp-zone enforcement mode list
# ------------------------------------------------------------
old_guard = '''    if cfg.use_grasp_zone and cfg.mode in ["auction_only", "fixed_intent", "conservative_wait"]:'''

new_guard = '''    if cfg.use_grasp_zone and cfg.mode in [
        "auction_only",
        "fixed_intent",
        "conservative_wait",
        "nearest_robot",
        "cost_based_assignment",
        "shared_intention_perfect",
        "cbba_like_consensus",
        "event_triggered_consensus",
    ]:'''

text = text.replace(old_guard, new_guard)

path.write_text(text)
print(f"Patched more methods into: {path}")
