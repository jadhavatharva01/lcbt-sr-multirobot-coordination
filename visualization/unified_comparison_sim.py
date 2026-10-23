"""
Unified multi-robot coordination simulator under asynchronous communication.

Design goal: an HONEST comparison.

  * Every decentralized method runs through the EXACT SAME stochastic
    communication model (independent per-message drop + Gaussian delay).
  * Methods differ ONLY in their decision policy (how long to wait, what
    coordination guards they use). They are not handicapped relative to
    each other.
  * The only "cheating" method is `oracle`, which is explicitly an
    unreachable upper bound (god-assignment, no comms) and is plotted as a
    reference line, never as a competitor we claim to beat.

The phenomenon we are trying to expose:
  Under lossy/delayed comms, several robots can each *believe* they are the
  winner (they never heard the truly-best robot's bid). Naive methods then
  commit on stale information -> duplicate activation, wasted travel energy,
  and (without a physical mutex) collisions in the critical grasp zone.
  Our layered method trades a little latency for safety by progressively
  catching these false winners: risk-aware wait -> intent -> heartbeat ->
  lease -> grasp-zone mutex.

Subcommands:
  compare   single comms regime, bar charts per metric
  sweep     vary drop_rate (and delay) -> metric-vs-degradation curves
  ablate    ours_full vs each layer removed -> isolate layer contributions

This file deliberately supersedes baseline_comparison_sim.py, whose
idealized baselines did not share the stochastic comms model.
"""

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
# World
# ----------------------------------------------------------------------------
POI_MAP = {
    1: (10.0, 10.0), 2: (20.0, 5.0), 3: (5.0, 25.0), 4: (30.0, 30.0),
    5: (15.0, 18.0), 6: (40.0, 12.0), 7: (8.0, 40.0), 8: (35.0, 42.0),
    9: (50.0, 25.0), 10: (22.0, 38.0),
}

# Base robot fleet. Positions are jittered per-trial so results are not an
# artifact of one fixed geometry. Profiles vary so bids are not all equal.
BASE_ROBOTS = {
    "R1": {"pos": (0.0, 0.0),   "battery": 0.90, "health": 0.95, "obstacle_density": 0.20},
    "R2": {"pos": (25.0, 10.0), "battery": 0.80, "health": 0.90, "obstacle_density": 0.40},
    "R3": {"pos": (18.0, 22.0), "battery": 0.85, "health": 0.90, "obstacle_density": 0.30},
    "R4": {"pos": (5.0, 30.0),  "battery": 0.78, "health": 0.85, "obstacle_density": 0.50},
    "R5": {"pos": (28.0, 28.0), "battery": 0.92, "health": 0.82, "obstacle_density": 0.35},
}


@dataclass
class Comms:
    delay_mean: float = 2.0
    delay_jitter: float = 1.0
    drop_rate: float = 0.10
    # Optional Gilbert-Elliott burst channel: the medium flips between a GOOD
    # state (drop=drop_rate) and a BAD state (drop=burst_drop) as a 2-state
    # Markov chain over time slots. Models correlated outages / transient
    # partitions, where consensus methods stall waiting for rounds to get
    # through. When burst=False, loss is i.i.d. at drop_rate.
    burst: bool = False
    burst_drop: float = 0.95
    p_good_to_bad: float = 0.15   # per slot
    p_bad_to_good: float = 0.30   # per slot -> mean bad spell ~ 1/0.30 slots
    slot: float = 1.0             # seconds per channel slot


def make_channel(comms, rng, horizon=400.0):
    """Returns drop_prob(t): the loss probability of a transmission sent at t.
    i.i.d. unless comms.burst, in which case a precomputed GE state sequence
    drives correlated bursts."""
    if not comms.burst:
        return lambda t: comms.drop_rate
    n_slots = int(horizon / comms.slot) + 2
    states = []
    bad = False
    for _ in range(n_slots):
        if bad:
            bad = rng.random() >= comms.p_bad_to_good
        else:
            bad = rng.random() < comms.p_good_to_bad
        states.append(bad)

    def drop_prob(t):
        idx = int(max(0.0, t) / comms.slot)
        idx = min(idx, n_slots - 1)
        return comms.burst_drop if states[idx] else comms.drop_rate
    return drop_prob


@dataclass
class Policy:
    name: str
    # auction window = window_factor * (delay_mean + jitter), unless rounds>1
    window_factor: float = 1.0
    consensus_rounds: int = 1          # legacy fixed-round bidding (unused by CBBA now)
    commit_wait_mode: str = "none"     # none | fixed | conservative | risk
    fixed_wait: float = 1.0
    abstain_confidence: float = 0.0    # if >0, abstain when confidence below it
    use_intent: bool = False           # pre-PRE backoff on better intent
    use_heartbeat: bool = False        # en-route backoff on better active robot
    use_grasp_zone: bool = False       # physical mutex at the critical section

    # CBBA-style consensus: iterate exchange rounds until agents AGREE on a
    # conflict-free assignment (or hit the cap). Does not act until converged,
    # so rounds/messages/latency all grow as comms degrade.
    consensus: bool = False
    consensus_max_rounds: int = 30

    # risk-aware wait params (only used when commit_wait_mode == "risk")
    min_wait: float = 0.5
    conf_scale: float = 2.0
    margin_scale: float = 2.0
    max_wait: float = 3.0


# The honest method zoo. Decentralized methods all share the comms model.
def build_policies():
    return {
        # Unreachable reference. God assignment, no comms cost, no conflict.
        "oracle": Policy("oracle"),

        # Commit on whatever you heard in a short window. No guards.
        "greedy_immediate": Policy(
            "greedy_immediate", window_factor=0.5, commit_wait_mode="none"),

        # Longer fixed auction window, still no guards.
        "auction_only": Policy(
            "auction_only", window_factor=1.0, commit_wait_mode="none"),

        # Auction + fixed intent wait + intent backoff.
        "auction_intent": Policy(
            "auction_intent", window_factor=1.0, commit_wait_mode="fixed",
            fixed_wait=1.0, use_intent=True),

        # CBBA-like: iterate consensus rounds until agents agree on a
        # conflict-free assignment, THEN act. Convergence (and thus rounds,
        # messages, and start latency) degrades as comms get lossier.
        "cbba_consensus": Policy(
            "cbba_consensus", consensus=True, commit_wait_mode="none"),

        # Wait long, and abstain if you did not hear enough of the fleet.
        "conservative_wait": Policy(
            "conservative_wait", window_factor=2.0, commit_wait_mode="conservative",
            fixed_wait=2.0, abstain_confidence=0.6),

        # Ours: risk-aware adaptive wait + ALGORITHMIC guards only (intent +
        # heartbeat). The physical grasp-zone mutex is intentionally OFF: it is
        # a deployment safety interlock any method could add, not an algorithmic
        # contribution, so collision-freedom must now be earned by coordination.
        "ours_layered": Policy(
            "ours_layered", window_factor=1.0, commit_wait_mode="risk",
            use_intent=True, use_heartbeat=True, use_grasp_zone=False),
    }


# ablation variants of ours
def build_ablations():
    base = build_policies()["ours_layered"]

    def variant(name, **overrides):
        p = Policy(**{**base.__dict__})
        p.name = name
        for k, v in overrides.items():
            setattr(p, k, v)
        return p

    return {
        "ours_full": variant("ours_full"),
        "ours_no_wait": variant("ours_no_wait", commit_wait_mode="none"),
        "ours_no_intent": variant("ours_no_intent", use_intent=False),
        "ours_no_heartbeat": variant("ours_no_heartbeat", use_heartbeat=False),
    }


# ----------------------------------------------------------------------------
# Physics / cost helpers
# ----------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def effective_speed(robot, nominal=5.0):
    return (nominal
            * (0.5 + 0.5 * robot["battery"])
            * (0.5 + 0.5 * robot["health"])
            * (1.0 / (1.0 + robot["obstacle_density"])))


def compute_bid(robot, poi_pos):
    return (dist(robot["pos"], poi_pos)
            + 10.0 * (1.0 - robot["battery"])
            + 10.0 * (1.0 - robot["health"]))


@dataclass
class Robot:
    rid: str
    pos: tuple
    bid: float
    travel_time: float
    distance: float
    # per-trial state
    received: dict = field(default_factory=dict)
    believes_winner: bool = False
    confidence: float = 0.0
    commit_wait: float = 0.0
    pre_start: float = math.inf
    reaches: float = math.inf
    enters_pre: bool = False
    enters_crit: bool = False
    done_time: float = 0.0
    blocked_reason: str = ""


def make_world(seed, true_poi, position_jitter, nominal_speed, crit_duration,
               search_visit_time, num_pois):
    rng = random.Random(seed)
    poi_pos = POI_MAP[true_poi]

    robots = {}
    for rid, base in BASE_ROBOTS.items():
        jx = rng.uniform(-position_jitter, position_jitter)
        jy = rng.uniform(-position_jitter, position_jitter)
        prof = {
            "pos": (base["pos"][0] + jx, base["pos"][1] + jy),
            "battery": base["battery"], "health": base["health"],
            "obstacle_density": base["obstacle_density"],
        }
        d = dist(prof["pos"], poi_pos)
        v = max(effective_speed(prof, nominal_speed), 0.01)
        robots[rid] = Robot(
            rid=rid, pos=prof["pos"], bid=compute_bid(prof, poi_pos),
            travel_time=d / v, distance=d)

    # search-to-detection time (random walk over POIs until true one found)
    pois = list(range(1, num_pois + 1))
    rng.shuffle(pois)
    visited = pois.index(true_poi) + 1
    detection_time = visited * search_visit_time

    return robots, detection_time, crit_duration


# ----------------------------------------------------------------------------
# CBBA-style consensus: gossip min-bid knowledge until all agents agree on the
# winner (single task) or a conflict-free assignment (multi task), or cap hit.
# Returns (rounds_used, messages, converged, per-agent knowledge dict).
# Knowledge is relayed/merged each round (multi-hop gossip), so under drops it
# simply takes MORE rounds to propagate -> more messages, more start latency.
# ----------------------------------------------------------------------------
def simulate_consensus(ids, value_of, comms, rng, max_rounds, tasks=None,
                       assign_fn=None, drop_prob=None, base_time=0.0):
    # known[i][j] = i's best-known info about agent j.
    #   single task: a scalar bid;  multi task: a {task: cost} vector
    known = {i: {i: value_of(i)} for i in ids}
    if drop_prob is None:
        drop_prob = lambda t: comms.drop_rate

    def agree_key(view):
        if tasks is None:                       # single task: agreed winner id
            return min(view, key=lambda j: (view[j], j))
        return tuple(sorted(assign_fn(view, tasks).items()))  # agreed assignment

    messages = rounds = 0
    converged = False
    for _ in range(max_rounds):
        # round r is exchanged at this wall-clock time, so a burst outage can
        # make a whole round's traffic fail and force more rounds.
        t = base_time + rounds * comms.delay_mean
        p = drop_prob(t)
        rounds += 1
        messages += len(ids)                    # broadcast medium: 1 send/agent
        snap = {i: dict(known[i]) for i in ids}
        for s in ids:
            for d in ids:
                if s == d or rng.random() < p:
                    continue
                for j, info in snap[s].items():  # relay everything s knows
                    if j not in known[d]:
                        known[d][j] = info
        # Converged when every agent computes the SAME winner / assignment.
        # (Agreement is CBBA's real stopping rule; we do not additionally
        # require full dissemination, which would unfairly inflate its rounds.)
        keys = [agree_key(known[i]) for i in ids]
        if len(set(keys)) == 1:
            converged = True
            break
    return rounds, messages, converged, known


# ----------------------------------------------------------------------------
# Core trial: ONE comms model, policy-driven behaviour
# ----------------------------------------------------------------------------
def run_trial(seed, comms: Comms, policy: Policy, cfg):
    robots, detection_time, crit_duration = make_world(
        seed, cfg["true_poi"], cfg["position_jitter"], cfg["nominal_speed"],
        cfg["crit_duration"], cfg["search_visit_time"], cfg["num_pois"])

    ids = list(robots.keys())
    n = len(ids)
    true_winner = min(ids, key=lambda r: robots[r].bid)

    # ---- Oracle: explicit unreachable upper bound ----
    if policy.name == "oracle":
        w = robots[true_winner]
        w.believes_winner = True
        w.enters_pre = True
        w.pre_start = detection_time
        w.reaches = detection_time + w.travel_time
        w.enters_crit = True
        w.done_time = w.reaches + crit_duration
        return summarize(robots, detection_time, messages=n, comms=comms,
                         policy=policy, universal_mutex=cfg["universal_mutex"],
                         crit_duration=crit_duration, true_winner=true_winner)

    # Same comms rng seed for every method on a given trial: the shared
    # bid-broadcast phase consumes the random stream identically.
    rng = random.Random(seed * 7919 + 13)
    drop_prob = make_channel(comms, rng)

    # deliver() only SAMPLES the channel (drop + delay). Message accounting is
    # done explicitly per protocol below, so it reflects real bandwidth use and
    # is not an artifact of how many times we happened to sample the channel.
    def deliver(send_time):
        if rng.random() < drop_prob(send_time):
            return None
        return send_time + max(0.0, rng.gauss(comms.delay_mean, comms.delay_jitter))

    # ---- CBBA consensus path: agree first, THEN act ----
    if policy.consensus:
        rounds_used, c_msgs, converged, known = simulate_consensus(
            ids, lambda i: robots[i].bid, comms, rng, policy.consensus_max_rounds,
            drop_prob=drop_prob, base_time=detection_time)
        # consensus completes only after `rounds_used` propagation cycles
        decision_time = detection_time + rounds_used * comms.delay_mean
        for r in robots.values():
            view = known[r.rid]
            r.confidence = len(view) / n
            r.believes_winner = (min(view, key=lambda j: (view[j], j)) == r.rid)
            if r.believes_winner:
                r.pre_start = decision_time          # no extra wait; act on agreement
                r.enters_pre = True
                r.reaches = r.pre_start + r.travel_time
        return summarize(robots, detection_time, messages=c_msgs, comms=comms,
                         policy=policy, universal_mutex=cfg["universal_mutex"],
                         crit_duration=crit_duration, true_winner=true_winner)

    window = policy.window_factor * (comms.delay_mean + comms.delay_jitter)
    rounds = max(1, policy.consensus_rounds)
    if rounds > 1:
        window = rounds * comms.delay_mean
    decision_time = detection_time + window

    # ---- Phase 1: bid broadcast (shared model) ----
    for r in robots.values():
        r.received[r.rid] = r.bid
    for s in ids:
        for d in ids:
            if s == d:
                continue
            # multi-round methods re-broadcast: receiver keeps it if ANY copy
            # arrives by the decision time (more rounds -> more robust, costlier)
            for rnd in range(rounds):
                arr = deliver(detection_time + rnd * comms.delay_mean)
                if arr is not None and arr <= decision_time:
                    robots[d].received[s] = robots[s].bid
                    break

    # ---- Decide believed winner + confidence ----
    for r in robots.values():
        r.confidence = len(r.received) / n
        local_winner = min(r.received, key=lambda x: (r.received[x], x))
        r.believes_winner = (local_winner == r.rid)

    # ---- Commit wait (per believed winner) ----
    for r in robots.values():
        if not r.believes_winner:
            continue
        if policy.abstain_confidence > 0 and r.confidence < policy.abstain_confidence:
            r.blocked_reason = "abstain_low_confidence"
            r.believes_winner = False
            continue
        r.commit_wait = compute_wait(r, policy)
        r.pre_start = decision_time + r.commit_wait

    # ---- Intent layer: pre-PRE backoff on a better intent ----
    intent_senders = 0
    if policy.use_intent:
        senders = [r for r in robots.values() if r.believes_winner]
        intent_senders = len(senders)
        for r in senders:
            for o in senders:
                if o.rid == r.rid or o.bid >= r.bid:
                    continue
                arr = deliver(decision_time)  # o's intent broadcast
                if arr is not None and arr <= r.pre_start:
                    r.blocked_reason = "intent_blocked"
                    r.believes_winner = False
                    break

    # ---- Enter PRE (drive toward object) ----
    for r in robots.values():
        if r.believes_winner:
            r.enters_pre = True
            r.reaches = r.pre_start + r.travel_time

    # ---- Heartbeat layer: en-route backoff if a better active robot is ahead ----
    # A better robot o emits heartbeats every hb_period from o.pre_start. The
    # trailing robot r backs off if ANY heartbeat is delivered before it reaches
    # the object. Slower hb_period => fewer emissions => lower catch probability
    # under loss => the safety/energy benefit of this layer fades.
    hb_period = cfg["hb_period"]
    if policy.use_heartbeat:
        active = sorted([r for r in robots.values() if r.enters_pre],
                        key=lambda x: x.pre_start)
        for r in active:
            for o in active:
                if o.rid == r.rid or o.bid >= r.bid or o.blocked_reason:
                    continue
                if o.pre_start > r.reaches:
                    continue
                t = o.pre_start
                caught = False
                while t < r.reaches:
                    arr = deliver(t)
                    if arr is not None and arr < r.reaches:
                        caught = True
                        break
                    t += hb_period
                if caught:
                    r.enters_pre = False
                    r.reaches = math.inf
                    r.blocked_reason = "heartbeat_blocked"
                    break

    # ---- Explicit message accounting (broadcast medium: one transmission per
    # sender reaches all neighbours and counts as one message) ----
    #   bidding  : every robot broadcasts its bid once PER consensus round
    #              -> rounds * N   (this is where CBBA's cost lives)
    #   intent   : each committed robot broadcasts one intent
    #   heartbeat: each robot in PRE broadcasts periodically over its active
    #              window (travel + crit) at hb_period -> the recurring cost of
    #              our liveness layer, which is NOT free.
    hb_period = cfg["hb_period"]
    messages = rounds * n
    messages += intent_senders
    if policy.use_heartbeat:
        for r in robots.values():
            if r.enters_pre:
                active_window = r.travel_time + crit_duration
                messages += max(1, round(active_window / hb_period))

    return summarize(robots, detection_time, messages=messages, comms=comms,
                     policy=policy, universal_mutex=cfg["universal_mutex"],
                     crit_duration=crit_duration, true_winner=true_winner)


def compute_wait(r: Robot, policy: Policy):
    if policy.commit_wait_mode == "none":
        return 0.0
    if policy.commit_wait_mode in ("fixed", "conservative"):
        return policy.fixed_wait
    if policy.commit_wait_mode == "risk":
        vals = sorted(r.received.values())
        margin = (vals[1] - vals[0]) if len(vals) >= 2 else 0.0
        wait = (policy.min_wait
                + (1.0 - r.confidence) * policy.conf_scale
                + policy.margin_scale / (1.0 + margin))
        return min(policy.max_wait, wait)
    return 0.0


# ----------------------------------------------------------------------------
# Critical section resolution + metrics
# ----------------------------------------------------------------------------
def summarize(robots, detection_time, messages, comms, policy, universal_mutex,
              crit_duration, true_winner):
    arrivals = sorted([r for r in robots.values() if r.enters_pre],
                      key=lambda x: x.reaches)

    # Physical mutex at the grasp zone. The policy may use it; a universal-mutex
    # experiment grants it to everyone (so the gap shifts to wasted energy).
    mutex = policy.use_grasp_zone or universal_mutex

    crit_robots = []
    if mutex:
        for i, r in enumerate(arrivals):
            if i == 0:
                r.enters_crit = True
                r.done_time = r.reaches + crit_duration
                crit_robots.append(r)
            else:
                r.blocked_reason = r.blocked_reason or "grasp_denied"
    else:
        # No mutex: anyone who reaches grabs. Overlapping grabs = collisions.
        for r in arrivals:
            r.enters_crit = True
            r.done_time = r.reaches + crit_duration
            crit_robots.append(r)

    pre_robots = [r for r in robots.values() if r.enters_pre]
    done_robots = [r for r in robots.values() if r.done_time > 0.0]

    # collisions: simultaneous occupancy of the single grasp zone
    crit_violations = 0
    if not mutex and len(crit_robots) > 1:
        crit_robots.sort(key=lambda x: x.reaches)
        for i in range(1, len(crit_robots)):
            if crit_robots[i].reaches < crit_robots[i - 1].done_time:
                crit_violations += 1

    success = 1 if done_robots else 0
    completion = max((r.done_time for r in done_robots), default=0.0)
    # wrong-object safety: a robot that is not the true best one physically grabbed
    wrong_grabs = sum(1 for r in crit_robots if r.rid != true_winner)

    energy = sum(r.distance for r in pre_robots)  # total travel of activated robots
    first_pre = min((r.pre_start for r in pre_robots), default=detection_time)
    planning_latency = max(0.0, first_pre - detection_time)
    local_winners = {min(r.received, key=lambda x: (r.received[x], x))
                     for r in robots.values() if r.received}

    return {
        "mode": policy.name,
        "drop_rate": comms.drop_rate,
        "delay_mean": comms.delay_mean,
        "mission_success": success,
        "mission_completion_time": completion,
        "energy_travel": energy,
        "messages": messages,
        "planning_latency": planning_latency,
        "robots_activated": len(pre_robots),
        "pre_duplicates": max(0, len(pre_robots) - 1),
        "crit_violations": crit_violations,
        "wrong_object_grabs": wrong_grabs,
        "winner_disagreement": 1 if len(local_winners) > 1 else 0,
        "intent_blocks": sum(1 for r in robots.values() if r.blocked_reason == "intent_blocked"),
        "heartbeat_blocks": sum(1 for r in robots.values() if r.blocked_reason == "heartbeat_blocked"),
        "grasp_denials": sum(1 for r in robots.values() if r.blocked_reason == "grasp_denied"),
        "abstentions": sum(1 for r in robots.values() if r.blocked_reason == "abstain_low_confidence"),
    }


# ----------------------------------------------------------------------------
# Aggregation / IO
# ----------------------------------------------------------------------------
def average(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {"mode": rows[0]["mode"]}
    for k in keys:
        out[k] = sum(r[k] for r in rows) / len(rows)
    return out


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_comms(args, drop_rate=None):
    return Comms(
        delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
        drop_rate=args.drop_rate if drop_rate is None else drop_rate,
        burst=args.burst, burst_drop=args.burst_drop)


def trial_cfg(args):
    return {
        "true_poi": args.true_poi,
        "num_pois": 10,
        "position_jitter": args.position_jitter,
        "nominal_speed": 5.0,
        "crit_duration": 2.0,
        "search_visit_time": 1.0,
        "hb_period": args.hb_period,
        "universal_mutex": args.universal_mutex,
    }


# ----------------------------------------------------------------------------
# Subcommand: compare (single regime)
# ----------------------------------------------------------------------------
def cmd_compare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comms = make_comms(args)
    policies = build_policies()
    cfg = trial_cfg(args)

    all_rows, avg_rows = [], []
    for name, pol in policies.items():
        rows = [run_trial(1000 + i, comms, pol, cfg) for i in range(args.trials)]
        all_rows.extend(rows)
        avg_rows.append(average(rows))

    save_csv(all_rows, out / "compare_raw.csv")
    save_csv(avg_rows, out / "compare_avg.csv")

    metrics = ["mission_completion_time", "energy_travel", "messages",
               "planning_latency", "robots_activated", "pre_duplicates",
               "crit_violations", "wrong_object_grabs", "winner_disagreement",
               "mission_success"]
    for m in metrics:
        bar_plot(avg_rows, m, out)

    print_table(avg_rows, metrics)
    print(f"\nSaved -> {out}")


# ----------------------------------------------------------------------------
# Subcommand: sweep (degradation curves)
# ----------------------------------------------------------------------------
def cmd_sweep(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policies = build_policies()
    cfg = trial_cfg(args)

    drops = [round(x, 2) for x in frange(0.0, args.max_drop, args.drop_step)]
    rows = []
    # curve data: metric -> mode -> [values across drop rates]
    curves = {m: {name: [] for name in policies} for m in SWEEP_METRICS}

    for dr in drops:
        comms = make_comms(args, drop_rate=dr)
        for name, pol in policies.items():
            trials = [run_trial(2000 + i, comms, pol, cfg) for i in range(args.trials)]
            avg = average(trials)
            avg["drop_rate"] = dr
            rows.append(avg)
            for m in SWEEP_METRICS:
                curves[m][name].append(avg[m])

    save_csv(rows, out / "sweep_drop.csv")
    for m in SWEEP_METRICS:
        line_plot(drops, curves[m], m, "drop_rate", out)
    print(f"Swept drop_rate over {drops}\nSaved -> {out}")


SWEEP_METRICS = ["crit_violations", "wrong_object_grabs", "pre_duplicates",
                 "energy_travel", "mission_completion_time", "messages",
                 "mission_success", "robots_activated"]


# ----------------------------------------------------------------------------
# Subcommand: hbsweep (heartbeat-rate trade-off)
# ----------------------------------------------------------------------------
def cmd_hbsweep(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comms = make_comms(args)

    # No physical mutex: the heartbeat layer is the only en-route collision
    # guard, so slowing it down should make collisions reappear. auction_intent
    # (no heartbeat) is the floor reference.
    sweep_pols = {
        "ours_layered": build_policies()["ours_layered"],
        "auction_intent(no-hb)": build_policies()["auction_intent"],
    }
    periods = [round(x, 2) for x in frange(args.min_hb, args.max_hb, args.hb_step)]
    metrics = ["crit_violations", "messages", "pre_duplicates",
               "energy_travel", "robots_activated"]
    curves = {m: {name: [] for name in sweep_pols} for m in metrics}
    rows = []

    for hb in periods:
        cfg = dict(trial_cfg(args))
        cfg["hb_period"] = hb
        for name, pol in sweep_pols.items():
            trials = [run_trial(6000 + i, comms, pol, cfg) for i in range(args.trials)]
            avg = average(trials)
            avg["hb_period"] = hb
            avg["mode"] = name
            rows.append(avg)
            for m in metrics:
                curves[m][name].append(avg[m])

    save_csv(rows, out / "hbsweep.csv")
    for m in metrics:
        line_plot(periods, curves[m], "hb_" + m, "hb_period (s)", out)
    print(f"Swept hb_period over {periods} at drop={args.drop_rate}\nSaved -> {out}")


# ----------------------------------------------------------------------------
# Subcommand: ablate
# ----------------------------------------------------------------------------
def cmd_ablate(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comms = make_comms(args)
    cfg = trial_cfg(args)
    variants = build_ablations()

    avg_rows = []
    for name, pol in variants.items():
        rows = [run_trial(3000 + i, comms, pol, cfg) for i in range(args.trials)]
        avg_rows.append(average(rows))

    save_csv(avg_rows, out / "ablation_avg.csv")
    metrics = ["crit_violations", "wrong_object_grabs", "pre_duplicates",
               "energy_travel", "mission_completion_time", "robots_activated"]
    for m in metrics:
        bar_plot(avg_rows, m, out, prefix="ablation_")
    print_table(avg_rows, metrics)
    print(f"\nSaved -> {out}")


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def bar_plot(avg_rows, metric, out, prefix=""):
    modes = [r["mode"] for r in avg_rows]
    vals = [r[metric] for r in avg_rows]
    plt.figure(figsize=(10, 5))
    colors = ["#888" if m == "oracle" else "#1f77b4" for m in modes]
    plt.bar(modes, vals, color=colors)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(metric.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(out / f"{prefix}{metric}.png", dpi=180)
    plt.close()


def line_plot(xs, mode_series, metric, xlabel, out):
    plt.figure(figsize=(9, 5.5))
    for name, ys in mode_series.items():
        style = "--" if name == "oracle" else "-"
        lw = 1.5 if name != "ours_layered" else 3.0
        plt.plot(xs, ys, style, marker="o", linewidth=lw, label=name)
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title(f"{metric.replace('_', ' ').title()} vs {xlabel}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / f"sweep_{metric}.png", dpi=180)
    plt.close()


def print_table(avg_rows, metrics):
    head = ["mode"] + metrics
    print("  ".join(f"{h:>22}" for h in head))
    for r in avg_rows:
        cells = [r["mode"]] + [f"{r[m]:.2f}" for m in metrics]
        print("  ".join(f"{c:>22}" for c in cells))


def frange(start, stop, step):
    x = start
    while x <= stop + 1e-9:
        yield x
        x += step


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--trials", type=int, default=200)
        sp.add_argument("--delay_mean", type=float, default=2.0)
        sp.add_argument("--delay_jitter", type=float, default=1.0)
        sp.add_argument("--drop_rate", type=float, default=0.2)
        sp.add_argument("--true_poi", type=int, default=5)
        sp.add_argument("--position_jitter", type=float, default=6.0)
        sp.add_argument("--hb_period", type=float, default=0.75,
                        help="heartbeat broadcast period (s); lower = more msgs")
        sp.add_argument("--universal_mutex", action="store_true",
                        help="OPTIONAL: grant grasp-zone mutex to ALL methods "
                             "(off by default; mutex is excluded from evaluation)")
        sp.add_argument("--burst", action="store_true",
                        help="use Gilbert-Elliott correlated burst loss instead of i.i.d.")
        sp.add_argument("--burst_drop", type=float, default=0.95)
        sp.add_argument("--output_dir", type=str,
                        default=str(Path.home() / "phase1a_ws/results/unified"))

    c = sub.add_parser("compare"); common(c)
    s = sub.add_parser("sweep"); common(s)
    s.add_argument("--max_drop", type=float, default=0.6)
    s.add_argument("--drop_step", type=float, default=0.1)
    a = sub.add_parser("ablate"); common(a)
    h = sub.add_parser("hbsweep"); common(h)
    h.add_argument("--min_hb", type=float, default=0.25)
    h.add_argument("--max_hb", type=float, default=4.0)
    h.add_argument("--hb_step", type=float, default=0.25)

    args = p.parse_args()
    if args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "sweep":
        cmd_sweep(args)
    elif args.cmd == "ablate":
        cmd_ablate(args)
    elif args.cmd == "hbsweep":
        cmd_hbsweep(args)


if __name__ == "__main__":
    main()
