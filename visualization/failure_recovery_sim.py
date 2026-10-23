"""
Failure-aware task RECOVERY under asynchronous communication.

The open seam (verified against the 2025-26 literature): existing decentralized
MRTA methods -- CBBA / CBBA-ETC / FORMICA / implicit-coordination -- stop at the
ALLOCATION decision and assume a robot, once it wins a task, executes it
correctly. None handle a robot that fails mid-execution under lossy comms, where
the team must DETECT the failure and re-cover the task. The hard part is that a
silent teammate is ambiguous:

    failed   (act -> recover the task)      vs   alive-but-unheard (act -> a
                                                  duplicate robot on one task =
                                                  collision + wasted travel).

Per-node failure detectors (phi-accrual, fixed timeout) watch each teammate's
heartbeat history in ISOLATION, so during a correlated outage (a comms burst
where everyone goes quiet) they conclude "everyone failed" and fire a storm of
false takeovers.

THE IDEA (this file's contribution):
    Silence is evidence whose meaning is set by its CORRELATION across the team.
      * isolated silence  (I hear the others, not X)  => X failed     => recover
      * correlated silence (I hear almost no one)      => I'm partitioned => hold
    Each robot estimates channel-health online (fraction of teammates heard
    recently) and scales its failure-belief by it, then takes over only when the
    expected coverage-hole cost beats the expected duplicate-conflict cost.
    Recovery uses a deterministic successor (next-best bidder, no negotiation),
    so re-coverage never itself creates a new conflict.

Methods compared (identical initial optimal assignment; they differ ONLY in how
they decide a silent teammate has failed):
    no_recovery       allocation-only (CBBA-ETC/FORMICA stand-in): never recover
    fixed_timeout      take over after a fixed silence timeout
    adaptive_pernode   phi-accrual-style: timeout adapts to the teammate's OWN
                       heartbeat history (no cross-teammate correlation)
    ours_correlated    channel-health-scaled failure belief + cost-calibrated
                       takeover (this file's method)
"""

import argparse
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from unified_comparison_sim import (  # noqa: E402
    POI_MAP, Comms, make_channel, dist, effective_speed, compute_bid,
    average, save_csv, frange, line_plot, print_table, simulate_consensus,
)


# 6 robots so K=3 tasks each get a distinct idle successor (spare capacity).
FLEET = {
    "R1": {"pos": (0.0, 0.0),   "battery": 0.90, "health": 0.95, "obstacle_density": 0.20},
    "R2": {"pos": (25.0, 10.0), "battery": 0.80, "health": 0.90, "obstacle_density": 0.40},
    "R3": {"pos": (18.0, 22.0), "battery": 0.85, "health": 0.90, "obstacle_density": 0.30},
    "R4": {"pos": (5.0, 30.0),  "battery": 0.78, "health": 0.85, "obstacle_density": 0.50},
    "R5": {"pos": (28.0, 28.0), "battery": 0.92, "health": 0.82, "obstacle_density": 0.35},
    "R6": {"pos": (45.0, 18.0), "battery": 0.88, "health": 0.88, "obstacle_density": 0.30},
}


def gen_fleet(seed, n):
    """Procedurally generate an N-robot fleet (for the scale sweep, N>6)."""
    rng = random.Random(seed * 131 + 7)
    return {f"R{i+1}": {
        "pos": (rng.uniform(0.0, 50.0), rng.uniform(0.0, 45.0)),
        "battery": rng.uniform(0.70, 0.95), "health": rng.uniform(0.80, 0.97),
        "obstacle_density": rng.uniform(0.20, 0.50),
    } for i in range(n)}


def gen_pois(seed, k):
    """Procedurally generate K task locations (for the scale sweep, K>10)."""
    rng = random.Random(seed * 977 + 41)
    return {i + 1: (rng.uniform(5.0, 50.0), rng.uniform(5.0, 45.0)) for i in range(k)}


def make_world(seed, num_tasks, jitter, nominal_speed, fleet_size=6):
    rng = random.Random(seed)
    # default 6/<=10 path reproduces the original fixed geometry byte-for-byte
    # (so earlier results are unchanged); larger sizes are generated procedurally.
    if fleet_size == 6 and num_tasks <= len(POI_MAP):
        base_fleet, poi_map = FLEET, POI_MAP
        tasks = rng.sample(list(poi_map.keys()), num_tasks)
    else:
        base_fleet, poi_map = gen_fleet(seed, fleet_size), gen_pois(seed, num_tasks)
        tasks = list(poi_map.keys())
    robots = {}
    for rid, base in base_fleet.items():
        prof = {
            "pos": (base["pos"][0] + rng.uniform(-jitter, jitter),
                    base["pos"][1] + rng.uniform(-jitter, jitter)),
            "battery": base["battery"], "health": base["health"],
            "obstacle_density": base["obstacle_density"],
        }
        v = max(effective_speed(prof, nominal_speed), 0.01)
        cost = {k: compute_bid(prof, poi_map[k]) for k in tasks}
        travel = {k: dist(prof["pos"], poi_map[k]) / v for k in tasks}
        robots[rid] = {"pos": prof["pos"], "cost": cost, "travel": travel,
                       "profile": prof, "v": v}
    return robots, tasks


def add_task_costs(robots, k):
    """Compute every robot's cost/travel for a (possibly new) task k in-place."""
    for r, rb in robots.items():
        if k not in rb["cost"]:
            rb["cost"][k] = compute_bid(rb["profile"], POI_MAP[k])
            rb["travel"][k] = dist(rb["pos"], POI_MAP[k]) / rb["v"]


def greedy_assign(robots, tasks, candidates=None):
    cand = candidates if candidates is not None else list(robots)
    pairs = sorted((robots[r]["cost"][k], r, k) for r in cand for k in tasks)
    taken_r, taken_k, res = set(), set(), {}
    for _, r, k in pairs:
        if r not in taken_r and k not in taken_k:
            res[r] = k
            taken_r.add(r)
            taken_k.add(k)
    return res  # robot -> task


# ----------------------------------------------------------------------------
def run_trial(seed, comms, method, cfg):
    robots, tasks = make_world(seed, cfg["num_tasks"], cfg["position_jitter"],
                               cfg["nominal_speed"], cfg.get("fleet_size", 6))
    ids = list(robots)
    n = len(ids)
    crit = cfg["crit_duration"]
    hb = cfg["hb_period"]
    start_t = 0.0
    # mission cost structure: a coverage hole (missed failure) costs c_miss; a
    # false takeover (duplicate robot on a live task) costs c_conflict; recovery
    # latency costs c_latency per unit. ONLY ours_correlated/channel_aware_cost
    # can use this asymmetry -- ours via a calibrated belief threshold, which a
    # fixed timeout cannot express.
    c_miss = cfg.get("c_miss", 1.0)
    c_conflict = cfg.get("c_conflict", 1.0)
    c_latency = cfg.get("c_latency", 0.0)
    tau = c_conflict / (c_miss + c_conflict)  # belief threshold that minimizes E[cost]
    # SPRT log-odds threshold: declare 'dead' when accumulated LLR >= Lambda.
    # Wald/Bayes-calibrated: Lambda = log( (c_conflict/c_miss) * (pi_alive/pi_dead) ).
    Lambda = cfg.get("lambda_sprt",
                     math.log((c_conflict / c_miss) *
                              ((1.0 - cfg.get("p_prior", 0.1)) / cfg.get("p_prior", 0.1))))

    rng = random.Random(seed * 7919 + 13)
    drop_prob = make_channel(comms, rng)

    # ---- identical initial assignment for every method (isolate RECOVERY) ----
    primary = greedy_assign(robots, tasks)                 # robot -> task
    task_primary = {k: r for r, k in primary.items()}
    idle = [r for r in ids if r not in primary]
    # deterministic, distinct successor per task = best idle robot, greedily
    succ = {}
    used = set()
    for k in sorted(tasks, key=lambda k: min(robots[r]["cost"][k] for r in idle)):
        cands = sorted([r for r in idle if r not in used], key=lambda r: robots[r]["cost"][k])
        succ[k] = cands[0] if cands else None
        if succ[k]:
            used.add(succ[k])

    # ---- draw failures: a primary may fail mid-task (between reach and done) ----
    horizon = start_t
    pstate = {}
    for k, p in task_primary.items():
        treach = start_t + robots[p]["travel"][k]
        tdone = treach + crit
        failed = rng.random() < cfg["p_fail"]
        # fail somewhere during travel or execution
        tfail = rng.uniform(start_t + 0.1, tdone) if failed else math.inf
        pstate[k] = {"p": p, "treach": treach, "tdone": tdone,
                     "failed": failed, "tfail": tfail}
        horizon = max(horizon, tdone)
    horizon += 40.0

    # ---- heartbeat reception: a robot emits every hb while ALIVE (not failed) ----
    # alive_until[j]: a primary that fails stops at tfail; everyone else alive whole run
    alive_until = {j: math.inf for j in ids}
    for k, st in pstate.items():
        if st["failed"]:
            alive_until[st["p"]] = st["tfail"]
    hb_ticks = [start_t + i * hb for i in range(int((horizon - start_t) / hb) + 1)]
    # delivered[s][j] = sorted emission times of j that s actually received
    delivered = {s: {j: [] for j in ids if j != s} for s in ids}
    for j in ids:
        for te in hb_ticks:
            if te >= alive_until[j]:
                break
            for s in ids:
                if s == j:
                    continue
                if rng.random() >= drop_prob(te):
                    delivered[s][j].append(te)

    def last_hb(s, j, t):
        lst = delivered[s][j]
        # latest emission <= t
        prev = start_t
        for te in lst:
            if te <= t:
                prev = te
            else:
                break
        return prev

    def channel_health(s, exclude, t, window):
        others = [j for j in ids if j != s and j != exclude]
        if not others:
            return 1.0
        heard = sum(1 for j in others if last_hb(s, j, t) >= t - window)
        return heard / len(others)

    def est_drop_rate(s, exclude, t, window):
        # per-slot drop probability the channel is currently inflicting on s,
        # estimated from the WHOLE team's reception pattern (the correlated-
        # silence coupling): expected heartbeat slots from others vs received.
        others = [j for j in ids if j != s and j != exclude]
        if not others:
            return 0.01
        slots = max(1, int(round(window / hb)))
        recv = sum(1 for j in others for te in delivered[s][j]
                   if t - window <= te <= t)
        expected = len(others) * slots
        return 1.0 - recv / expected if expected else 0.5

    def adaptive_timeout(s, j, t, kstd=3.0):
        # phi-accrual-ish: mean + kstd*std of received inter-arrival gaps so far
        times = [te for te in delivered[s][j] if te <= t]
        if len(times) < 2:
            return kstd * hb
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        mean = sum(gaps) / len(gaps)
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        return mean + kstd * math.sqrt(var) + hb

    def lrt_belief(s, p, t):
        # sequential likelihood ratio: P(silence | p failed)=1 ;
        # P(silence | p alive)=q^k (k missed slots, q=team-estimated drop rate).
        # belief = P(failed | silence). channel_aware_timeout cannot express q^k.
        q = min(max(est_drop_rate(s, p, t, window), 1e-3), 0.999)
        k_missed = min((t - last_hb(s, p, t)) / hb, 40.0)
        prior_f = cfg.get("p_prior", 0.1)
        odds = (prior_f / (1.0 - prior_f)) * (q ** (-k_missed))  # failed : alive
        return odds / (1.0 + odds)

    # ---- per-task: simulate the successor's takeover decision over time ----
    dt = 0.25
    window = 3.0 * hb
    T_fixed = 3.0 * hb

    covered, false_takeovers, correct_recoveries, missed = 0, 0, 0, 0
    recovery_latencies, detection_latencies, completion_times = [], [], []
    wasted_travel = 0.0   # successor travel spent driving to LIVE tasks (false takeovers)
    hb_msgs = sum(len([te for te in hb_ticks if te < alive_until[j]]) * (n - 1) for j in ids)

    # optional sensitivity knob to trace each detector's ROC (None = defaults)
    sens = cfg.get("sens", None)

    for k, st in pstate.items():
        p, s = st["p"], succ[k]
        took_over_at = None
        if s is not None and method != "no_recovery":
            t = start_t
            L_acc = 0.0   # channel-conditioned sequential evidence accumulator
            while t <= horizon:
                # task observably done by primary -> stop monitoring (no takeover)
                if (not st["failed"]) and t >= st["tdone"]:
                    break
                silence = t - last_hb(s, p, t)
                fire = False
                if method == "fixed_timeout":
                    fire = silence > (sens if sens is not None else 3.0) * hb
                elif method == "adaptive_pernode":
                    fire = silence > adaptive_timeout(
                        s, p, t, sens if sens is not None else 3.0)
                elif method == "channel_aware_timeout":
                    # uses the SAME channel-health signal as ours, but applies it
                    # as a timeout inflation rather than a hold. Isolates whether
                    # the gap is from channel-awareness generally vs ours' rule.
                    ch = channel_health(s, exclude=p, t=t, window=window)
                    base = (sens if sens is not None else 3.0) * hb
                    fire = silence > base * (1.0 + 4.0 * (1.0 - ch))
                elif method == "channel_aware_cost":
                    # strongest fair baseline: channel-aware timeout that ALSO
                    # tries to use the cost asymmetry, by scaling the timeout with
                    # tau. But silence-time is not a probability, so the cost->
                    # timeout map is ad hoc; this isolates whether ours' edge is
                    # the CALIBRATED belief vs merely "be cost-aware somehow".
                    ch = channel_health(s, exclude=p, t=t, window=window)
                    T_eff = T_fixed * (1.0 + 4.0 * (1.0 - ch))
                    T_eff *= max(0.2, min(2.0, tau / 0.5))  # misses costly -> fire sooner
                    fire = silence > T_eff
                elif method == "ours_correlated":
                    s_raw = max(0.0, min(1.0, (silence - 2.0 * hb) / (4.0 * hb)))
                    ch = channel_health(s, exclude=p, t=t, window=window)
                    failure_belief = s_raw * ch
                    # cost-calibrated: take over when belief that X failed exceeds
                    # the cost-optimal threshold tau = c_conflict/(c_miss+c_conflict).
                    # balanced costs -> tau=0.5 (== old behavior); misses costly ->
                    # tau low (fire eagerly); conflicts costly -> tau high (hold).
                    fire = failure_belief >= (sens if sens is not None else tau)
                elif method == "ours_lrt":
                    fire = lrt_belief(s, p, t) >= (sens if sens is not None else tau)
                elif method in ("ours_sprt", "ours_sprt_lin"):
                    # CHANNEL-CONDITIONED SEQUENTIAL TEST.
                    # Each slot we miss p contributes evidence for "p dead".
                    # Per-slot LLR of a miss = log[ P(miss|dead)/P(miss|alive) ]
                    #                        = log[ 1 / (1 - delta) ],
                    # where delta = instantaneous delivery prob, estimated model-free
                    # by the team-heard fraction ch (correlation-aware: in a burst
                    # ch->0 so a miss carries ~0 evidence and the outage is explained
                    # away; on a good channel ch->1 so each miss is strong evidence).
                    # We ACCUMULATE across the silence run, so evidence piles up only
                    # during good-channel slots -- unbounded, hence no structural miss.
                    ch = channel_health(s, exclude=p, t=t, window=window)
                    if silence < hb:
                        L_acc = 0.0          # heard p -> reset evidence
                    else:
                        if method == "ours_sprt":
                            incr = -math.log(1.0 - min(ch, 0.9))   # exact LLR (capped)
                        else:
                            incr = ch                              # robust linearization
                        L_acc += incr * (dt / hb)
                    fire = L_acc >= (sens if sens is not None else Lambda)
                if fire:
                    took_over_at = t
                    break
                t += dt

        # ---- outcomes ----
        if not st["failed"]:
            covered += 1
            completion_times.append(st["tdone"])
            if took_over_at is not None:
                # primary was ALIVE -> takeover is a false alarm = duplicate/conflict.
                # The successor drives all the way to a task that did not need it:
                # that travel is pure wasted energy, attributable to the detector.
                false_takeovers += 1
                wasted_travel += robots[s]["travel"][k]
        else:
            if took_over_at is not None:
                covered += 1
                correct_recoveries += 1
                done = took_over_at + robots[s]["travel"][k] + crit
                completion_times.append(done)
                recovery_latencies.append(done - st["tfail"])
                # clamp: a takeover fired before tfail is premature (drop-noise that
                # happened to precede a real failure); count 0, not negative latency.
                detection_latencies.append(max(0.0, took_over_at - st["tfail"]))
            else:
                missed += 1  # coverage hole

    K = len(tasks)
    n_failed = sum(1 for st in pstate.values() if st["failed"])
    total_recovery_latency = sum(recovery_latencies)
    total_cost = (false_takeovers * c_conflict + missed * c_miss
                  + total_recovery_latency * c_latency)
    return {
        "mode": method,
        "drop_rate": comms.drop_rate,
        "burst": 1 if comms.burst else 0,
        "p_fail": cfg["p_fail"],
        "coverage_frac": covered / K,
        "false_takeovers": false_takeovers,
        "correct_recoveries": correct_recoveries,
        "missed_failures": missed,
        "n_failed": n_failed,
        "recovery_rate": (correct_recoveries / n_failed) if n_failed else 1.0,
        "mean_recovery_latency": (sum(recovery_latencies) / len(recovery_latencies))
                                 if recovery_latencies else 0.0,
        "mean_detection_latency": (sum(detection_latencies) / len(detection_latencies))
                                  if detection_latencies else 0.0,
        "makespan": max(completion_times, default=0.0),
        "wasted_travel": wasted_travel,
        "hb_messages": hb_msgs,
        "total_cost": total_cost,
    }


# ============================================================================
# SHARED-POOL variant of run_trial. Same fixed allocator, same detectors -- the
# ONLY change is that recovery agents come from ONE shared idle pool instead of
# a distinct pre-assigned successor per task. Now a FALSE takeover physically
# STEALS a robot: it is busy driving to a live task during [t, t+travel+crit],
# so a genuine failure elsewhere that fires in that window finds no free robot
# and goes UNCOVERED. This is the regime where detector quality (false-takeover
# rate) finally propagates into success rate and mission makespan.
#
# Detection is run from a single always-alive coordinator (an idle pool robot);
# only primaries fail, so the coordinator's view is well defined for all methods.
# Fire *times* are computed per task exactly as in run_trial (independent of the
# pool); then fires are resolved in chronological order against the shared pool.
# ============================================================================
def run_trial_pool(seed, comms, method, cfg):
    robots, tasks = make_world(seed, cfg["num_tasks"], cfg["position_jitter"],
                               cfg["nominal_speed"], cfg.get("fleet_size", 6))
    ids = list(robots)
    crit = cfg["crit_duration"]
    hb = cfg["hb_period"]
    start_t = 0.0
    c_miss = cfg.get("c_miss", 1.0)
    c_conflict = cfg.get("c_conflict", 1.0)
    tau = c_conflict / (c_miss + c_conflict)
    Lambda = cfg.get("lambda_sprt",
                     math.log((c_conflict / c_miss) *
                              ((1.0 - cfg.get("p_prior", 0.1)) / cfg.get("p_prior", 0.1))))

    rng = random.Random(seed * 7919 + 13)
    drop_prob = make_channel(comms, rng)

    primary = greedy_assign(robots, tasks)
    task_primary = {k: r for r, k in primary.items()}
    pool = [r for r in ids if r not in primary]   # ONE shared successor pool
    if not pool:                                   # degenerate: no spares at all
        pool = []

    # ---- draw failures (identical structure to run_trial) ----
    horizon = start_t
    pstate = {}
    for k, p in task_primary.items():
        treach = start_t + robots[p]["travel"][k]
        tdone = treach + crit
        failed = rng.random() < cfg["p_fail"]
        tfail = rng.uniform(start_t + 0.1, tdone) if failed else math.inf
        pstate[k] = {"p": p, "treach": treach, "tdone": tdone,
                     "failed": failed, "tfail": tfail}
        horizon = max(horizon, tdone)
    horizon += 40.0

    # ---- heartbeats ----
    alive_until = {j: math.inf for j in ids}
    for k, st in pstate.items():
        if st["failed"]:
            alive_until[st["p"]] = st["tfail"]
    hb_ticks = [start_t + i * hb for i in range(int((horizon - start_t) / hb) + 1)]
    delivered = {s: {j: [] for j in ids if j != s} for s in ids}
    for j in ids:
        for te in hb_ticks:
            if te >= alive_until[j]:
                break
            for s in ids:
                if s == j:
                    continue
                if rng.random() >= drop_prob(te):
                    delivered[s][j].append(te)

    # coordinator = a fixed, always-alive idle robot (falls back to any id)
    coord = pool[0] if pool else ids[0]

    def last_hb(s, j, t):
        prev = start_t
        for te in delivered[s][j]:
            if te <= t:
                prev = te
            else:
                break
        return prev

    def channel_health(s, exclude, t, window):
        others = [j for j in ids if j != s and j != exclude]
        if not others:
            return 1.0
        heard = sum(1 for j in others if last_hb(s, j, t) >= t - window)
        return heard / len(others)

    def adaptive_timeout(s, j, t, kstd=3.0):
        times = [te for te in delivered[s][j] if te <= t]
        if len(times) < 2:
            return kstd * hb
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        mean = sum(gaps) / len(gaps)
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        return mean + kstd * math.sqrt(var) + hb

    dt = 0.25
    window = 3.0 * hb

    def fire_time(k):
        """First time the coordinator's detector declares task k's primary dead
        (or None). Identical decision rule to run_trial; pool-independent."""
        if method == "no_recovery":
            return None
        p = pstate[k]["p"]
        s = coord
        t = start_t
        L_acc = 0.0
        st = pstate[k]
        while t <= horizon:
            if (not st["failed"]) and t >= st["tdone"]:
                return None
            silence = t - last_hb(s, p, t)
            fire = False
            if method == "fixed_timeout":
                fire = silence > 3.0 * hb
            elif method == "adaptive_pernode":
                fire = silence > adaptive_timeout(s, p, t, 3.0)
            elif method == "channel_aware_timeout":
                ch = channel_health(s, exclude=p, t=t, window=window)
                fire = silence > (3.0 * hb) * (1.0 + 4.0 * (1.0 - ch))
            elif method == "ours_correlated":
                s_raw = max(0.0, min(1.0, (silence - 2.0 * hb) / (4.0 * hb)))
                ch = channel_health(s, exclude=p, t=t, window=window)
                fire = s_raw * ch >= tau
            elif method in ("ours_sprt", "ours_sprt_lin"):
                ch = channel_health(s, exclude=p, t=t, window=window)
                if silence < hb:
                    L_acc = 0.0
                else:
                    incr = (-math.log(1.0 - min(ch, 0.9))
                            if method == "ours_sprt" else ch)
                    L_acc += incr * (dt / hb)
                fire = L_acc >= Lambda
            if fire:
                return t
            t += dt
        return None

    # ---- resolve fires in time order against the ONE shared pool ----
    events = sorted((ft, k) for k in tasks if (ft := fire_time(k)) is not None)
    busy_until = {r: -math.inf for r in pool}
    covered = 0
    false_takeovers = correct_recoveries = missed = 0
    stolen_misses = 0     # genuine failures uncovered specifically for lack of a free robot
    wasted_travel = 0.0
    completion_times = []
    resolved = set()

    for tf, k in events:
        st = pstate[k]
        free = [r for r in pool if busy_until[r] <= tf]
        if not free:
            # no spare available RIGHT NOW (pool drained, e.g. by a false takeover)
            if st["failed"]:
                missed += 1
                stolen_misses += 1
            # if alive: harmless false alarm that couldn't even grab a robot
            resolved.add(k)
            continue
        r = min(free, key=lambda r: robots[r]["travel"][k])
        busy_until[r] = tf + robots[r]["travel"][k] + crit
        resolved.add(k)
        if st["failed"]:
            correct_recoveries += 1
            covered += 1
            completion_times.append(busy_until[r])
        else:
            false_takeovers += 1
            wasted_travel += robots[r]["travel"][k]

    # ---- tally tasks not resolved by a fire ----
    for k, st in pstate.items():
        if not st["failed"]:
            covered += 1
            completion_times.append(st["tdone"])
        elif k not in resolved:
            missed += 1     # failed, detector never fired -> coverage hole

    K = len(tasks)
    n_failed = sum(1 for st in pstate.values() if st["failed"])
    return {
        "mode": method,
        "drop_rate": comms.drop_rate,
        "burst": 1 if comms.burst else 0,
        "p_fail": cfg["p_fail"],
        "coverage_frac": covered / K,
        "false_takeovers": false_takeovers,
        "correct_recoveries": correct_recoveries,
        "missed_failures": missed,
        "stolen_misses": stolen_misses,
        "n_failed": n_failed,
        "recovery_rate": (correct_recoveries / n_failed) if n_failed else 1.0,
        "makespan": max(completion_times, default=0.0),
        "wasted_travel": wasted_travel,
        "pool_size": len(pool),
    }


# ============================================================================
# ROUTE mode: the realistic "no spares" model. POIs >> robots, so every robot
# starts loaded with an ORDERED ROUTE (built by cheapest-insertion). One robot
# is silenced mid-mission. Survivors RE-AUCTION its remaining POIs by MARGINAL
# INSERTION COST (same backend for every detector) and detour to absorb them.
#
# The honest cost model, fixed BEFORE looking at results:
#   * genuine death, detector fires  -> orphans re-auctioned & covered; each
#       absorbing survivor's finish grows by its detour + service time.
#   * genuine death, never fires     -> orphans NEVER visited = coverage holes
#       (this is where a too-calm detector, e.g. ours during a long burst, can
#        actually MISS a real failure -- we do NOT hide that).
#   * ALIVE but silenced (burst), detector fires = FALSE takeover -> survivors
#       detour toward phantom orphans until the heartbeat resumes at t_resume
#       (read from the actual delivery record), then CANCEL. The drive taken in
#       [tf, t_resume] plus the return is sunk: it delays the survivor's real
#       route (makespan) and wastes energy, but coverage is unharmed (the alive
#       robot serviced its own POIs and the survivor resumes its own route).
#   * alive, never fires             -> clean, no perturbation.
# Only the DETECTOR varies; routing + re-auction are identical for all methods.
# ============================================================================
def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_length(start, route, poi_pos):
    pts = [start] + [poi_pos[k] for k in route]
    return sum(_euclid(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _best_insertion(start, route, k, poi_pos):
    """Cheapest position to insert POI k; returns (added_distance, new_route)."""
    base = _route_length(start, route, poi_pos)
    best_add, best_route = math.inf, None
    for pos in range(len(route) + 1):
        cand = route[:pos] + [k] + route[pos:]
        add = _route_length(start, cand, poi_pos) - base
        if add < best_add:
            best_add, best_route = add, cand
    return best_add, best_route


def run_trial_route(seed, comms, method, cfg):
    rng = random.Random(seed)
    N = cfg.get("fleet_size", 6)
    M = cfg["num_pois"]
    speed = cfg["nominal_speed"]
    crit = cfg["crit_duration"]
    hb = cfg["hb_period"]
    field = cfg.get("field", 100.0)
    c_miss = cfg.get("c_miss", 1.0)
    c_conflict = cfg.get("c_conflict", 1.0)
    tau = c_conflict / (c_miss + c_conflict)
    Lambda = cfg.get("lambda_sprt",
                     math.log((c_conflict / c_miss) *
                              ((1.0 - cfg.get("p_prior", 0.1)) / cfg.get("p_prior", 0.1))))

    ids = list(range(N))
    robot_pos = {i: (rng.uniform(0, field), rng.uniform(0, field)) for i in ids}
    poi_pos = {k: (rng.uniform(0, field), rng.uniform(0, field)) for k in range(M)}

    # ---- cheapest-insertion routing (identical for every method) ----
    routes = {i: [] for i in ids}
    for k in range(M):
        best = None
        for i in ids:
            add, nr = _best_insertion(robot_pos[i], routes[i], k, poi_pos)
            # tie-break toward the emptier robot for balance
            score = add + 0.01 * len(routes[i])
            if best is None or score < best[0]:
                best = (score, i, nr)
        _, i, nr = best
        routes[i] = nr

    def finish_time(i, route):
        return _route_length(robot_pos[i], route, poi_pos) / speed + crit * len(route)

    base_finish = {i: finish_time(i, routes[i]) for i in ids}

    # ---- victim: the most heavily loaded robot (orphaning it matters most) ----
    victim = max(ids, key=lambda i: len(routes[i]))
    survivors = [i for i in ids if i != victim]
    coord = survivors[0]

    # victim per-POI arrival schedule
    v_arr = {}
    t_acc = 0.0
    prev = robot_pos[victim]
    for k in routes[victim]:
        t_acc += _euclid(prev, poi_pos[k]) / speed
        v_arr[k] = t_acc            # arrival at POI k
        t_acc += crit
        prev = poi_pos[k]
    v_finish = base_finish[victim]

    dead = rng.random() < cfg["p_fail"]
    # if it dies, it does so partway through its route
    tfail = rng.uniform(0.15 * v_finish, 0.85 * v_finish) if dead else math.inf
    alive_until = tfail if dead else math.inf

    # ---- heartbeats: coord's reception of every teammate (burst-correlated) ----
    rng2 = random.Random(seed * 7919 + 13)
    drop_prob = make_channel(comms, rng2)
    horizon = max(base_finish.values()) + 60.0
    hb_ticks = [i * hb for i in range(int(horizon / hb) + 1)]
    delivered = {j: [] for j in ids if j != coord}
    for j in ids:
        if j == coord:
            continue
        for te in hb_ticks:
            if te >= (alive_until if j == victim else math.inf):
                break
            if rng2.random() >= drop_prob(te):
                delivered[j].append(te)

    window = 3.0 * hb
    dt = 0.25

    def last_hb(j, t):
        prev = 0.0
        for te in delivered[j]:
            if te <= t:
                prev = te
            else:
                break
        return prev

    def channel_health(exclude, t):
        others = [j for j in ids if j != coord and j != exclude]
        if not others:
            return 1.0
        heard = sum(1 for j in others if last_hb(j, t) >= t - window)
        return heard / len(others)

    def adaptive_timeout(j, t, kstd=3.0):
        times = [te for te in delivered[j] if te <= t]
        if len(times) < 2:
            return kstd * hb
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        mean = sum(gaps) / len(gaps)
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        return mean + kstd * math.sqrt(var) + hb

    # ---- when does the detector declare the victim dead? ----
    def fire_time():
        if method == "no_recovery":
            return None
        t, L_acc = 0.0, 0.0
        while t <= horizon:
            silence = t - last_hb(victim, t)
            fire = False
            if method == "fixed_timeout":
                fire = silence > 3.0 * hb
            elif method == "adaptive_pernode":
                fire = silence > adaptive_timeout(victim, t, 3.0)
            elif method == "channel_aware_timeout":
                ch = channel_health(victim, t)
                fire = silence > (3.0 * hb) * (1.0 + 4.0 * (1.0 - ch))
            elif method == "ours_correlated":
                s_raw = max(0.0, min(1.0, (silence - 2.0 * hb) / (4.0 * hb)))
                fire = s_raw * channel_health(victim, t) >= tau
            elif method in ("ours_sprt", "ours_sprt_lin"):
                ch = channel_health(victim, t)
                if silence < hb:
                    L_acc = 0.0
                else:
                    incr = (-math.log(1.0 - min(ch, 0.9))
                            if method == "ours_sprt" else ch)
                    L_acc += incr * (dt / hb)
                fire = L_acc >= Lambda
            if fire:
                return t
            t += dt
        return None

    tf = fire_time()

    # ---- resolve ----
    finish = dict(base_finish)
    false_takeovers = correct_recoveries = missed = 0
    duplicate_visits = 0
    team_detour = 0.0
    recovery_latencies = []

    # POIs the victim actually completes on its own
    if dead:
        victim_done = [k for k in routes[victim] if v_arr[k] + crit <= tfail]
        orphans = [k for k in routes[victim] if k not in victim_done]
    else:
        victim_done = list(routes[victim])   # alive -> covers its whole route
        orphans = [k for k in routes[victim] if v_arr[k] >= (tf if tf is not None else math.inf)]

    covered = (M - len(routes[victim])) + len(victim_done)
    finish[victim] = tfail if dead else base_finish[victim]

    if tf is not None and orphans:
        # re-auction orphans among survivors by marginal insertion cost (shared backend)
        sr_routes = {i: list(routes[i]) for i in survivors}
        detour_dist = {i: 0.0 for i in survivors}
        absorb_count = {i: 0 for i in survivors}
        for k in orphans:
            best = None
            for i in survivors:
                add, nr = _best_insertion(robot_pos[i], sr_routes[i], k, poi_pos)
                if best is None or add < best[0]:
                    best = (add, i, nr)
            add, i, nr = best
            sr_routes[i] = nr
            detour_dist[i] += add
            absorb_count[i] += 1

        if dead:
            # genuine recovery: survivors truly service the orphans
            correct_recoveries = 1
            covered += len(orphans)
            for i in survivors:
                finish[i] = finish_time(i, sr_routes[i])
            recovery_latencies.append(max(0.0, tf - tfail))
            team_detour = sum(detour_dist.values())
        else:
            # FALSE takeover: alive victim silenced by a burst. Survivors chase
            # phantoms until the heartbeat resumes, then cancel. Sunk = the drive
            # taken in [tf, t_resume] + the return, capped by the detour length.
            false_takeovers = 1
            t_resume = next((te for te in delivered[victim] if te > tf), horizon)
            chase = speed * (t_resume - tf)
            for i in survivors:
                sunk = min(detour_dist[i], chase)      # distance driven toward phantom
                waste_time = 2.0 * sunk / speed        # out-and-back, no service done
                finish[i] = base_finish[i] + waste_time
                team_detour += 2.0 * sunk
                if sunk >= detour_dist[i] - 1e-9 and absorb_count[i] > 0:
                    duplicate_visits += absorb_count[i]  # reached before cancel
    elif dead and tf is None:
        # never detected a real death -> its remaining POIs are coverage holes
        missed = len(orphans)

    K = M
    return {
        "mode": method,
        "drop_rate": comms.drop_rate,
        "burst": 1 if comms.burst else 0,
        "p_fail": cfg["p_fail"],
        "coverage_frac": covered / K,
        "makespan": max(finish.values()),
        "false_takeovers": false_takeovers,
        "correct_recoveries": correct_recoveries,
        "missed_pois": missed,
        "duplicate_visits": duplicate_visits,
        "team_detour": team_detour,
        "victim_dead": 1 if dead else 0,
        "recovery_rate": (correct_recoveries / 1) if dead else 1.0,
        "mean_recovery_latency": (sum(recovery_latencies) / len(recovery_latencies))
                                 if recovery_latencies else 0.0,
    }


# ============================================================================
# UNIFIED dynamic engine: failures (negative-information events) and new POIs
# (positive-information events) both become "a task entering the pool", handled
# by the SAME (re)allocation routine. Detection is fixed to correlated-silence
# (the contribution); we vary the RECOVERY MODE to test whether moulding
# recovery into incremental task-allocation helps:
#   succession   deterministic next-best, comms-free, but acts on a possibly
#                stale view of who is still available -> can DOUBLE-BOOK an
#                already-consumed robot under loss.
#   incremental  re-allocate the orphan via a mini-consensus among robots that
#                actively bid (only truly-available ones do) -> never
#                double-books, picks the best CURRENT robot, but needs comms and
#                stalls under burst.
#   adaptive     incremental when channel-health is good, succession when not.
# ============================================================================
RECOVERY_MODES = ["succession", "incremental", "adaptive"]


def run_dynamic_trial(seed, comms, recovery_mode, cfg):
    robots, tasks = make_world(seed, cfg["num_tasks"], cfg["position_jitter"],
                               cfg["nominal_speed"])
    ids = list(robots); n = len(ids); crit = cfg["crit_duration"]; hb = cfg["hb_period"]
    rng = random.Random(seed * 7919 + 13)
    drop_prob = make_channel(comms, rng)

    primary = greedy_assign(robots, tasks)
    task_primary = {k: r for r, k in primary.items()}
    idle0 = set(r for r in ids if r not in primary)

    # ---- primary failures ----
    horizon = 0.0
    pstate = {}
    for k, p in task_primary.items():
        tdone = robots[p]["travel"][k] + crit
        failed = rng.random() < cfg["p_fail"]
        tfail = rng.uniform(0.1, tdone) if failed else math.inf
        pstate[k] = {"p": p, "tdone": tdone, "failed": failed, "tfail": tfail}
        horizon = max(horizon, tdone)
    horizon += 30.0

    # ---- new POIs arriving mid-mission (positive-information events) ----
    new_events = []
    pool = [x for x in POI_MAP if x not in tasks]
    for kk in rng.sample(pool, min(cfg["n_new"], len(pool))):
        add_task_costs(robots, kk)
        new_events.append((rng.uniform(1.0, max(2.0, horizon - 25.0)), "new", kk))

    # ---- heartbeats (for detection + availability belief) ----
    alive_until = {j: math.inf for j in ids}
    for k, st in pstate.items():
        if st["failed"]:
            alive_until[st["p"]] = st["tfail"]
    ticks = [i * hb for i in range(int(horizon / hb) + 1)]
    delivered = {s: {j: [] for j in ids if j != s} for s in ids}
    for j in ids:
        for te in ticks:
            if te >= alive_until[j]:
                break
            for s in ids:
                if s != j and rng.random() >= drop_prob(te):
                    delivered[s][j].append(te)

    def last_hb(s, j, t):
        prev = 0.0
        for te in delivered[s][j]:
            if te <= t:
                prev = te
            else:
                break
        return prev

    window = 3.0 * hb

    def channel_health(s, exclude, t):
        others = [j for j in ids if j != s and j != exclude]
        if not others:
            return 1.0
        return sum(1 for j in others if last_hb(s, j, t) >= t - window) / len(others)

    # ---- detect failures via correlated silence -> detection time per task ----
    succ = {}
    used = set()
    for k in sorted(tasks, key=lambda k: min((robots[r]["cost"][k] for r in idle0), default=0)):
        cands = sorted([r for r in idle0 if r not in used], key=lambda r: robots[r]["cost"][k])
        succ[k] = cands[0] if cands else None
        if succ[k]:
            used.add(succ[k])

    recover_events = []
    dt = 0.25
    for k, st in pstate.items():
        if not st["failed"] or succ[k] is None:
            continue
        s = succ[k]; t = 0.0; t_detect = None
        while t <= horizon:
            silence = t - last_hb(s, st["p"], t)
            s_raw = max(0.0, min(1.0, (silence - 2.0 * hb) / (4.0 * hb)))
            if s_raw * channel_health(s, st["p"], t) >= 0.5:
                t_detect = t; break
            t += dt
        if t_detect is not None:
            recover_events.append((t_detect, "recover", k))

    # ---- event-ordered (re)allocation over a shared, contended pool ----
    coordinator = ids[0]
    consumed = {}          # robot -> time it was committed to a recovery/new task
    covered_extra = 0      # orphans + new tasks successfully (re)allocated
    total_extra = len(recover_events) + len(new_events)
    rec_costs, rec_latencies, conflicts = [], [], 0
    completion = [st["tdone"] for st in pstate.values() if not st["failed"]]

    def truly_available(t):
        out = []
        for r in ids:
            if r in consumed:
                continue
            if r in idle0:
                out.append(r); continue
            # a primary is available only after it finished its own task (and didn't fail)
            kp = task_primary.get(r)
            if kp is not None and not pstate[kp]["failed"] and pstate[kp]["tdone"] <= t:
                out.append(r)
        return out

    use_incremental = (recovery_mode == "incremental")
    for t_evt, kind, k in sorted(new_events + recover_events):
        add_task_costs(robots, k)
        avail = truly_available(t_evt)
        if not avail:
            continue  # no spare capacity -> uncovered (same limit for all modes)
        mode = recovery_mode
        if mode == "adaptive":
            mode = "incremental" if channel_health(coordinator, None, t_evt) >= 0.5 else "succession"

        chosen, latency, conflict = None, 0.0, 0
        if mode == "incremental":
            rounds, _m, conv, _k = simulate_consensus(
                avail, (lambda kk: (lambda r: robots[r]["cost"][kk]))(k),
                comms, rng, max_rounds=8, drop_prob=drop_prob, base_time=t_evt)
            latency = rounds * comms.delay_mean
            chosen = min(avail, key=lambda r: robots[r]["cost"][k])
            if not conv:
                conflict = 1  # disagreement -> a duplicate robot also acts
        else:  # succession: deterministic best-by-cost among BELIEVED-free robots
            # belief: a consumed robot is known-consumed only if its commit was heard
            believed_free = []
            for r in ids:
                if r in idle0 or (task_primary.get(r) and not pstate[task_primary[r]]["failed"]
                                  and pstate[task_primary[r]]["tdone"] <= t_evt):
                    if r in consumed:
                        # known-consumed iff coordinator heard a recent heartbeat-gap update;
                        # approximate: known with prob (1 - drop at consume time)
                        if rng.random() < (1.0 - drop_prob(consumed[r])):
                            continue  # known consumed -> skip
                    believed_free.append(r)
            if not believed_free:
                continue
            chosen = min(believed_free, key=lambda r: robots[r]["cost"][k])
            if chosen in consumed:
                conflict = 1          # DOUBLE-BOOK: picked an already-consumed robot
                conflicts += conflict
                continue              # this (re)allocation fails -> task uncovered

        conflicts += conflict
        consumed[chosen] = t_evt
        done = t_evt + latency + robots[chosen]["travel"][k] + crit
        completion.append(done)
        rec_costs.append(robots[chosen]["travel"][k])
        rec_latencies.append(done - t_evt)
        covered_extra += 1

    base_cov = sum(1 for st in pstate.values() if not st["failed"])
    total_tasks = len(tasks) + len(new_events)
    return {
        "mode": recovery_mode,
        "drop_rate": comms.drop_rate, "burst": 1 if comms.burst else 0,
        "p_fail": cfg["p_fail"], "n_new": cfg["n_new"],
        "coverage_frac": (base_cov + covered_extra) / total_tasks,
        "recovery_cost": sum(rec_costs) / len(rec_costs) if rec_costs else 0.0,
        "recovery_latency": sum(rec_latencies) / len(rec_latencies) if rec_latencies else 0.0,
        "conflicts": conflicts,
        "makespan": max(completion, default=0.0),
    }


# ============================================================================
# STAGE 3: GRADED / PARTIAL partitions. Stage 1b showed ours == channel_aware_
# timeout because a full burst makes channel-health BIMODAL (~0 during the
# outage, ~1 otherwise) -- both a timeout-inflation and a belief-threshold see
# the same clear signal. The only place ours' CONTINUOUS belief could beat a
# hard timeout is the INTERMEDIATE zone (channel-health ~0.5), produced by a
# PARTIAL partition: a transient group split that cuts a SUBSET of the team off
# from a successor. An alive primary on the far side then looks silent to s --
# indistinguishable from a failure by silence alone -- but s's channel-health
# drops (it loses that whole group). The discriminator: a FAILED primary is
# silent to everyone (s still hears the rest -> channel-health high -> recover);
# a PARTITIONED-ALIVE primary is silent because s lost its group (channel-health
# low -> hold). We sweep the cut severity so channel-health is graded, and ask
# whether ours' belief separates failed-vs-partitioned better than the timeout.
# ============================================================================
def run_partition_trial(seed, comms, method, cfg):
    robots, tasks = make_world(seed, cfg["num_tasks"], cfg["position_jitter"],
                               cfg["nominal_speed"], cfg.get("fleet_size", 6))
    ids = list(robots); n = len(ids); crit = cfg["crit_duration"]; hb = cfg["hb_period"]
    rng = random.Random(seed * 7919 + 13)
    drop_prob = make_channel(comms, rng)
    c_miss = cfg.get("c_miss", 1.0); c_conflict = cfg.get("c_conflict", 1.0)
    tau = c_conflict / (c_miss + c_conflict)
    Lambda = cfg.get("lambda_sprt",
                     math.log((c_conflict / c_miss) *
                              ((1.0 - cfg.get("p_prior", 0.1)) / cfg.get("p_prior", 0.1))))
    cut_drop = cfg["cut_drop"]; part_len = cfg["partition_len"]

    # global group split (A/B), guaranteed non-trivial
    side = {}
    for j in ids:
        side[j] = "A" if rng.random() < 0.5 else "B"
    if len(set(side.values())) == 1:           # force both groups non-empty
        side[ids[0]] = "A"; side[ids[-1]] = "B"

    primary = greedy_assign(robots, tasks)
    task_primary = {k: r for r, k in primary.items()}
    idle = [r for r in ids if r not in primary]
    succ, used = {}, set()
    for k in sorted(tasks, key=lambda k: min(robots[r]["cost"][k] for r in idle)):
        cands = sorted([r for r in idle if r not in used], key=lambda r: robots[r]["cost"][k])
        succ[k] = cands[0] if cands else None
        if succ[k]:
            used.add(succ[k])

    horizon = 0.0
    pstate = {}
    for k, p in task_primary.items():
        treach = robots[p]["travel"][k]; tdone = treach + crit
        failed = rng.random() < cfg["p_fail"]
        tfail = rng.uniform(0.1, tdone) if failed else math.inf
        pstate[k] = {"p": p, "treach": treach, "tdone": tdone,
                     "failed": failed, "tfail": tfail}
        horizon = max(horizon, tdone)
    horizon += 40.0

    # one transient partition window overlapping execution
    t_p0 = rng.uniform(0.5, max(1.0, 0.5 * horizon))
    t_p1 = t_p0 + part_len

    def cut(j, s, te):
        return (t_p0 <= te < t_p1) and side[j] != side[s]

    alive_until = {j: math.inf for j in ids}
    for k, st in pstate.items():
        if st["failed"]:
            alive_until[st["p"]] = st["tfail"]
    hb_ticks = [i * hb for i in range(int(horizon / hb) + 1)]
    delivered = {s: {j: [] for j in ids if j != s} for s in ids}
    for j in ids:
        for te in hb_ticks:
            if te >= alive_until[j]:
                break
            for s in ids:
                if s == j:
                    continue
                if rng.random() < drop_prob(te):
                    continue                       # base loss
                if cut(j, s, te) and rng.random() < cut_drop:
                    continue                       # partition loss
                delivered[s][j].append(te)

    def last_hb(s, j, t):
        prev = 0.0
        for te in delivered[s][j]:
            if te <= t:
                prev = te
            else:
                break
        return prev

    window = 3.0 * hb; T_fixed = 3.0 * hb

    def channel_health(s, exclude, t):
        others = [j for j in ids if j != s and j != exclude]
        if not others:
            return 1.0
        return sum(1 for j in others if last_hb(s, j, t) >= t - window) / len(others)

    def est_drop_rate(s, exclude, t):
        others = [j for j in ids if j != s and j != exclude]
        if not others:
            return 0.01
        slots = max(1, int(round(window / hb)))
        recv = sum(1 for j in others for te in delivered[s][j]
                   if t - window <= te <= t)
        expected = len(others) * slots
        return 1.0 - recv / expected if expected else 0.5

    def lrt_belief(s, p, t):
        q = min(max(est_drop_rate(s, p, t), 1e-3), 0.999)
        k_missed = min((t - last_hb(s, p, t)) / hb, 40.0)
        prior_f = cfg.get("p_prior", 0.1)
        odds = (prior_f / (1.0 - prior_f)) * (q ** (-k_missed))
        return odds / (1.0 + odds)

    def adaptive_timeout(s, j, t):
        times = [te for te in delivered[s][j] if te <= t]
        if len(times) < 2:
            return 3.0 * hb
        gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
        mean = sum(gaps) / len(gaps)
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        return mean + 3.0 * math.sqrt(var) + hb

    dt = 0.25
    false_takeovers = correct_recoveries = missed = 0
    false_on_partition = 0     # false takeovers specifically on alive cross-side primaries
    n_partitioned = 0
    for k, st in pstate.items():
        p, s = st["p"], succ[k]
        # is this an "ambiguous" case: alive primary, cross-side from s, silenced
        # during the window for at least part of its execution?
        ambiguous = (s is not None and not st["failed"] and side[p] != side[s]
                     and t_p0 < st["tdone"] and t_p1 > 0.0)
        if ambiguous:
            n_partitioned += 1
        took_over_at = None
        if s is not None and method != "no_recovery":
            t = 0.0
            L_acc = 0.0
            while t <= horizon:
                if (not st["failed"]) and t >= st["tdone"]:
                    break
                silence = t - last_hb(s, p, t)
                fire = False
                if method == "fixed_timeout":
                    fire = silence > T_fixed
                elif method == "adaptive_pernode":
                    fire = silence > adaptive_timeout(s, p, t)
                elif method == "channel_aware_timeout":
                    ch = channel_health(s, p, t)
                    fire = silence > T_fixed * (1.0 + 4.0 * (1.0 - ch))
                elif method == "ours_correlated":
                    s_raw = max(0.0, min(1.0, (silence - 2.0 * hb) / (4.0 * hb)))
                    fire = s_raw * channel_health(s, p, t) >= tau
                elif method == "ours_lrt":
                    fire = lrt_belief(s, p, t) >= tau
                elif method in ("ours_sprt", "ours_sprt_lin"):
                    ch = channel_health(s, p, t)
                    if silence < hb:
                        L_acc = 0.0
                    else:
                        incr = (-math.log(1.0 - min(ch, 0.9))
                                if method == "ours_sprt" else ch)
                        L_acc += incr * (dt / hb)
                    fire = L_acc >= (cfg.get("sens") if cfg.get("sens") is not None
                                     else Lambda)
                if fire:
                    took_over_at = t
                    break
                t += dt

        if not st["failed"]:
            if took_over_at is not None:
                false_takeovers += 1
                if ambiguous:
                    false_on_partition += 1
        else:
            if took_over_at is not None:
                correct_recoveries += 1
            else:
                missed += 1

    n_failed = sum(1 for st in pstate.values() if st["failed"])
    total_cost = false_takeovers * c_conflict + missed * c_miss
    return {
        "mode": method, "cut_drop": cut_drop, "burst": 1 if comms.burst else 0,
        "p_fail": cfg["p_fail"],
        "false_takeovers": false_takeovers,
        "false_on_partition": false_on_partition,
        "n_partitioned": n_partitioned,
        "missed_failures": missed,
        "recovery_rate": (correct_recoveries / n_failed) if n_failed else 1.0,
        "total_cost": total_cost,
    }


STAGE3_METHODS = ["fixed_timeout", "adaptive_pernode", "channel_aware_timeout",
                  "ours_correlated", "ours_sprt", "ours_sprt_lin"]
STAGE3_METRICS = ["false_takeovers", "false_on_partition", "missed_failures",
                  "recovery_rate", "total_cost"]


def cmd_stage3(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg0 = trial_cfg(args)
    cfg0["partition_len"] = args.partition_len
    cfg0["crit_duration"] = args.crit_duration   # longer execution -> the
    # alive-but-partitioned ambiguity has time to bite before the task completes
    # sweep partition severity -> channel-health goes from ~1 (no cut) to graded
    cuts = [round(x, 2) for x in frange(0.0, args.max_cut, args.cut_step)]
    curves = {m: {mt: [] for mt in STAGE3_METRICS} for m in STAGE3_METHODS}
    rows = []
    for cd in cuts:
        comms = Comms(delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
                      drop_rate=args.drop_rate, burst=args.burst,
                      burst_drop=args.burst_drop)
        cfg = dict(cfg0); cfg["cut_drop"] = cd
        for m in STAGE3_METHODS:
            tr = [run_partition_trial(9900 + i, comms, m, cfg) for i in range(args.trials)]
            avg = average(tr); avg["cut_drop"] = cd; avg["mode"] = m
            rows.append(avg)
            for mt in STAGE3_METRICS:
                curves[m][mt].append(avg[mt])
    tag = "burst" if args.burst else "iid"
    save_csv(rows, out / f"stage3_partition_{tag}.csv")
    for mt in STAGE3_METRICS:
        line_plot(cuts, {m: curves[m][mt] for m in STAGE3_METHODS},
                  f"stage3_{tag}_{mt}", "partition cut severity", out)

    print(f"\n===== STAGE 3 GRADED PARTITION ({tag}, base drop={args.drop_rate}, "
          f"p_fail={args.p_fail}, part_len={args.partition_len}) =====")
    print("  false takeovers ON alive-but-partitioned primaries (lower=better):")
    print("  cut " + "".join(f"{m[:16]:>18}" for m in STAGE3_METHODS))
    for i, cd in enumerate(cuts):
        print(f"  {cd:>4}" + "".join(f"{curves[m]['false_on_partition'][i]:>18.3f}"
                                     for m in STAGE3_METHODS))
    print("  total_cost (false_takeovers + missed, balanced costs; lower=better):")
    print("  cut " + "".join(f"{m[:16]:>18}" for m in STAGE3_METHODS))
    for i, cd in enumerate(cuts):
        print(f"  {cd:>4}" + "".join(f"{curves[m]['total_cost'][i]:>18.3f}"
                                     for m in STAGE3_METHODS))
    print(f"\nSaved CSV + plots -> {out}")


# ----------------------------------------------------------------------------
# no_recovery is MOTIVATION/context (shows why recovery is needed), NOT a
# competitor -- like the oracle reference. The fair comparison is among the
# recovery methods below it.
METHODS = ["no_recovery", "fixed_timeout", "adaptive_pernode",
           "channel_aware_timeout", "ours_correlated"]
SWEEP_METRICS = ["coverage_frac", "false_takeovers", "recovery_rate",
                 "mean_recovery_latency", "missed_failures", "makespan"]


def make_comms(args, drop_rate=None):
    return Comms(delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
                 drop_rate=args.drop_rate if drop_rate is None else drop_rate,
                 burst=args.burst, burst_drop=args.burst_drop)


def trial_cfg(args):
    return {"num_tasks": args.num_tasks, "position_jitter": args.position_jitter,
            "nominal_speed": 5.0, "crit_duration": 2.0, "hb_period": args.hb_period,
            "p_fail": args.p_fail, "n_new": getattr(args, "n_new", 0),
            "fleet_size": getattr(args, "fleet_size", 6),
            "c_miss": getattr(args, "c_miss", 1.0),
            "c_conflict": getattr(args, "c_conflict", 1.0),
            "c_latency": getattr(args, "c_latency", 0.0)}


DYN_METRICS = ["coverage_frac", "recovery_cost", "recovery_latency",
               "conflicts", "makespan"]


def cmd_dynamic(args):
    """Recovery-mode comparison (detection fixed to correlated-silence) with
    mid-mission new-POI arrival, swept over failure rate."""
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg0 = trial_cfg(args)
    pfs = [round(x, 2) for x in frange(0.0, args.max_pfail, args.pfail_step)]
    curves = {m: {mode: [] for mode in RECOVERY_MODES} for m in DYN_METRICS}
    rows = []
    for pf in pfs:
        comms = make_comms(args)
        cfg = dict(cfg0); cfg["p_fail"] = pf
        for mode in RECOVERY_MODES:
            trials = [run_dynamic_trial(9300 + i, comms, mode, cfg) for i in range(args.trials)]
            avg = average(trials); avg["p_fail"] = pf; avg["mode"] = mode
            rows.append(avg)
            for m in DYN_METRICS:
                curves[m][mode].append(avg[m])
    tag = ("burst" if args.burst else "iid") + f"_nnew{args.n_new}"
    save_csv(rows, out / f"dyn_{tag}.csv")
    for m in DYN_METRICS:
        line_plot(pfs, curves[m], f"dyn_{tag}_{m}", "failure rate", out)
    # also print the table at the requested p_fail
    print(f"=== recovery modes ({tag}, drop={args.drop_rate}) ===")
    for mode in RECOVERY_MODES:
        comms = make_comms(args); cfg = dict(cfg0); cfg["p_fail"] = args.p_fail
        trials = [run_dynamic_trial(9300 + i, comms, mode, cfg) for i in range(args.trials)]
        print_table([average(trials)], DYN_METRICS)
    print(f"Saved -> {out}")


def cmd_compare(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    comms = make_comms(args); cfg = trial_cfg(args)
    rows = []
    for m in METHODS:
        trials = [run_trial(9000 + i, comms, m, cfg) for i in range(args.trials)]
        rows.append(average(trials))
    save_csv(rows, out / "fr_compare.csv")
    print_table(rows, SWEEP_METRICS)
    print(f"\n(burst={args.burst}, drop={args.drop_rate}, p_fail={args.p_fail})\nSaved -> {out}")


POOL_METHODS = ["no_recovery", "fixed_timeout", "adaptive_pernode",
                "channel_aware_timeout", "ours_correlated"]
POOL_METRICS = ["coverage_frac", "makespan", "false_takeovers",
                "stolen_misses", "missed_failures", "recovery_rate"]


def cmd_pool(args):
    """SHARED-POOL comparison: one contended successor pool, so false takeovers
    steal recovery capacity. Fixed allocator; only the detector varies. This is
    the regime where success rate (coverage) and makespan can finally separate."""
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    comms = make_comms(args)
    cfg = trial_cfg(args)
    cfg["fleet_size"] = args.fleet_size
    rows = []
    for m in POOL_METHODS:
        trials = [run_trial_pool(9500 + i, comms, m, cfg) for i in range(args.trials)]
        rows.append(average(trials))
    save_csv(rows, out / "fr_pool.csv")
    print(f"=== SHARED-POOL (fleet={args.fleet_size}, tasks={args.num_tasks}, "
          f"pool={rows[0].get('pool_size', '?')}, burst={args.burst}, "
          f"drop={args.drop_rate}, p_fail={args.p_fail}) ===")
    print_table(rows, POOL_METRICS)
    print(f"\nSaved -> {out}")


ROUTE_METHODS = ["no_recovery", "fixed_timeout", "adaptive_pernode",
                 "channel_aware_timeout", "ours_correlated"]
ROUTE_METRICS = ["coverage_frac", "makespan", "false_takeovers",
                 "missed_pois", "team_detour", "duplicate_visits"]


def cmd_route(args):
    """ROUTE mode: no spares. Every robot starts with a cheapest-insertion route
    over POIs>>robots; one robot is silenced mid-mission and its remaining POIs
    are re-auctioned by marginal insertion cost. Fixed backend; detector varies.
    This is the regime where a false takeover perturbs the WHOLE team's routes,
    so success rate AND makespan can genuinely separate the detectors."""
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    comms = make_comms(args)
    cfg = trial_cfg(args)
    cfg["fleet_size"] = args.fleet_size
    cfg["num_pois"] = args.num_pois
    cfg["crit_duration"] = args.crit_duration
    rows = []
    for m in ROUTE_METHODS:
        trials = [run_trial_route(9700 + i, comms, m, cfg) for i in range(args.trials)]
        rows.append(average(trials))
    save_csv(rows, out / "fr_route.csv")
    print(f"=== ROUTE mode (fleet={args.fleet_size}, POIs={args.num_pois}, "
          f"burst={args.burst}, drop={args.drop_rate}, p_fail={args.p_fail}) ===")
    print_table(rows, ROUTE_METRICS)
    print(f"\nSaved -> {out}")


def cmd_sweep_fail(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg0 = trial_cfg(args)
    pfs = [round(x, 2) for x in frange(0.0, args.max_pfail, args.pfail_step)]
    curves = {m: {meth: [] for meth in METHODS} for m in SWEEP_METRICS}
    rows = []
    for pf in pfs:
        comms = make_comms(args)
        cfg = dict(cfg0); cfg["p_fail"] = pf
        for meth in METHODS:
            trials = [run_trial(9100 + i, comms, meth, cfg) for i in range(args.trials)]
            avg = average(trials); avg["p_fail"] = pf; avg["mode"] = meth
            rows.append(avg)
            for m in SWEEP_METRICS:
                curves[m][meth].append(avg[m])
    tag = "burst" if args.burst else "iid"
    save_csv(rows, out / f"fr_sweep_pfail_{tag}.csv")
    for m in SWEEP_METRICS:
        line_plot(pfs, curves[m], f"fr_{tag}_{m}", "failure rate", out)
    print(f"Swept p_fail {pfs} ({tag}, drop={args.drop_rate})\nSaved -> {out}")


# ============================================================================
# STAGE 1: detection isolation (E1) + recovery-backend isolation (E2), each run
# under BOTH iid and burst comms. The honest story lives in the iid->burst
# contrast: under iid every method looks fine; the gaps only open under burst
# (correlated outage), which is the whole premise. We freeze one axis and vary
# the other so each result is attributable to a single cause:
#   E1  backend held constant (deterministic successor) ; vary DETECTION
#       -> false_takeovers/recovery_rate differences are caused by detection.
#   E2  detection held constant (ours_correlated)       ; vary BACKEND
#       -> conflicts/coverage differences are caused by the reallocation backend.
# ============================================================================
def _sweep_detection(pfs, comms, cfg0, trials):
    rows = {meth: [] for meth in METHODS}
    for pf in pfs:
        cfg = dict(cfg0); cfg["p_fail"] = pf
        for meth in METHODS:
            tr = [run_trial(9100 + i, comms, meth, cfg) for i in range(trials)]
            avg = average(tr); avg["p_fail"] = pf; avg["mode"] = meth
            rows[meth].append(avg)
    return rows


def _sweep_backend(pfs, comms, cfg0, trials):
    rows = {mode: [] for mode in RECOVERY_MODES}
    for pf in pfs:
        cfg = dict(cfg0); cfg["p_fail"] = pf
        for mode in RECOVERY_MODES:
            tr = [run_dynamic_trial(9300 + i, comms, mode, cfg) for i in range(trials)]
            avg = average(tr); avg["p_fail"] = pf; avg["mode"] = mode
            rows[mode].append(avg)
    return rows


def _row_at(rows_by_key, pf):
    """Pull the per-key averaged row at a given p_fail for a summary table."""
    out = []
    for key, lst in rows_by_key.items():
        match = next((r for r in lst if abs(r["p_fail"] - pf) < 1e-6), None)
        if match:
            out.append(match)
    return out


def cmd_stage1(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg0 = trial_cfg(args)
    pfs = [round(x, 2) for x in frange(0.0, args.max_pfail, args.pfail_step)]
    hi = max(pfs)  # representative high-failure operating point for the tables
    regimes = [("iid", False), ("burst", True)]

    e1_metrics = ["coverage_frac", "false_takeovers", "recovery_rate",
                  "mean_recovery_latency", "missed_failures"]
    e2_metrics = ["coverage_frac", "conflicts", "recovery_latency", "makespan"]
    flat = []

    for tag, burst in regimes:
        comms = Comms(delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
                      drop_rate=args.drop_rate, burst=burst, burst_drop=args.burst_drop)

        # ---- E1: detection isolation ----
        e1 = _sweep_detection(pfs, comms, cfg0, args.trials)
        for meth, lst in e1.items():
            for r in lst:
                flat.append({"experiment": "E1_detection", "regime": tag, **r})
        for m in e1_metrics:
            line_plot(pfs, {k: [r[m] for r in v] for k, v in e1.items()},
                      f"stage1_E1_{tag}_{m}", "failure rate", out)
        print(f"\n===== E1 DETECTION ISOLATION  ({tag}, drop={args.drop_rate}, "
              f"p_fail={hi}) =====")
        print("  backend held constant; only the detector varies")
        print_table(_row_at(e1, hi), e1_metrics)

        # ---- E2: backend isolation ----
        e2 = _sweep_backend(pfs, comms, cfg0, args.trials)
        for mode, lst in e2.items():
            for r in lst:
                flat.append({"experiment": "E2_backend", "regime": tag, **r})
        for m in e2_metrics:
            line_plot(pfs, {k: [r[m] for r in v] for k, v in e2.items()},
                      f"stage1_E2_{tag}_{m}", "failure rate", out)
        print(f"\n===== E2 BACKEND ISOLATION  ({tag}, drop={args.drop_rate}, "
              f"p_fail={hi}) =====")
        print("  detection held constant (ours_correlated); only the backend varies")
        print_table(_row_at(e2, hi), e2_metrics)

    # E1 and E2 have different columns -> separate CSVs
    save_csv([r for r in flat if r["experiment"] == "E1_detection"],
             out / "stage1_E1_detection.csv")
    save_csv([r for r in flat if r["experiment"] == "E2_backend"],
             out / "stage1_E2_backend.csv")

    # ---- the contrast that is the actual story: iid vs burst false-takeovers ----
    print("\n===== STAGE 1 HEADLINE: false-takeovers iid -> burst (E1, p_fail="
          f"{hi}) =====")
    print(f"  {'detector':<24}{'iid':>10}{'burst':>10}{'burst/iid':>12}")
    e1_iid = {r["mode"]: r for r in flat
              if r["experiment"] == "E1_detection" and r["regime"] == "iid"
              and abs(r["p_fail"] - hi) < 1e-6}
    e1_burst = {r["mode"]: r for r in flat
                if r["experiment"] == "E1_detection" and r["regime"] == "burst"
                and abs(r["p_fail"] - hi) < 1e-6}
    for meth in METHODS:
        if meth == "no_recovery":
            continue
        fi = e1_iid[meth]["false_takeovers"]; fb = e1_burst[meth]["false_takeovers"]
        ratio = (fb / fi) if fi > 1e-9 else float("inf")
        rs = "inf" if ratio == float("inf") else f"{ratio:.1f}x"
        print(f"  {meth:<24}{fi:>10.3f}{fb:>10.3f}{rs:>12}")
    print(f"\nSaved CSV + plots -> {out}")


# ============================================================================
# STAGE 1b: does ours' calibrated belief earn its keep over a trivial channel-
# aware timeout? Two tests, both under burst (the regime where it matters):
#   COST sweep:  vary the cost asymmetry c_miss/c_conflict. ours adapts its
#                takeover threshold tau = c_conflict/(c_miss+c_conflict);
#                fixed detectors can't. If ours tracks min total_cost across the
#                whole range while the fixed ones are only optimal near balanced
#                costs, the calibrated belief is a real contribution.
#   SCALE sweep: vary fleet size N and #simultaneous tasks K (=N/2). If per-node
#                detectors' false-takeovers grow with K (a real "storm") while
#                ours stays flat, the motivating story is earned at scale.
# ============================================================================
COST_METHODS = ["fixed_timeout", "adaptive_pernode", "channel_aware_timeout",
                "channel_aware_cost", "ours_correlated"]

# ============================================================================
# ROC: the structural upper hand. A per-node detector has ONE knob (how long to
# wait): short -> fast detection but many false alarms; long -> few false alarms
# but slow detection. It cannot escape that trade-off because it cannot tell
# "silent=dead" from "silent=channel". Channel-aware detectors CAN, so for any
# false-alarm rate they detect real failures faster. We sweep each detector's
# sensitivity knob across its full range and plot detection-latency vs false-
# takeover rate; if our curve sits to the lower-left of the per-node curves, we
# dominate the ROC -- a result that holds at EVERY operating point, not a tuned
# one. Downstream metrics (recovery makespan, wasted travel/energy) inherit it.
# ============================================================================
ROC_SENS = {
    "fixed_timeout":         [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],   # timeout x hb
    "adaptive_pernode":      [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0],  # k*std
    "channel_aware_timeout": [1.5, 2.0, 3.0, 4.0, 6.0, 8.0],        # base x hb
    "ours_correlated":       [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],  # tau
    "ours_sprt":             [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0],  # Lambda (LLR)
    "ours_sprt_lin":         [1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 13.0, 18.0],  # Lambda (lin)
}


def cmd_roc(args):
    import matplotlib.pyplot as plt
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg0 = trial_cfg(args)
    rows, curves = [], {}
    for tag, burst in [("iid", False), ("burst", True)]:
        comms = Comms(delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
                      drop_rate=args.drop_rate, burst=burst, burst_drop=args.burst_drop)
        curves[tag] = {}
        for meth, sens_list in ROC_SENS.items():
            pts = []
            for sv in sens_list:
                cfg = dict(cfg0); cfg["sens"] = sv
                tr = [run_trial(9000 + i, comms, meth, cfg) for i in range(args.trials)]
                avg = average(tr)
                pt = {"regime": tag, "mode": meth, "sens": sv,
                      "false_takeovers": avg["false_takeovers"],
                      "detection_latency": avg["mean_detection_latency"],
                      "missed_failures": avg["missed_failures"],
                      "recovery_rate": avg["recovery_rate"],
                      "makespan": avg["makespan"]}
                pts.append(pt); rows.append(pt)
            curves[tag][meth] = pts

        plt.figure(figsize=(8, 6))
        for meth, pts in curves[tag].items():
            xs = [p["false_takeovers"] for p in pts]
            ys = [p["detection_latency"] for p in pts]
            lw = 3.0 if meth.startswith("ours") else 1.6
            plt.plot(xs, ys, marker="o", linewidth=lw, label=meth)
        plt.xlabel("false takeovers / trial  (false-alarm rate ->)")
        plt.ylabel("detection latency on REAL failures (s)")
        plt.title(f"Detector ROC ({tag}) -- lower-left is better")
        plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(out / f"roc_{tag}.png", dpi=180); plt.close()

        print(f"\n===== ROC ({tag}, drop={args.drop_rate}, p_fail={args.p_fail}) "
              f": detection-latency vs false-alarms (sweep each knob) =====")
        for meth, pts in curves[tag].items():
            print(f"  {meth}")
            for p in pts:
                print(f"      sens={p['sens']:<5} false_tk={p['false_takeovers']:.3f}"
                      f"  det_lat={p['detection_latency']:.2f}s"
                      f"  missed={p['missed_failures']:.3f}")
    save_csv(rows, out / "roc_all.csv")
    print(f"\nSaved CSV + ROC plots -> {out}")


def cmd_stage1b(args):
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg0 = trial_cfg(args)
    burst = Comms(delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
                  drop_rate=args.drop_rate, burst=True, burst_drop=args.burst_drop)

    # ---------- COST-ASYMMETRY sweep (burst, balanced fleet) ----------
    ratios = [0.11, 0.25, 0.5, 1.0, 2.0, 4.0, 9.0]  # c_miss / c_conflict
    cost_curves = {m: [] for m in COST_METHODS}
    cost_rows = []
    for r in ratios:
        cfg = dict(cfg0); cfg["c_conflict"] = 1.0; cfg["c_miss"] = r
        for m in COST_METHODS:
            tr = [run_trial(9500 + i, burst, m, cfg) for i in range(args.trials)]
            avg = average(tr); avg["ratio"] = r; avg["mode"] = m
            cost_rows.append({"experiment": "cost_sweep", "ratio": r, **avg})
            cost_curves[m].append(avg["total_cost"])
    save_csv(cost_rows, out / "stage1b_cost_sweep.csv")
    line_plot(ratios, cost_curves, "stage1b_cost_total_cost",
              "c_miss / c_conflict", out)

    print(f"\n===== STAGE 1b COST-ASYMMETRY (burst, drop={args.drop_rate}, "
          f"p_fail={args.p_fail}) : total_cost (lower=better) =====")
    hdr = "  ratio " + "".join(f"{m[:14]:>16}" for m in COST_METHODS)
    print(hdr)
    for i, r in enumerate(ratios):
        line = f"  {r:>5} " + "".join(f"{cost_curves[m][i]:>16.3f}" for m in COST_METHODS)
        print(line)
    # who is best at each ratio
    print("  best-> " + "".join(
        f"{min(COST_METHODS, key=lambda m: cost_curves[m][i])[:14]:>16}"
        for i in range(len(ratios))))

    # ---------- SCALE sweep (burst, balanced costs) ----------
    Ks = [3, 6, 9, 12, 16]
    scale_methods = ["fixed_timeout", "adaptive_pernode",
                     "channel_aware_timeout", "ours_correlated",
                     "ours_sprt", "ours_sprt_lin"]
    scale_curves = {m: [] for m in scale_methods}
    scale_rows = []
    for K in Ks:
        cfg = dict(cfg0); cfg["num_tasks"] = K; cfg["fleet_size"] = 2 * K
        cfg["c_miss"] = 1.0; cfg["c_conflict"] = 1.0
        for m in scale_methods:
            tr = [run_trial(9700 + i, burst, m, cfg) for i in range(args.trials)]
            avg = average(tr); avg["K"] = K; avg["mode"] = m
            scale_rows.append({"experiment": "scale_sweep", "K": K, "N": 2 * K, **avg})
            scale_curves[m].append(avg["false_takeovers"])
    save_csv(scale_rows, out / "stage1b_scale_sweep.csv")
    line_plot(Ks, scale_curves, "stage1b_scale_false_takeovers",
              "#simultaneous tasks K (N=2K)", out)

    print(f"\n===== STAGE 1b SCALE (burst, drop={args.drop_rate}, p_fail="
          f"{args.p_fail}) : false-takeovers/trial vs K =====")
    print("  K  N  " + "".join(f"{m[:16]:>18}" for m in scale_methods))
    for i, K in enumerate(Ks):
        print(f"  {K:<3}{2*K:<3}" + "".join(f"{scale_curves[m][i]:>18.3f}"
                                            for m in scale_methods))
    print(f"\nSaved CSV + plots -> {out}")


def cmd_downstream(args):
    """Downstream metrics that INHERIT the detection gap, swept over team size
    under burst: (a) wasted travel/energy = successor travel spent on false
    takeovers (a monotone function of the false-takeover storm), and (b)
    recovery latency / makespan. We print both raw (default sensitivity) and,
    for latency, the fair matched-false-alarm view via the ROC, since ours
    deliberately confirms before acting and is only faster at matched FA."""
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg0 = trial_cfg(args)
    burst = Comms(delay_mean=args.delay_mean, delay_jitter=args.delay_jitter,
                  drop_rate=args.drop_rate, burst=True, burst_drop=args.burst_drop)

    Ks = [3, 6, 9, 12, 16]
    methods = ["fixed_timeout", "adaptive_pernode",
               "channel_aware_timeout", "ours_correlated"]
    metrics = ["wasted_travel", "false_takeovers", "makespan",
               "mean_recovery_latency", "mean_detection_latency"]
    curves = {mt: {m: [] for m in methods} for mt in metrics}
    rows = []
    for K in Ks:
        cfg = dict(cfg0); cfg["num_tasks"] = K; cfg["fleet_size"] = 2 * K
        cfg["c_miss"] = 1.0; cfg["c_conflict"] = 1.0
        for m in methods:
            tr = [run_trial(9900 + i, burst, m, cfg) for i in range(args.trials)]
            avg = average(tr); avg["K"] = K; avg["mode"] = m
            rows.append({"experiment": "downstream_scale", "K": K, "N": 2 * K, **avg})
            for mt in metrics:
                curves[mt][m].append(avg[mt])
    save_csv(rows, out / "downstream_scale.csv")

    # energy figure (the clean downstream win: wasted travel inherits the storm)
    import matplotlib.pyplot as plt
    labels = {"fixed_timeout": "fixed timeout",
              "adaptive_pernode": r"$\varphi$-accrual (per-node)",
              "channel_aware_timeout": "channel-aware timeout",
              "ours_correlated": "ours (channel-aware belief)"}
    styles = {"fixed_timeout": ("#1f77b4", "o", "-"),
              "adaptive_pernode": ("#d62728", "s", "-"),
              "channel_aware_timeout": ("#2ca02c", "^", "--"),
              "ours_correlated": ("#9467bd", "D", "-")}
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m in methods:
        c, mk, ls = styles[m]
        ax.plot(Ks, curves["wasted_travel"][m], color=c, marker=mk, ls=ls,
                lw=2.0, ms=6, label=labels[m])
    ax.set_xlabel("team size: #simultaneous tasks $K$  ($N=2K$ robots)")
    ax.set_ylabel("wasted travel per trial  (energy on false takeovers)")
    ax.set_title("Downstream energy inherits the false-takeover storm (burst)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_energy.png", dpi=150)
    plt.close(fig)

    def table(mt, label):
        print(f"\n===== DOWNSTREAM (burst, drop={args.drop_rate}, "
              f"p_fail={args.p_fail}) : {label} vs K =====")
        print("  K  N  " + "".join(f"{m[:16]:>18}" for m in methods))
        for i, K in enumerate(Ks):
            print(f"  {K:<3}{2*K:<3}" + "".join(
                f"{curves[mt][m][i]:>18.3f}" for m in methods))

    table("wasted_travel", "wasted travel / energy (lower=better)")
    table("false_takeovers", "false takeovers/trial (the source gap)")
    table("makespan", "mission makespan (raw, default sensitivity)")
    table("mean_recovery_latency", "mean recovery latency (raw)")
    table("mean_detection_latency", "mean detection latency (raw)")
    print(f"\nSaved CSV + energy plot -> {out}")
    print("NOTE: raw makespan/latency are at DEFAULT sensitivity (different "
          "false-alarm operating points); the fair speed comparison is the "
          "matched-false-alarm ROC (see `roc`).")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--trials", type=int, default=400)
        sp.add_argument("--num_tasks", type=int, default=3)
        sp.add_argument("--delay_mean", type=float, default=1.0)
        sp.add_argument("--delay_jitter", type=float, default=0.5)
        sp.add_argument("--drop_rate", type=float, default=0.2)
        sp.add_argument("--p_fail", type=float, default=0.3)
        sp.add_argument("--position_jitter", type=float, default=6.0)
        sp.add_argument("--hb_period", type=float, default=0.75)
        sp.add_argument("--burst", action="store_true")
        sp.add_argument("--burst_drop", type=float, default=0.97)
        sp.add_argument("--output_dir", type=str,
                        default=str(Path.home() / "phase1a_ws/results/failure_recovery"))

    c = sub.add_parser("compare"); common(c)
    s = sub.add_parser("sweepfail"); common(s)
    s.add_argument("--max_pfail", type=float, default=0.6)
    s.add_argument("--pfail_step", type=float, default=0.1)
    d = sub.add_parser("dynamic"); common(d)
    d.add_argument("--n_new", type=int, default=2, help="new POIs arriving mid-mission")
    d.add_argument("--max_pfail", type=float, default=0.6)
    d.add_argument("--pfail_step", type=float, default=0.1)
    s1 = sub.add_parser("stage1"); common(s1)
    s1.add_argument("--n_new", type=int, default=2, help="new POIs (E2 backend test)")
    s1.add_argument("--max_pfail", type=float, default=0.6)
    s1.add_argument("--pfail_step", type=float, default=0.1)
    s1b = sub.add_parser("stage1b"); common(s1b)
    s3 = sub.add_parser("stage3"); common(s3)
    s3.add_argument("--partition_len", type=float, default=10.0,
                    help="duration of the transient group split")
    s3.add_argument("--crit_duration", type=float, default=8.0,
                    help="critical-execution duration (longer = ambiguity bites)")
    s3.add_argument("--max_cut", type=float, default=1.0)
    s3.add_argument("--cut_step", type=float, default=0.2)
    rc = sub.add_parser("roc"); common(rc)
    ds = sub.add_parser("downstream"); common(ds)
    pl = sub.add_parser("pool"); common(pl)
    pl.add_argument("--fleet_size", type=int, default=6,
                    help="total robots; shared successor pool = fleet_size - num_tasks")
    rt = sub.add_parser("route"); common(rt)
    rt.add_argument("--fleet_size", type=int, default=6)
    rt.add_argument("--num_pois", type=int, default=40)
    rt.add_argument("--crit_duration", type=float, default=2.0)

    args = p.parse_args()
    if args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "sweepfail":
        cmd_sweep_fail(args)
    elif args.cmd == "stage1":
        cmd_stage1(args)
    elif args.cmd == "stage1b":
        cmd_stage1b(args)
    elif args.cmd == "stage3":
        cmd_stage3(args)
    elif args.cmd == "roc":
        cmd_roc(args)
    elif args.cmd == "downstream":
        cmd_downstream(args)
    elif args.cmd == "pool":
        cmd_pool(args)
    elif args.cmd == "route":
        cmd_route(args)
    else:
        cmd_dynamic(args)


if __name__ == "__main__":
    main()
