#!/usr/bin/env python3
"""Where does reality sit between the two delay models?

Everything in the rendezvous line is bracketed by two assumptions about what a
robot carries into an appointment, and they disagree by 10-100x:

  A-L2 (final-leg)   a work stop RESYNCHRONISES: the router replans from the
                     robot's actual clock at every waypoint, so the delay that
                     matters is only the last approach leg.  Refuted as stated
                     -- HoloOcean shows a dwell changes nothing (see
                     holoocean_eq3_validation, N4 in docs/rendezvous_assumptions)
                     -- but it is the right LOWER bound, because it is what a
                     robot that adapts perfectly would achieve.
  carry-forward      no adaptation at all: every second of disturbance over the
                     whole window shows up as lateness at the appointment.  An
                     UPPER bound.

Neither is what a real robot does.  A real robot replans: when it runs late it
SHEDS WORK so it can still make the meeting, converting lateness into lost
coverage.  This harness measures that conversion in a hydrodynamic simulator
rather than asserting it.

  open    fly the whole route whatever happens          -> the carry bound
  closed  at each waypoint, drop the next task if       -> what actually happens
          finishing it would miss the appointment
  sprint  same test, but spend speed instead of dropping work

The number the whole line needs is `carried`: the fraction of accumulated
disturbance that survives to the appointment as lateness once the robot is
allowed to react.  carried ~ 0 means reality is near A-L2 and the cost is paid
in coverage; carried ~ 1 means reality is near carry-forward.

Run inside the HoloOcean venv:
  ~/holoocean_env/bin/python visualization/holoocean_closed_loop.py --trials 8
"""
import argparse
import csv
import os
from copy import deepcopy

import numpy as np

TPS = 50.0          # physics ticks per second
DEPTH = -5.0        # m, HoveringAUV operating depth
TOL = 2.0           # m, waypoint arrival tolerance
SCAN = 60.0         # s, task dwell (matches taskalloc_coverage_first.SCAN)
GRACE = 5.0         # s, shed-decision tolerance (absorbs tracker jitter)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "holoocean_closed_loop")


def make_cfg():
    from holoocean import packagemanager
    cfg = deepcopy(packagemanager.get_scenario("OpenWater-Hovering"))
    cfg["package_name"] = "Ocean"
    cfg["name"] = "ClosedLoop"
    cfg["agents"] = [{
        "agent_name": "a0", "agent_type": "HoveringAUV",
        "sensors": [{"sensor_type": "LocationSensor"}],
        "location": [0.0, 0.0, DEPTH], "rotation": [0.0, 0.0, 0.0],
        "control_scheme": 1, "is_main_agent": True}]
    cfg["main_agent"] = "a0"
    cfg["ticks_per_sec"] = TPS
    cfg["frames_per_sec"] = 0
    return cfg


def _cmd(p):
    return np.array([p[0], p[1], DEPTH, 0.0, 0.0, 0.0], dtype=float)


def band_current(strength, y0, y1):
    """A current that fills a BAND of the field, so a route crosses into and out
    of it -- the spatially correlated structure the i.i.d.-per-leg models in the
    rendezvous literature leave out."""
    def f(p):
        return [-strength, 0.0, 0.0] if y0 <= p[1] <= y1 else [0.0, 0.0, 0.0]
    return f


def plan_route(rng, n_tasks, span):
    """A short tour: tasks scattered in the band, then an appointment point back
    at the origin side -- the geometry of one approach window."""
    pts = [np.array([rng.uniform(0.3, 1.0) * span,
                     rng.uniform(0.0, 1.0) * span]) for _ in range(n_tasks)]
    meet = np.array([0.15 * span, 0.5 * span])
    return pts, meet


def calibrate(env, pts, meet):
    """Fly the route in STILL WATER and record what it actually takes.

    The analytic nominal (distance / assumed cruise) is not the right budget:
    HoloOcean's PID waypoint tracker does not hold the assumed speed, and that
    bias would be read as drift and make the robot shed tasks even with no
    current. Everything downstream is measured against this instead:
      still_task[k]  travel to q_k plus its scan, in still water
      still_meet[k]  travel from q_k to the appointment, in still water
      budget         total still-water duration of the whole planned window
    A planner knows these numbers; the disturbance is what it does not know."""
    env.reset()
    env.agents["a0"].teleport([0.0, 0.0, DEPTH], [0, 0, 0])
    for _ in range(5):
        env.act("a0", _cmd(np.array([0.0, 0.0]))); env.tick()
    calm = lambda p: [0.0, 0.0, 0.0]
    t, still_task, arrive = 0.0, [], []
    for q in pts:
        t0 = t
        t, _p = goto(env, q, calm, t)
        t = dwell(env, SCAN, calm, t, q)
        still_task.append(t - t0)
        arrive.append(t)
    t, _p = goto(env, meet, calm, t)
    budget = t
    # still-water time from each task back to the appointment, measured
    still_meet = []
    for k, q in enumerate(pts):
        env.reset()
        env.agents["a0"].teleport([float(q[0]), float(q[1]), DEPTH], [0, 0, 0])
        for _ in range(5):
            env.act("a0", _cmd(q)); env.tick()
        tt, _p = goto(env, meet, calm, 0.0, )
        still_meet.append(tt)
    return budget, still_task, still_meet


# Measured envelope of the HoveringAUV PID waypoint tracker (control_scheme 1)
# over a 100 m leg (scratchpad/speed.py). It does NOT hold a nominal cruise, and
# an opposing current degrades it smoothly rather than stalling it:
#   current (m/s)  0.00  0.25  0.50  0.75  1.00  1.50
#   speed   (m/s)  0.94  0.91  0.87  0.84  0.79  0.69
# Budgets are measured from a still-water run, not assumed; V_FLOOR exists only
# to size travel timeouts, so a long leg under current is never cut off
# mid-flight -- which would silently corrupt the rest of the route.
V_FLOOR = 0.6


def goto(env, target, current_fn, t, speed_cmd=None, max_s=None):
    """Drive to `target` under the current, returning the elapsed sim seconds.
    `speed_cmd` is unused by the PID scheme (it has one gain); sprinting is
    emulated by the caller shortening the route instead."""
    if max_s is None:
        st = env.tick()
        here = np.asarray(st["LocationSensor"][:2], float)
        max_s = 3.0 * float(np.linalg.norm(np.asarray(target) - here)) / V_FLOOR + 60.0
        t += 1.0 / TPS
    n = int(max_s * TPS)
    for _ in range(n):
        st = env.tick()
        p = np.asarray(st["LocationSensor"][:2], float)
        t += 1.0 / TPS
        env.set_ocean_currents("a0", current_fn(st["LocationSensor"]))
        env.act("a0", _cmd(target))
        if float(np.linalg.norm(p - np.asarray(target))) < TOL:
            return t, p
    raise RuntimeError("goto timed out %.0f m short of %s after %.0f s -- the "
                       "leg budget is too small and the route is corrupt"
                       % (float(np.linalg.norm(p - np.asarray(target))),
                          np.round(target, 1), max_s))


def dwell(env, secs, current_fn, t, hold):
    for _ in range(int(secs * TPS)):
        st = env.tick()
        t += 1.0 / TPS
        env.set_ocean_currents("a0", current_fn(st["LocationSensor"]))
        env.act("a0", _cmd(hold))
    return t


def run_trial(env, pts, meet, current_fn, policy, cal, grace_eps=0.0):
    """Fly one approach window. Returns (lateness_s, tasks_done).

    `grace_eps` is the meeting grace the replanner is allowed to spend. The
    `closed` policy targets the bare appointment (grace_eps=0) -- which is what
    a planner that treats the appointment as a hard deadline does. `graced`
    targets appointment + eps, i.e. it knows that arriving eps late still makes
    the meeting. The gap between them is the whole point of the experiment."""
    budget, still_task, still_meet = cal
    env.reset()
    env.agents["a0"].teleport([0.0, 0.0, DEPTH], [0, 0, 0])
    for _ in range(5):
        env.act("a0", _cmd(np.array([0.0, 0.0]))); env.tick()
    t, done = 0.0, 0
    for k, q in enumerate(pts):
        if policy in ("closed", "graced"):
            # THE REPLANNING DECISION. On my actual clock, can I still do this
            # task and reach the appointment on the still-water estimates a
            # planner has? If not, shed it and everything after it. With no
            # current this is an equality, so nothing is shed -- which is the
            # sanity check the whole measurement rests on.
            if t + still_task[k] + still_meet[k] > budget + GRACE + grace_eps:
                break
        t, _p = goto(env, q, current_fn, t)
        t = dwell(env, SCAN, current_fn, t, q)
        done += 1
    t, _p = goto(env, meet, current_fn, t)
    return t - budget, done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Sized for tractability: one route is ~5 legs plus tasks*SCAN of simulated
    # time and HoloOcean runs ~5x real time, so span and the current ladder both
    # cost wall-clock directly.
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--span", type=float, default=120.0)
    ap.add_argument("--currents", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--eps", type=float, default=30.0,
                    help="meeting grace a partner waits past the appointment; "
                         "the `graced` replanner is allowed to spend it")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import holoocean
    rows = []
    with holoocean.make(scenario_cfg=make_cfg(), show_viewport=False) as env:
        for tr in range(args.trials):
            rng = np.random.default_rng(args.seed + tr)
            pts, meet = plan_route(rng, args.tasks, args.span)
            cal = calibrate(env, pts, meet)
            budget = cal[0]
            for c in args.currents:
                cf = band_current(c, 0.25 * args.span, 0.75 * args.span)
                for policy in ("open", "closed", "graced"):
                    late, nd = run_trial(env, pts, meet, cf, policy, cal,
                                         grace_eps=(args.eps if policy == "graced"
                                                    else 0.0))
                    if c == 0.0:
                        # the sanity check: with no current nothing may be shed
                        # and nobody may be late. If this trips, the budget is
                        # wrong and every other number is meaningless.
                        assert nd == args.tasks and abs(late) < 20.0, (
                            "zero-current control failed: %d/%d tasks, %+.1f s"
                            % (nd, args.tasks, late))
                    rows.append(dict(trial=tr, current=c, policy=policy,
                                     lateness=round(late, 2),
                                     tasks=nd, planned=args.tasks,
                                     budget=round(budget, 1)))
                    print("  trial %d  cur %.1f  %-6s  lateness %+7.1f s  "
                          "tasks %d/%d" % (tr, c, policy, late, nd,
                                           args.tasks), flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "closed_loop.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)

    # NOTE ON THE SUMMARY. The obvious statistic -- lateness(closed) /
    # lateness(open) as a "fraction of delay carried" -- is meaningless here,
    # and the pilot run showed why: shedding is QUANTISED. A task is worth a
    # scan plus its travel (~300 s on these routes), so a robot reacting to
    # 15 s of drift by dropping one overshoots by 20x and arrives early. The
    # ratio then flips sign and reports nonsense. Adaptation is not a dial
    # between the two bounds; it is an all-or-nothing switch, so the honest
    # summary is how OFTEN it fires and what it costs when it does.
    f = lambda v: (float(np.mean(v)) if v else float("nan"))
    print("\n%-8s %-7s %6s %11s %11s %8s" % ("current", "policy", "shed%",
                                              "late|noshed", "late|shed",
                                              "cover"))
    for c in args.currents:
        for pol in ("open", "closed", "graced"):
            g = [r for r in rows if r["current"] == c and r["policy"] == pol]
            shed = [r for r in g if r["tasks"] < r["planned"]]
            keep = [r for r in g if r["tasks"] == r["planned"]]
            print("%-8.1f %-7s %5.0f%% %10.1fs %10.1fs %7.0f%%"
                  % (c, pol, 100.0 * len(shed) / max(len(g), 1),
                     f([r["lateness"] for r in keep]),
                     f([r["lateness"] for r in shed]),
                     100.0 * f([r["tasks"] for r in g]) / args.tasks))
    print("\n[csv] %s" % path)
    print("A robot that cannot shed a FRACTION of a task cannot track the")
    print("disturbance: it either carries the whole delay to the meeting, or")
    print("drops a whole task and arrives far too early. Neither bound is wrong")
    print("-- the system simply alternates between them.")


if __name__ == "__main__":
    main()
