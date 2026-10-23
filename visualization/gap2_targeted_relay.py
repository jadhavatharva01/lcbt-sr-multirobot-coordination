#!/usr/bin/env python3
"""
Gap-2 fix, cheap version: locality-aware TARGETED relay vs FLOODING relay.

Setup (spatial occlusion): an ALIVE robot X sits where its monitor S cannot hear
it (bad direct link), but robots physically NEAR X hear it fine. To avoid
false-recovering X, S must learn second-hand that X is alive.

  - no relay      : S uses only its own (bad) hearing of X  -> false-recovers.
  - flooding      : S asks the WHOLE team "do you hear X?"   -> max suppression,
                    high message cost (N-1 queried per attempt).
  - targeted      : S asks only the k robots it BELIEVES are nearest X's expected
                    position -> low cost (k queried). Works iff the position
                    model correctly points at robots that actually hear X.
  - targeted+stale: same, but a fraction p_stale of robots' positions in S's
                    model are outdated, so S may query the wrong robots.

We measure BOTH false-takeover rate on the alive X (suppression) and messages
used (cost), to test whether locality buys the suppression cheaply and whether
staleness quietly breaks it.
"""
import math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HB = 0.75
DT = 0.25
WINDOW = 3.0 * HB
HORIZON = 30.0
N = 12
R_HEAR = 15.0            # within this radius of X -> good link to X
D_GOOD = 0.10
D_FAR = 0.92            # out-of-range link to X (incl. S's direct link)
T_TAKEOVER = 5.0 * HB  # take over if no evidence X alive for this long
T_SUSPECT = 2.0 * HB   # start querying once silent this long
REQUERY = 1.0          # seconds between relay queries while suspicious
K = 2                  # targeted: number of robots queried
TRIALS = 600
BOX = 100.0


def recv_times(rng, drop):
    out, t = [], 0.0
    while t <= HORIZON:
        if rng.random() >= drop:
            out.append(t)
        t += HB
    return out


def heard_within(times, t, w=WINDOW):
    return any(t - w <= te <= t for te in times)


def last_at(times, t):
    prev = -1e9
    for te in times:
        if te <= t:
            prev = te
        else:
            break
    return prev


def make_world(seed):
    rng = random.Random(seed)
    pX = (50.0, 50.0)
    pos = {}
    # 3 genuine neighbours within R of X (they hear X well)
    for j in range(3):
        ang, r = rng.uniform(0, 2 * math.pi), rng.uniform(2, R_HEAR - 2)
        pos[f"nb{j}"] = (pX[0] + r * math.cos(ang), pX[1] + r * math.sin(ang))
    pos["S"] = (8.0, 8.0)                       # monitor: far from X
    for j in range(N - 4):                      # rest: spread out (mostly far)
        pos[f"r{j}"] = (rng.uniform(0, BOX), rng.uniform(0, BOX))
    others = [k for k in pos if k != "S"]
    # each robot's reception of X depends on its true distance to X
    rx = {}
    for u in pos:
        d = math.dist(pos[u], pX)
        rx[u] = recv_times(rng, D_GOOD if d <= R_HEAR else D_FAR)
    return rng, pos, pX, others, rx


def trial(seed, strategy, p_stale=0.0):
    rng, pos, pX, others, rx = make_world(seed)
    s_direct = rx["S"]                          # S's own (bad) hearing of X
    # S's believed positions (for targeted selection): true, but stale ones random
    believed = {}
    for u in others:
        if u != "S" and rng.random() < p_stale:
            believed[u] = (rng.uniform(0, BOX), rng.uniform(0, BOX))   # outdated
        else:
            believed[u] = pos[u]
    cand = [u for u in others if u != "S"]
    by_belief = sorted(cand, key=lambda u: math.dist(believed[u], pX))

    last_alive = last_at(s_direct, 0.0)
    last_query = -1e9
    messages = 0
    took_over = 0
    t = 0.0
    while t <= HORIZON:
        # refresh from S's own direct hearing
        d_last = last_at(s_direct, t)
        if d_last > last_alive:
            last_alive = d_last
        silent_for = t - last_alive
        # issue a relay query if suspicious and due
        if strategy != "none" and silent_for > T_SUSPECT and (t - last_query) >= REQUERY:
            last_query = t
            if strategy == "flooding":
                asked = cand
            else:                                # targeted / targeted+stale
                asked = by_belief[:K]
            messages += len(asked)
            for u in asked:
                # u confirms iff it actually heard X recently AND its reply reaches S
                if heard_within(rx[u], t) and rng.random() >= D_GOOD:
                    last_alive = t               # learned X alive (now)
                    break
            silent_for = t - last_alive
        if silent_for > T_TAKEOVER:
            took_over = 1                        # FALSE: X is alive
            break
        t += DT
    return took_over, messages


STRATS = [("none", 0.0), ("flooding", 0.0),
          ("targeted", 0.0), ("targeted", 0.2),
          ("targeted", 0.4), ("targeted", 0.6), ("targeted", 0.8)]


if __name__ == "__main__":
    print(f"Targeted vs flooding relay (N={N}, k={K}, {TRIALS} trials). "
          f"X alive, occluded from S.\n")
    print(f"  {'strategy':<22}{'false-takeover':>16}{'avg messages':>16}")
    rows = []
    for strat, ps in STRATS:
        fts, msgs = 0, 0
        for i in range(TRIALS):
            ft, m = trial(7000 + i, strat, ps)
            fts += ft; msgs += m
        name = strat if strat != "targeted" else f"targeted (stale {int(ps*100)}%)"
        ft_rate, avg_msg = fts / TRIALS, msgs / TRIALS
        rows.append((name, ft_rate, avg_msg))
        print(f"  {name:<22}{ft_rate:>16.3f}{avg_msg:>16.2f}")

    # plot: suppression vs cost
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    names = [r[0] for r in rows]
    a1.bar(range(len(rows)), [r[1] for r in rows], color="#c62828")
    a1.set_xticks(range(len(rows))); a1.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    a1.set_ylabel("false takeovers on ALIVE X (rate)"); a1.set_title("(a) suppression")
    a2.bar(range(len(rows)), [r[2] for r in rows], color="#1565c0")
    a2.set_xticks(range(len(rows))); a2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    a2.set_ylabel("avg relay messages / trial"); a2.set_title("(b) message cost")
    fig.suptitle("Locality-aware targeted relay: cheap Gap-2 fix, vs staleness", fontweight="bold")
    fig.tight_layout()
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "results", "gap2")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_targeted_relay.png"), dpi=150)
    print("\nwrote results/gap2/fig_targeted_relay.png")
