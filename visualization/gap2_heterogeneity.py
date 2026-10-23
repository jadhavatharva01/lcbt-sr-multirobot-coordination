#!/usr/bin/env python3
"""
Gap-2 diagnostic: per-ROBOT channel heterogeneity.

The main paper assumes a channel that is the SAME for everyone, so it cannot
show Gap 2: team-wide channel health eta is blind to ONE robot having a uniquely
bad link. Here we model exactly that.

Scenario: successor S monitors primary X, which is ALIVE the whole time (so ANY
takeover is FALSE). X's DIRECT link to S is bad (drop = d_bad, swept), while the
rest of the team is clear to S (drop = d_good) so eta stays high. A neighbour Nb
hears X over a good link.

We measure, per detector, the false-takeover rate on the alive-but-poorly-
connected X as d_bad grows. If it climbs while eta stays high, Gap 2 is real.

We also test a RELAYED-LIVENESS variant: S additionally learns "X alive" second-
hand from Nb (change the question from "can I hear X?" to "can anyone hear X?").
"""
import math, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HB = 0.75
DT = 0.25
WINDOW = 3.0 * HB
HORIZON = 30.0
TAU = 0.5
LAMBDA = math.log(9.0)        # SPRT default threshold (balanced costs, prior 0.1)
BASE = 3.0 * HB               # timeout base
N_OTHERS = 5                  # extra clear teammates (for eta), besides X and Nb
D_GOOD = 0.10
TRIALS = 600
D_BADS = [0.10, 0.30, 0.50, 0.70, 0.85, 0.95, 0.99]


def recv_times(rng, drop, horizon=HORIZON):
    """Slots (heartbeat ticks) at which a receiver hears this sender."""
    out = []
    t = 0.0
    while t <= horizon:
        if rng.random() >= drop:
            out.append(t)
        t += HB
    return out


def last_at(times, t):
    prev = 0.0
    for te in times:
        if te <= t:
            prev = te
        else:
            break
    return prev


def run_trial(seed, d_bad, detector, relay=False):
    rng = random.Random(seed)
    # S's receptions
    s_from_x = recv_times(rng, d_bad)                     # bad direct link to X
    others = [recv_times(rng, D_GOOD) for _ in range(N_OTHERS + 1)]  # Nb + others, clear
    nb_idx = 0
    # neighbour Nb's receptions of X (good link), and S's receptions of Nb's relay
    nb_from_x = recv_times(rng, D_GOOD)
    s_from_nb_relay = recv_times(rng, D_GOOD)

    def eta(t):
        heard = sum(1 for o in others if last_at(o, t) >= t - WINDOW)
        return heard / len(others)

    def last_heard_x(t):
        direct = last_at(s_from_x, t)
        if not relay:
            return direct
        # relayed liveness: the most recent moment S learned (via Nb) that X was
        # alive = Nb's last-heard-X as of the last relay S received from Nb.
        relay_rx = last_at(s_from_nb_relay, t)            # when S last heard Nb
        relayed_knowledge = last_at(nb_from_x, relay_rx)  # Nb's last X as of then
        return max(direct, relayed_knowledge)

    L = 0.0
    t = 0.0
    while t <= HORIZON:
        sigma = t - last_heard_x(t)
        e = eta(t)
        fire = False
        if detector == "fixed_timeout":
            fire = sigma > BASE
        elif detector == "channel_aware_timeout":
            fire = sigma > BASE * (1.0 + 4.0 * (1.0 - e))
        elif detector == "ours_correlated":
            rho = max(0.0, min(1.0, (sigma - 2.0 * HB) / (4.0 * HB)))
            fire = rho * e >= TAU
        elif detector == "ours_sprt_lin":
            if sigma < HB:
                L = 0.0
            else:
                L += e * (DT / HB)
            fire = L >= LAMBDA
        if fire:
            return 1               # FALSE takeover (X is alive)
        t += DT
    return 0


DETECTORS = ["fixed_timeout", "channel_aware_timeout", "ours_correlated", "ours_sprt_lin"]


def sweep(relay=False):
    res = {d: [] for d in DETECTORS}
    for d_bad in D_BADS:
        for det in DETECTORS:
            ft = sum(run_trial(1000 + i, d_bad, det, relay=relay) for i in range(TRIALS))
            res[det].append(ft / TRIALS)
    return res


if __name__ == "__main__":
    print(f"Gap-2 test: ONE alive robot with a bad direct link (d_bad swept), "
          f"team clear (d_good={D_GOOD}). {TRIALS} trials.\n")
    no = sweep(relay=False)
    yes = sweep(relay=True)

    def table(title, res):
        print(f"=== {title}: FALSE-takeover rate on the alive bad-link robot ===")
        print("  d_bad " + "".join(f"{d[:16]:>18}" for d in DETECTORS))
        for i, d_bad in enumerate(D_BADS):
            print(f"  {d_bad:>5} " + "".join(f"{res[d][i]:>18.3f}" for d in DETECTORS))
        print()

    table("NO RELAY (current method)", no)
    table("WITH RELAYED LIVENESS (proposed fix)", yes)

    # plot: ours_correlated with vs without relay, plus channel_aware
    plt.figure(figsize=(7, 4.5))
    plt.plot(D_BADS, no["ours_correlated"], "o-", color="#9467bd", lw=2,
             label="ours (no relay)")
    plt.plot(D_BADS, no["channel_aware_timeout"], "^--", color="#2ca02c", lw=2,
             label="channel-aware (no relay)")
    plt.plot(D_BADS, yes["ours_correlated"], "s-", color="#1565c0", lw=2,
             label="ours + relayed liveness")
    plt.xlabel("drop rate on the one robot's link to its monitor (team stays clear)")
    plt.ylabel("false takeovers on the ALIVE robot (rate)")
    plt.title("Gap 2: team-average channel health is blind to one bad link")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    import os
    out = os.path.join(os.path.dirname(__file__), "..", "results", "gap2")
    os.makedirs(out, exist_ok=True)
    plt.savefig(os.path.join(out, "fig_gap2.png"), dpi=150)
    print("wrote results/gap2/fig_gap2.png")
