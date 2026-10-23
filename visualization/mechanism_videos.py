#!/usr/bin/env python3
"""
mechanism_videos.py
-------------------
Illustrative animations of the channel-aware correlated-silence failure
detector used in the paper. These are MECHANISM ILLUSTRATIONS (they visualise
the decision rule beta = rho * eta against the cost-calibrated threshold tau);
the quantitative results in the paper come from failure_recovery_sim.py.

Renders three GIFs into results/videos/:
  1. vid1_isolated_recovery.gif   isolated silence -> eta high -> recover
  2. vid2_correlated_hold.gif     correlated burst -> eta collapses -> hold
                                  (a per-node detector storms; ours holds)
  3. vid3_burst_then_recover.gif  burst masks a real death -> hold, then
                                  recover once the channel returns

No ffmpeg required: saved as GIF via Pillow.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ----------------------------------------------------------------------
# parameters (match the paper's notation)
# ----------------------------------------------------------------------
H = 1.0            # heartbeat period (s)
W = 3.0            # channel-health window (s)
TAU = 0.5          # cost-calibrated threshold (balanced costs)
PERNODE_K = 3.0    # per-node (phi-accrual proxy) fires at silence >= K*H
DT = 0.2
FPS = 8

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "videos")
os.makedirs(OUT, exist_ok=True)

# fixed 6-robot layout. R0 is the monitoring successor.
POS = {
    0: (1.0, 5.0),   # monitor / successor
    1: (3.0, 8.6),
    2: (5.6, 7.2),
    3: (8.6, 5.6),
    4: (7.0, 2.0),
    5: (3.4, 1.6),
}
MONITOR = 0
COL_OK = "#2e7d32"
COL_FAIL = "#c62828"
COL_HOLD = "#f9a825"
COL_MON = "#1565c0"


def rho_of(sigma):
    return float(np.clip((sigma - 2 * H) / (4 * H), 0.0, 1.0))


# ----------------------------------------------------------------------
# A scenario produces, per timestep, the "last heard" time of every robot
# at the monitor, plus which robot(s) are the monitored primary(ies).
# ----------------------------------------------------------------------
class Scenario:
    def __init__(self, name, title, tmax, primary, delivers, takeover_label,
                 caption):
        self.name = name
        self.title = title
        self.tmax = tmax
        self.primary = primary            # monitored primary id
        self.delivers = delivers          # delivers(j, t) -> bool (heard now)
        self.takeover_label = takeover_label
        self.caption = caption

    def timeline(self):
        ts = np.arange(0.0, self.tmax + DT / 2, DT)
        n = len(POS)
        last_heard = {j: 0.0 for j in POS}
        rows = []
        pernode_fired = set()       # primaries a per-node detector already fired
        pernode_count = 0
        took_over_at = None
        for t in ts:
            for j in POS:
                if j == MONITOR:
                    continue
                if self.delivers(j, t):
                    last_heard[j] = t
            p = self.primary
            sigma = t - last_heard[p]
            rho = rho_of(sigma)
            others = [j for j in POS if j not in (MONITOR, p)]
            heard = sum(1 for j in others if last_heard[j] >= t - W)
            eta = heard / max(1, len(others))
            beta = rho * eta
            # per-node proxy: fires on raw silence, ignores eta
            for q in [k for k in POS if k not in (MONITOR,)]:
                if q in pernode_fired:
                    continue
                if (t - last_heard[q]) >= PERNODE_K * H:
                    pernode_fired.add(q)
                    pernode_count += 1
            if took_over_at is None and beta >= TAU:
                took_over_at = t
            rows.append(dict(t=t, sigma=sigma, rho=rho, eta=eta, beta=beta,
                             last_heard=dict(last_heard),
                             pernode_count=pernode_count,
                             took_over_at=took_over_at))
        return ts, rows


# ----------------------------------------------------------------------
# scenario definitions
# ----------------------------------------------------------------------
def make_scenarios():
    # 1) isolated failure: R3 dies at t=3; everyone else keeps being heard.
    def deliv1(j, t):
        if j == 3:
            return t < 3.0
        return True
    s1 = Scenario(
        "vid1_isolated_recovery",
        "Situation 1: Isolated silence -> correct recovery",
        tmax=12.0, primary=3, delivers=deliv1,
        takeover_label="R0 takes over R3's task",
        caption=("R3 fails; the rest of the team is still heard so channel "
                 "health stays high.\n"
                 "Silence from R3 is therefore explained by FAILURE: "
                 "beta = rho*eta crosses tau and R0 recovers."))

    # 2) correlated burst: at t=3..9 NOTHING is heard (whole team masked).
    #    R3 is actually still alive here -> any takeover would be FALSE.
    def deliv2(j, t):
        if 3.0 <= t < 9.0:
            return False           # burst masks the entire team
        return True                # everyone alive before and after
    s2 = Scenario(
        "vid2_correlated_hold",
        "Situation 2: Correlated burst -> hold (no false-takeover storm)",
        tmax=12.0, primary=3, delivers=deliv2,
        takeover_label="(no takeover -- correctly holds)",
        caption=("A burst silences the WHOLE team at once; every primary looks "
                 "dead to a per-node detector,\n"
                 "which fires a storm of FALSE takeovers. Channel health eta "
                 "collapses, so our belief beta is suppressed: R0 holds."))

    # 3) burst masks a REAL death: burst t=3..8; R3 truly dies at t=4 and never
    #    returns. During the burst eta~0 -> hold; after t=8 the team returns but
    #    R3 stays silent -> eta recovers, beta crosses tau -> correct (late)
    #    recovery. A per-node detector already false-fired during the burst.
    def deliv3(j, t):
        if 3.0 <= t < 8.0:
            return False           # burst masks everyone
        if j == 3:
            return t < 4.0         # R3 genuinely dead from t=4
        return True
    s3 = Scenario(
        "vid3_burst_then_recover",
        "Situation 3: Burst hides a real death -> hold, then recover",
        tmax=14.0, primary=3, delivers=deliv3,
        takeover_label="R0 recovers R3 after the channel returns",
        caption=("R3 truly dies during a burst. While eta~0 the ambiguity is "
                 "unresolved, so R0 HOLDS (no premature takeover).\n"
                 "Once the channel returns eta recovers; R3 is still silent, so "
                 "beta crosses tau and R0 recovers -- delayed but correct."))
    return [s1, s2, s3]


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------
def render(scn):
    ts, rows = scn.timeline()
    n = len(ts)

    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)
    axA = fig.add_subplot(gs[0, 0])   # arena
    axB = fig.add_subplot(gs[0, 1])   # belief panel

    # precompute monitor takeover motion target = primary position
    p = scn.primary
    target = np.array(POS[p])
    home = np.array(POS[MONITOR])

    def draw(i):
        r = rows[i]
        t = r["t"]
        axA.clear(); axB.clear()

        # ---------- arena ----------
        axA.set_xlim(-0.5, 10.5); axA.set_ylim(-0.5, 10.0)
        axA.set_xticks([]); axA.set_yticks([])
        axA.set_title(scn.title, fontsize=11, fontweight="bold", loc="left")

        # monitor position (moves on takeover)
        tko = r["took_over_at"]
        mon_pos = home.copy()
        if tko is not None:
            frac = np.clip((t - tko) / 3.0, 0.0, 1.0)
            mon_pos = home + frac * (target - home)

        # comm links + robots
        for j, pos in POS.items():
            if j == MONITOR:
                continue
            heard = r["last_heard"][j] >= t - W
            lc = COL_OK if heard else COL_FAIL
            ls = "-" if heard else "--"
            alpha = 0.55 if heard else 0.25
            axA.plot([mon_pos[0], pos[0]], [mon_pos[1], pos[1]],
                     color=lc, ls=ls, lw=1.4, alpha=alpha, zorder=1)

        for j, pos in POS.items():
            dead = (r["last_heard"][j] < t - 50)  # never; we mark deaths below
            if j == MONITOR:
                axA.scatter(*mon_pos, s=520, c=COL_MON, edgecolors="k",
                            zorder=3)
                axA.annotate("R0\n(monitor)", mon_pos, ha="center", va="center",
                             fontsize=8, color="white", fontweight="bold",
                             zorder=4)
                continue
            heard = r["last_heard"][j] >= t - W
            is_primary = (j == p)
            # is this primary actually silent right now (candidate dead)?
            silent = (t - r["last_heard"][j]) > 1.5 * H
            face = COL_OK if heard else (COL_FAIL if is_primary else COL_HOLD)
            axA.scatter(*pos, s=460, c=face, edgecolors="k", zorder=3)
            lab = f"R{j}"
            if is_primary:
                lab += "\n(primary)"
            axA.annotate(lab, pos, ha="center", va="center", fontsize=8,
                         color="white", fontweight="bold", zorder=4)
            if is_primary and silent:
                axA.annotate("?", (pos[0] + 0.5, pos[1] + 0.5), fontsize=15,
                             color=COL_FAIL, fontweight="bold", zorder=5)

        # legend
        axA.scatter([], [], s=120, c=COL_OK, edgecolors="k", label="heard (link up)")
        axA.scatter([], [], s=120, c=COL_FAIL, edgecolors="k", label="silent / link down")
        axA.scatter([], [], s=120, c=COL_MON, edgecolors="k", label="monitoring successor")
        axA.legend(loc="lower right", fontsize=7, framealpha=0.9)
        axA.text(0.0, -0.12, f"t = {t:4.1f} s", transform=axA.transAxes,
                 fontsize=10, fontweight="bold")

        # ---------- belief panel ----------
        axB.set_xlim(0, 1.0); axB.set_ylim(-0.5, 4.2)
        axB.set_yticks([0, 1, 2, 3])
        axB.set_yticklabels(["rho  (silence)", "eta  (channel health)",
                             "beta = rho*eta", "decision"], fontsize=9)
        axB.set_xlabel("value", fontsize=9)
        axB.set_title("Channel-aware belief", fontsize=11, fontweight="bold")

        bars = [("rho", r["rho"], "#7e57c2", 0),
                ("eta", r["eta"], "#26a69a", 1),
                ("beta", r["beta"], "#1565c0", 2)]
        for _, val, col, y in bars:
            axB.barh(y, val, height=0.55, color=col, edgecolor="k")
            axB.text(min(val + 0.02, 0.92), y, f"{val:0.2f}", va="center",
                     fontsize=9, fontweight="bold")
        # tau threshold line
        axB.axvline(TAU, color=COL_FAIL, ls="--", lw=1.8)
        axB.text(TAU + 0.01, 3.7, f"tau = {TAU:0.2f}", color=COL_FAIL,
                 fontsize=9, fontweight="bold")

        # decision row
        fired = r["took_over_at"] is not None
        dtxt = ("ACT: " + scn.takeover_label) if fired else "HOLD (beta < tau)"
        dcol = COL_OK if fired else COL_HOLD
        axB.barh(3, 0.0, height=0.001)  # keep scaling
        axB.text(0.02, 3.0, dtxt, color="white", fontsize=9.5,
                 fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc=dcol, ec="k"))

        # per-node storm counter (the contrast)
        axB.text(0.02, -0.35,
                 f"per-node (phi-accrual) takeovers fired: {r['pernode_count']}"
                 f"   |   ours: {1 if fired else 0}",
                 fontsize=8.5, color="#444")

        # caption (figure-level, inside the bottom margin)
        if getattr(fig, "_cap", None) is not None:
            fig._cap.remove()
        fig._cap = fig.text(0.04, 0.02, scn.caption, fontsize=8.6,
                            color="#222", va="bottom", ha="left")
        fig.subplots_adjust(bottom=0.20, top=0.92, left=0.04, right=0.98)
        return []

    anim = FuncAnimation(fig, draw, frames=n, interval=1000 / FPS, blit=False)
    path = os.path.join(OUT, scn.name + ".gif")
    anim.save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print("wrote", os.path.relpath(path))
    return path


if __name__ == "__main__":
    for scn in make_scenarios():
        render(scn)
    print("done ->", os.path.relpath(OUT))
