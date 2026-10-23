#!/usr/bin/env python3
"""
Rendezvous pipeline -> clean 2-D top-down EXPLAINER.

Purpose: understand the mechanism, not admire a 3-D scene. It reuses the exact
same plan as the HoloOcean driver (build_plan) but renders it top-down with
ROLE-BASED colours (only three actors matter) and plain-language captions:

    grey/blue  = ordinary survivor USV (stays on its own plan)
    RED        = the USV that fails (the victim)
    GREEN      = a taker -- a survivor that takes over orphaned work

    POI light-grey = not searched yet
    POI blue       = searched by its owner
    POI orange     = ORPHAN (victim's leftover, awaiting takeover)
    POI green      = orphan RECOVERED by a taker
    gold ring      = a rendezvous (meeting) happening now
    red ring       = the MISSED rendezvous = the moment of detection

Outputs (into results/holoocean_rendezvous/):
    explainer_storyboard.png   6 captioned panels: deploy -> coverage ->
                               failure -> detection -> recovery -> done
    explainer.gif              the same story, animated and slow

Run:
    python3 rendezvous_explainer.py                 # both, default scenario
    python3 rendezvous_explainer.py --no_gif        # storyboard only (fast)
    python3 rendezvous_explainer.py --num_pois 45 --victim 4 --t_fail 330
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rendezvous_holoocean_sim import build_plan, _pkey, OUT  # noqa: E402
from encounter_planning_probe import pos_at  # noqa: E402
from full_pipeline_sim import DELTA, EPS  # noqa: E402

# ------------------------------------------------------------ role colours
SURV   = "#5b7fa6"      # ordinary survivor
VICT   = "#e03131"      # the victim
TAKER  = "#2f9e44"      # a taker (rescuer)
PEND   = "#d0d0d0"      # POI not searched
DONE   = "#4c8bf5"      # POI searched by owner
ORPH   = "#f08c00"      # POI orphaned, awaiting takeover
RECOV  = "#37b24d"      # POI recovered by a taker
RDV    = "#f2c33c"      # rendezvous ring
MISS   = "#e03131"      # missed rendezvous ring


def make_args(a):
    """Build the plan; also return deploy geometry so the explainer can show
    the fleet launching from one dock and fanning out."""
    P = build_plan(a)
    K = P["mesh"]["K"]
    T_dep = max(1e-6, a.deploy_time)
    dock = (a.L / 2.0, a.dock_y)
    route_start = {r: pos_at(P["replay"][r], 0.0) for r in range(K)}
    cols = int(math.ceil(math.sqrt(K)))
    sp = max(a.dock_spread, 12.0)
    dock_pos = {}
    for r in range(K):
        rr, cc = divmod(r, cols)
        dock_pos[r] = (dock[0] + (cc - (cols - 1) / 2.0) * sp, dock[1] - rr * sp)
    P.update(T_dep=T_dep, dock=dock, route_start=route_start, dock_pos=dock_pos)
    return P


def rpos(P, r, t):
    """USV r position at time t; t<0 is the dock fan-out phase."""
    if t < 0.0:
        a = max(0.0, min(1.0, (t + P["T_dep"]) / P["T_dep"]))
        dx, dy = P["dock_pos"][r]
        rx, ry = P["route_start"][r]
        return dx + a * (rx - dx), dy + a * (ry - dy)
    return pos_at(P["replay"][r], t)


def poi_col(P, k, t):
    scan_t, t_f, orphan_keys = P["scan_t"], P["t_f"], P["orphan_keys"]
    searched = scan_t[k] is not None and scan_t[k] <= t
    is_orphan = k in orphan_keys
    if searched:
        return RECOV if is_orphan else DONE
    if is_orphan and t >= t_f:
        return ORPH
    return PEND


def robot_color(P, r, t):
    if r == P["victim"]:
        return VICT
    first_det = P["fail"]["first_det"]
    if r in P["takers"] and first_det is not None and t >= first_det:
        return TAKER
    return SURV


def draw_scene(ax, P, t):
    """Render the whole field at time t (mesh coords) onto ax."""
    mesh, victim, t_f = P["mesh"], P["victim"], P["t_f"]
    first_det = P["fail"]["first_det"]
    L, K = P["mesh"]["L"], mesh["K"]

    # POIs (small dots)
    for k, p in P["pois"].items():
        ax.plot(p[0], p[1], "o", ms=5, color=poi_col(P, k, t),
                zorder=2, mec="white", mew=0.4)

    # rendezvous rings (gold) + the missed one (red) around this instant
    for ap in mesh["appts"]:
        involves_victim = victim in (ap["a"], ap["b"])
        if not involves_victim and abs(t - ap["t"]) < 0.6 * DELTA and t >= 0:
            ax.add_patch(plt.Circle(ap["pt"], 22, fill=False, ec=RDV,
                                    lw=2.2, zorder=3))
        if involves_victim and ap["t"] > t_f and abs(t - (ap["t"] + 30)) < 0.6 * DELTA:
            ax.add_patch(plt.Circle(ap["pt"], 30, fill=False, ec=MISS,
                                    lw=2.6, ls="--", zorder=4))

    # per-USV route line (only the not-yet-searched POIs), in the role colour
    for r in range(K):
        if r == victim and t >= t_f:
            continue
        col = robot_color(P, r, t)
        cx, cy = rpos(P, r, t)
        xs, ys = [cx], [cy]
        for p in mesh["routes"][r]:
            k = _pkey(p)
            if P["scan_t"][k] is not None and P["scan_t"][k] <= t:
                continue
            xs.append(p[0])
            ys.append(p[1])
        ax.plot(xs, ys, "-", color=col, lw=1.0, alpha=0.45, zorder=1)

    # takers' takeover arrows -> their not-yet-recovered orphans
    for r in P["takers"]:
        if first_det is None or t < first_det:
            continue
        cx, cy = rpos(P, r, t)
        from rendezvous_holoocean_sim import _TAKER_ORPHANS
        for p in _TAKER_ORPHANS.get(r, []):
            k = _pkey(p)
            if not (P["scan_t"][k] is not None and P["scan_t"][k] <= t):
                ax.annotate("", xy=(p[0], p[1]), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle="->", color=TAKER,
                                            lw=1.6, alpha=0.8), zorder=5)

    # the USVs themselves
    for r in range(K):
        cx, cy = rpos(P, r, t)
        col = robot_color(P, r, t)
        if r == victim and t >= t_f:
            ax.plot(cx, cy, "x", ms=13, mew=3.5, color=VICT, zorder=7)
            ax.text(cx + 12, cy + 12, f"U{r} DEAD", color=VICT, fontsize=8,
                    fontweight="bold", zorder=8)
        else:
            ax.plot(cx, cy, "o", ms=11, color=col, mec="white", mew=1.2, zorder=7)
            ax.text(cx + 10, cy + 10, f"U{r}", color=col, fontsize=7.5,
                    fontweight="bold", zorder=8)

    ax.set_xlim(-30, L + 30)
    ax.set_ylim(P["dock"][1] - 40, L + 30)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def legend_handles():
    return [
        Line2D([0], [0], marker="o", color="w", mfc=SURV, ms=9, label="survivor USV"),
        Line2D([0], [0], marker="o", color="w", mfc=VICT, ms=9, label="victim USV"),
        Line2D([0], [0], marker="o", color="w", mfc=TAKER, ms=9, label="taker USV"),
        Line2D([0], [0], marker="o", color="w", mfc=PEND, ms=9, label="POI pending"),
        Line2D([0], [0], marker="o", color="w", mfc=DONE, ms=9, label="POI searched"),
        Line2D([0], [0], marker="o", color="w", mfc=ORPH, ms=9, label="POI orphaned"),
        Line2D([0], [0], marker="o", color="w", mfc=RECOV, ms=9, label="POI recovered"),
        Line2D([0], [0], marker="o", color="w", mec=RDV, mfc="w", ms=11,
               label="rendezvous"),
        Line2D([0], [0], marker="o", color="w", mec=MISS, mfc="w", ms=11,
               label="missed (detected)"),
    ]


def _approach(P, r, t_end, window, npts=26):
    """Sampled trajectory of USV r over [t_end-window, t_end] -- shows how it
    travels IN to a rendezvous point (from its true space-time plan)."""
    return [pos_at(P["replay"][r], t_end - window + window * i / (npts - 1))
            for i in range(npts)]


def mechanism_figure(P, a):
    """Two zoomed panels that answer 'how do they know?': a KEPT rendezvous
    (both partners travel to the agreed point and meet) vs the MISSED one after
    the victim dies (the survivor keeps the appointment, the victim never comes
    -> that no-show is the detection). No path-crossing luck involved."""
    mesh, victim, t_f = P["mesh"], P["victim"], P["t_f"]
    appts = mesh["appts"]
    # Panel A: a normal rendezvous between two survivors (both always alive).
    generic = sorted((ap for ap in appts if victim not in (ap["a"], ap["b"])),
                     key=lambda x: x["t"])
    # Panel B: the victim's first meeting after death -> the missed one.
    missed = sorted((ap for ap in appts if victim in (ap["a"], ap["b"])
                     and ap["t"] > t_f), key=lambda x: x["t"])
    if not generic or not missed:
        print("mechanism: no suitable rendezvous to illustrate -- adjust scenario.")
        return None
    ka, ma = generic[0], missed[0]
    win = mesh["delta"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 7.6))

    # ---- Panel A: a rendezvous that HAPPENS -------------------------------
    a_i, a_j = ka["a"], ka["b"]
    pts = []
    for r in (a_i, a_j):
        col = VICT if r == victim else SURV
        tr = _approach(P, r, ka["t"], win)
        pts += tr
        axA.plot([p[0] for p in tr], [p[1] for p in tr], "-", color=col, lw=2.2,
                 alpha=0.9, zorder=3)
        axA.annotate("", xy=ka["pt"], xytext=tr[-3],
                     arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2), zorder=4)
        st = tr[0]
        axA.plot(*st, "o", ms=8, color=col, mec="white", mew=1, zorder=5)
        axA.text(st[0], st[1] + 6, f"U{r}", color=col, fontsize=10,
                 fontweight="bold", ha="center")
    axA.add_patch(plt.Circle(ka["pt"], 10, fill=False, ec=RDV, lw=3, zorder=6))
    axA.plot(*ka["pt"], "o", ms=13, color=RDV, mec="black", mew=0.8, zorder=7)
    axA.set_title(f"A rendezvous that HAPPENS  (t = {ka['t']:.0f} s)",
                  fontsize=13, fontweight="bold")
    axA.set_xlabel("Both U{} and U{} travel to the SAME pre-agreed point at the\n"
                   "SAME scheduled time, meet (a mutual heartbeat), then carry on."
                   .format(a_i, a_j), fontsize=10.5)

    # ---- Panel B: the rendezvous that is MISSED = detection ----------------
    surv = ma["b"] if ma["a"] == victim else ma["a"]
    death_pos = pos_at(P["replay"][victim], t_f)
    tr = _approach(P, surv, ma["t"] + EPS, win)
    ptsB = tr + [death_pos, ma["pt"]]
    axB.plot([p[0] for p in tr], [p[1] for p in tr], "-", color=SURV, lw=2.2,
             zorder=3)
    axB.annotate("", xy=ma["pt"], xytext=tr[-3],
                 arrowprops=dict(arrowstyle="-|>", color=SURV, lw=2.2), zorder=4)
    axB.plot(*tr[0], "o", ms=8, color=SURV, mec="white", mew=1, zorder=5)
    axB.text(tr[0][0], tr[0][1] + 6, f"U{surv}", color=SURV, fontsize=10,
             fontweight="bold", ha="center")
    # survivor waiting AT the agreed point; red ring = waited eps, no-show
    axB.plot(*ma["pt"], "o", ms=11, color=SURV, mec="white", mew=1.2, zorder=6)
    axB.add_patch(plt.Circle(ma["pt"], 12, fill=False, ec=MISS, lw=3,
                             ls="--", zorder=7))
    axB.annotate("agreed point\n(survivor waits)", xy=ma["pt"],
                 xytext=(ma["pt"][0] - 22, ma["pt"][1] + 40), color=MISS,
                 fontsize=9, ha="right", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=MISS, lw=0.8))
    # the dead victim, frozen far away, that never comes
    axB.plot(*death_pos, "x", ms=15, mew=4, color=VICT, zorder=6)
    axB.text(death_pos[0], death_pos[1] - 34, f"U{victim} DEAD\n(frozen here)",
             color=VICT, fontsize=9.5, ha="center", fontweight="bold")
    axB.annotate("", xy=ma["pt"], xytext=death_pos,
                 arrowprops=dict(arrowstyle="->", color=VICT, lw=1.6, ls=":",
                                 alpha=0.8), zorder=2)
    axB.text((death_pos[0] + ma["pt"][0]) / 2, (death_pos[1] + ma["pt"][1]) / 2 - 8,
             "never arrives", color=VICT, fontsize=9, style="italic", ha="center")
    axB.set_title(f"The rendezvous that is MISSED = DETECTION  (t = {ma['t']:.0f} s)",
                  fontsize=13, fontweight="bold")
    axB.set_xlabel(f"U{surv} keeps the appointment and waits {EPS:.0f} s. U{victim} "
                   f"is dead\nelsewhere and never comes. The no-show IS the alarm "
                   f"-- no radio.", fontsize=10.5)

    for ax, pp in ((axA, pts), (axB, ptsB)):
        xs = [p[0] for p in pp]
        ys = [p[1] for p in pp]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mx = max(40, 0.15 * (x1 - x0 + 1))
        my = max(40, 0.15 * (y1 - y0 + 1))
        x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my
        # avoid an extreme letterbox: keep height >= 0.5 * width
        xspan, yspan = x1 - x0, y1 - y0
        if yspan < 0.5 * xspan:
            cy = 0.5 * (y0 + y1)
            y0, y1 = cy - 0.25 * xspan, cy + 0.25 * xspan
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("How a robot knows another failed -- with zero communication",
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.16, wspace=0.10)
    out = OUT / "explainer_mechanism.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"mechanism -> {out}")
    return out


def storyboard(P, a):
    victim = P["victim"]
    t_f = P["t_f"]
    fd = P["fail"]["first_det"]
    span = P["span"]
    lat = P["fail"]["latency"]
    res = P["res"]
    fd_or = fd if fd is not None else t_f + DELTA

    panels = [
        (-0.5 * P["T_dep"],
         "1. DEPLOY",
         f"All {P['mesh']['K']} USVs launch from one dock and fan out to their "
         f"sectors. Each keeps a precomputed schedule of rendezvous with its "
         f"neighbours."),
        (0.6 * t_f,
         "2. COVERAGE",
         "Everyone searches its own POIs (turning blue) and keeps its meetings "
         "(gold rings). No radios needed -- the plan is common knowledge."),
        (t_f,
         "3. FAILURE",
         f"USV {victim} dies at t={t_f:.0f}s. It stops and goes silent. Its "
         f"un-searched POIs are now ORPHANS (orange)."),
        (fd_or,
         "4. DETECTION (no comms)",
         f"A neighbour shows up for its scheduled rendezvous with USV {victim} "
         f"and no one is there (red ring). That missed meeting = detection, "
         f"{(lat if lat else 0):.0f}s after death."),
        (0.5 * (fd_or + span),
         "5. RECOVERY (precomputed)",
         f"By table lookup, {len(P['takers'])} takers (green) drive to the "
         f"orphans (arrows) while still keeping their OWN meetings. Orphans turn "
         f"green as they are recovered."),
        (span,
         "6. COMPLETE",
         f"Every POI covered. {res['appts_kept_pct']:.0f}% of post-failure "
         f"meetings kept, all {res['n_orphans']} orphans recovered -- with zero "
         f"communication."),
    ]

    import textwrap
    fig, axes = plt.subplots(2, 3, figsize=(18, 14))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.10,
                        wspace=0.10, hspace=0.34)
    for ax, (t, title, cap) in zip(axes.ravel(), panels):
        draw_scene(ax, P, t)
        ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
        ax.set_xlabel(textwrap.fill(cap, 58), fontsize=10.5, labelpad=10,
                      linespacing=1.4)
    fig.suptitle("Rendezvous as a comms-free failure detector + recovery",
                 fontsize=18, fontweight="bold", y=0.965)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=9,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.02))
    out = OUT / "explainer_storyboard.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"storyboard -> {out}")
    return out


def phase_caption(P, t):
    t_f, fd = P["t_f"], P["fail"]["first_det"]
    if t < 0:
        return "DEPLOY -- fleet fans out from the dock", "#333"
    if t < t_f:
        return "COVERAGE -- each USV searches + keeps its rendezvous", SURV
    if fd is not None and t < fd:
        return f"USV {P['victim']} HAS FAILED -- not yet detected", VICT
    if fd is not None and t < fd + DELTA:
        return "DETECTED via a missed rendezvous -- recovery begins", VICT
    if t < P["span"]:
        return "RECOVERY -- takers absorb the orphaned POIs", TAKER
    return "COMPLETE -- all POIs covered, comms-free", TAKER


def animate(P, a):
    span, T_dep = P["span"], P["T_dep"]
    t_f, fd = P["t_f"], P["fail"]["first_det"]
    n = a.frames
    base = [(-T_dep) + (span + T_dep) * i / (n - 1) for i in range(n)]
    # dwell (repeat frames) on the key moments so they don't blip past
    keys = {min(range(n), key=lambda i: abs(base[i] - k))
            for k in (t_f, fd, span) if k is not None}
    times = []
    for i in range(n):
        times.append(base[i])
        if i in keys:
            times += [base[i]] * a.dwell
    fig, ax = plt.subplots(figsize=(9, 9))

    def render(t):
        ax.clear()
        draw_scene(ax, P, t)
        cap, col = phase_caption(P, t)
        ax.set_title(f"t = {max(t,0):5.0f}s   |   {cap}", fontsize=12,
                     fontweight="bold", color=col)

    from matplotlib.animation import FuncAnimation, PillowWriter
    anim = FuncAnimation(fig, lambda i: render(times[i]), frames=len(times))
    out = OUT / "explainer.gif"
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=a.fps))
    plt.close(fig)
    print(f"animation -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="2-D top-down rendezvous explainer.")
    ap.add_argument("--Gr", type=int, default=3)
    ap.add_argument("--Gc", type=int, default=3)
    ap.add_argument("--num_pois", type=int, default=45)
    ap.add_argument("--L", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--victim", type=int, default=None)
    ap.add_argument("--t_fail", type=float, default=None)
    ap.add_argument("--deploy_time", type=float, default=120.0)
    ap.add_argument("--dock_y", type=float, default=-40.0)
    ap.add_argument("--dock_spread", type=float, default=14.0)
    ap.add_argument("--frames", type=int, default=160, help="GIF frame count")
    ap.add_argument("--fps", type=int, default=12, help="GIF frames per second")
    ap.add_argument("--dwell", type=int, default=10,
                    help="frames to hold on failure/detection/finish moments")
    ap.add_argument("--no_gif", action="store_true", help="storyboard only")
    a = ap.parse_args()

    P = make_args(a)
    print(f"Plan: victim U{P['victim']} @ t={P['t_f']:.0f}s, "
          f"{P['res']['n_orphans']} orphans, takers {sorted(P['takers'])}, "
          f"detection latency "
          f"{P['fail']['latency']:.0f}s" if P['fail']['latency'] else "n/a")
    mechanism_figure(P, a)
    storyboard(P, a)
    if not a.no_gif:
        animate(P, a)


if __name__ == "__main__":
    main()
