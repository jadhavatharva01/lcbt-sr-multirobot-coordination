#!/usr/bin/env python3
"""Theory and architecture figures for the aligned-rendezvous paper.

fig_alignment.png : the admissible alignment set in the (approach duration,
                    approach separation) plane, the two corners, and how the
                    relaxation grows as the alignment budget tightens.
fig_bt.png        : the per-robot behaviour tree that executes the frozen mesh.
Run: python3 paper/make_paper_figs.py
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.optimize import brentq
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "taskalloc")
EPS, V, ELL, TREF = 30.0, 2.0, 150.0, 300.0


def budget(kappa, alpha):
    return EPS ** 2 / (kappa ** 2 * norm.ppf(1 - alpha / 2) ** 2)


def d_c4(B, T=TREF):
    arg = 1 - B / (2 * T * T)
    return -ELL * math.log(arg) if arg > 0 else float("inf")


def s_cross(B):
    f = lambda s: 2 * (s / (2 * V)) ** 2 * (1 - math.exp(-s / (2 * ELL))) - B
    return brentq(f, 1e-9, 1e5) if f(1e5) > 0 else float("inf")


def fig_alignment(path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    kappa, alpha = 4.0 * EPS / TREF, 0.2
    B = budget(kappa, alpha)

    # --- A: the admissible set in (t, d) ------------------------------------
    t = np.linspace(1, 320, 700)
    d = np.linspace(0.3, 320, 700)
    TT, DD = np.meshgrid(t, d)
    varfac = 2 * TT ** 2 * (1 - np.exp(-DD / ELL))
    axA.contourf(TT, DD, (varfac <= B).astype(float), levels=[0.5, 1.5],
                 colors=["#2ca02c"], alpha=0.18)
    axA.contour(TT, DD, varfac, levels=[B], colors=["#2ca02c"], linewidths=2.0)
    dc = d_c4(B)
    axA.plot([TREF], [dc], "D", color="#d62728", ms=9, zorder=5)
    axA.annotate("separation-only corner (C4)\nt fixed at T = %.0f s  ->  d* = %.1f m"
                 % (TREF, dc), (TREF, dc), textcoords="offset points",
                 xytext=(-18, 42), ha="right", fontsize=8.6, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
    ss = np.linspace(1, 640, 400)
    axA.plot(ss / (2 * V), ss / 2, "--", color="#1f77b4", lw=1.8,
             label="crossing locus  t = s/2V,  d = s/2")
    sx = s_cross(B)
    axA.plot([sx / (2 * V)], [sx / 2], "*", color="#1f77b4", ms=17, zorder=5)
    axA.annotate("crossing corner\ns* = %.0f m  ->  d = %.0f m" % (sx, sx / 2),
                 (sx / (2 * V), sx / 2), textcoords="offset points",
                 xytext=(10, -26), fontsize=8.6, color="#1f77b4")
    axA.set_xlim(0, 320); axA.set_ylim(0, 320)
    axA.set_xlabel("approach duration  t  (s)")
    axA.set_ylabel("approach separation  d  (m)")
    axA.set_title("(a) admissible alignment set  2t$^2$(1-e$^{-d/\\lambda}$) $\\leq$ B\n"
                  "shaded = meets the target; the crossing locus enters it "
                  "%.0fx further out" % ((sx / 2) / dc), fontsize=10)
    axA.legend(fontsize=8.5, loc="upper right"); axA.grid(alpha=0.2)

    # --- B: relaxation vs how tight the budget is ---------------------------
    Bs, rat, dcs, sxs = [], [], [], []
    for sxf in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        for al in (0.4, 0.2, 0.1, 0.05, 0.02):
            Bv = budget(sxf * EPS / TREF, al)
            dd, s2 = d_c4(Bv), s_cross(Bv) / 2
            if not np.isfinite(dd) or dd <= 0:
                continue
            Bs.append(Bv); rat.append(s2 / dd); dcs.append(dd); sxs.append(s2)
    o = np.argsort(Bs)
    Bs, rat = np.array(Bs)[o], np.array(rat)[o]
    axB.loglog(Bs, rat, "o", color="#2ca02c", ms=5, label="exact ratio  d$_{cross}$/d*")
    ref = rat[len(rat) // 2] * (Bs / Bs[len(Bs) // 2]) ** (-2.0 / 3.0)
    axB.loglog(Bs, ref, "--", color="#555", lw=1.4, label="$B^{-2/3}$ reference slope")
    axB.set_xlabel("alignment budget  B = $\\epsilon^2/(\\kappa^2 z^2)$   (s$^2$)   "
                   "<- stronger current / stricter $\\alpha$")
    axB.set_ylabel("relaxation factor")
    axB.set_title("(b) the tighter the budget, the more the crossing corner wins\n"
                  "letting t co-vary with d turns $\\Theta$(B) into $\\Theta(B^{1/3})$",
                  fontsize=10)
    axB.legend(fontsize=9); axB.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _node(ax, xy, w, h, text, kind):
    face = {"seq": "#dbe7f5", "fb": "#fde6d2", "act": "#e3f2e1",
            "cond": "#f5f0da"}[kind]
    edge = {"seq": "#2c5f9e", "fb": "#c1671a", "act": "#2f7a2a",
            "cond": "#9a8b1f"}[kind]
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=face, ec=edge, lw=1.3, zorder=3))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=8.1,
            zorder=4, linespacing=1.25)


def _edge(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-", lw=1.1, color="#666",
                                 zorder=1))


def fig_bt(path):
    fig, ax = plt.subplots(figsize=(13.6, 6.0))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.0); ax.axis("off")
    Y0, Y1, Y2, Y3 = 5.45, 4.35, 3.05, 1.75

    _node(ax, (7.0, Y0), 2.6, 0.52, "? FALLBACK  (root)", "fb")
    l1 = [(2.1, "$\\rightarrow$ contingency", "seq", 2.3),
          (6.3, "$\\rightarrow$ appointment", "seq", 2.3),
          (10.3, "$\\rightarrow$ coverage work", "seq", 2.3),
          (13.0, "idle / dock", "act", 1.6)]
    for x, t, k, w in l1:
        _node(ax, (x, Y1), w, 0.52, t, k)
        _edge(ax, (7.0, Y0 - 0.26), (x, Y1 + 0.26))
    ax.text(7.0, Y1 + 0.66, "tick priority, left to right  =  the lexicographic "
            "objective of Eq. (2)", ha="center", fontsize=8.5, style="italic",
            color="#333", zorder=6,
            bbox=dict(fc="white", ec="none", pad=1.6))

    kids = [
        (2.1, [(1.1, 1.9, "PartnerMissed?\n(no show at $\\tau+\\epsilon$)", "cond"),
               (3.2, 2.0, "ApplyContingency\n$f(v,c)$ table lookup", "act")]),
        (6.3, [(4.9, 1.6, "Appointment\ndue within $\\Delta$?", "cond"),
               (6.6, 1.4, "? anchor", "fb"),
               (8.4, 1.8, "TravelToMeeting\n+ AwaitPartner($\\epsilon$)", "act")]),
        (10.3, [(10.2, 1.8, "FitsBefore\nDeadline?", "cond"),
                (12.2, 1.7, "ScanNearestPOI", "act")]),
    ]
    for px, ch in kids:
        for x, w, t, k in ch:
            _node(ax, (x, Y2), w, 0.62, t, k)
            _edge(ax, (px, Y1 - 0.26), (x, Y2 + 0.31))
    for x, w, t in ((5.9, 1.8, "ScanReservedAnchor\n(the crossing POI --\nalignment paid in WORK)"),
                    (7.9, 1.7, "DwellInCorridor\n(fallback --\nalignment paid in IDLE)")):
        _node(ax, (x, Y3), w, 0.78, t, "act")
        _edge(ax, (6.6, Y2 - 0.31), (x, Y3 + 0.39))

    ax.text(0.15, 1.02, "The tree is execution architecture, not a contribution: "
            "reactive fault recovery is the founding use case of behaviour trees. "
            "What matters is the ORDER it\nenforces -- recovery preempts the "
            "appointment, the appointment preempts coverage work "
            "(\"meeting-wins\"), and the anchor scan is tried before the idle "
            "dwell, so\nalignment is paid in work whenever the reservation "
            "survived and in idle only when it did not. Every branch is a lookup "
            "on the frozen mesh: no runtime negotiation.",
            fontsize=8.4, va="top", color="#333")
    ax.text(0.15, 0.28, "?  fallback (run children until one succeeds)      "
            "$\\rightarrow$  sequence (run all children in order)      "
            "green  action      yellow  condition",
            fontsize=8.0, va="top", color="#666", style="italic")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(fig_alignment(os.path.join(OUT, "fig_alignment.png")))
    print(fig_bt(os.path.join(OUT, "fig_bt.png")))
