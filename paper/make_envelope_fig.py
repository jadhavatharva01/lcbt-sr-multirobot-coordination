#!/usr/bin/env python3
"""Fig. 6: the admissible-mission envelope, and the absorption floor."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "visualization"))
OUT = os.path.join(HERE, "..", "results", "taskalloc")

# measured with taskalloc_coverage_first.sweep_envelope(model="measured")
POL = [("unaligned mesh", 38.7, [0.81, 3.83, 6.65, 9.36]),
       ("absorb +60 s", 56.0, [0.01, 0.58, 1.48, 2.92]),
       ("absorb +120 s", 94.5, [0.00, 0.16, 0.81, 1.65]),
       ("absorb +240 s", 224.4, [0.00, 0.07, 0.36, 0.86]),
       ("C4 corridor", 56.0, [0.00, 0.01, 0.06, 0.16]),
       ("crossings h=1", 35.4, [0.00, 0.00, 0.21, 0.61]),
       ("crossings h=4", 62.9, [0.00, 0.02, 0.10, 0.22])]
SIG = [1.0, 2.0, 3.0, 4.0]
DL = [1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.5, 3.5]
FA_T = 1.0


def build(out):
    fig = plt.figure(figsize=(15.6, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.0], wspace=0.30)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # --- A: the envelope ----------------------------------------------------
    code = np.zeros((len(DL), len(SIG)))
    for i, D in enumerate(DL):
        for j, sx in enumerate(SIG):
            adm = [p for p, ov, fa in POL
                   if (1 + ov / 100.0) <= D and fa[j] <= FA_T]
            others = [p for p in adm if not p.startswith("crossings")]
            if not adm:
                code[i, j] = 0
            elif not others:
                code[i, j] = 1          # only our construction
            else:
                code[i, j] = 2
    cmap = matplotlib.colors.ListedColormap(["#c9ccd1", "#2ca02c", "#bcd8ef"])
    axA.imshow(code, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto", origin="lower")
    for i, D in enumerate(DL):
        for j, sx in enumerate(SIG):
            n = sum(1 for p, ov, fa in POL
                    if (1 + ov / 100.0) <= D and fa[j] <= FA_T)
            txt = "none" if n == 0 else ("ONLY\ncrossings" if code[i, j] == 1
                                         else "%d policies" % n)
            axA.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                     fontweight="bold" if code[i, j] == 1 else "normal",
                     color="white" if code[i, j] == 1 else "#222")
    axA.set_xticks(range(len(SIG))); axA.set_xticklabels(["%g×" % s for s in SIG])
    axA.set_yticks(range(len(DL))); axA.set_yticklabels(["%.2g×" % d for d in DL])
    axA.set_xlabel("disturbance strength  σ/ε")
    axA.set_ylabel("deadline  (× no-rendezvous makespan)")
    axA.set_title("(a) which policies can fly the mission?\n"
                  "100%% coverage + cadence, deadline, false alarms <= %g%%"
                  % FA_T, fontsize=10)
    axA.legend(handles=[Patch(fc="#2ca02c", label="only natural crossings"),
                        Patch(fc="#bcd8ef", label="several policies work"),
                        Patch(fc="#c9ccd1", label="no policy admissible")],
               fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)

    # --- B: the absorption floor -------------------------------------------
    ab = [(38.7, 9.36), (56.0, 2.92), (94.5, 1.65), (224.4, 0.86)]
    axB.plot([a for a, _ in ab], [b for _, b in ab], "o-", color="#d62728",
             lw=1.9, ms=6, label="absorption (arrive early)")
    axB.axhline(0.82, ls=":", color="#d62728", lw=1.5)
    axB.text(120, 0.95, "floor 0.82%: past +240 s the\nreserved slack strands work "
             "(Prop. 3)", fontsize=8, color="#d62728")
    for nm, ov, fa in POL:
        if nm.startswith(("C4", "crossings")):
            axB.plot([ov], [max(fa[3], 1e-2)], "*", ms=15,
                     color="#2ca02c" if "crossings" in nm else "#1f77b4")
            axB.annotate(nm, (ov, max(fa[3], 1e-2)), fontsize=7.6,
                         textcoords="offset points", xytext=(6, 6))
    axB.set_yscale("log"); axB.set_xlabel("makespan over no-rendezvous plan (%)")
    axB.set_ylabel("benign false alarms (%), measured response")
    axB.set_title("(b) absorption has a floor; alignment does not\n"
                  "(σ/ε = 4)", fontsize=10)
    axB.legend(fontsize=8); axB.grid(alpha=0.25, which="both")

    # --- C: why the Gaussian model hid it -----------------------------------
    names = ["unaligned", "absorb\n+60 s", "absorb\n+120 s", "C4\ncorridor",
             "crossings\nh=1"]
    gaus = [10.70, 0.48, 0.02, 0.00, 0.00]
    meas = [9.36, 2.92, 1.65, 0.16, 0.61]
    x = np.arange(len(names))
    axC.bar(x - 0.2, np.maximum(gaus, 1e-2), 0.4, color="#9ecae1",
            label="Eq. 3 as written (Gaussian)")
    axC.bar(x + 0.2, np.maximum(meas, 1e-2), 0.4, color="#d62728",
            label="measured response (HoloOcean)")
    axC.set_yscale("log"); axC.set_xticks(x); axC.set_xticklabels(names, fontsize=8)
    axC.set_ylabel("benign false alarms (%)")
    axC.set_title("(c) the Gaussian model flattered absorption\n"
                  "by up to 80×, and only absorption", fontsize=10)
    axC.legend(fontsize=8); axC.grid(alpha=0.25, axis="y", which="both")

    fig.suptitle("Fig. 6  The admissible-mission envelope: where cheap alignment "
                 "is not an economy but the only option", fontsize=12,
                 fontweight="bold")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    build(os.path.join(OUT, "fig_envelope.png"))
