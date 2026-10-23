#!/usr/bin/env python3
"""Figure 5 of the paper: Eq. (3) validated against HoloOcean dynamics.
Reads the CSVs written by visualization/holoocean_eq3_validation.py."""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "holoocean_eq3")
LAM = 150.0


def rd(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else None


def build(out):
    resp, tim = rd("eq3_response.csv"), rd("eq3_time.csv")
    fig, ax = plt.subplots(1, 4, figsize=(19.4, 4.5))

    # (a) response curve: where the linearisation holds
    c = np.array([float(r["current"]) for r in resp])
    th = np.array([float(r["theta_s"]) for r in resp])
    m = np.abs(c) <= 1.0
    slope = np.linalg.lstsq(np.c_[c[m]], th[m], rcond=None)[0][0]
    ax[0].plot(c, th, "o-", color="#1f77b4", lw=1.8, ms=5, label="HoloOcean AUV")
    ax[0].plot(c, slope * c, "--", color="#d62728", lw=1.7,
               label="Eq. 3: linear in $u$")
    ax[0].axvspan(-0.3, 0.3, color="#2ca02c", alpha=0.14)
    ax[0].annotate("linear\nregime", (0, 0), xytext=(0, 60), fontsize=8,
                   ha="center", color="#2ca02c")
    ax[0].set_xlabel("along-track current (m/s)")
    ax[0].set_ylabel(r"arrival offset $\theta$ (s)")
    ax[0].set_title("(a) the $u$ factor FAILS at scale:\n"
                    "head currents cost what tail currents cannot save",
                    fontsize=9.8)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

    # (b) theta proportional to exposure time
    for cc, col in ((-1.0, "#d62728"), (-0.5, "#ff7f0e"),
                    (0.5, "#1f77b4"), (1.0, "#2ca02c")):
        sub = [r for r in tim if abs(float(r["current"]) - cc) < 1e-9]
        t0 = [float(r["t0_s"]) for r in sub]
        rat = [abs(float(r["theta_over_t0"])) for r in sub]
        ax[1].plot(t0, rat, "o-", color=col, lw=1.6, ms=5,
                   label="c = %+.1f m/s" % cc)
    ax[1].set_xlabel("exposure time  $t$  (s)")
    ax[1].set_ylabel(r"$|\theta| / t$")
    ax[1].set_ylim(0, 0.22)
    ax[1].set_title("(b) the $t$ factor HOLDS: $\\theta\\propto t$\n"
                    "flat lines = constant ratio over a 6$\\times$ range",
                    fontsize=9.8)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)

    # (c) correlation inheritance, both amplitudes
    for amp, col, mk in (("0.30", "#2ca02c", "o"), ("1.00", "#9467bd", "s")):
        cr = rd("eq3_field_corr_A%s.csv" % amp)
        if not cr:
            continue
        d = [float(r["d_m"]) for r in cr]
        ax[2].plot(d, [float(r["corr_theta"]) for r in cr], mk, color=col, ms=7,
                   label=r"Corr($\theta_i,\theta_j$), A=%s" % amp)
        ax[2].plot(d, [float(r["corr_u"]) for r in cr], "--", color=col, lw=1.2,
                   alpha=0.75, label="realised Corr($u_i,u_j$), A=%s" % amp)
    dd = np.linspace(0, 330, 100)
    ax[2].plot(dd, np.exp(-dd / LAM), "-", color="#d62728", lw=2.0,
               label=r"Eq. 3: $\rho(d)=e^{-d/\lambda}$")
    ax[2].set_xlabel("locale separation  $d$  (m)")
    ax[2].set_ylabel("correlation")
    ax[2].set_title("(c) the $u$ factor's STRUCTURE holds:\n"
                    "offsets inherit the field's correlation", fontsize=9.8)
    ax[2].legend(fontsize=7.2); ax[2].grid(alpha=0.25)

    # (d) the quantity Prop. 1 actually uses
    for amp, col, mk in (("0.30", "#2ca02c", "o"), ("1.00", "#9467bd", "s")):
        vr = rd("eq3_field_var_A%s.csv" % amp)
        if not vr:
            continue
        d = [float(r["d_m"]) for r in vr]
        ax[3].plot(d, [float(r["var_measured"]) for r in vr], mk, color=col,
                   ms=7, label="measured, A=%s" % amp)
        ax[3].plot(d, [float(r["var_eq3"]) for r in vr], "-", color=col, lw=1.6,
                   alpha=0.8, label="Eq. 3 prediction, A=%s" % amp)
    ax[3].set_xlabel("locale separation  $d$  (m)")
    ax[3].set_ylabel(r"Var($\theta_i-\theta_j$)  (s$^2$)")
    ax[3].set_yscale("log")
    ax[3].set_title("(d) Prop. 1's variance:\n"
                    r"$\kappa^2(t_i^2+t_j^2-2t_it_j\rho(d))$", fontsize=9.8)
    ax[3].legend(fontsize=7.2); ax[3].grid(alpha=0.25, which="both")

    fig.suptitle("Fig. 5  Eq. (3) against HoloOcean rigid-body hydrodynamics "
                 "(HoveringAUV, built-in PID waypoint controller, "
                 "set_ocean_currents) — the structure survives, the linearity "
                 "does not", fontsize=11.6, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    build(os.path.join(RES, "fig_eq3_validation.png"))
