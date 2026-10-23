#!/usr/bin/env python3
"""
Validate Eq. (3) of the aligned-rendezvous paper against HoloOcean physics.

Eq. (3) claims a robot's arrival offset at a rendezvous is
        theta_k = kappa * t_k * u(a_k),
i.e. LINEAR in the exposure time t_k of its final approach and LINEAR in a
unit-variance correlated disturbance field u sampled at the approach locale a_k,
so that for two partners
        Var(theta_i - theta_j) = kappa^2 ( t_i^2 + t_j^2 - 2 t_i t_j rho(d) ).
That formula is the whole basis of Proposition 1 and Theorem 1, and it was
written down by us; this script tests it against dynamics we did not write.

WHY THIS IS NOT CIRCULAR.  We never impose an effective-speed model.  We set a
genuine ocean-current VECTOR on the vehicle through HoloOcean's own
set_ocean_currents, let its rigid-body hydrodynamics and its built-in PID
waypoint controller do whatever they do, and measure the ACTUAL wall-clock
arrival time at a capture radius.  Drag, thrust saturation, heading dynamics and
controller behaviour are all outside Eq. (3) and are free to break it.

PLATFORM NOTE (measured, see --exp probe).  HoloOcean's SurfaceVessel is inert to
set_ocean_currents -- a 3 m/s current moves it 0.18 m in 20 s -- so the USV of
the paper cannot be used and we validate on the HoveringAUV, whose response is
physical (8.0 m of drift in 20 s under the same current).  The claim under test
is about a vehicle transiting a current field, which the AUV is.

Experiments:
  probe : platform check + the raw theta-vs-current response curve
  time  : is theta linear in exposure time t?            (Eq. 3, the t factor)
  field : is Corr(theta_i, theta_j) = rho(d) = exp(-d/L)? (Eq. 3, the u factor)
          and does Var(theta_i - theta_j) match the combined formula?
  all   : everything, then writes the figure + CSVs

Run inside the HoloOcean venv:
  ~/holoocean_env/bin/python visualization/holoocean_eq3_validation.py --exp all
"""
import argparse
import csv
import math
import os
from copy import deepcopy

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "results", "holoocean_eq3")

TPS = 50          # physics ticks per simulated second
DEPTH = -5.0      # AUV operating depth (m)
LEG = 60.0        # default leg length (m)
TOL = 2.0         # waypoint capture radius (m)
LAMBDA = 150.0    # disturbance correlation length (m) -- the paper's value


# ----------------------------------------------------------------- scenario --
def make_cfg(n_agents, agent_type="HoveringAUV", scheme=1, lanes=None):
    from holoocean import packagemanager
    cfg = deepcopy(packagemanager.get_scenario("OpenWater-Hovering"))
    cfg["package_name"] = "Ocean"
    cfg["name"] = "Eq3Validation"
    z = DEPTH if agent_type == "HoveringAUV" else 0.0
    agents = []
    for k in range(n_agents):
        y = 0.0 if lanes is None else float(lanes[k])
        agents.append({
            "agent_name": f"a{k}", "agent_type": agent_type,
            "sensors": [{"sensor_type": "LocationSensor"}],
            "location": [0.0, y, z], "rotation": [0.0, 0.0, 0.0],
            "control_scheme": scheme, "is_main_agent": (k == 0)})
    cfg["agents"] = agents
    cfg["main_agent"] = "a0"
    cfg["ticks_per_sec"] = TPS
    cfg["frames_per_sec"] = 0
    return cfg


def _state(st, name):
    """HoloOcean returns a flat state for a single agent and a name-keyed dict
    for several; normalise."""
    return st[name] if name in st else st


def _cmd(x, y):
    return np.array([x, y, DEPTH, 0.0, 0.0, 0.0], dtype=float)


def run_legs(env, n, lanes, leg, current_fn, max_t=600.0, tol=TOL):
    """Fly n AUVs down parallel +x legs at once, each feeling the current that
    `current_fn(k, pos)` returns at its own position. Returns arrival times."""
    env.reset()
    for k in range(n):
        env.agents[f"a{k}"].teleport([0.0, float(lanes[k]), DEPTH], [0, 0, 0])
    for _ in range(5):
        for k in range(n):
            env.act(f"a{k}", _cmd(leg, lanes[k]))
        env.tick()
    arrive = [None] * n
    nmax = int(max_t * TPS)
    for step in range(nmax):
        st = env.tick()
        t = (step + 1) / TPS
        done = True
        for k in range(n):
            name = f"a{k}"
            p = _state(st, name)["LocationSensor"]
            if arrive[k] is None:
                if abs(p[0] - leg) < tol:
                    arrive[k] = t
                else:
                    done = False
            env.set_ocean_currents(name, list(current_fn(k, p)))
            env.act(name, _cmd(leg, lanes[k]))
        if done:
            break
    return [a if a is not None else float("nan") for a in arrive]


# -------------------------------------------------------------- disturbance --
def gp_field(xs, ys, lam, rng):
    """A unit-variance Gaussian field on a grid with rho(d)=exp(-d/lam)."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()])
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    R = np.exp(-D / lam)
    for jit in (1e-8, 1e-6, 1e-4):
        try:
            L = np.linalg.cholesky(R + jit * np.eye(len(P)))
            break
        except np.linalg.LinAlgError:
            continue
    else:
        w, U = np.linalg.eigh(R)
        L = U @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
    return (L @ rng.standard_normal(len(P))).reshape(X.shape)


def bilinear(grid, xs, ys, x, y):
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
    tx = (x - xs[i]) / (xs[i + 1] - xs[i])
    ty = (y - ys[j]) / (ys[j + 1] - ys[j])
    tx, ty = np.clip(tx, 0, 1), np.clip(ty, 0, 1)
    return ((1 - tx) * (1 - ty) * grid[i, j] + tx * (1 - ty) * grid[i + 1, j]
            + (1 - tx) * ty * grid[i, j + 1] + tx * ty * grid[i + 1, j + 1])


def _save(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  wrote", os.path.abspath(path))


# ------------------------------------------------------------ experiment: probe
def exp_probe(args):
    """Platform check, then the raw response of arrival offset to a uniform
    current -- the shape Eq. (3) assumes to be a straight line through 0."""
    import holoocean
    print("[probe] platform check: does set_ocean_currents move the hull?")
    rows = []
    for atype, scheme, act in (("SurfaceVessel", 0, [0.0, 0.0]),
                               ("HoveringAUV", 0, [0.0] * 8)):
        env = holoocean.make(scenario_cfg=make_cfg(1, atype, scheme),
                             show_viewport=False, verbose=False)
        env.reset()
        for _ in range(5):
            env.act("a0", np.array(act)); env.tick()
        p0 = _state(env.tick(), "a0")["LocationSensor"].copy()
        for _ in range(20 * TPS):
            env.set_ocean_currents("a0", [3.0, 0.0, 0.0])
            env.act("a0", np.array(act)); st = env.tick()
        p1 = _state(st, "a0")["LocationSensor"].copy()
        env.__exit__(None, None, None)
        drift = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        print("  %-14s free drift under 3 m/s current, 20 s: %6.2f m  -> %s"
              % (atype, drift, "USABLE" if drift > 1.0 else "INERT, unusable"))
        rows.append(dict(agent=atype, drift_20s_m=round(drift, 3),
                         usable=drift > 1.0))
    _save(os.path.join(OUT, "eq3_platform.csv"), rows)

    print("[probe] response curve: arrival offset vs uniform along-track current")
    env = holoocean.make(scenario_cfg=make_cfg(1), show_viewport=False,
                         verbose=False)
    t0 = run_legs(env, 1, [0.0], LEG, lambda k, p: [0, 0, 0])[0]
    rows = []
    cs = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0,
          0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for c in cs:
        t = run_legs(env, 1, [0.0], LEG, lambda k, p, c=c: [c, 0, 0])[0]
        rows.append(dict(current=c, arrival_s=round(t, 3),
                         theta_s=round(t - t0, 3)))
        print("  current %+5.2f m/s -> arrival %7.2f s, theta %+8.2f s"
              % (c, t, t - t0), flush=True)
    env.__exit__(None, None, None)
    # local slope about zero (the kappa*t of Eq. 3, in seconds per unit current)
    sm = [r for r in rows if abs(r["current"]) <= 1.0]
    A = np.array([[r["current"], 1.0] for r in sm])
    b = np.array([r["theta_s"] for r in sm])
    slope, icpt = np.linalg.lstsq(A, b, rcond=None)[0]
    pred = A @ np.array([slope, icpt])
    ss = 1 - np.sum((b - pred) ** 2) / max(np.sum((b - b.mean()) ** 2), 1e-12)
    print("  nominal (no-current) transit t0 = %.2f s" % t0)
    print("  linear fit on |c| <= 1: theta = %+.3f * c   (R^2 = %.4f)"
          % (slope, ss))
    hi = [r["theta_s"] for r in rows if r["current"] == -2.0][0]
    lo = [r["theta_s"] for r in rows if r["current"] == 2.0][0]
    print("  ASYMMETRY at |c|=2: head %+.1f s vs tail %+.1f s (ratio %.1fx) "
          "-> Eq. 3 is a small-signal linearisation" % (hi, lo, abs(hi / lo)))
    _save(os.path.join(OUT, "eq3_response.csv"), rows)
    return dict(t0=t0, slope=float(slope), r2=float(ss), rows=rows)


# ------------------------------------------------------------- experiment: time
def exp_time(args):
    """Eq. (3) says theta is proportional to exposure time t. Fly legs of
    different length under the same current and test theta / t = const."""
    import holoocean
    print("[time] is theta proportional to exposure time?")
    legs = [20.0, 40.0, 60.0, 90.0, 120.0]
    rows = []
    for c in (-1.0, -0.5, 0.5, 1.0):
        for L in legs:
            env = holoocean.make(scenario_cfg=make_cfg(1), show_viewport=False,
                                 verbose=False)
            t0 = run_legs(env, 1, [0.0], L, lambda k, p: [0, 0, 0])[0]
            t = run_legs(env, 1, [0.0], L, lambda k, p, c=c: [c, 0, 0])[0]
            env.__exit__(None, None, None)
            rows.append(dict(current=c, leg_m=L, t0_s=round(t0, 3),
                             theta_s=round(t - t0, 3),
                             theta_over_t0=round((t - t0) / t0, 5)))
            print("  c=%+.1f  leg %5.1f m  t0 %6.2f s  theta %+7.2f s  "
                  "theta/t0 %+.4f" % (c, L, t0, t - t0, (t - t0) / t0),
                  flush=True)
    # a power law theta ~ t0^p : Eq. 3 predicts p = 1
    for c in (-1.0, -0.5, 0.5, 1.0):
        sub = [r for r in rows if r["current"] == c]
        lt = np.log([r["t0_s"] for r in sub])
        lth = np.log([abs(r["theta_s"]) for r in sub])
        p, _ = np.polyfit(lt, lth, 1)
        rat = [r["theta_over_t0"] for r in sub]
        print("  c=%+.1f : exponent p = %.2f (Eq. 3 predicts 1.00);  "
              "theta/t0 spread %.4f..%.4f" % (c, p, min(rat), max(rat)))
    _save(os.path.join(OUT, "eq3_time.csv"), rows)
    return rows


# ------------------------------------------------------------ experiment: field
def exp_field(args):
    """The u factor. Fly a rank of parallel legs through one realization of a
    genuine correlated field, repeat over realizations, and test both
    Corr(theta_i, theta_j) = exp(-d/lambda) and the combined variance formula."""
    import holoocean
    lanes = [0.0, 15.0, 35.0, 70.0, 110.0, 160.0, 230.0, 320.0]
    n = len(lanes)
    xs = np.arange(-20.0, LEG + 30.0, 10.0)
    ys = np.arange(-30.0, max(lanes) + 40.0, 10.0)
    print("[field] %d parallel legs, lambda=%.0f m, amplitude A=%.2f, "
          "%d realizations" % (n, args.lam, args.amp, args.reals))
    env = holoocean.make(scenario_cfg=make_cfg(n, lanes=lanes),
                         show_viewport=False, verbose=False)
    base = np.array(run_legs(env, n, lanes, LEG, lambda k, p: [0, 0, 0]))
    print("  baseline arrivals (no current): %s" % np.round(base, 2))

    TH, US = [], []
    for s in range(args.reals):
        rng = np.random.default_rng(args.seed + 977 * s)
        g = gp_field(xs, ys, args.lam, rng)
        cf = lambda k, p, g=g: [args.amp * float(bilinear(g, xs, ys, p[0], p[1])),
                                0.0, 0.0]
        t = np.array(run_legs(env, n, lanes, LEG, cf))
        # the field value each lane actually saw, at its own leg locale
        u = np.array([bilinear(g, xs, ys, LEG / 2.0, y) for y in lanes])
        TH.append(t - base); US.append(u)
        if (s + 1) % 10 == 0:
            print("    %d/%d realizations" % (s + 1, args.reals), flush=True)
    env.__exit__(None, None, None)
    TH, US = np.array(TH), np.array(US)
    ok = np.all(np.isfinite(TH), axis=1)
    TH, US = TH[ok], US[ok]
    print("  usable realizations: %d/%d" % (len(TH), args.reals))

    rows = []
    print("\n  lane   d from lane0   rho(d) model   Corr(theta_0,theta_j) "
          " Corr(u_0,u_j)")
    for j in range(n):
        d = abs(lanes[j] - lanes[0])
        rho = math.exp(-d / args.lam)
        ct = float(np.corrcoef(TH[:, 0], TH[:, j])[0, 1]) if j else 1.0
        cu = float(np.corrcoef(US[:, 0], US[:, j])[0, 1]) if j else 1.0
        rows.append(dict(lane=j, d_m=d, rho_model=round(rho, 4),
                         corr_theta=round(ct, 4), corr_u=round(cu, 4),
                         sd_theta_s=round(float(np.std(TH[:, j])), 3)))
        print("  %4d   %10.0f   %12.3f   %20.3f   %13.3f"
              % (j, d, rho, ct, cu))
    _save(os.path.join(OUT, "eq3_field_corr_A%.2f.csv" % args.amp), rows)

    # combined formula: Var(theta_i - theta_j) vs kappa^2 t^2 * 2(1 - rho(d))
    t_nom = float(np.mean(base))
    kappa = float(np.std(TH[:, 0]) / t_nom)          # calibrated, not assumed
    print("\n  calibrated kappa = sd(theta)/t = %.4f  (t = %.1f s)"
          % (kappa, t_nom))
    print("  d (m)   measured Var(dtheta)   Eq.3 prediction   ratio")
    rows2 = []
    for j in range(1, n):
        d = abs(lanes[j] - lanes[0])
        rho = math.exp(-d / args.lam)
        meas = float(np.var(TH[:, 0] - TH[:, j]))
        pred = kappa ** 2 * (2 * t_nom ** 2) * (1 - rho)
        rows2.append(dict(d_m=d, var_measured=round(meas, 3),
                          var_eq3=round(pred, 3),
                          ratio=round(meas / max(pred, 1e-9), 3)))
        print("  %5.0f   %19.2f   %17.2f   %6.2f"
              % (d, meas, pred, meas / max(pred, 1e-9)))
    _save(os.path.join(OUT, "eq3_field_var_A%.2f.csv" % args.amp), rows2)
    # skew: Eq. 3 makes theta Gaussian; the asymmetric response should not be
    sk = float(np.mean(((TH[:, 0] - TH[:, 0].mean()) / (TH[:, 0].std() + 1e-9)) ** 3))
    print("\n  skewness of theta = %+.2f  (Eq. 3 assumes 0; positive = the late "
          "tail is heavier)" % sk)
    return dict(lanes=lanes, TH=TH, US=US, rows=rows, rows2=rows2,
                kappa=kappa, t_nom=t_nom, skew=sk)


# ------------------------------------------------------------------- figure --
def render(probe, field, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.7))

    c = np.array([r["current"] for r in probe["rows"]])
    th = np.array([r["theta_s"] for r in probe["rows"]])
    ax[0].plot(c, th, "o-", color="#1f77b4", lw=1.7, ms=5, label="HoloOcean")
    lin = probe["slope"] * c
    ax[0].plot(c, lin, "--", color="#d62728", lw=1.6,
               label="Eq. 3 (linear, fit on |c|$\\leq$1)")
    ax[0].axvspan(-1, 1, color="#2ca02c", alpha=0.10)
    ax[0].text(0, min(th) * 0.55, "linear regime", ha="center", fontsize=8.5,
               color="#2ca02c")
    ax[0].set_xlabel("along-track current (m/s)")
    ax[0].set_ylabel("arrival offset $\\theta$ (s)")
    ax[0].set_title("(a) Eq. 3 is a small-signal linearisation\n"
                    "head currents cost more than tail currents save",
                    fontsize=10)
    ax[0].legend(fontsize=8.5); ax[0].grid(alpha=0.25)

    d = np.array([r["d_m"] for r in field["rows"]])
    ax[1].plot(d, [r["rho_model"] for r in field["rows"]], "-", color="#d62728",
               lw=2.0, label="Eq. 3:  $\\rho(d)=e^{-d/\\lambda}$")
    ax[1].plot(d, [r["corr_theta"] for r in field["rows"]], "o", color="#1f77b4",
               ms=7, label="measured Corr($\\theta_0,\\theta_j$)")
    ax[1].plot(d, [r["corr_u"] for r in field["rows"]], "x", color="#7f7f7f",
               ms=7, label="field Corr($u_0,u_j$) (sanity)")
    ax[1].set_xlabel("locale separation  d  (m)")
    ax[1].set_ylabel("correlation")
    ax[1].set_title("(b) the $u$ factor: offsets inherit the field's\n"
                    "correlation, as Eq. 3 assumes", fontsize=10)
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.25)

    dd = [r["d_m"] for r in field["rows2"]]
    ax[2].plot(dd, [r["var_eq3"] for r in field["rows2"]], "-", color="#d62728",
               lw=2.0, label="Eq. 3 prediction")
    ax[2].plot(dd, [r["var_measured"] for r in field["rows2"]], "o",
               color="#1f77b4", ms=7, label="HoloOcean measured")
    ax[2].set_xlabel("locale separation  d  (m)")
    ax[2].set_ylabel("Var($\\theta_i-\\theta_j$)   (s$^2$)")
    ax[2].set_title("(c) the quantity Prop. 1 actually uses\n"
                    "$\\kappa^2(t_i^2+t_j^2-2t_it_j\\rho(d))$", fontsize=10)
    ax[2].legend(fontsize=8.5); ax[2].grid(alpha=0.25)

    fig.suptitle("Validating Eq. (3) against HoloOcean rigid-body dynamics "
                 "(HoveringAUV, built-in PID waypoint controller, "
                 "set_ocean_currents)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print("  wrote", os.path.abspath(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all",
                    choices=["probe", "time", "field", "all"])
    ap.add_argument("--reals", type=int, default=60)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=LAMBDA)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    probe = field = None
    if args.exp in ("probe", "all"):
        probe = exp_probe(args)
    if args.exp in ("time", "all"):
        exp_time(args)
    if args.exp in ("field", "all"):
        field = exp_field(args)
    if probe and field:
        render(probe, field, os.path.join(OUT, "fig_eq3_validation.png"))


if __name__ == "__main__":
    main()
