#!/usr/bin/env python3
"""
PI-facing baseline comparison: success rate + mission completion time + the
false-takeover / energy differentiators, ours vs the baselines (burst channel).
Produces results/comparison/fig_comparison.png + comparison.csv + a table.
"""
import importlib, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
frs = importlib.import_module("failure_recovery_sim")
from unified_comparison_sim import Comms

TRIALS = 500
K, N = 6, 12
burst = Comms(delay_mean=1.0, delay_jitter=0.5, drop_rate=0.2,
              burst=True, burst_drop=0.97)

METHODS = ["no_recovery", "fixed_timeout", "adaptive_pernode",
           "channel_aware_timeout", "ours_correlated"]
LABEL = {"no_recovery": "no recovery\n(alloc-only)",
         "fixed_timeout": "fixed\ntimeout",
         "adaptive_pernode": r"$\varphi$-accrual",
         "channel_aware_timeout": "channel-aware\ntimeout",
         "ours_correlated": "ours\n(belief)",
         "ours_sprt_lin": "ours\n(sequential)"}
OURS = {"channel_aware_timeout", "ours_correlated", "ours_sprt_lin"}


def cfg_at(pf):
    return {"num_tasks": K, "position_jitter": 6.0, "nominal_speed": 5.0,
            "crit_duration": 2.0, "hb_period": 0.75, "p_fail": pf, "n_new": 0,
            "fleet_size": N, "c_miss": 1.0, "c_conflict": 1.0, "c_latency": 0.0}


def run(method, pf):
    tr = [frs.run_trial(4200 + i, burst, method, cfg_at(pf)) for i in range(TRIALS)]
    return frs.average(tr)


# detailed comparison at p_fail = 0.4
PF = 0.4
rows = {m: run(m, PF) for m in METHODS}

print(f"\n=== BASELINE COMPARISON (burst, K={K}, N={N}, p_fail={PF}, {TRIALS} trials) ===")
hdr = f"  {'method':<22}{'success%':>10}{'recover%':>10}{'mission_time':>14}{'false_tk':>10}{'energy':>9}"
print(hdr)
for m in METHODS:
    a = rows[m]
    print(f"  {m:<22}{100*a['coverage_frac']:>9.1f}{100*a['recovery_rate']:>10.1f}"
          f"{a['makespan']:>14.1f}{a['false_takeovers']:>10.3f}{a['wasted_travel']:>9.2f}")

# success rate vs p_fail (shows recovery necessity)
PFS = [0.0, 0.2, 0.4, 0.6]
succ = {m: [run(m, pf)["coverage_frac"] for pf in PFS]
        for m in ["no_recovery", "adaptive_pernode", "ours_correlated"]}

# ---------------- figure ----------------
fig = plt.figure(figsize=(12, 7.5))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)
colors = ["#9e9e9e" if m == "no_recovery" else
          ("#1565c0" if m in OURS else "#c62828") for m in METHODS]
xs = range(len(METHODS))
names = [LABEL[m] for m in METHODS]

# (a) success rate
ax = fig.add_subplot(gs[0, 0])
ax.bar(xs, [100 * rows[m]["coverage_frac"] for m in METHODS], color=colors)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylabel("mission success rate (%)"); ax.set_title("(a) success rate  (higher = better)")
ax.set_ylim(0, 105); ax.axhline(100, color="k", lw=0.6, ls=":")

# (b) mission completion time
ax = fig.add_subplot(gs[0, 1])
ax.bar(xs, [rows[m]["makespan"] for m in METHODS], color=colors)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylabel("mission completion time (s)"); ax.set_title("(b) completion time  (lower = better)")

# (c) false takeovers (the differentiator)
ax = fig.add_subplot(gs[1, 0])
ax.bar(xs, [rows[m]["false_takeovers"] for m in METHODS], color=colors)
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylabel("false takeovers / trial"); ax.set_title("(c) wasted recovery  (lower = better)")

# (d) success rate vs failure rate
ax = fig.add_subplot(gs[1, 1])
ax.plot(PFS, [100*v for v in succ["no_recovery"]], "o--", color="#9e9e9e", lw=2, label="no recovery")
ax.plot(PFS, [100*v for v in succ["adaptive_pernode"]], "s-", color="#c62828", lw=2, label=r"$\varphi$-accrual")
ax.plot(PFS, [100*v for v in succ["ours_correlated"]], "D-", color="#1565c0", lw=2, label="ours")
ax.set_xlabel("robot failure probability"); ax.set_ylabel("mission success rate (%)")
ax.set_title("(d) success vs failure rate"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"Failure recovery under bursty comms: ours vs baselines "
             f"(K={K} tasks, N={N} robots)", fontweight="bold", fontsize=13)
out = os.path.join(os.path.dirname(__file__), "..", "results", "comparison")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_comparison.png"), dpi=150, bbox_inches="tight")
frs.save_csv([{"method": m, **rows[m]} for m in METHODS],
             __import__("pathlib").Path(out) / "comparison.csv")
print(f"\nwrote {os.path.relpath(os.path.join(out, 'fig_comparison.png'))}")
