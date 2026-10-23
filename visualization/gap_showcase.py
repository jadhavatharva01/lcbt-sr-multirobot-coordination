#!/usr/bin/env python3
"""
Showcase the regime where the channel-aware BELIEF genuinely pulls ahead of the
simple channel-aware timeout (and crushes the per-node baselines): PARTIAL /
graded network partitions, where channel health is graded (~0.5) rather than the
bimodal 0/1 of a full burst. Metric: false takeovers on alive-but-partitioned
primaries (wrongly declaring a temporarily-cut-off robot dead). Lower = better.
"""
import importlib, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
frs = importlib.import_module("failure_recovery_sim")
from unified_comparison_sim import Comms

TRIALS = 600
CUTS = [0.2, 0.4]
iid = Comms(delay_mean=1.0, delay_jitter=0.5, drop_rate=0.2, burst=False)

METHODS = ["adaptive_pernode", "fixed_timeout", "channel_aware_timeout", "ours_correlated"]
LABEL = {"adaptive_pernode": r"$\varphi$-accrual" + "\n(per-node)",
         "fixed_timeout": "fixed\ntimeout",
         "channel_aware_timeout": "ours: simple\n(timeout)",
         "ours_correlated": "ours: belief\n(full)"}
COL = {"adaptive_pernode": "#c62828", "fixed_timeout": "#e57373",
       "channel_aware_timeout": "#64b5f6", "ours_correlated": "#1565c0"}


def cfg(cut):
    return {"num_tasks": 4, "position_jitter": 6.0, "nominal_speed": 5.0,
            "crit_duration": 8.0, "hb_period": 0.75, "p_fail": 0.3, "n_new": 0,
            "fleet_size": 6, "c_miss": 1.0, "c_conflict": 1.0, "c_latency": 0.0,
            "cut_drop": cut, "partition_len": 10.0}


res = {m: {} for m in METHODS}
for cut in CUTS:
    for m in METHODS:
        tr = [frs.run_partition_trial(9900 + i, iid, m, cfg(cut)) for i in range(TRIALS)]
        res[m][cut] = frs.average(tr)["false_on_partition"]

print("=== false takeovers on alive-but-partitioned robots (lower=better) ===")
print(f"  {'method':<24}" + "".join(f"cut={c:<8}" for c in CUTS))
for m in METHODS:
    print(f"  {m:<24}" + "".join(f"{res[m][c]:<12.3f}" for c in CUTS))
# gap vs the simple channel-aware timeout
for c in CUTS:
    g = res["channel_aware_timeout"][c] / max(res["ours_correlated"][c], 1e-6)
    print(f"  belief vs simple-timeout gap @cut={c}: {g:.1f}x fewer; "
          f"vs phi-accrual {res['adaptive_pernode'][c]/max(res['ours_correlated'][c],1e-6):.1f}x")

# grouped bar chart
fig, ax = plt.subplots(figsize=(8, 4.8))
w = 0.18
import numpy as np
x = np.arange(len(CUTS))
for i, m in enumerate(METHODS):
    ax.bar(x + (i - 1.5) * w, [res[m][c] for c in CUTS], w,
           label=LABEL[m].replace("\n", " "), color=COL[m])
ax.set_xticks(x); ax.set_xticklabels([f"cut severity {c}" for c in CUTS])
ax.set_ylabel("false takeovers on alive,\npartitioned robots  (lower = better)")
ax.set_title("Partial network partitions: the belief separates failed from\n"
             "partitioned-but-alive robots that a simple timeout cannot", fontsize=11)
ax.legend(fontsize=9, title="detector")
fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "results", "comparison")
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig_partition_gap.png"), dpi=150)
print("\nwrote results/comparison/fig_partition_gap.png")
