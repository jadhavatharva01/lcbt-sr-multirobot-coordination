#!/usr/bin/env python3
"""
Does making channel health GRADED (instead of bimodal) widen the family gap
(belief vs simple channel-aware timeout)? We sweep burst severity: a high
burst_drop (~0.97) silences the bad state completely -> eta is bimodal 0/1 ->
belief == simple timeout. A moderate burst_drop lets some traffic through in the
bad state -> eta is GRADED -> the belief should pull ahead.
Metric: false takeovers on alive primaries (burst, p_fail=0.4).
"""
import importlib
frs = importlib.import_module("failure_recovery_sim")
from unified_comparison_sim import Comms

TRIALS = 500
K, N = 6, 12
DROPS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97]
METHODS = ["channel_aware_timeout", "ours_correlated"]


def cfg():
    return {"num_tasks": K, "position_jitter": 6.0, "nominal_speed": 5.0,
            "crit_duration": 2.0, "hb_period": 0.75, "p_fail": 0.4, "n_new": 0,
            "fleet_size": N, "c_miss": 1.0, "c_conflict": 1.0, "c_latency": 0.0}


print("=== false takeovers vs burst severity (lower burst_drop = more GRADED eta) ===")
print(f"  {'burst_drop':<12}{'simple(timeout)':>18}{'ours(belief)':>16}{'gap (x)':>10}")
for bd in DROPS:
    burst = Comms(delay_mean=1.0, delay_jitter=0.5, drop_rate=0.2,
                  burst=True, burst_drop=bd)
    vals = {}
    for m in METHODS:
        tr = [frs.run_trial(4300 + i, burst, m, cfg()) for i in range(TRIALS)]
        vals[m] = frs.average(tr)["false_takeovers"]
    gap = vals["channel_aware_timeout"] / max(vals["ours_correlated"], 1e-6)
    grade = "GRADED" if bd <= 0.7 else ("bimodal" if bd >= 0.9 else "mixed")
    print(f"  {bd:<12}{vals['channel_aware_timeout']:>18.3f}"
          f"{vals['ours_correlated']:>16.3f}{gap:>9.2f}x   {grade}")
