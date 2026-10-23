#!/usr/bin/env python3
"""
Bootstrap confidence intervals for the K=16 matched-false-alarm win
(sequential test vs channel-aware vs bounded belief), burst channel.

Method: run the sim ONCE storing per-trial (false_takeovers, mean_detection_
latency, missed_failures) for each (detector, sensitivity). Then PAIRED bootstrap
over trial indices: for each replicate, recompute each detector's
false-alarm/latency/miss frontier, interpolate latency+miss at the target
false-alarm rate, and record the sequential-minus-channel-aware latency
difference. Report 95% percentile CIs.
"""
import importlib, numpy as np
frs = importlib.import_module("failure_recovery_sim")
from unified_comparison_sim import Comms

N = 500          # trials
B = 4000         # bootstrap replicates
TARGETS = [0.05, 0.02]
SEED0 = 9000

cfg = {"num_tasks": 16, "position_jitter": 6.0, "nominal_speed": 5.0,
       "crit_duration": 2.0, "hb_period": 0.75, "p_fail": 0.4, "n_new": 0,
       "fleet_size": 32, "c_miss": 1.0, "c_conflict": 1.0, "c_latency": 0.0}
burst = Comms(delay_mean=1.0, delay_jitter=0.5, drop_rate=0.2,
              burst=True, burst_drop=0.97)

SENS = {"channel_aware_timeout": [1.5, 2, 3, 4, 6, 8, 10, 12, 14],
        "ours_correlated":       [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "ours_sprt_lin":         [1, 2, 3, 4, 6, 9, 13, 18, 24]}
LABEL = {"channel_aware_timeout": "channel-aware",
         "ours_correlated": "bounded belief",
         "ours_sprt_lin": "sequential"}

# ---- one simulation pass: store per-trial arrays ----
print(f"simulating (N={N} trials, K=16 burst)...", flush=True)
data = {m: {} for m in SENS}            # data[m][s] = (ft[N], dl[N], ms[N])
for m in SENS:
    for s in SENS[m]:
        c = dict(cfg); c["sens"] = s
        ft = np.empty(N); dl = np.empty(N); ms = np.empty(N)
        for i in range(N):
            r = frs.run_trial(SEED0 + i, burst, m, c)
            ft[i] = r["false_takeovers"]
            dl[i] = r["mean_detection_latency"]
            ms[i] = r["missed_failures"]
        data[m][s] = (ft, dl, ms)
    print(f"  done {m}", flush=True)

def interp_at(idx, m, tgt):
    """Interp (latency, miss) at target false-alarm for resampled indices idx."""
    pts = []
    for s in SENS[m]:
        ft, dl, ms = data[m][s]
        pts.append((ft[idx].mean(), dl[idx].mean(), ms[idx].mean()))
    pts.sort()                                   # by false-alarm
    for (x0, l0, m0), (x1, l1, m1) in zip(pts, pts[1:]):
        if (x0 - tgt) * (x1 - tgt) <= 0 and x1 != x0:
            w = (tgt - x0) / (x1 - x0)
            return l0 + w * (l1 - l0), m0 + w * (m1 - m0)
    return None                                  # target not bracketed

# ---- point estimates (full sample) ----
full = np.arange(N)
print("\n=== POINT ESTIMATES (full sample) ===")
for tgt in TARGETS:
    print(f" FA={tgt}: " + "  ".join(
        f"{LABEL[m]} {('%.2fs[%.3f]'%interp_at(full,m,tgt)) if interp_at(full,m,tgt) else 'unreach'}"
        for m in SENS))

# ---- paired bootstrap ----
rng = np.random.default_rng(0)
boot = {tgt: {m: {"lat": [], "miss": []} for m in SENS} for tgt in TARGETS}
diff = {tgt: [] for tgt in TARGETS}              # sequential - channel-aware latency
for _ in range(B):
    idx = rng.integers(0, N, N)                  # paired resample (same across methods)
    for tgt in TARGETS:
        vals = {}
        for m in SENS:
            r = interp_at(idx, m, tgt)
            if r is None:
                vals[m] = None; continue
            vals[m] = r
            boot[tgt][m]["lat"].append(r[0]); boot[tgt][m]["miss"].append(r[1])
        if vals["ours_sprt_lin"] and vals["channel_aware_timeout"]:
            diff[tgt].append(vals["ours_sprt_lin"][0] - vals["channel_aware_timeout"][0])

def ci(a):
    a = np.asarray(a); return np.percentile(a, 2.5), np.median(a), np.percentile(a, 97.5)

print(f"\n=== 95% BOOTSTRAP CIs (B={B}, paired) ===")
for tgt in TARGETS:
    print(f"\n FA target = {tgt}")
    for m in SENS:
        lo, md, hi = ci(boot[tgt][m]["lat"])
        mlo, mmd, mhi = ci(boot[tgt][m]["miss"])
        valid = 100 * len(boot[tgt][m]["lat"]) / B
        print(f"   {LABEL[m]:>14}: latency {md:.2f}s [{lo:.2f}, {hi:.2f}]"
              f"   miss {mmd:.3f} [{mlo:.3f}, {mhi:.3f}]   (bracketed {valid:.0f}%)")
    d = np.asarray(diff[tgt])
    lo, md, hi = ci(d)
    pwin = 100 * (d < 0).mean()
    print(f"   --> sequential - channel-aware latency: {md:.2f}s "
          f"[{lo:.2f}, {hi:.2f}]   P(sequential faster) = {pwin:.1f}%")
