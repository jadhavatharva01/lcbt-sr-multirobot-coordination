# Simulator audit, fixes, and limitations

Audit of `unified_comparison_sim.py` (single-task) and `unified_multitask_sim.py`
(multi-task). Read this before trusting any number out of these sims.

## Calculation bugs found and fixed

1. **CBBA was undercounted (messages) — fixed earlier.** The old phase-1 loop
   `break`-ed on first successful delivery, so consensus rounds stopped
   accruing. Replaced with an explicit per-protocol broadcast count.

2. **CBBA used a fake fixed 3 rounds — fixed.** Now a real consensus loop
   (`simulate_consensus`) gossips/merges bid knowledge until all agents agree
   on the assignment (or a 30-round cap). Rounds, messages, and start latency
   now grow with packet loss, as a real consensus protocol does.

3. **CBBA convergence criterion was too strict — fixed.** It required every
   agent to have heard *everyone* in addition to agreeing. Agreement alone is
   CBBA's real stopping rule; the extra condition inflated its rounds and
   unfairly handicapped it. Relaxed to agreement-only.

4. **Heartbeat was a single ping, not periodic — fixed in both sims.** A
   trailing robot now gets caught if ANY heartbeat emitted at
   `pre_start + k*hb_period` is delivered before it reaches the zone. So
   `hb_period` now genuinely affects catch probability under loss, not just the
   message count. (Previously only the single-task sim was periodic.)

5. **Grasp-zone mutex removed from the evaluation.** It is a physical mutual-
   exclusion interlock any method could bolt on, not an algorithmic
   contribution; with it, every method trivially gets zero collisions. Now
   OFF for all methods (`--universal_mutex` remains as an optional analysis
   toggle only). Collision-freedom must now be EARNED by coordination.

## What removing the mutex revealed (the honest headline)

With no physical safety net, ours' previously-headline wins largely evaporate:
- collision-freedom was the mutex, not the algorithm (ours now has real,
  nonzero collisions: ~0.03 at drop 0.4 up to ~0.4–1.1 under heavy/bursty loss);
- full coverage was partly propped up (ours degrades to ~0.82–0.96).

A faithfully-modeled CBBA **dominates on every quality axis** (coverage,
collisions, energy, and even messages at low/moderate drop) *whenever it is
allowed to finish*. Its only weakness is **unbounded latency**: it refuses to
act until consensus, so makespan grows without bound under loss (≈30s at
drop 0.85, ≈51s under long blackouts) and recovers as soon as the network is
eventually connected — which the burst channel still allows.

**Ours' single robust, honest advantage is therefore BOUNDED RESPONSE TIME.**
It acts in ~18–21s regardless of how bad comms get, at the cost of degraded
coverage and some collisions. This is a latency-vs-quality trade, only
worthwhile for hard-real-time / deadline-driven missions. If you can wait,
CBBA is the better method. State this plainly; do not reclaim the mutex wins.

## Remaining modeling limitations (NOT yet addressed)

These bound how far the current results can be pushed:

- **No spatial motion or path conflicts.** Robots are points with travel
  times; a "collision" is only a temporal overlap at the single grasp point of
  a task. En-route congestion, path crossing, and localization error are not
  modeled.
- **One-shot burst of K tasks.** All tasks appear at once. Streaming / dynamic
  task arrival (true online MRTA) — the strongest form of the "asynchronous"
  claim — is not modeled.
- **Single task per robot.** No task bundles or sequential routes. This
  actually undercuts CBBA's designed strength (bundle bidding); a bundle
  setting could change the comparison.
- **Messages are unweighted broadcasts.** CBBA's packets carry full
  winning-bid/assignment vectors (larger); ours' heartbeats are tiny. Counting
  equal "messages" understates CBBA's bandwidth and overstates ours'. A
  byte-weighted metric would be fairer.
- **Comms model.** i.i.d. drop or a shared-medium Gilbert-Elliott burst. There
  is no per-link / spatial *persistent* partition (a fixed subset that can
  never hear the rest). GE bursts always recover, so CBBA always eventually
  converges; a true permanent partition is the one regime where consensus
  provably cannot finish and a reactive method must win.
- **Reassignment approximations.** A bumped robot re-routes from its ORIGINAL
  position, not its bumped mid-route position (slightly under-counts energy),
  and re-assignment is granted only to intent/heartbeat-aware methods (a
  fairness choice — `auction_intent` is the fair "auction+reassign" baseline).
- **Perfect sensing/estimation.** Detection, bids, travel-time estimates, and
  teammate costs (once received) are exact. No sensor noise or estimation
  error.
- **CBBA is gossip-to-agreement, simplified.** Real CBBA's bundle
  conflict-resolution (CBAA/CBBA rules with timestamps) is approximated by
  merge-min gossip until assignment agreement.

## Suggested next experiments (to make the contribution fair AND honest)

1. **Anytime/deadline CBBA**: cut consensus off at a mission deadline; CBBA then
   acts on its (possibly inconsistent) partial assignment. This is the fair
   fight for time-critical settings and the regime where ours' guards may
   degrade more gracefully than deadline-truncated consensus.
2. **Persistent spatial partition**: a fixed subset permanently unreachable.
   The one regime where consensus cannot converge by construction.
3. **Streaming task arrival** and **byte-weighted messages**.
