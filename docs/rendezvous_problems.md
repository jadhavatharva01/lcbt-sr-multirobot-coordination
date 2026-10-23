# Rendezvous concept — problem register

> **STATUS 2026-09-25 — this line is on hold; a load-bearing assumption was refuted.**
> HoloOcean showed that a fixed-duration stop neither resets nor absorbs accumulated
> delay, so assumption **A-L2** (exposure = the final approach leg) is false. Under the
> supported carry-forward model, correlation-alignment buys only ~25–33% relative
> false-alarm reduction while simply arriving early buys 10–100×. See
> `rendezvous_assumptions.md` **N4 (REFUTED)** and `HANDOFF.md` §3 before acting on
> anything below. The design reasoning here is still readable as design reasoning; its
> quantitative conclusions are not current.


*Companion to `rendezvous_allocation_idea.md` (v2). Every known weakness of the*
*concept, ranked by severity. We solve these one by one; each entry carries a*
*status. Statuses: **OPEN** / **SOLUTION PROPOSED** (design exists, not*
*simulated) / **SOLVED** (verified in simulation).*

## Scope assumptions (this iteration — declared, revisit in later iterations)

*The full idealization catalogue (every unnatural thing in the pipeline, with a
phased plan to retire them toward deployability) now lives in
`rendezvous_assumptions.md`. A1–A3 below are the three oldest entries; they map
to F3, P5, and N1 there.*

- **A1 — single failure at a time.** No second death in the same
  neighborhood before the first repair (slot inheritance + orphan
  absorption) has settled. Simultaneous/adjacent failures and
  successor-also-dies chains are future iterations, not this one.
- **A2 — scan/survey missions.** Duplicated work is wasted travel, never a
  conflict (idea doc §8).
- **A3 — navigation is good enough** that schedule error stays within the
  drift envelope measured in Q3b (≤~5% speed error with ε=60 s).

---

## P1. The plan repairs itself using the thing that just broke
**Severity: make-or-break.  Status: SOLVED — first-order pipeline sim
(2026-07-23), now FORWARD-EXECUTED end-to-end (2026-07-24, Stage X): after a
death the whole repaired mesh is played forward with each survivor's other
appointments interleaved as hard constraints, and recovery keeps 100% of
appointments with 100% total ownership — the precomputed takeover is verified by
execution, not asserted on the schedule graph. Design below, results at the end
of this entry.**

The guarantees come from everyone sharing one deterministic schedule, but the
first failure invalidates it: the successor inherits POIs, its route stretches,
its future contact times/places shift — while robots that haven't heard the
news still navigate by the old schedule. Stale robot goes to the old meeting
point, re-planned robot to the new one → missed contact → false alarm →
cascade. The repair mechanism (contacts) is the same machinery the failure
invalidated.

### Proposed solution: contract-preserving re-planning (Atharva's overlap rule, formalized)

Root idea (Atharva): *any newly generated path must overlap at least the next
two consecutive contacts of the old plan*, so robots still on stale plans are
guaranteed to physically meet a robot carrying the new plan at the old
rendezvous points; late orphan pickup is harmless because the succession table
gives every orphan exactly one designated taker — lateness is delay, never a
race.

Formalized as three rules:

1. **Pairwise contact contracts.** The next contact between any pair (time +
   place) is whatever the two agreed at their *previous* contact (or the
   pre-mission plan). **No unilateral changes, ever.** A re-plan must honor
   all outstanding contracts; contacts may be moved only by face-to-face
   agreement at the preceding meeting of that same pair. Consequence: a
   missed contact can never be caused by staleness — the schedule updates
   propagate exactly as fast as news does, by construction.
2. **Slot inheritance (pre-agreed schedule repair — REVISED 2026-07-08).**
   *(v1 of this rule — the discoverer physically substitutes into the dead
   robot's appointments — was WRONG: in the brick-wall the dead robot's
   other meeting runs in the same round as the discoverer's own meeting, so
   substitution is double-booked / arrives far beyond ε, violating rule 1.
   Caught by Atharva.)*
   Instead, the repair is **precomputed and needs no messenger**: the dead
   robot's two partners each discover the death *independently* through
   their own missed meeting with it (both within one round of each other —
   detection is naturally redundant). Upon confirming, each switches to the
   pre-agreed repaired schedule from the mission "will": **the two partners
   of a dead robot become partners with each other, meeting in the dead
   robot's vacated slots, initially at the dead robot's old meeting
   points.** Three structural gifts:
   - the vacated slots are **free in both partners' calendars by
     construction** (those were exactly the times each spent meeting the
     dead robot) → zero conflicts, zero delay, no ε burden, no contract
     violated;
   - both partners can deterministically compute the same first joint slot
     (each knows when the *other* will have discovered the miss, from the
     shared schedule) → the chain is re-welded without any robot carrying
     the news;
   - **false alarms self-heal**: if the "dead" robot was merely late, it
     keeps its old plan and shows up at its old meeting points — which is
     exactly where its two ex-partners now meet → it is rediscovered and
     reintegrated at the first repaired meeting.
   First repaired meeting is a **grace round** (attend, but no accusations
   if the partner is absent — its verification may still be pending); alarm
   only from the second repaired slot onward.
3. **Orphans yield to contracts.** Orphaned POIs are inserted only where they
   cannot violate outstanding contracts (in slack, or deferred to the end).
   Always feasible: keeping the old route timing and appending orphans last
   is a valid fallback (the old plan met its contracts by definition).
   Exclusivity needs no timeliness: only the table-designated successor ever
   touches an orphan, no matter who learns what when — a stale robot never
   grabs orphans that aren't table-assigned to it.

Cost (probe-scale arithmetic): the repaired pair meets across the dead
robot's band, so meeting trips lengthen by ~1 band width (~150–300 m at
2 m/s = 75–150 s each way) until the successor's absorption of the territory
pulls the pair closer; this is travel cost only — no schedule conflict.

What this buys: no staleness-induced missed contacts (by construction);
no robot's existing contracts are ever touched by a repair, so re-plan
cascades cannot start; the chain re-welds itself without a messenger; both
sides of the hole detect the death within one round of each other (far-side
bound ≤ 2Δ); checkpoint asymmetry (one partner holds a fresher checkpoint
than the successor) costs at most one Δ-interval of re-scanned POIs — benign.

### Residual open items under P1
- Asymmetric confirmation: one partner's verification succeeds while the
  other's is delayed (or it blames itself per the P4 rule) → it skips the
  repaired slot; the grace round absorbs one such miss, longer divergence
  unhandled.
- Simultaneous adjacent failures: **out of scope by assumption A1** (kept
  for a later iteration; the repaired-pair rule applied recursively is the
  likely shape).
- Note (2026-07-08): the contact structure is now a *distributed appointment
  set* meeting requirements R1–R4 (idea doc §4.2), not rigid pairs/rounds —
  the P1 rules (contracts, slot inheritance, orphans-yield) are unchanged by
  this; they only ever needed pairwise appointments, partner diversity (R3),
  and temporal connectivity (R2).
- **Validation — BUILT (2026-07-23):** `visualization/full_pipeline_sim.py`
  (outputs in `results/full_pipeline/`). One deterministic offline pass
  freezes the whole mesh (every robot's trajectory + every appointment's
  time & place = global common knowledge); the live run replays it and kills
  a robot mid-mission. Over 120 seeds (K=6, N=30, L=600 m, Δ=300 s, ε=30 s):
  - **Detection bound holds.** Every death detected within 2Δ+ε (100%);
    within the tight Δ+ε bound 85.8% (the ~14% over are the two end robots,
    whose bound is 2Δ+ε — see the R3 caveat below). Mean latency 204 s.
  - **Zero false alarms** on living robots (no drift in this pass — by
    construction, since everyone replays the frozen mesh).
  - **Redundant detection 64%** (both neighbours independently catch it);
    the non-redundant cases are end robots (one neighbour only) and late
    deaths with a single remaining appointment.
  - **Total-ownership held 100%** — every orphaned POI owned exactly once.
  - **Chain re-welds 100%** — every survivor still has a partner after the
    death + slot-inheritance link is applied.
  - **Overlap window (Stage E, news propagation).** A survivor stays on its OLD
    track for W scheduled contacts after detecting the death before diverging
    to recovery, so it hands the news to robots still on the old plan (a stale
    robot is only ever at old contact points). W=0 (diverge immediately) causes
    ~2 false-alarm cascades/death and informs only ~33% of the team; **W=1
    preserved contact → exactly zero divergence false alarms**; full-team
    propagation rises with mission length (21%→93% as N grows 30→120), bounded
    by remaining rounds not by W (the R2 (K−2)·Δ certificate). Atharva's "~2
    POIs" is a sound margin; the measured floor is one preserved *contact*
    (≈1·Δ). [propagation numbers corrected 2026-07-23 after a seeding/causality
    bugfix in propagate_news.]
  - **Takeover is a precomputed lookup (2026-07-23).** Recovery is now
    `f(dead robot v, last checkpoint c)` — one deterministic pattern per
    (dead robot, death-progress point), built offline from the frozen mesh, so
    every survivor applies the *same* assignment (the design Atharva insisted
    on; it is what keeps agreement intact once live views diverge under noise).
    Orphan set = POIs after v's last checkpoint (a safe superset — re-scanning
    the checkpoint gap is benign). Each orphan has exactly one taker, the
    surviving neighbor it is *forward-absorbable* for; behind-orphans go to an
    end cleanup (which also protects the survivors' other contracts). Over 120
    deaths: total-ownership 100%, chain re-weld 100%, **~65% of orphans grabbed
    forward** at ~48% of makespan delay, ~35% deferred (~84%); worst-case tail
    p95 130%; survivor travel overhead mean 37% / p95 96%. (The numbers are
    higher than the earlier true-death-time model because the checkpoint-based
    orphan set is deliberately conservative — it re-scans POIs v may have
    finished but couldn't report.)
  Caveats making this "first-order," not final: forward/behind uses a
  sweep-progress (y) heuristic + backtrack tolerance; no timing drift or
  close-range link loss in this pass (P6/Q3b); single failure only (A1); the
  re-welded L–R appointment link and survivors' *other* contracts are asserted
  on the schedule graph, not yet forward-executed for slack feasibility
  (assumptions ledger R2).
- **New limitation surfaced by the sim — end-robot blind spot (R3).** On a
  *line* of bands the two end robots have only ONE neighbour, so R3 (≥2
  distinct partners) fails there: their detection bound is 2Δ+ε and their
  single partner is a single point of monitoring failure. The fix is the
  rotating-buddy / ring / hierarchical topology already flagged for §7 of the
  idea doc — deferred, but now demonstrated rather than suspected.

---

## P2. Every meeting is a hard deadline inside a soft mission
**Severity: make-or-break.  Status: SOLUTION PROPOSED — demonstrated
(2026-07-23, Stage D); PARKED under the equal-scan-time assumption (2026-07-24)
— re-opens only when noise is added (Phase 1).**

> **Scoping (2026-07-24):** for now scan times are taken to be *equal and
> deterministic*, so the frozen plan is exact — there is no such thing as a task
> overrun, and the "meeting-wins + slack" machinery below does nothing. P2 is
> therefore trivially satisfied in the nominal case and is *parked*. It comes
> back the moment we add variance (drift, lossy links, scan-time jitter — Phase
> 1 of the ledger), which is the only regime where slack does any work; that is
> where SLACK is re-introduced and the deferred-work tail must be bounded.
> The rule and its Stage-D demonstration below are retained for that point.

Contacts convert a flexible coverage mission into a real-time system. Task
overruns, obstacle detours, battery-saving speed changes are lumpy and
one-sided — they blow through ε=30–60 s unless the robot abandons work to make
the meeting, so the monitoring layer can steal priority from the mission. The
drift probe (Q3b) assumed small Gaussian per-robot speed error — kind to the
concept. Needs: an explicit meeting-vs-task priority rule; slack scheduled
after uncertain tasks; honest re-accounting of the travel/time tax with that
slack included. (Note: P1's contract rule sharpens this — contracts are
inviolable, so the priority rule must be "meeting wins," making the slack
budget mandatory.)

**Proposed rule (demonstrated): "meeting wins" + mandatory slack budget.**
Each robot leaves for an appointment at a fixed leave-time (t_meet − travel −
SLACK) no matter what it is doing; a task still running is abandoned and its
scan becomes a *self-orphan* completed later. Consequence: a task overrun can
never make a robot late, so it can never cause a false missed-contact alarm;
only navigation/travel error that exceeds SLACK can. Stage D (200 seeds/point,
SLACK=30 s, ε=30 s) sweeps the overrun magnitude:

| mean overrun | meeting-wins false-alarm | task-wins false-alarm | meeting-wins defers a scan |
|---|---|---|---|
| 30 s  | 0.0% | 0.0%  | 50% |
| 60 s  | 0.0% | 49.5% | 92% |
| 120 s | 0.0% | 90.5% | 98% |
| 240 s | 0.0% | 96.0% | 98% |

Meeting-wins keeps the detection layer false-alarm-free at every overrun size
by converting task-time variance into deferred work; task-wins leaks that
variance straight into false accusations of healthy robots. The price of
meeting-wins is the deferred-scan pileup (those self-orphans lengthen the
tail, same mechanism as P1's orphans). Still open: choosing SLACK against a
measured task-time distribution, and bounding the deferred-work tail (it
competes with orphan work for the same end-of-mission slack).

## P3. Who is the actual customer? (economics)
**Severity: high (framing).  Status: OPEN.**

Detection needs ~1 byte/min — surface vessels can usually get that via
satellite (Iridium), killing the USV case for a fraction of a +100% travel
tax; relay buoys/aerial relays compete too. The honest niche: platforms with
no side channel at all AND adequate navigation — e.g. AUVs are comms-denied
but nav-drift-prone (attacks P2's tolerances). The concept must argue a real
platform where "no comms" and "good nav" hold simultaneously, or accept the
narrow window explicitly.

## P4. The observer's own faults look like everyone else's death
**Severity: high.  Status: OPEN.**

If my radio (or clock/nav) fails, every contact I attend looks missed — I
accuse partner after partner while being the problem. Needs the self-suspicion
protocol: two consecutive missed contacts with *different* partners → suspect
self → fail-safe behavior (go loud / go home / hold accusations). Sketched in
the idea doc §5, no protocol, no test. Related: pair-simultaneous deaths go
unnoticed one extra round (bound 2Δ); a discoverer dying before its next
contact takes the news with it.

## P5. Recovery breaks its own preconditions (stale succession table)
**Severity: high.  Status: OPEN (partially eased by P1 solution).**

The succession table was computed for original territories; after the first
recovery, territories changed, so the table is stale for the *second* failure
— the "1–2 contenders per orphan" property may no longer hold. Successor
overload is eased (2026-07-08) by the **balanced split** rule: the table
divides a dead robot's orphans among its feasible neighbors rather than
dumping them on one successor, and the total-ownership guarantee (idea doc
§6) ensures a late-served POI is done at the end rather than dropped. The
*stale-table-for-the-second-failure* part remains open (and is softened by
scope assumption A1 for this iteration).
P1's contracts stop the *schedule* from breaking, but table refresh must ride
the news chain: everyone can deterministically recompute the table from the
death report — needs a canonical recompute rule so all robots converge on the
same new table as the news reaches them. Design needed + double-failure sim
(next-experiment #2).

## P6. "Perfect within 60 m" and "data fits in a pass-by" are assumptions
**Severity: medium.  Status: OPEN.**

Real close-range links drop packets (waves, multipath); a 90%-reliable contact
link reintroduces false misses at the layer that's supposed to have none
(mitigation: contact = a window of many heartbeat exchanges, not one shot —
quantify). Data ferry assumes accumulated scan data transfers in a ~30 s
pass-by; sonar/imagery may need loiters, reintroducing the stopping cost —
or ferry only checkpoints + thumbnails, full data on recovery of the vehicle.

## P7. Scaling is mediocre
**Severity: medium.  Status: OPEN.**

News staleness is linear in K on a line ((K−2)·Δ; K=10, Δ=300 s → ~40 min for
full-team consistency); grids only improve to √K. **Measured (Stage E2/E3):**
at K=6 the news reaches the whole team in ≈3 rounds ≈ 8–10 POIs when it
completes, and completes on 48% of deaths thanks to *multi-source* detection
(every missed-meeting partner seeds the spread) vs 6% for a single discoverer —
the multi-source win is *completion*, not speed (both take ~3.4 rounds when they
finish). Growth is **linear** on the line (~constant %/round), not exponential.
Higher-degree topologies *do* make it exponential (Stage E3, K=16: hypercube
3.6 rounds ≈ log K vs line 11.6) but at 3.75× the contact-travel and meetings
spanning up to the whole field — self-defeating for physical robots, and
low-value since late news is harmless (total-ownership). **The worthwhile fix is
the mild one: a 2-D grid** (degree 4, meetings still local, √K diameter, no
blind ends) — same fix as R4. P1's substitution keeps the chain connected, but
staleness still grows with K. Early finishers unhandled:
keep commuting to meetings (waste) or exit the schedule (rewires the chain —
though under P1 rules an exit can be *negotiated* at a contact, which may be
the clean answer). Large-K version likely needs hierarchical territories.

## P8. Weaknesses in our own evidence
**Severity: meta.  Status: OPEN (standing caveat).**

The +100% travel figure comes from crude greedy planners on an abstract
kinematic sim (no currents, dynamics, real links) — an estimate, not a bound.
The regime map used a hard 150 m disk; graded links create a middle regime
where a hybrid likely beats both pure strategies. Prior-art adjacency is real:
co-observation planning (arXiv 2403.13266) already treats a missed planned
meeting as the detectable event — defensible ground is the regime map + cost
Pareto + recovery coupling.

Update (2026-07-23): the end-to-end pipeline *has* now been simulated
(`full_pipeline_sim.py`, first-order — see P1's validation entry), which
retires the "never simulated" caveat but not the abstraction caveat: still no
currents/dynamics/real acoustic links, orphans appended at end (upper bound),
single failure only, and recovery costed on survivor routes rather than a full
mesh re-solve. The headline claims (bounded detection, total ownership, chain
re-weld) now hold in that first-order sim; the travel/delay *magnitudes* remain
estimates, and the HoloOcean pipeline is the next fidelity step.
