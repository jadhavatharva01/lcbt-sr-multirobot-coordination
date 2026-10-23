# Rendezvous-Scheduled Coverage with Checkpointed, Communication-Free Failure Recovery

> **STATUS 2026-09-25 — this line is on hold; a load-bearing assumption was refuted.**
> HoloOcean showed that a fixed-duration stop neither resets nor absorbs accumulated
> delay, so assumption **A-L2** (exposure = the final approach leg) is false. Under the
> supported carry-forward model, correlation-alignment buys only ~25–33% relative
> false-alarm reduction while simply arriving early buys 10–100×. See
> `rendezvous_assumptions.md` **N4 (REFUTED)** and `HANDOFF.md` §3 before acting on
> anything below. The design reasoning here is still readable as design reasoning; its
> quantitative conclusions are not current.


*Design note — idea by the author (Atharva Jadhav), v2.1.*
*v1 (2026-06): concept draft. v2 (2026-07-08): revised after the feasibility probe.*
*v2.1 (2026-07-08): distributed appointments replace rigid pairs (§4.2); balanced*
*orphan split + total-ownership guarantee (§6); scope assumptions A1–A3 declared*
*in `rendezvous_problems.md`.*
*(`visualization/encounter_planning_probe.py`, outputs in `results/encounter_probe/`)*
*and design discussion. Sections are marked **[VERIFIED]** (simulated), **[FOLLOWS]***
*(simple counting/geometry), or **[UNTESTED]** (assumption to validate).*

---

## 1. One-paragraph summary

Plan the fleet's POI routes **before** the mission so that pairs of robots are
**scheduled** to pass within reliable comm range (a *vicinity contact*, ~60 m —
they never need to physically meet or stop) at known times, with known timing
tolerance. Every contact does three jobs at once:

1. **Detection with a deadline.** A missed scheduled contact is failure
   evidence that no radio burst can explain away. Every robot is verified at
   least every Δ seconds — a *deterministic* detection bound.
2. **Checkpoint + data ferry.** A successful contact proves the partner was
   alive *and* carries away a copy of its collected data. A lost robot costs
   at most one Δ-interval of data, and its orphaned POIs are exactly the ones
   scheduled *after* its last confirmed contact.
3. **News chain.** Every robot keeps appointments with ≥2 distinct territory
   neighbors, so any event witnessed by one robot reaches the whole team
   within a bounded, plan-certified time — by gossip along contacts that
   already exist, no long-range comms and no dedicated messenger.

Recovery is **negotiation-free**: a precomputed succession table (computable
before launch, since territories fix who is near what) maps every POI to its
one or two feasible substitute robots. The robot that discovers a missed
contact is, by construction, the territory neighbor — i.e. the designated
successor — so recovery-critical news has zero propagation delay.

**We never claim shortest paths.** The scheme deliberately trades extra travel
for guaranteed detection, bounded staleness, and loss-proof data in comm
regimes where radios cannot do the job.

---

## 2. What the feasibility probe established (2026-07-08)

The v1 draft rested on two premises. One died, one survived; two new facts
emerged. World: USVs, L=600 m field, 16–32 POIs, v=2 m/s, 60 s scans, R=20–60 m
reliable range, Gilbert-Elliott long-range channel (paper-consistent params).

- **[VERIFIED — NEGATIVE] "Detection for free" is dead.** Efficient
  (cheapest-insertion) coverage routes partition space; across K=4–8 and
  L=150–1200 m, 95–100% of robots got **zero** mid-mission close-range
  contacts (max check gap ≈ the whole mission). Widening the vicinity from
  20 m to 60 m changes nothing — the obstacle is the ~hundreds of meters
  between territories, not the last 60 m. Encounters must be **planned and
  paid for**, never harvested. (Q1; `q1_natural_density.csv`)
- **[VERIFIED] Price of a Δ-bounded contact schedule.** Post-hoc patching of
  meetings into finished routes does not converge (each detour lengthens the
  mission, spawning new gaps). Comms-blind allocation + meetings costs +76%
  → +1422% travel (Δ=500→120 s). **Buddy-banded allocation** (paired robots,
  adjacent territory) + synced meetings costs **+99% → +209%**, with roughly
  neutral makespan (banding load-balances). Rule of thumb: the cost is
  governed by **comm range ÷ neighbor-territory spacing**; near 1, contacts
  get cheap. (Q2; `q2_bound_cost.csv`)
- **[VERIFIED] Regime map — when this idea pays.** On long (~25 min) missions,
  whenever long-range packets arrive at all (even with ~50 s outages), a
  duration-re-tuned statistical detector (β=ρ·η family) **dominates**
  rendezvous: FA=0 at 61–88 s latency vs rendezvous 174 s mean. Packets are
  cheap; travel is expensive. **But under range-limited comms (150 m)** —
  silence spatially induced — statistical detection collapses (sensitive
  tunings: 4–5 false accusations/trial; conservative: 62–77% of real failures
  never detected), while the missed-contact detector keeps **FA=0, 0% missed,
  latency bounded by Δ+ε**. Planned contacts are the *detection floor* when
  the channel gives nothing; the channel-aware belief rides on top when it
  does. (Q3a; `q3a_latency.csv`)
  - Side-finding for the main paper: the paper's β calibration does not
    transfer to long missions (per-slot false-alarm probability accumulates);
    duration-aware thresholds are needed.
- **[VERIFIED] Drift is the failure mode, and it is manageable.** With
  contacts re-zeroing a pair's relative schedule error: 2% speed error → 0%
  false misses at ε=30 s tolerance; 5% needs ε=60 s; 10% leaves ~12% even at
  ε=60 s. (Q3b; `q3b_drift.csv`) Two moving robots stay inside a 60 m disk
  for ~30 s at 2 m/s — enough for heartbeats + data exchange with no stopping;
  scheduling contacts against a partner's 60 s scan dwell (a sitting target)
  absorbs most timing error for free.

---

## 3. Setup and assumptions

- A fleet of USVs and a set of POIs to scan (POIs ≫ robots); each robot gets
  an **ordered route**; POI sets are **disjoint** (no overlap).
- **Target comm regime: range-limited.** Reliable ("perfect") link within
  ~30 m of each robot → pairs exchange data within a ~60 m vicinity; beyond
  that, assume nothing (this is where the scheme pays — see regime map).
- Routing is computed pre-mission and is deterministic and shared: every
  robot knows every teammate's planned space-time track, the full contact
  schedule, and the succession table.
- Timing tolerance ε per contact (ε ≈ 30–60 s for 2–5% speed error).
- Mission type: **scan/survey** (duplicated work is wasted travel, not a
  conflict). See §8 for why this matters.

---

## 4. The plan structure

### 4.1 Territory partition (banded, buddy-adjacent)
POIs are split into per-robot territories arranged so that each robot has
**adjacent neighbors** (bands for a line layout; a grid of cells for large K).
Adjacency is what makes contacts cheap (Q2) and makes the succession table
short (§6). Within a territory, the route is a standard cheapest-insertion
path. **[VERIFIED** for bands; grid variant untested**]**

### 4.2 Contact schedule (distributed appointments, not rigid pairs)
There is **no global pairing structure and no synchronized rounds** (v2.1,
2026-07-08 — an earlier draft used a rigid alternating-pairs "brick wall";
demoted to an *example*, see below). The schedule is simply a **distributed
set of pairwise appointments**: each robot holds a list of contacts
(partner, time ± ε, place), agreed pre-mission, with whatever neighbors
geometry makes cheap — sometimes the same neighbor twice, sometimes three
different ones, asynchronous across the team. Each contact is a **pass-by
within the 60 m vicinity**; nobody stops; ideally timed against the
partner's scan dwell.

Any valid appointment set must satisfy four requirements — these, not any
particular pattern, are the spec:

- **R1 (detection bound):** every robot has *some* appointment at least
  every Δ → missed-contact detection within Δ+ε.
- **R2 (news flow):** the appointment graph, unrolled over time, connects
  every robot to every other → all news (deaths, checkpoints, re-plans)
  reaches everyone by normal gossip along contacts that already exist; no
  dedicated messenger, ever. The propagation bound is a *per-plan
  certificate* (compute the temporal diameter of the schedule and report
  it), not a fixed formula.
- **R3 (partner diversity):** every robot meets **≥2 distinct partners**
  per cycle. This is what makes a death get discovered *twice,
  independently*, keeps the graph connected when a node drops out, and
  lets slot-inheritance repair work with no messenger (§P1 register).
- **R4 (slack):** enough margin before each appointment that plausible
  task overruns don't break it (see problem P2).

The alternating-neighbors "brick wall" is just the simplest schedule
satisfying R1–R4 on a line of bands (it gives Δ+ε detection and ≈(K−2)·Δ
propagation) — useful for analysis, not part of the concept. The general
"chain calculation" — choosing a cheap appointment set satisfying R1–R4 for
irregular territories and unequal loads — is the research-grade scheduling
problem. **[UNTESTED beyond bands]**

### 4.3 What a contact carries
Heartbeat/liveness (implicit in the contact itself) + **checkpoint** (which
POIs I've completed) + **data** (scan results so far, ferried redundantly) +
**news** (everything I've learned from previous contacts: failures,
checkpoints, reassignments). No new message types; rate is per-contact, not
per-second.

---

## 5. Detection: a missed contact is the alarm

At a scheduled contact, if the partner does not appear within the vicinity by
T+ε:

1. **Self-suspicion first.** The observer checks its own schedule error
   (it knows its own drift) before accusing the partner. **[UNTESTED]**
2. **Verify, don't fire.** A missed contact triggers a *verification*, not an
   instant takeover: a short wait / local sweep / targeted-relay-style probe
   (the Alg-3 machinery from the main paper). A false miss costs a probe, not
   a wasted recovery. **[UNTESTED as protocol; relay verified in gap2]**
3. **Declare + scope.** On confirmation, the failure is declared with its
   **alive-interval**: dead sometime between its last confirmed contact and
   the missed one. Orphaned work = the POIs scheduled after the last
   checkpoint — at most one Δ-interval's worth is ambiguous. The searcher
   also knows *where* to look: one known route segment. **[FOLLOWS]**

---

## 6. Recovery: precomputed succession, no negotiation

Before launch, compute **one deterministic takeover pattern**
`f(dead robot v, v's last checkpoint c) → {orphan POIs, and the single taker of
each}`. It is a function of the frozen mesh alone (common knowledge), so every
survivor that learns of the death looks up the **same** pattern and executes
only its slice — no negotiation, and no disagreement even once live views
diverge under noise. Two ingredients:

- **Orphans from the last checkpoint.** Each contact is a progress report, so
  the last contact where v was seen alive fixes its last reported progress;
  the orphan set is *everything after that checkpoint* — a safe superset (any
  POI v quietly finished afterward is at worst re-scanned, benign for scan
  missions). This is the checkpoint-inference insight: knowing where v was
  last seen tells us exactly what it left behind.
- **Forward-preferred, one owner each.** Each orphan goes to exactly one
  taker — the surviving neighbor it is *forward-absorbable* for (ahead in that
  robot's sweep, woven into its remaining route as a marginal detour). Orphans
  that are *behind* every survivor's sweep front are deferred to an end
  cleanup, which also protects the survivors' other contracts from backtrack
  disruption. Balancing future load across the ≤2 feasible neighbors, never
  dumping everything on one.

Everyone else updates bookkeeping as the news reaches them through the chain.
*Validated in the first-order pipeline sim (`full_pipeline_sim.py`): the
takeover is a pure table lookup at runtime; over 120 deaths total-ownership
held 100%, ~65% of orphans were grabbed forward, and the chain re-welded every
time. See `rendezvous_problems.md` P1 and `rendezvous_assumptions.md` R2/R3.*

**Total-ownership guarantee:** every POI has exactly one designated owner at
all times (original owner, else its table successor, recursively). So no POI
can ever be dropped or double-taken, no matter how late the news travels —
the worst case is that a POI is done *at the end* by its owner instead of on
schedule. Timeliness and correctness are decoupled by construction.

**Why this is safe here (and wasn't before) — the pieces need each other:**

- Stage-1 E2 (main project) showed deterministic succession loses to
  negotiated reallocation because *statistical* detection under loss makes
  false death-calls, and a fixed table then double-books. Here detection is
  the missed-contact detector: **FA≈0 and 0% missed in the range-limited
  regime [VERIFIED]** — feeding a near-perfect detector into the table
  removes the table's historical weakness.
- Conversely, in the range-limited regime there are **no comms to negotiate
  over** — precomputed succession isn't just acceptable, it's the only
  option. Reliable detection makes succession safe; dead comms make it
  necessary.
- **Discoverer = successor.** The robot that finds the missed contact is the
  territory neighbor, which is exactly the precomputed successor — so
  recovery-critical news has ~zero propagation delay. The chain delay to the
  rest of the team (a per-plan certificate; ≈(K−2)·Δ for the banded example)
  affects only bookkeeping. **[FOLLOWS]**
- Late news is tolerable *for this mission type*: the worst outcome of a
  stale belief is a POI scanned twice or an orphan done late — wasted travel,
  never a conflict, and never a lost POI (the table maps each orphan to
  exactly one first successor). **[FOLLOWS, given §8 scope]**

---

## 7. Re-synchronizing stale robots

Handled structurally by the appointment requirements: R1 means no robot runs
on stale information longer than its next contact (≤ Δ), and R2 means
full-team consistency lags by at most the schedule's temporal diameter (the
per-plan certificate). Additionally, contract-preserving re-planning (P1
register) guarantees staleness can never cause a *missed* meeting — a stale
robot's own appointments are, by rule, still honored by everyone else.

**The overlap window (makes the above concrete).** A survivor that detects a
death does *not* diverge to the recovery plan immediately: it stays on its OLD
frozen track for its next **W** scheduled contacts (the overlap window), using
them to hand the death report + the new schedule to robots still on the old
plan. Only after W preserved contacts does it switch to the re-welded contact
points. This is the v1 "new path must overlap ≥2 consecutive old contacts"
rule, now precise: a stale robot is only ever at OLD contact points, so old
contacts are the *only* channel from an updated robot to a stale one — diverge
too early and a stale robot arrives at an old point, finds nobody, and falsely
concludes *it* died (a cascade). **Verified (Stage E, `full_pipeline_sim.py`):**
W=0 (immediate divergence) gives ~2 false-alarm cascades per death and informs
only ~33% of the team; **W=1 preserved contact drives divergence false alarms
to exactly zero**, and full-team propagation then rises with mission length
(21% of teams at N=30 → 80% at N=90 → 93% at N=120 POIs) — bounded by remaining
contact-rounds, *not* by W, exactly the R2 certificate. The design's "~2 POIs" instinct
is a sound margin (the measured floor is one preserved *contact*, ≈ 1·Δ, a bit
more than 2 POIs of travel). Incomplete propagation on short missions is
harmless by the total-ownership guarantee (§6): an uninformed robot never
causes a lost POI. **[VERIFIED for the propagation/false-alarm property;
window's coupling to orphan-absorption timing is Phase-2, ledger R2]**

**How long to reach the whole team, and the shape of the spread (Stage E2).**
Detection is inherently **multi-source**: every partner that misses a meeting
with the victim lights up independently (≥2 on a line), so the news starts from
several points and fans outward — its main benefit is *reach*, not raw speed:
at K=6 the multi-source spread informs the whole team on **48% of deaths vs
only 6% for a single discoverer** that must cross the line alone (when both
complete they take the same ~3.4 rounds — the gain is completion, not speed).
When it completes, it takes **≈3 rounds ≈ 8–10 POIs (~800–950 s)** at K=6–10
(p95 ≈4 rounds). But the growth is **linear, not exponential**: each round adds
a roughly constant slice of the team (r1 23% → r2 65% → r3 91% → r4 99%),
because the brick-wall line gives each robot only 2 distinct partners per cycle,
so the frontier advances ~2 hops/round.

**Higher-degree topologies (Stage E3) — the exponential is real but not worth
it here.** Raising the contact-graph degree does make the spread exponential:
at K=16 a hypercube schedule reaches the whole team in **3.6 rounds (~log K)**
vs the line's 11.6, and a small-world (local + occasional shortcuts) in 5.8.
But the price is steep and *self-defeating for physical robots*: hypercube
costs **3.75× the line's contact-travel** and its meetings span up to **half
the field** (small-world/ring up to the *whole* field via one wraparound),
which destroys the cheap-adjacency premise that made contacts affordable in the
first place (Q2), and the long meeting trips accrue more drift (Q3b). And the
benefit is low-value here anyway: propagation speed is *not* the binding
constraint — late news is harmless by total-ownership (§6). The sensible
upgrade is therefore the *mild* one: not a 1-D high-degree schedule, but the
field's own **2-D structure — a grid** (degree 4, all meetings still local,
√K diameter, no blind ends). So the "exponential" intuition is reachable, but
it's a property of the schedule's degree, and for this problem the cost
outweighs the gain.

**The grid, built and run through the whole pipeline (Stage G).**
`build_grid_mesh` puts one robot per cell of a Gr×Gc grid with an edge-coloured
schedule (each robot meets its ≤4 neighbours once per 4-round cycle, every
meeting at a shared cell edge). Full pipeline, 4×4 grid vs 1×16 line at K=16:
the **blind spot is gone** (min distinct partners 1 → 2, so every robot —
corners included — gets redundant detection); **propagation is ~1.8× faster**
(6.4 vs 11.5 rounds, all meetings still local); **recovery is better** (orphans
grabbed forward 94% vs 54%, survivor overhead 21% vs 30%, because compact cells
put more orphans ahead of a survivor's sweep); total-ownership and chain
re-weld stay 100%. The **honest cost**: contact-travel overhead rises to 114%
(from the line's 46%) — degree 4 is ~2× the meeting trips and square cells give
longer hops than thin bands — and detection is marginally slower (mean 184→229
s, within-Δ+ε 91%→80%, since a given partner is met once per 4-round cycle). So
the grid buys resilience (no blind spot, redundant detection everywhere, faster
news, cheaper recovery) at roughly 2× the contact tax — the right trade when
fault-coverage matters more than the travel budget.

**Generalizing the schedule to an arbitrary contact GRAPH — and why the grid
stays the sweet spot (Stage S, 2026-07-24).** The line and grid are just
particular graphs; the honest way to think about "parallel not series" is as
graph invariants — min-degree (witnesses per death), single-/two-death
connectivity (survivors stay reachable), algebraic connectivity λ₂ (parallel
spread rate), diameter (worst-case news hops). We built the **small-world
hybrid** the design would naïvely reach for — grid backbone **+ a few
long-range shortcuts** (`build_smallworld_mesh`) — and measured all of these.
Going line → grid → small-world: min-degree 1→2→2, single-death-safe
12→100→100%, two-death-safe 2→97→99%, λ₂ 0.04→0.59→1.03, diameter 15→6→3.9.
The **line → grid** step is the real prize and it is nearly free. The **grid →
small-world** step does *not* earn its keep for a one- or two-death model:
- news does **not** get faster (7.9 vs 6.4 rounds — *slower*) because a
  higher-degree graph needs more colour classes, so the longer schedule cycle
  eats the smaller diameter (chromatic index ≥ max degree — fundamental, not a
  tuning artifact);
- recovery is **identical** (takeover was already graph-native — each orphan to
  the single nearest capable survivor — so it stays local even as news goes
  global);
- the grid **already** survives 1–2 deaths (100 / 97%), so shortcuts add only
  marginal multi-death robustness (→ 99 / 100%) at **+20% travel**.

So a general graph is the right *frame*, but the 2-D grid is the operating
point; the small-world is a lever to reach only for *high-order* fault tolerance
(many concurrent deaths) or a continuous/high-frequency gossip regime. Frontier,
knee, and the full comparison: `results/full_pipeline/stageS2_frontier.png`,
`stageS_compare.csv`.

---

## 8. Scope and honest limits

*Full problem register with per-problem status: `docs/rendezvous_problems.md`.*

- **Scan/survey missions only (for now).** Duplicated work is benign waste.
  For retrieval/manipulation, duplicated takeovers are real conflicts and
  late news bites — that variant needs the physical-mutex layer or stronger
  agreement before takeover.
- **Second-order failures are the weak point.** If the designated successor
  also dies, the backup is farther down the chain and acts on older
  information. First place the table can actually break — simulate before
  trusting beyond one failure deep. **[UNTESTED]**
- **The travel tax is real.** Budget roughly +100% travel for Δ=300 s at
  L=600 m / K=4 density with current planners. It shrinks with team density
  (comm-range/spacing ratio → 1) and possibly with phase-synchronized
  sweeps (neighbors sweeping their bands in step so tracks run parallel and
  pass close without dedicated detours) — promising, **[UNTESTED]**.
- **Do not deploy this where radios work.** The regime map is explicit:
  with usable long-range packets, re-tuned statistical detection is better
  and free. This scheme is the floor for range-limited/partitioned regimes,
  and a complement — not a competitor — to the main paper's detector.
- **End-to-end consistency is the biggest untested claim**: the full pipeline
  (bands + distributed appointments + checkpoints + succession) has not been
  simulated as a system. Next experiments, in order:
  1. full pipeline under range-limited comms vs the paper's statistical
     stack: coverage, duplicate work, orphan-completion delay;
  2. double-failure (successor also dies);
  3. phase-synchronized sweep planner (contact cost at higher density);
  4. self-suspicion / verification protocol under drift.

---

## 9. Relation to existing work (cite, don't claim)

- **Scheduled contacts / contact plans:** DTN contact-graph routing
  (arXiv 2108.07092).
- **Missed meeting as detectable event:** co-observation planning
  (arXiv 2403.13266; HoLA Robots arXiv 2301.10704) — closest prior art for §5.
- **Recurrent/intermittent connectivity planning:** Hollinger & Singh
  (periodic connectivity); Kantaros & Zavlanos (intermittent connectivity);
  intermittent rendezvous with synchronization agreements (arXiv 2309.13494).
- **Communication-free reallocation from shared models:** FORMICA; implicit
  coordination (arXiv 2501.03907, 2302.10393).
- **Indirect verification of a silent node:** SWIM (DSN 2002) — already cited
  in the main paper's relay section.

**Novelty check additions (2026-07-08, verified by reading, not just abstracts):**
- arXiv 2403.13266 (co-observation): missed scheduled observation detects
  **adversarial plan-deviation**, not faults; **no recovery**, no
  detection-latency bounds, no travel-cost analysis, proximity abstract.
- arXiv 2309.13494 (intermittent rendezvous exploration): has pre-agreed +
  renegotiated-at-meetings schedules (≈ our contracts), but **assumes all
  robots reach every rendezvous** — no failure detection, no takeover, no
  staleness bounds; objective is exploration/map-sharing.
- Boustrophedon coverage redistribution (Sensors 2024, PMC11644315): the
  orphan-redistribution half (neighbor-proportional, no negotiation) — but
  **assumes reliable communication**, detection unspecified, no bounds.
- arXiv 2305.05485 (resilient LTL planning): reassignment on failure,
  failure assumed known/communicated.
- Reliability-aware coverage path planning (patent US 11868931): proactive
  redundancy (overlap visits) instead of detect+recover — contrast baseline.
- Data-gathering under intermittent comms (arXiv 1706.02092): a missed
  meeting is treated as a *cancelled meeting* (max-wait to avoid deadlock),
  not as failure evidence.

**Whitespace this note targets** (per June 2026 lit check): the *regime map*
(when packets beat planning and vice versa), the quantified **Δ-bound vs
travel Pareto for failure detection** (not data exchange), and the coupling of
scheduled contacts to **checkpointed, precomputed, negotiation-free recovery**
with provable detection/staleness bounds under range-limited comms.
