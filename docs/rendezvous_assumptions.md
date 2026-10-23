# Rendezvous concept — realism ledger (idealizations → deployable)

*Companion to `rendezvous_allocation_idea.md` (design) and
`rendezvous_problems.md` (ranked conceptual weaknesses P1–P8). This file is
different in kind: it lists every **unnatural thing** currently baked into the
pipeline (`visualization/full_pipeline_sim.py`) and the concept, declares each
as an explicit assumption, and orders them for removal. The rule: **every
result we report must name which of these it still rests on.** We retire them
one by one, and "real-life usable" = enough of them retired that the remaining
ones are honest scope, not hidden cheating.*

Status tags:
- **ASSUMED** — idealized in the pipeline right now; on the roadmap to retire.
- **PROBED** — stressed in an *isolated* experiment but not yet inside the
  full pipeline (so the pipeline's headline numbers don't yet feel it).
- **PERMANENT SCOPE** — deliberately not solved; removing it changes the
  problem (e.g. deletes the niche). Kept honest, not retired.
- **RETIRED** — now modeled realistically in the pipeline.
- **REFUTED** — we tested it against independent physics and it is *false*.
  Anything that rested on it is withdrawn, not merely caveated.

The headline claims every assumption is measured against:
**(C1)** detection latency bounded by Δ+ε · **(C2)** ~zero false alarms ·
**(C3)** total-ownership (no POI dropped or double-taken) ·
**(C4)** chain re-welds after a death.

---

## Group N — Navigation & timing

**N1. Perfect navigation — robots hit the frozen appointment point exactly on
time.**  *Status: PROBED (2026-07-24, Stage M) — the fix is a liveness detector,
and it favours the grid; not yet wired into the pipeline.*
Reality: **currents/wind push robots off course → they arrive late/displaced and
miss the rendezvous *window*** (vicinity comms themselves stay perfect — L1 — so
this, not link loss, is the true source of "a live robot looks absent"). The
key physics is that this disturbance is spatially + temporally **correlated** (a
current fills a region for a while), which changes everything vs the i.i.d.
link-loss contrast (Stage L):
- **A current that delays *both* partners equally does not break their meeting** —
  they are co-located, both late, and still find each other. Misses come only
  from **differential** delay (one in the current, one out / across a gradient).
- So **multi-witness confirmation is the wrong tool here** — a current makes a
  whole region miss together, so the witnesses agree on a *benign* absence. Stage
  M (K=16, peak delay swept in units of ε): single-miss FA runs 0→89%, and even
  2-witness still hits 44% at the strongest current — the *opposite* of the
  i.i.d. case where independent witnesses helped.
- **A liveness detector fixes it**: declare a robot dead only if it made **zero**
  contacts for `persist` cycles running. A delayed-but-alive robot always still
  meets a *co-disturbed* neighbour, so it is never fully silent → FA stays ≈ 0
  (≤ 1.5% up to 3ε, ~6% at 4ε), while a true death (permanent silence) is still
  caught 100% at a cost of `persist−1` cycles of latency.
- **The grid beats the small-world** here: local meetings *share* the disturbance
  (equal delay → still meet), while the small-world's long shortcuts straddle
  current boundaries and add differential misses (2-witness FA 20% vs the grid's
  6% at 2ε). Currents *reinforce* grid + liveness as the design.

To retire N1 for real: fold the current/wind delay + the **liveness (persistence)
detector** into the pipeline's `run_failure`, then re-measure C2 as *survives
currents*, not *zero by construction*. **Load-bearing for C2. Priority #1.**
Files: `results/full_pipeline/stageM_environment.{png,csv}`.

**N4. A stop resets the disturbance — "accrual assumption" A-L2.**
*Status: **REFUTED** (2026-09-25, HoloOcean). This one was load-bearing for the
whole correlation-alignment line, and it is false.*
The assumption, introduced when the alignment theory was folded onto the
coverage allocation: because the router replans from the robot's actual clock
whenever it finishes a scan, a completed stop is a *resynchronisation point*, so
the delay a robot carries into an appointment is set only by its **final
approach leg** — offset = κ·t_leg·u(locale). Every result in the alignment line
(the joint duration–separation condition, the Θ(B^1/3) relaxation, the
"natural tour crossing" construction, the admissible-mission envelope) depends
on it, because all of them work by making that final leg short.

**The measurement.** Drive an AUV A→B→dwell(h)→C under a constant current and
compare against the same path with no current (`--exp` stop test; the harness is
`visualization/holoocean_eq3_validation.py`, the probe script is in the session
scratchpad). If a stop resets accrual, the offset at C should depend only on
B→C. It does not:

| dwell h | 0 s | 5 s | 30 s | 60 s |
|---|---|---|---|---|
| offset at −1 m/s | +13.62 | +13.12 | +13.12 | +13.12 |
| offset at −2 m/s | +46.66 | +45.08 | +45.08 | +45.08 |

The offset is the **sum of both legs' delays**, unchanged by the dwell. A
fixed-duration stop neither resets the accumulated delay nor absorbs it — only
*clock-based* waiting (arrive early, hold until the appointment) absorbs, and
then only up to the slack it buys.

**What follows.** Re-running every false-alarm figure under the supported
carry-forward model (`c4_metrics(..., model="carry")`, exposure = every travel
segment in the window) at σ/ε = 4: unaligned 24.1%, corridor 23.0%, crossings
h=1 20.2%, h=4 17.2% — against absorb +120 s 4.9% and +240 s 2.0%. **Alignment
buys ~25–33% relative; absorption buys 10–100×,** with no crossover anywhere
from 0.5× to 4×. Theorem 1 is mathematically sound but about the wrong
quantity: final-leg duration is not the operative exposure.

**Caveat, stated because it cuts both ways.** The carry model applies the
disturbance to a trajectory planned *without* it, so the robot never adapts —
an **upper** bound on false alarms. A-L2 assumed instantaneous full adaptation —
a **lower** bound. A real robot sheds work when it runs late (meeting-wins),
converting delay into lost coverage rather than lateness. Reality is between,
and settling it needs **closed-loop replanning under disturbance**, which
neither model does. That experiment must come *first* in any resumption of this
line, and note that it cannot rescue the crossing construction either way: if
reality is near A-L2, a 5 s dwell corridor gets the same protection at bare-mesh
cost (+36.6%, 0.00% FA) and dominates crossings; if reality is near carry-forward,
alignment barely helps at all.

**UPDATE 2026-10-20 — the closed-loop experiment is DONE, and the bracket does not
close.** `visualization/holoocean_closed_loop.py` flies the window in HoloOcean with
a replanner that sheds work when it runs late. Reality is **not** between the two
bounds: adaptation is **quantised**. A task is worth ~150 s, the drift it corrects
is 9-45 s, so the system alternates between the bounds instead of interpolating.
A replanner that targets the bare appointment destroys 20% of coverage at 1 m/s to
avoid 12 s of lateness that the ε=30 s grace already covered; one that targets
appointment+ε is strictly dominant to 1 m/s but still misses at 2 m/s. Task-granular
shedding never lands on target. The actuator has to be continuous — speed, or the
`absorb` margin. Full table in `HANDOFF.md`; raw rows in
`results/holoocean_closed_loop/closed_loop.csv`.

**N2. Clocks are perfectly synchronized.**  *Status: ASSUMED.*
Reality: clock skew looks exactly like drift (N1) — a robot "on time" by its
own clock is late by yours. Buys us: appointments are absolute-time with no
skew term. To retire: fold a clock-skew term into N1 (same machinery).

**N3. Position is known exactly (no localization error).**  *Status: ASSUMED.*
Reality: GPS/INS error means the meeting *point* itself is fuzzy; two robots
aiming at the same coordinate can be tens of metres apart. Buys us: the 60 m
contact range is treated as a hard clean disk. To retire: add localization
noise; couples to L1 (the contact link already has to tolerate a gap).

---

## Group L — Links & sensing

**L1. The close-range contact link is perfect within R = 60 m.**  *Status:
ASSUMED — kept deliberately (user, 2026-07-24): "perfect comms in the vicinity."*
When two robots are actually within range at a rendezvous they exchange
flawlessly; delay and apparent loss come from the *environment* (currents/wind →
N1), not from the radio/acoustic link. So the real detection stressor lives in
N1, not here.

*Illustrative contrast — why the mechanism matters (Stage L, i.i.d. link loss).*
Before adopting the "perfect vicinity" model we probed the *other* extreme — an
**independent** per-contact drop w.p. p_loss — precisely because it gives the
OPPOSITE answer to the correlated environmental model (N1), and the contrast is
the point:
- under **i.i.d.** loss, drops are independent, so **multi-witness confirmation
  helps**: single-miss FA ≈ degree·p_loss (27→77 per 100 robots/cycle, grid), a
  2-witness rule takes it quadratic (3→35);
- under **correlated** currents (N1), a whole region misses together, so
  witnesses agree on a benign absence and **multi-witness does *not* help** — a
  liveness/persistence detector does.

So the confirmation rule must be chosen for the *correlation structure* of the
real disturbance. Since the user's model is perfect vicinity comms + correlated
currents, N1's **liveness detector** is the operative fix, and the i.i.d.
results above are retained only as the contrasting bookend.
Files: `results/full_pipeline/stageL_*` (i.i.d. contrast),
`stageM_environment.*` (the real, correlated model).

**L2. Data ferrying at a contact is instantaneous and always succeeds.**
*Status: ASSUMED (P6).*  Reality: sonar/imagery may not transfer in a ~30 s
pass-by; the link can drop mid-transfer. Buys us: checkpoints/orphan bounds
assume the ferry always lands. To retire: ferry only checkpoints + thumbnails
(small), full data on vehicle recovery; quantify the transfer budget vs dwell.

**L3. There is *no* long-range comms at all.**  *Status: **PERMANENT SCOPE**.*
This is the niche, not a bug: the regime map (Q3a) shows that if *any*
long-range packets arrive, statistical channel-aware detection dominates and
rendezvous detection doesn't pay. Removing this assumption deletes the reason
the method exists. We keep it as a **declared operating regime** (shadow zones,
covert, deep/cluttered water), never "solve" it.

---

## Group F — Failure model

**F1. A failure is a clean instantaneous stop-dead.**  *Status: ASSUMED.*
Reality: partial/degraded failures — dead radio but live motors, a robot
circling, drifting with current, or intermittently alive. These are *worse*
than death: a half-alive robot can attend some contacts and miss others,
confusing detection. Buys us: detection = "did it show, yes/no." To retire:
add degraded-failure modes and check they still resolve (probably via P4
self-suspicion + verification). **Priority #2 — the nastiest realism gap.**

**F2. Death is permanent; the victim never returns.**  *Status: ASSUMED.*
Reality: reboots, temporary occlusion, a robot that surfaces late. The design
*claims* false alarms self-heal ("a late-not-dead robot reappears where its
ex-partners now meet") — but the pipeline never tests a comeback. To retire:
inject transient failures and verify reintegration actually happens.

**F3. Failures are independent of place and of each other.**  *Status:
ASSUMED (overlaps A1).*  Reality: hazards (shallows, nets, traffic) cause
*spatially/temporally correlated* failures — exactly the case A1 ("single
failure at a time") assumes away. Buys us: one death, settled before the next.
To retire: this is retiring **A1** — simultaneous/adjacent deaths, the
repaired-pair rule applied recursively, stale-table refresh (P5).

**F4. The observer itself never faults while observing.**  *Status: ASSUMED
(this is P4).*  Reality: if my radio/clock/nav fails, every contact I attend
looks missed and I accuse partner after partner while being the problem. Buys
us: a missed contact unambiguously implicates the *other* robot. To retire:
the self-suspicion protocol — two missed contacts with *different* partners →
suspect self → fail-safe (go loud / go home / hold accusations). **Load-bearing
for C2, bundles with N1/L1 in Priority #1.**

---

## Group R — Schedule, recovery & topology

**R1. The frozen offline mesh is perfectly executable and identically known to
all.**  *Status: "identically known to all" PROBED via gossip (2026-07-24, Stage
K); "perfectly executable" still ASSUMED (= N1/L1).*  Reality: plan distribution
can differ per robot; execution deviates (N1/L1). The **common-knowledge** half
is now backed by an explicit mechanism instead of an assumption: every rendezvous
is a mutual **heartbeat** (each robot hears a live heartbeat from ≥2 distinct
partners on the grid — liveness multiply-confirmed, no single-source blind spot)
PLUS a **full-state merge** (who is alive + every robot's scan progress).
`gossip_check` (Stage K, grid 4×4, 150 deaths): survivors reach a **complete,
consistent global view in 100%** of deaths, converging in **~7 rounds** (the
staleness bound), and the victim's last-ferried state is still held by a survivor
in **100%** of deaths — **a single death loses no information** (it was shared at
its last meeting). This is exactly what the checkpoint/orphan logic depends on
(orphans = POIs the victim hadn't ferried as done by its last meeting). The
residual is whether two robots can *disagree* about the current plan under noise
(N1/L1) and how contracts reconcile it. Files:
`results/full_pipeline/stageK_gossip.csv`.

**R2. Recovery — takeover precomputed AND now forward-executed.**
*Status: RETIRED (first-order) 2026-07-24 (Stage X).*  The takeover is a
precomputed contingency table `f(v, c)` (dead robot × last checkpoint → orphan
set + single taker each), built offline from the frozen mesh and applied at
runtime by pure lookup — one shared pattern, so survivors cannot disagree even
once live views diverge. The remaining residual (the survivors' *other*
appointments were only checked on the schedule graph, not played forward) is now
**closed**: `execute_repaired_mesh` plays every taker's post-death timeline
forward with its scheduled appointments interleaved as hard time+place
constraints under meeting-wins, and checks by EXECUTION — not assertion — that
the pre-planned orphan pickup does not make a survivor late for any *other*
meeting. Result (grid 4×4, 150 deaths): **appointments kept 100% (min 100%),
total ownership completes within the mission 100%**, at a true worst-taker
makespan cost of 25% mean / 46% p95 — **+6.5pp above** the isolated-route
estimate (18.6%) that ignored the appointment diversions. *Why it holds* (and
why the design was right to be simple): meeting-wins defers work not meetings,
and the slots a survivor used to spend meeting the **dead** robot are now free —
so the orphan pickup largely fits into time the survivor already had. On the
grid (operating point) the "re-weld" needs no separate appointment at all —
single-death connectivity is automatic (Stage S: single-death-safe 100%), so
forward-execution reduces to confirming takers keep their existing contracts,
which they do. The transition-consistency half was already modeled by the
**overlap window** (Stage E: W=1 → zero divergence false alarms). *Note:* the
runtime weave (`_weave_forward`) is a deterministic function of common-knowledge
inputs, hence equivalent to a lookup; it could be stored literally in the table
if a zero-computation runtime is wanted. Remaining scope on recovery is
multi-failure (P5/A1), not this. Files: `results/full_pipeline/stageX_*`.

**R3. Orphan pickup model.**  *Status: RETIRED → forward/behind model
(2026-07-23), one residual.*  ~~Original shortcut: every survivor finished its
whole own route, then all orphans were tacked on at the tail (delay mean 78% /
p95 120% of makespan) — a true upper bound but NOT the intended design.~~ Now
implemented as designed: the instant a survivor detects the death it looks at
where it is and grabs the dead robot's POIs that are **ahead in its sweep**
(forward-absorbable) *inline*, weaving them into its remaining route; only POIs
**behind** its sweep front are deferred to an end cleanup. Exclusivity via the
succession table (one owner per orphan), preferring the survivor the orphan is
forward for. Orphan set is checkpoint-based (POIs after v's last checkpoint —
see R2), a deliberate safe superset. Result (120 seeds): **~65% of orphans
grabbed forward** at ~48% of makespan delay (done on the way), ~35% deferred
(~84%); worst-case tail p95 130% driven by the behind minority + the
conservatively re-scanned checkpoint gap. Residuals: "forward" uses a
sweep-progress (y-latitude) heuristic with a backtrack tolerance =
REACH_FRAC·band-width (fine for dock-at-bottom sweeps, needs generalising for
non-monotone routes); and deferring behind orphans protects the survivor's
*other* contracts from backtrack disruption — that coupling to P2 slack is
asserted, not yet forward-simulated (see R2).

**R4. Topology — the line's end robots have only one partner.**  *Status:
RETIRED via 2-D grid (2026-07-23), with an honest cost.*  The line's two ends
had only 1 partner (R3 fails → 2Δ+ε bound, single point of monitoring failure).
Topology study first (Stage E3, K=16): raising the contact-graph degree *does*
turn propagation from linear (line 11.6 rounds) into exponential (hypercube 3.6
≈ log K; small-world 5.8), but 1-D high-degree schedules cost 3.75× the
contact-travel and force meetings that span up to the whole field —
self-defeating for physical robots, and low-value (late news is harmless by
total-ownership). So the fix taken is the field's own 2-D structure: **a grid**
(`build_grid_mesh`, edge-coloured schedule, Stage G). Full pipeline, 4×4 grid vs
1×16 line at K=16:
- **Blind spot gone:** min distinct partners **1 → 2** (every robot, corners
  included, has ≥2 partners → redundant detection everywhere).
- **Propagation ~1.8× faster:** 6.4 vs 11.5 rounds to reach the whole team,
  with all meetings still *local* (cell edges, no field-crossing) — the sweet
  spot vs hypercube's 3.6 rounds at 3.75× travel.
- **Recovery better:** orphans grabbed forward **94% vs 54%**, survivor travel
  overhead **21% vs 30%** (compact cells → more orphans forward-absorbable).
- **Correctness held:** total-ownership 100%, chain re-weld 100% (both).
- **The cost (honest negative):** mesh contact-travel overhead **114% vs 46%**
  — degree 4 means ~2× the meeting trips, and square cells give longer per-trip
  travel than the line's thin bands; detection is also marginally slower (mean
  184→229 s, within-Δ+ε 91%→80%, since each specific partner is met once per
  4-round cycle). So the grid buys resilience (no blind spot, redundant
  detection, faster news, cheaper recovery) at roughly **2× the contact tax** —
  worth it when the deployment values fault-coverage over the travel budget.

**R6. The contact structure is a fixed low-degree chain/lattice.**  *Status:
GENERALIZED to an arbitrary contact GRAPH (2026-07-24); small-world hybrid probed
and found NOT worth it for the single-death model — the grid stays the sweet
spot.*  The line and even the grid bake in a *series* structure (fixed
neighbours, hop-by-hop news). The real object is a graph: robots are independent
nodes, meetings are edges we design, and the properties we care about are graph
invariants. Built the **small-world hybrid** (`build_smallworld_mesh`, Stage S):
grid *local* backbone (min-degree ≥ 2 even at corners) **+ n_short random
long-range shortcuts** between non-adjacent cells (greedy edge-colouring
schedule; shortcut meetings cross the field). Recast the metrics as invariants
(`_graph_invariants`: min-degree, single-/two-death connectivity, algebraic
connectivity λ₂, diameter). Line → grid → small-world (K=16):

| invariant | line | grid | small-world |
|---|---|---|---|
| min degree (≥2 = no blind spot) | 1 | 2 | 2 |
| single-death keeps team connected | 12% | 100% | 100% |
| two-death keeps team connected | 2% | 97% | 99% |
| algebraic connectivity λ₂ | 0.04 | 0.59 | 1.03 |
| diameter (worst news hops) | 15 | 6 | 3.9 |

**Honest verdict — the big win is line → grid, and it is nearly free; grid →
small-world does NOT pay off for a 1–2 death model:**
- **News does NOT get faster.** small-world 7.9 vs grid 6.4 *rounds* to inform
  the whole team — *slower*, despite the shorter diameter. A higher-degree graph
  needs more colour classes ⇒ a longer schedule cycle ⇒ each edge fires less
  often, and that eats the diameter gain. This is fundamental (chromatic index ≥
  max degree), not a tuning artifact. (Measured on the bare unbounded-round
  schedule — the short coverage mission truncates propagation to nothing.)
- **Recovery is identical** (96% forward, 22% overhead both): the recovery
  machinery was already graph-native (each orphan → the single nearest capable
  survivor; news seeded by *every* misser), so takeover stays *local* even as
  detection/news go *global*.
- **The grid already survives 1–2 deaths** (100 / 97%); shortcuts lift two-death
  survival only marginally (→ 99 / 100% at +16 shortcuts) at **+20% travel**
  (132% vs 113% mesh overhead). Frontier (`sweep_shortcuts`): λ₂ rises ~linearly
  but *diameter* saturates (6→4 by +8 shortcuts, floor 3), and diameter — not
  λ₂ — bounds worst-case news, so the knee (if used at all) is ≈ +8 shortcuts.

So the **2-D grid is the sweet spot**; the small-world is a lever to pull only
for *high-order* fault tolerance (many concurrent deaths) or a continuous /
high-frequency gossip regime where the diameter, not the schedule cycle,
dominates. Files: `results/full_pipeline/stageS_compare.csv`,
`stageS2_shortcuts.csv`, `stageS2_frontier.png`, `smallworld_instance.png`.

**R5. The succession table, computed once offline, stays valid.**  *Status:
ASSUMED (this is P5).*  Reality: after the first recovery, territories changed,
so the table is stale for the *second* failure. Softened by the balanced-split
rule and by A1 for now; full fix is a canonical recompute-from-death-report so
all robots converge on the same new table. To retire: bundles with F3/A1.

---

## Group P — Physics & environment (fidelity phase)

**P1. Constant speed, instant turns, straight-line travel.**  *Status:
ASSUMED.*  Reality: turning radius, acceleration, momentum. To retire: vehicle
dynamics (HoloOcean).

**P2. No currents, wind, or waves.**  *Status: ASSUMED.*  Reality: a current
biases every leg and every arrival time (feeds N1). To retire: current field in
the sim / HoloOcean.

**P3. No obstacles, coastlines, or keep-out zones; field is a clean square with
vertical-strip bands.**  *Status: ASSUMED.*  Reality: irregular survey polygons,
islands, traffic-separation zones. To retire: real geometry + obstacle-aware
routing (may break the tidy band partition).

**P4. Every POI takes an identical, deterministic 60 s scan.**  *Status:
ASSUMED.*  Reality: scan time varies by target/sensor and can require re-scans —
this is the *source* of the P2-problem overruns. Partly handled: Stage D shows
"meeting wins" converts overruns to deferred work. To retire: a scan-time
distribution feeding both the schedule slack and the deferred-work tail.

**P5. Surface, 2D world.**  *Status: **PERMANENT SCOPE** (for now).*  Matches
the USV scan-mission framing (A2). AUV/3D is a separate platform study.

---

## Group E — Evidence quality

**E1. Travel/delay magnitudes are estimates, not bounds.**  *Status: ASSUMED
(this is P8).*  Crude greedy planners on an abstract kinematic sim. The
*structural* claims (C1–C4) hold; the *magnitudes* (+50% mesh tax, 78% orphan
delay) are estimates. To retire: better planners + higher-fidelity sim.

**E2. Results are at one parameter point (K=6, N=30, L=600, Δ=300).**  *Status:
ASSUMED (scaling = P7).*  News staleness is linear in K on a line. To retire:
sweep K/N/L/Δ; likely needs hierarchical territories at large K.

---

## Roadmap — the order we retire these

Ordering principle: retire first whatever a headline claim is *silently
resting on*, respecting dependencies. Each phase ends with the pipeline's
result labels rewritten to drop the retired footnotes.

| Phase | Retire | Makes honest | Cluster |
|---|---|---|---|
| **1. Harden detection** | N1, N2, L1, F4 | C2 ("~0 FA") becomes *survives noise + verification*, not *zero by construction* | drift + lossy contact link + self-suspicion + grace/verification, all in the live replay |
| **2. Make recovery real** — **DONE (first-order)** | R2, R3, R4, R6 | C3/C4 now proven by *forward execution* (Stage X: appointments kept 100%, ownership 100%), not asserted on the schedule graph; end-robot tail closed by the grid; topology generalized (small-world probed, grid is the operating point) | forward-execute the repaired mesh **done**; grid topology **done** |
| **3. Relax failures** | F1, F2, F3(=A1), R5(=P5) | detection/recovery survive degraded, transient, and correlated/simultaneous failures | partial-failure modes; comeback reintegration; recursive repair + table refresh |
| **4. Physical fidelity** | N3, L2, P1, P2, P3, P4 | magnitudes become trustworthy under dynamics/currents/obstacles/real links | HoloOcean port |
| **5. Scale & write** | E1, E2 | claims hold across K/N and the numbers are defensible | scaling sweep, hierarchical territories, then the paper |
| **Never** | L3, P5 | these *are* the scope (no long-range comms; surface scan missions) — declared, not solved | — |

**Where we are now:** everything above Phase 1 is ASSUMED except L3/P5
(permanent scope), R3 (retired — forward/behind orphan pickup), R4 (retired —
2-D grid fixes the blind spot), R6 (generalized to an arbitrary contact graph;
small-world probed and found *not* worth it — grid is the sweet spot), N1
(probed under correlated currents/wind — Stage M — the fix is a liveness
detector, which favours the grid; not yet wired into the pipeline; L1 kept
ASSUMED = perfect vicinity comms per the user, with Stage L's i.i.d. link-loss
kept only as the contrasting bookend), R2 (RETIRED first-order — the repaired
mesh is now forward-executed, Stage X: appointments kept 100%, ownership 100%),
and the Stage-D
slack/meeting-wins handling of P4-the-problem (partial, now *parked* under the
equal-scan-time assumption — P2). C1/C3/C4 hold in the first-order sim (line,
grid, small-world, and at K=36 scale where the line→grid win only sharpens); C2
holds only *by construction* until the confirmation rule lands (Phase 1).
Nothing is being hidden — each result in `rendezvous_problems.md` and the idea
doc cites the assumptions it rests on.
