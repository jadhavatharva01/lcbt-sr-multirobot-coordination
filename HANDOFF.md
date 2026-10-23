# Project handoff — state as of 2026-09-25

**Purpose.** Hand a fresh session the *complete* state in one document, so nothing
is re-derived. Everything below is verified by simulation or measurement, not
asserted. Where a claim has been withdrawn, it says so.

---

## 1. Read this first

The project has run through **three method lines**. Two are dead. One is a
finished paper.

| line | what it is | verdict |
|---|---|---|
| **Era 1 — layered LCBT** | BT + intent-wait + bid-margin + adaptive wait, vs CBBA | **dead** — pre-empted by published prior art |
| **Era 2 — channel-aware failure recovery** | correlated-silence detector, β = ρ·η | **SHIPPABLE.** `paper/main.tex` is submission-ready |
| **Era 3 — comms-free rendezvous** | meetings as the only failure signal; correlation-aligned rendezvous | **refuted** — the load-bearing assumption is false |

**If you do one thing: compile and submit Era 2.** It is the only shippable
artifact five months of work has produced, and it was finished while attention
was on Era 3.

---

## 2. Era 2 — the paper (DO THIS)

**Idea.** *Silence is evidence whose meaning is set by its correlation across the
team.* Isolated silence (the monitor hears everyone but X) = X failed. Correlated
silence (the monitor hears almost no one) = an outage, hold.

- silence severity `ρ = clip((σ − 2h)/4h, 0, 1)`, σ = time since last heartbeat, h = heartbeat period
- channel health `η` = fraction of *other* teammates heard in window W
- **failure belief `β = ρ·η`**, take over when `β ≥ τ`, with `τ = c_conflict/(c_miss + c_conflict)`

η is free — a by-product of heartbeats already received. Multiplicative coupling
is the principle: long silence means failure **only when** the rest of the team
is audible.

### Results (all re-verified 2026-09-25)
1. **Scaling gap.** False takeovers/trial, burst, drop 0.2: φ-accrual **0.695 → 2.487**
   as the team grows N=6 → 32; channel-aware stays flat **0.072–0.107**. Gap
   **9.0× → 25.5×**. Source: `results/stage1b/stage1b_scale_sweep.csv` (K is pairs, N=2K).
2. **φ-accrual false-alarm floor.** Under burst it cannot go below ≈0.46–0.49 at
   any tuning. Verified 9× past the paper's swept range: **k = 14 → 120 leaves the
   rate pinned**, latency balloons 5.3 → 14.2 s, then recovery collapses 100% → 55%.
3. **ROC dominance** in the practical regime; ~15–20% faster than channel-aware
   timeout, ~30–45% faster than fixed timeout at matched false alarms.
4. **Downstream energy**: wasted travel flat <1 vs φ-accrual 15.2 at K=16 (~20×).

### An overclaim found and fixed (2026-09-25)
The paper attributed the floor to **per-node** detection as such. That is false:
`fixed_timeout` is equally per-node and reaches **0.000** false takeovers at
multiplier 16 (100% recovery, 11.4 s latency). The floor belongs to the
**variance-adaptive** detector. φ-accrual sets each teammate's threshold from
that teammate's *observed* gap variance; a burst contributes no samples, so the
teammates whose delivered traffic looks most regular get the tightest thresholds
and are exactly the ones a burst silences. Raising k inflates every threshold
without repairing the ranking. §ROC now says this. The corrected claim is
narrower, mechanistic, and more interesting — *adaptivity is the liability* — and
ours still wins the fair comparison (~3× faster than the fixed timeout).

### Honest negatives (keep them, they are credibility)
- Cost-calibration doesn't distinguish our formula; plain `channel_aware_timeout` ties or wins.
- Channel-gated *allocation* confidence gives no benefit — the principle is detection-only.
- The LRT variant is *worse*; its i.i.d. assumption is violated by the correlation we exploit.
- Raw makespan/latency at fixed sensitivity is **not** a win. Fair speed = matched-false-alarm ROC.

### Status: submission-ready
- Affiliation filled (Oregon State University, CoRIS Institute).
- **All four placeholder bib entries resolved.** `cbbaetc` = Aznar, Pujol, Díez,
  *Computer Communications* 251 (2026) — this is also the Era-1 killer, arXiv 2604.06813.
  `edcbba` = Sao, Ho, Bhore, Sujit, arXiv:2509.06481. `formica` = Lopez, Muirhead,
  Pinciroli, ANTS 2026, LNCS 16515. `implicit` = Stulp, Isik, Beetz, **ICRA 2006**
  (the old stub said 2023; it is a 2006 robot-soccer paper, still correct for the claim).
- LaTeX validated structurally: no undefined cites/refs, balanced environments, all 7 figures present.
- ~3,870 body words → ~4 pages of text + 7 figures + 3 tables + 3 algorithms in
  IEEEtran 2-column. Inside a 6-page limit.
- **Only step left: compile on Overleaf** (`pdflatex → bibtex → pdflatex → pdflatex`).
  No TeX toolchain locally, so float placement is the one thing unchecked.

---

## 3. Era 3 — refuted, and exactly why

The idea: no comms at all; a scheduled rendezvous that does not happen is the only
failure signal. Since the disturbance (current) is spatially correlated, benign
silence is regional — so route partners' approaches close together, their delays
correlate and cancel, and a missed meeting becomes trustworthy.

A lot was built (see §5) and the theory was developed properly: a joint
duration–separation alignment condition, a Θ(B^1/3) vs Θ(B) relaxation theorem, a
"natural tour crossing" construction, an admissible-mission envelope. **All of it
rested on assumption A-L2** — that a completed stop is a resynchronisation point,
so the disturbance a robot carries into a meeting comes only from its *final
approach leg*.

**A-L2 is false.** HoloOcean, A→B→dwell(h)→C under constant current: the arrival
offset is +13.12 s at −1 m/s and +45.08 s at −2 m/s, **identical for h = 0, 5, 30
and 60 s**, and equal to the sum of both legs' delays. A fixed-duration stop
neither resets nor absorbs accumulated delay.

Under the supported carry-forward model (σ/ε = 4): unaligned 24.1%, corridor
23.0%, crossings h=1 20.2%, h=4 17.2%, vs absorb +120 s 4.9% and +240 s 2.0%.
**Alignment buys ~25–33% relative; absorption buys 10–100×,** no crossover from
0.5× to 4×. The theorem is sound but about the wrong quantity.

Full detail, including the both-ways caveat, is in `docs/rendezvous_assumptions.md`
under **N4 (REFUTED)**. Do not restart this line without reading it.

### What survives from Era 3
- **Coverage-first allocation** (L0/L1): balanced Hungarian partition inside Lloyd,
  100% coverage by construction, open-TSP routes, makespan rebalancing, single-failure
  recovery table. Sound and reusable.
- **Rotating-buddy mesh**: exact maximum matching per round (greedy maximal strands
  non-adjacent robots and opens a 2Δ gap).
- **Cadence feasibility bound**: Δ ≥ 2·r_max/V + S. On compact Voronoi regions this
  is 380 s, so the pipeline's habitual Δ=300 is **infeasible** — it strands work,
  coverage silently drops, and stranded robots sit at meeting points reporting
  perfect alignment. Guarded in code now.
- **The HoloOcean validation methodology** — the most valuable thing here.
- **A real tension**: false-alarm suppression needs schedule slack, and the cadence
  bound caps how much slack a Δ-cadence mesh can carry (~240 s at Δ=600 → ~2.0%).
- **The elastic schedule** (added 2026-10-19, `SCHEDULE` in
  `taskalloc_coverage_first.py`). The mesh's cost was being inflated by the *clock*,
  not by the meetings. Three changes, all at an unchanged worst-case detection gap:
  - `pack` — choose *which* of a robot's own tasks fill the approach window by exact
    subset search (Held–Karp over the 12 nearest), not nearest-first. This is what
    lets one partner take two near tasks while the other takes one far one.
  - `float_meet` — hold the meeting when the later partner actually arrives, not when
    the clock says. The pair pays `|t_a − t_b|`, not `2·(τ − arrival)`. Legal because
    the plan is deterministic, so both robots compute the same time offline.
  - `rolling` — anchor the next deadline to the *last meeting* instead of to the
    clock. **This is what makes floating free.** Consecutive meetings are then ≤ Δ
    apart by construction. Floating on an absolute grid looks better still (26.8% vs
    29.1%) but its worst gap is 660 s at Δ=600 — it borrows almost exactly the gain.

  Measured, Δ=600, N=90, K=4, 10–12 seeds, with absorption off: mesh cost
  **37.1% → 27.4%** over coverage-only, fleet idle **30.5 → 6.2 min**, worst gap
  **600 s in both**. Improves every variant (plain, slack, loiter, coopt, cross) by
  5–12 pp and every tested (N, K, L, Δ) — 118/132 seeds win individually.
  `run_dry_l2` test 8 asserts both halves (cheaper **and** no wider gap), because
  the tempting wrong version passes the first alone.
- **What the idle was actually buying** (the non-obvious half). A hold to a time
  fixed *in advance* is the only delay-**absorbing** mechanism left once HoloOcean
  rules out a stop resetting accrued delay. Removing the wait removes the absorption:
  benign false alarms go 21.5% → 37.5% (carry) / 7.6% → 18.4% (A-L2). So `absorb`
  was added — every partner holds at least that long past the later arrival — which
  turns the grid's accidental slack into a priced knob. `sweep_absorb` traces it
  (`results/taskalloc/l2_absorb.csv`, figure `schedule_elastic.png`).

  **Shipped operating point: `ABSORB_STAR = 3·SCAN = 180 s`**, which is what
  `schedule="elastic"` now means. Chosen as the point where mission cost reaches its
  floor (flat from ~90 s) with false alarms as low as the deadline clamp allows.
  **It is deliberately conservative**: on the tolerance-free measure below, 80% of
  the reducible false alarms are already gone by ~75 s. The extra holding is free in
  makespan (30.8% at β=60 vs 31.0% at β=180) and buys 2.7 pp of false alarms, but it
  costs 8 min of fleet station-keeping, so a deployment that prices energy should use
  60–90 s instead.

  **Do not repeat the B1 experiment** (run 2026-10-19, ~1 h compute; scripts in the
  session scratchpad `b1*.py`). It asked whether absorb\* is *predictable* from a
  config statistic — which would have turned a constant into a design rule and
  carried a paper. **It is not.** Measured on a tolerance-free knee (β removing 50%
  / 80% of the *reducible* false alarms, so the varying floor cannot bias it), over
  10 configurations spanning σ_x 1→8, Δ 500→800, K 4→6, N 90→120, L 600→900 and
  detector grace ε 10→120 s:
  - **differential arrival spread — refuted.** Dead flat (p50 = 18 s, p90 = 94 s)
    across the whole σ_x sweep while the requirement moved 2–3×. The mechanism
    sentence previously written here and in memory was simply wrong: absorption acts
    by *clipping offsets to zero* in `lateness = max(0, offset − slack)`, so the
    relevant quantity is the carried offset, not the arrival spread.
  - **carried-offset quantiles (p50/p90/p95/σ) — all worse than no rescaling.**
    Rescaling β by any of them *increases* the scatter of the collapsed curves
    (0.047–0.102) versus leaving β in raw seconds (0.003).
  - **detector grace ε — refuted.** b50/ε falls monotonically 2.12 → 0.31 as ε goes
    10 → 120 s, i.e. b50 is nearly ε-invariant (21 → 37 s over a 12× change).
  - **"it is a constant" — the best fit of the lot, and still false.** Over the
    original grid b50 = 24 s at CV 0.13, but extending σ_x to 8 breaks it: b50 goes
    13 → 31 s and b80 goes 32 → 108 s. The growth is **sublinear, ≈ σ^0.4 (carry) /
    σ^0.6 (Gaussian)**.
  - **the saturation explanation offered for the invariance — refuted by
    construction.** `calibrate_amp` forces `std(φ(A·u)) = κ` exactly, so the measured
    and Gaussian responses have matched dispersion and no saturation survives to
    explain anything. Both models scale sublinearly for a reason not yet identified
    (most likely the `max(0,·)` clipping geometry plus common-mode cancellation in
    the partner difference).

  What survives and is worth one sentence in whichever paper: **the absorption
  requirement is insensitive to cadence, fleet size, field size and detector grace,
  and only sublinearly sensitive to disturbance strength — so a fixed margin is a
  sound default that degrades gracefully and does not need retuning per
  deployment.** That is a robustness claim, not a design rule, and it is not enough
  to carry a paper on its own.

  At absorb\*, Δ=600, N=90, K=4, 10 seeds: cost **37.1% → 31.0%**, false alarms
  **7.6% → 7.3% (A-L2)** and **21.5% → 23.1% (carry-forward)**, worst gap 600 s in
  both. Across (N,K,L,Δ) the pattern is consistent: **−3 to −10 pp cost for +1.3 to
  +2.9 pp false alarms under carry, roughly neutral under A-L2.** It wins **10/12
  seeds** at absorb\*; at `absorb=0` it wins **12/12** but costs ~15 pp of false
  alarms. Everything below absorb\* is a deliberate speed-for-false-alarms trade.
- **A pairwise co-planning objective was tried and is NOT kept.** `_pack_pair`
  searches both partners' count-vs-arrival frontiers against each other,
  `argmin max(F_a(c_a), F_b(c_b)) − credit·(c_a+c_b)`. It is implemented, tested and
  correct, but measures 29.3% against independent packing's 27.4%: once the deadline
  rolls, arriving early is *already* rewarded, so an explicit "minimise the later
  arrival" term double-counts and the robots under-work. It also needs a
  no-starvation floor (at low `credit` the optimum is for both robots to do nothing
  for ever). Kept in the file as a documented negative; `joint=False` by default.

### The closed-loop experiment (2026-10-20) — the blocking item, resolved
`visualization/holoocean_closed_loop.py`, HoveringAUV, band current crossing the
route, budget calibrated from a measured still-water run (never from an assumed
cruise — the PID tracker does ~0.94 m/s, not the 2.0 the planner assumed, and that
bias alone made a first version shed tasks at **zero** current). Zero-current
control passes exactly: 0% shed, 0.0 s late, 100% coverage.

Three policies: `open` (fly the plan regardless = the carry bound), `closed`
(shed the next task if finishing it would miss the appointment), `graced` (same,
but target appointment **+ ε**, knowing a partner waits ε past it).

```
current  policy   shed%  late|noshed  late|shed  coverage     (5 tasks, span 150)
1.0      open        0%       12.1s       --        100%
1.0      closed    100%         --     -118.8s       80%
1.0      graced      0%       12.1s       --        100%
1.5      open        0%       25.9s       --        100%
1.5      closed    100%         --     -105.6s       80%
1.5      graced     17%       23.6s     -90.0s       97%
2.0      open        0%       45.3s       --        100%
2.0      closed    100%         --      -87.0s       80%
2.0      graced     67%       37.4s     -58.2s       87%
```

**The question "where does reality sit between A-L2 and carry-forward?" has no
answer, because adaptation is QUANTISED.** A task is worth a scan plus its travel
(~150 s here); the drift it is correcting is 9-45 s. The system does not
interpolate between the bounds, it alternates between them.

**Task-granular shedding cannot track the disturbance at any current.** Below ε it
over-corrects catastrophically (at 1 m/s `closed` destroys 20% of coverage to avoid
12 s of lateness it would never have been punished for, since ε=30 s). Above ε it
*under*-corrects (at 2 m/s `graced` drops a whole task and is **still** 37.4 s late,
past the grace). There is no current at which it lands on target.

**The one-line fix, and its limit.** Replanning against appointment **+ ε** instead
of the bare appointment is strictly dominant up to 1 m/s: 0% shed, 100% coverage,
worst lateness 12.1 s. It is not a cure — at 2 m/s it still pays 13% of coverage and
still misses. But `closed` is never right at any current tested.

**What this says about the actuator.** The mismatch is between the granularity of
the planner's actuator (whole tasks) and that of the protocol's tolerance (seconds).
The right actuator is continuous: cruise speed, or the absorption margin `absorb`
from the elastic schedule. That is the same conclusion the elastic-schedule work
reached from the opposite side, and it is the strongest single argument for treating
routing and the communication protocol as one object rather than two.

### Platform fact worth remembering
**HoloOcean's SurfaceVessel is inert to `set_ocean_currents`** — a 3 m/s current
displaces it 0.18 m in 20 s. Any current study must use the **HoveringAUV** (8.08 m
under the same current). This invalidates the USV framing for current work.

---

## 4. Prior art (verified)

- **Era 1 killer**: Aznar, Pujol & Díez, *Comput. Commun.* 251 (2026) — BT +
  event-triggered + adaptive time-fallback + CBBA baselines. ≈ the old layered method.
- **φ-accrual** (Hayashibara 2004) — the standard adaptive detector, per-node.
- **Era 3 mechanism pre-empted**: co-observation planning (arXiv 2403.13266) uses the
  identical missed-meeting signal; Zhong, Rossi & Shell (*Auton. Robots* 2025) plan
  synchronisation with active observations.
- CBBA (Choi/Brunet/How 2009), edcbba, FORMICA, implicit coordination — all stop at
  allocation and assume reliable execution.

---

## 5. Files map

**Paper (Era 2, the one to ship)** — `paper/main.tex`, `paper/references.bib`, 7 figures
(`fig_scale`, `fig_roc`, `fig_energy`, `fig_comparison`, `fig_partition_gap`,
`fig_relay`, `fig_sprt`).

**Era-3 paper (do not submit)** — `paper/aligned_rendezvous.pdf` +
`paper/build_aligned_rendezvous_pdf.py`, `paper/make_paper_figs.py`,
`paper/make_envelope_fig.py`, `paper/make_eq3_fig.py`. Central claims are
known-false; kept for the machinery and the negative results.

**Simulators**
- `visualization/failure_recovery_sim.py` — Era 2. Subcommands `stage1 stage1b stage3
  roc downstream dynamic compare sweepfail`. Methods incl. `adaptive_pernode` (φ-accrual),
  `fixed_timeout`, `channel_aware_timeout`, `ours_correlated`, `ours_sprt_lin`.
- `visualization/taskalloc_coverage_first.py` — Era 3 L0/L1/L2. `--dry_run --l2 --sweep`.
  Error models: `gaussian` (Eq. 3 as written), `measured` (HoloOcean response),
  `carry` (the supported one).
- `visualization/holoocean_eq3_validation.py` — the validation harness. `--exp probe|time|field|all`.
  Run inside `~/holoocean_env`.
- `visualization/full_pipeline_sim.py` — Era-3 frozen-mesh pipeline, stages A–X.
- `visualization/car_fcd_sim.py` — the original CAR-FCD prototype.

**Docs** — `docs/rendezvous_assumptions.md` (realism ledger, **read N4**),
`docs/rendezvous_problems.md`, `docs/rendezvous_allocation_idea.md`,
`visualization/LIMITATIONS.md` (Era-1 simulator audit; the mutex finding is there).

**Results** — `results/stage1b/` (scaling), `results/roc/`, `results/downstream/`,
`results/holoocean_eq3/` (the validation), `results/taskalloc/` (Era-3 L2).

---

## 6. What to do next

1. **Compile and submit Era 2.** Overleaf; check float placement; submit.
2. ~~**Closed-loop replanning experiment**~~ — **DONE 2026-10-20**, see below. It
   did not settle which bound is right; it showed the question was malformed.
3. **Only if (2) warrants it**: the coverage-versus-false-alarm frontier. Under
   disturbance a robot either arrives late (false alarm) or sheds tasks (coverage
   loss). That trade is genuine, unexplored, and uses everything that survives.
   The `absorb` knob is now the clean handle on it — `sweep_absorb` already traces
   cost against false alarms under both disturbance models, so this experiment is
   mostly a matter of running it closed-loop.
   Check novelty before starting.

Do **not** resume the crossing construction. It loses in both branches: if reality
is near A-L2 a 5 s dwell corridor dominates it; if near carry-forward, alignment
barely helps.

---

## 7. Working principles (the user demands these)

Brutal honesty. No inflated or strawman comparisons. Act as an honest skeptic who
**tests empirically** rather than asserting. When tempted to claim a win, run it
first. Report negatives as negatives. Don't hurt the method to "stay real", but
don't fabricate a gap either — find the condition where it genuinely wins.

**The meta-lesson, learned twice and expensively:** every modelling assumption
authored in-house that made the disturbance *easier to escape* flattered whichever
policy escaped that way. Gaussian symmetry flattered absorption; final-leg exposure
flattered alignment. Both survived internal review until they were measured against
independent physics. **Do not build a theory on a self-authored noise model without
testing it first** — the A-L2 test took ten minutes and was available for months.

Corollary that held up: the results that survived scrutiny were all *measurements
about existing methods* (the φ-accrual floor, multi-witness failing under correlation,
the Gaussian flattering absorption), not newly invented mechanisms.
