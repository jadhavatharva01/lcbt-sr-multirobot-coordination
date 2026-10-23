#!/usr/bin/env python3
"""
Build paper/car_fcd_method.pdf -- a method + literature-review draft for the
comms-free rendezvous line (CAR-FCD), written to share with an advisor.

No LaTeX toolchain in this environment, so the PDF is assembled with fpdf2:
body text in DejaVu Serif (Unicode -> Greek/math symbols render inline), display
equations rendered crisply via matplotlib mathtext to PNG, and the already-
generated result figures (results/car_fcd/) embedded.  Single column, justified.

Run:  python3 paper/build_car_fcd_pdf.py
"""
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(ROOT, "results", "car_fcd")
EQDIR = os.path.join(HERE, "_eq_cache")
os.makedirs(EQDIR, exist_ok=True)

# ---- fonts -----------------------------------------------------------------
def _find(*names):
    for n in names:
        for base in ("/usr/share/fonts/truetype/dejavu",
                     "/usr/share/fonts/truetype/dejavu/"):
            p = os.path.join(base, n)
            if os.path.exists(p):
                return p
    hits = glob.glob(f"/usr/share/fonts/**/{names[0]}", recursive=True)
    return hits[0] if hits else None

SERIF = _find("DejaVuSerif.ttf")
SERIF_B = _find("DejaVuSerif-Bold.ttf")
SERIF_I = _find("DejaVuSerif-Italic.ttf")
SANS_B = _find("DejaVuSans-Bold.ttf")
SANS = _find("DejaVuSans.ttf")

# ---- equation rendering (mathtext) -----------------------------------------
_eqn = 0
def eq_png(tex, fontsize=17):
    """Render a LaTeX-ish mathtext string to a tight transparent PNG; return
    (path, width_pt, height_pt)."""
    global _eqn
    _eqn += 1
    path = os.path.join(EQDIR, f"eq{_eqn}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${tex}$", fontsize=fontsize)
    fig.savefig(path, dpi=220, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    # px at 220 dpi -> mm
    w_mm = bb.width / 220 * 25.4
    h_mm = bb.height / 220 * 25.4
    return path, w_mm, h_mm


# ---- PDF document ----------------------------------------------------------
class Doc(FPDF):
    def header(self):
        pass

    def mc(self, w, h, txt, **kw):
        kw.setdefault("new_x", XPos.LMARGIN)
        kw.setdefault("new_y", YPos.NEXT)
        return self.multi_cell(w, h, txt, **kw)

    def footer(self):
        self.set_y(-12)
        self.set_font("serif", "I", 8)
        self.set_text_color(120)
        self.cell(0, 6, f"CAR-FCD method draft  --  page {self.page_no()}",
                  align="C")
        self.set_text_color(0)


LH = 5.1          # body line height mm
BODY = 10.3       # body font size
GfillR = (247, 247, 247)


def setup(pdf):
    pdf.add_font("serif", "", SERIF)
    pdf.add_font("serif", "B", SERIF_B or SERIF)
    pdf.add_font("serif", "I", SERIF_I or SERIF)
    pdf.add_font("sans", "B", SANS_B or SANS)
    pdf.set_margins(22, 20, 22)
    pdf.set_auto_page_break(True, margin=18)


def h1(pdf, n, title):
    pdf.ln(2.5)
    pdf.set_font("sans", "B", 12.5)
    pdf.mc(0, 6.2, f"{n}   {title}")
    pdf.ln(1.2)
    pdf.set_font("serif", "", BODY)


def h2(pdf, title):
    pdf.ln(1.2)
    pdf.set_font("serif", "B", 10.8)
    pdf.mc(0, 5.4, title)
    pdf.ln(0.6)
    pdf.set_font("serif", "", BODY)


def para(pdf, text, gap=1.6):
    pdf.set_font("serif", "", BODY)
    pdf.mc(0, LH, text, align="J", markdown=True)
    pdf.ln(gap)


def bullet(pdf, text):
    pdf.set_font("serif", "", BODY)
    x = pdf.get_x()
    pdf.set_x(x + 4)
    pdf.mc(0, LH, "•  " + text, align="J", markdown=True)
    pdf.set_x(x)
    pdf.ln(0.8)


def numitem(pdf, k, text):
    pdf.set_font("serif", "", BODY)
    x = pdf.get_x()
    pdf.set_x(x + 4)
    pdf.mc(0, LH, f"**{k})**  " + text, align="J", markdown=True)
    pdf.set_x(x)
    pdf.ln(0.8)


def equation(pdf, tex, fontsize=17, gap=2.2):
    path, w, h = eq_png(tex, fontsize)
    pdf.ln(1.4)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    maxw = page_w * 0.86
    if w > maxw:
        h = h * maxw / w
        w = maxw
    x = pdf.l_margin + (page_w - w) / 2
    pdf.image(path, x=x, w=w, h=h)
    pdf.ln(h + gap)


def boxnote(pdf, title, text):
    """A light-filled 'Proposition' box."""
    pdf.ln(1.0)
    x0, y0 = pdf.get_x(), pdf.get_y()
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_fill_color(*GfillR)
    # measure height by writing to a scratch: simpler -- just write with fill
    pdf.set_font("serif", "B", BODY)
    pdf.mc(page_w, LH, title, align="L", fill=True)
    pdf.set_font("serif", "I", BODY)
    pdf.mc(page_w, LH, text, align="J", fill=True)
    pdf.ln(2.0)
    pdf.set_font("serif", "", BODY)


def figure(pdf, path, caption, width_frac=0.92):
    if not os.path.exists(path):
        return
    pdf.ln(1.5)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    w = page_w * width_frac
    from PIL import Image
    iw, ih = Image.open(path).size
    h = w * ih / iw
    if pdf.get_y() + h + 14 > pdf.h - pdf.b_margin:
        pdf.add_page()
    x = pdf.l_margin + (page_w - w) / 2
    pdf.image(path, x=x, w=w)
    pdf.ln(1.2)
    pdf.set_font("serif", "I", 8.8)
    pdf.mc(0, 4.2, caption, align="J")
    pdf.ln(2.2)
    pdf.set_font("serif", "", BODY)


def build():
    pdf = Doc(format="letter", unit="mm")
    setup(pdf)
    pdf.add_page()

    # ---- title block ----
    pdf.set_font("sans", "B", 15.5)
    pdf.mc(0, 7.2,
        "Failure Detection without a Radio: Correlation-Aligned Rendezvous "
        "for Multi-Robot Teams under Range-Limited Communication", align="C")
    pdf.ln(1.5)
    pdf.set_font("serif", "", 10.5)
    pdf.mc(0, 5, "Atharva Jadhav", align="C")
    pdf.set_font("serif", "I", 9)
    pdf.mc(0, 4.6,
        "Method and literature-review draft  --  for discussion.  "
        "Claims are labelled by what is, and is not, novel.", align="C")
    pdf.ln(3)

    # ---- abstract ----
    pdf.set_font("serif", "B", 9.6)
    pdf.mc(0, 4.8, "Abstract", align="L")
    pdf.set_font("serif", "", 9.6)
    pdf.mc(0, 4.6,
        "Multi-robot teams operating under range-limited or intermittent "
        "communication cannot always use a radio heartbeat to decide whether a "
        "teammate has failed. A comms-free alternative is a schedule of planned "
        "physical rendezvous: a survivor keeps a pre-agreed appointment, and a "
        "partner that never arrives is presumed failed, with detection bounded by "
        "the meeting cadence. This turns failure detection into a routing "
        "problem, but it inherits the same ambiguity the radio setting has: an "
        "environmental disturbance (currents, wind) can delay an alive robot "
        "past its meeting window, so a missed rendezvous is ambiguous exactly as "
        "a silent radio is. We observe that the disambiguating signal is the one "
        "we already exploit over the radio channel -- the correlation of the "
        "silence across the team -- but that here it is a property of physical "
        "space that we can design. Because a disturbance field is spatially "
        "correlated, two robots that approach a meeting within one correlation "
        "length accrue nearly equal delay and still meet; a missed meeting "
        "between such a pair is therefore strong evidence of failure rather than "
        "disturbance. We formalize this as a scheduling constraint "
        "(correlation-aligned rendezvous, CAR) and prove it yields a per-meeting "
        "false-alarm bound that depends only on a lower bound on the disturbance "
        "correlation length -- not on the unknown disturbance realization. In "
        "simulation the bound holds: the aligned schedule holds a 5% target "
        "across disturbance strengths for which an unaligned schedule diverges "
        "toward certain false alarm, and the measured rate matches the analytic "
        "prediction. We are explicit about scope. The detection mechanism (a "
        "missed meeting as an alarm) is not new; our field-consistency sequential "
        "detector improves on a plain liveness rule only in a bounded regime of "
        "disturbance persistence, and in the high-persistence regime no "
        "comms-free detector can separate failure from disturbance -- a limit we "
        "characterize and that motivates a radio/relay fallback. The contribution "
        "is the correlation-aligned design, its realization-invariant guarantee, "
        "and the regime map for when it pays.", align="J")
    pdf.ln(3)

    # ================= 1 Introduction =================
    h1(pdf, "1", "Introduction")
    para(pdf,
        "Surface and underwater vehicles increasingly operate in teams over "
        "areas far larger than their reliable communication range. When a "
        "teammate stops moving -- a thruster fault, a grounding, a dead battery "
        "-- the team must *detect* the stoppage and *recover* the work, but the "
        "radio that a decentralized allocator relies on may simply not reach the "
        "silent robot. A comms-free alternative is to schedule *planned "
        "rendezvous*: at design time the team fixes a set of appointments, each a "
        "(point, time) pair known to all; at run time a survivor keeps its "
        "appointments, waits a slack window, and if a partner never arrives, "
        "presumes it has failed. Detection is then bounded by the appointment "
        "cadence, and no message is ever required.")
    para(pdf,
        "The difficulty is that a missed rendezvous is *ambiguous* in exactly "
        "the way a silent radio is. A partner may be dead, or it may be alive but "
        "delayed past its meeting window by an environmental disturbance -- a "
        "current or wind field that pushes it off schedule. Acting on every "
        "missed meeting produces a storm of false takeovers; waiting long enough "
        "to be sure erodes the detection guarantee. This is the same ambiguity we "
        "confronted for the radio channel, where a silent teammate may have "
        "failed or may merely be unheard during a correlated outage [12].")
    para(pdf,
        "The disambiguating signal is also the same: the *correlation of the "
        "silence across the team*. Over the radio we estimated a channel-health "
        "statistic online and scaled the failure belief by it, so that a "
        "correlated outage (many teammates unheard at once) suppressed the belief "
        "while isolated silence triggered it [12]. In the rendezvous setting the "
        "'channel' is physical space, and its correlation structure is not merely "
        "*observed* -- it can be *designed into the routes*. A disturbance field "
        "is spatially correlated: two robots close together experience nearly the "
        "same delay. If the schedule forces both partners of a meeting to "
        "approach along nearby corridors, they are co-disturbed and still meet "
        "even in a strong current; a missed meeting between them is therefore "
        "hard to explain by the environment, and is strong evidence of failure. "
        "Correlation, which we could only measure over the radio, becomes a "
        "*design variable* over rendezvous.")
    para(pdf, "We make four claims, and are careful about their scope.")
    numitem(pdf, "1",
        "**Correlation-aligned rendezvous (CAR).** A scheduling constraint that "
        "co-designs the meeting geometry with the disturbance's spatial "
        "correlation, and a design rule that turns it into a per-meeting "
        "false-alarm bound which is *invariant to the disturbance realization* -- "
        "it requires only a lower bound on the correlation length, never a "
        "forecast of the field (Sec. 3.2).")
    numitem(pdf, "2",
        "**A field-consistency sequential test (FCST).** The comms-free analogue "
        "of our channel-conditioned sequential detector [12]: it weights a missed "
        "meeting by the locally observed disturbance rather than by a count of "
        "witnesses (Sec. 3.3).")
    numitem(pdf, "3",
        "**A regime map.** Multi-witness confirmation, which is correct under "
        "i.i.d. loss, is the *wrong* tool under correlated disturbance; FCST "
        "improves on a plain liveness rule only at *moderate* disturbance "
        "persistence; and in the high-persistence regime no comms-free detector "
        "separates failure from disturbance -- an information limit, not a tuning "
        "artifact (Sec. 4).")
    numitem(pdf, "4",
        "**The price of the guarantee.** Alignment costs travel; we give the "
        "false-alarm-vs-travel Pareto that results (Sec. 3.2, 4).")
    para(pdf,
        "We state plainly what we do *not* claim. The mechanism of a missed "
        "planned observation as a detection signal is established prior art "
        "[4, 5]; the idea of routing to detect anomalies under a false-alarm "
        "budget exists for *environmental* targets [6]; and the coverage "
        "allocation we build on is standard [1]. Our contribution is the "
        "correlation-aligned *design* and its guarantee, not the detection "
        "mechanism or the allocator.")

    # ================= 2 Related Work =================
    h1(pdf, "2", "Related Work")
    h2(pdf, "2.1  Comms-free failure detection by planned observation")
    para(pdf,
        "The closest prior art is *co-observation planning* [4], which schedules "
        "mutual observations offline as common knowledge and treats a *missed* "
        "co-observation as the detectable event -- structurally identical to our "
        "missed-rendezvous alarm. Its problem, however, is adversarial: a "
        "compromised robot cannot leave a region without missing a co-observation. "
        "It assumes the observation itself is reliable and does not model an "
        "environmental disturbance that makes an honest observation fail, so it "
        "has no notion of designing the schedule against a disturbance "
        "correlation. *Planned synchronization with active observations* [5] "
        "decides *when* it is worth re-synchronizing team state through "
        "observations, optimizing the value of observing; we instead fix the "
        "schedule and read *misses* as failures. We adopt the missed-observation "
        "mechanism from this line and cite it as such; it is not ours. A separate "
        "body of work uses rendezvous to beat range-limited communication for "
        "*data exchange* and map merging [8, 9]; there the meeting exists to move "
        "bits, and failure handling reduces to a maximum-wait or a cancel "
        "message, both assuming communication -- the opposite of our setting.")
    h2(pdf, "2.2  Failure detectors in distributed systems")
    para(pdf,
        "The adaptive *phi-accrual* detector [2] fits the inter-arrival "
        "distribution of a single monitored process; SWIM [3] lowers false "
        "positives with *indirect probing*, confirming a suspect through other "
        "members before declaring it failed. Both are built for *i.i.d.* loss, "
        "where independent witnesses are informative and multi-witness "
        "confirmation is the right tool. Our earlier channel-aware detector [12] "
        "already showed that under *correlated* radio loss a per-node detector "
        "hits a false-alarm floor it cannot cross, because a single node's "
        "silence cannot separate a dead node from a bad channel. The present work "
        "carries that insight to physical space and, as Sec. 4 shows, reaches the "
        "stronger conclusion that under *spatially* correlated disturbance "
        "multi-witness confirmation is actively counter-productive: a whole region "
        "misses together, so agreeing witnesses corroborate a *benign* absence.")
    h2(pdf, "2.3  Routing for detection, and correlated travel times")
    para(pdf,
        "The nearest work in spirit is *spatial quickest detection* / persistent "
        "surveillance [6], where a team routes to detect anomalies in minimum "
        "time subject to a bound on the expected time between false alarms -- "
        "routing co-designed with a false-alarm budget, as here. But its "
        "detection *target* is an anomaly in the environment sensed by a moving "
        "robot, not the *failure of a robot* inferred from a missed rendezvous, "
        "and it does not design the routes against a disturbance correlation "
        "kernel. Stochastic vehicle routing with *correlated travel times* [7] "
        "models the same physical fact we exploit -- neighbouring paths have "
        "correlated delays -- but optimizes delivery feasibility (chance "
        "constraints on arrival), not the false-alarm rate of a detector. To our "
        "knowledge, co-designing the routing/rendezvous schedule against the "
        "disturbance's spatial correlation so as to bound a comms-free failure "
        "detector's false-alarm rate is not addressed by either line. We flag "
        "[6, 7] as the works a reviewer is most likely to reach for, and a "
        "full reading of both is the immediate next step before any novelty claim "
        "is finalized.")
    h2(pdf, "2.4  Task allocation and coverage (the substrate)")
    para(pdf,
        "The allocation on which the schedule rides -- balanced partitioning, "
        "per-robot open routes, makespan balancing, and single-failure "
        "reassignment -- is standard multi-robot task allocation built from "
        "consensus auctions [1], team-orienteering / adaptive large-neighbourhood "
        "search [10], and synchronized vehicle routing [11]. We use it as a "
        "substrate and claim no novelty for it; it is held fixed so that any "
        "detection result is attributable to the schedule design, not the "
        "allocator -- the same discipline we applied to the re-allocation backend "
        "in [12].")
    h2(pdf, "2.5  Relation to our own prior work")
    para(pdf,
        "This draft is the comms-free counterpart of our channel-aware "
        "correlated-silence detector [12]. There, silence arrives over a radio "
        "and its meaning is set by a *measured* channel-health statistic; here, "
        "silence arrives as a missed physical meeting and its meaning is set by a "
        "*designed* spatial correlation plus a locally *estimated* disturbance. "
        "The field-consistency test of Sec. 3.3 is, term for term, the "
        "rendezvous analogue of the channel-conditioned sequential test of [12] "
        "(Sec. 3.3).")

    # ================= 3 Method =================
    h1(pdf, "3", "Method")
    h2(pdf, "3.1  Problem formulation")
    para(pdf,
        "A team of K robots must cover a set of points of interest and, "
        "throughout, remain monitorable for failure without communication. At "
        "design time we fix (i) a partition of the coverage work giving every "
        "robot an open route, and (ii) a *rendezvous schedule*: a set of "
        "appointments, each an agreed point m and time, such that every robot "
        "meets a partner at least once per detection interval Δ. Detection "
        "latency is bounded by Δ + ε, where ε is the slack a "
        "survivor waits before declaring a no-show. A single robot may fail at an "
        "unknown time, after which it keeps no further appointments.")
    para(pdf,
        "The environment imparts a per-robot arrival-time offset. We model the "
        "disturbance as a stationary field with an exponential spatial "
        "correlation kernel and correlation length ℓ,")
    equation(pdf, r"\rho(d)=e^{-d/\ell},\quad \tau_i=\sigma\,u_i,\quad u\sim\mathcal{N}(0,R),\quad R_{ij}=\rho(\Vert x_i-x_j\Vert)", fontsize=15)
    para(pdf,
        "so that τ is the delay a robot accrues over its approach leg, "
        "σ is the disturbance amplitude, and two robots that accrue their "
        "delay at points a distance d apart have correlated offsets. A rendezvous "
        "between partners i and j is *made* iff their offsets differ by at most "
        "the slack,")
    equation(pdf, r"\left|\tau_i-\tau_j\right| \leq \varepsilon")
    para(pdf,
        "The key consequence: two robots delayed *equally* (co-located in a "
        "current) still meet -- only *differential* delay, across a gradient, "
        "breaks a meeting. Writing d for the separation of the two partners' "
        "approach corridors,")
    equation(pdf, r"\mathrm{Var}(\tau_i-\tau_j)=2\sigma^2\left(1-\rho(d)\right)")
    para(pdf,
        "The offline objective is lexicographic: 100% coverage (hard) first, "
        "then minimum makespan, then feasible single-failure recovery, then "
        "minimum false-alarm rate. The last term is what CAR addresses.")

    h2(pdf, "3.2  Correlation-aligned rendezvous and its guarantee")
    para(pdf,
        "From the variance above, the probability that an *alive* pair misses a "
        "scheduled meeting -- a benign false alarm -- is")
    equation(pdf, r"p(d)=2\,\Phi\!\left(\frac{-\varepsilon}{\sqrt{2\sigma^2\,(1-\rho(d))}}\right)")
    para(pdf,
        "which is monotone in d: co-located approaches (d → 0, ρ → "
        "1) never miss, distant approaches decay to the independent-loss floor. "
        "This monotonicity is the design lever. We add to the offline schedule "
        "the *correlation-alignment* constraint (C4), which requires every "
        "scheduled meeting to satisfy")
    equation(pdf, r"\rho(d_{ij}) \geq \rho_{\min}")
    para(pdf,
        "realized by routing both partners' final leg down a shared corridor so "
        "their approach points sit within d* = −ℓ ln ρ_min of each "
        "other. Given a disturbance amplitude σ, correlation length ℓ, "
        "and a target false-alarm rate α, the required alignment follows in "
        "closed form from p(d*) ≤ α:")
    equation(pdf, r"1-\rho^{*}=\frac{\varepsilon^2}{2\sigma^2\,z_{1-\alpha/2}^{2}},\quad d^{*}=-\ell\,\ln\rho^{*}")
    boxnote(pdf, "Proposition (realization-invariant false-alarm bound).",
        "If a rendezvous schedule satisfies (C4) with rho_min = rho* chosen as "
        "above, then the per-meeting benign false-alarm probability is at most "
        "alpha for every realization of any disturbance field whose correlation "
        "length is at least ell. The bound depends on the field only through ell "
        "and the amplitude sigma; it requires no knowledge or forecast of the "
        "field itself. A robot that has truly failed is absent from every "
        "appointment regardless of rho, and so is caught by persistence within a "
        "bounded number of intervals.")
    para(pdf,
        "The proof is immediate from the model: the offset difference of an "
        "aligned pair is Gaussian with variance 2σ²(1−ρ(d*)) "
        "≤ 2σ²(1−ρ*), so P(|τ_i−τ_j| > "
        "ε) ≤ α by the definition of ρ*; nothing in the "
        "argument uses the realized field, only the correlation length that "
        "lower-bounds ρ(d*). The value of the statement is operational: an "
        "operator cannot forecast a current field, but its correlation length is "
        "an estimable oceanographic prior, and designing against a conservative "
        "lower bound on ℓ makes the guarantee robust to that estimate.")
    para(pdf,
        "Alignment is not free. Forcing partners to converge before a meeting "
        "steals travel that would otherwise cover territory, so (C4) induces a "
        "false-alarm-vs-travel Pareto: a tighter α, or a stronger current, "
        "demands a smaller d* and thus more alignment travel. We quantify this "
        "trade in Sec. 4; it is the honest cost of the guarantee.")

    h2(pdf, "3.3  Field-consistency sequential test (FCST)")
    para(pdf,
        "Alignment lowers the per-meeting false-alarm rate but cannot drive it to "
        "zero when currents are strong or ℓ is small. A temporal detector "
        "closes the gap. The wrong choice is multi-witness confirmation: under "
        "*correlated* disturbance a whole region misses together, so witnesses "
        "agree on a *benign* absence and confirmation inflates, rather than "
        "suppresses, false alarms (Sec. 4). The right choice mirrors our radio "
        "detector. A robot that has truly failed meets *no one*, so any single "
        "made meeting is near-proof of life and resets the evidence; only a "
        "*fully silent* interval is incriminating, and then only in proportion to "
        "how *calm* the local field is. Let q_i(t) be the observed miss-rate "
        "among survivor-survivor meetings near robot i -- a local, comms-free "
        "estimate of how much the disturbance can explain i's silence. On each "
        "interval in which i meets no one (and resetting L_i to 0 the moment i "
        "is met), FCST accumulates")
    equation(pdf, r"L_i\ \leftarrow\ L_i\;-\;\mathrm{deg}(i)\,\log\!\left(\mathrm{clip}(q_i(t))\right)")
    para(pdf,
        "and declares i failed when L_i ≥ Λ. Silence during a rough "
        "region (q → 1) contributes almost nothing; silence during a calm "
        "region (q small) contributes strongly. This is, term for term, the "
        "comms-free analogue of the channel-conditioned sequential test of our "
        "radio work [12], which accumulates −log(1−η) per missed "
        "heartbeat with η the channel health: here the local field miss-rate "
        "q plays the role of the channel-badness (1−η), and both reduce "
        "to a plain persistence/liveness rule when their conditioning variable is "
        "constant. We are explicit (Sec. 4) that FCST's advantage over plain "
        "liveness is real but *bounded*: it appears only when the disturbance is "
        "moderately persistent.")

    # ================= 4 Preliminary results =================
    h1(pdf, "4", "Preliminary Results")
    para(pdf,
        "We evaluate on a K = 16 team on a 4x4 territory grid, meeting slack "
        "ε = 30 s, correlation length ℓ = 150 m, under a stationary "
        "Gaussian disturbance field with exponential kernel and an AR(1) temporal "
        "component; a meeting is made iff the partners' offsets differ by at most "
        "ε. The model reproduces and generalizes the correlated-current "
        "behaviour of our earlier surface-vessel study. Figures are produced by "
        "`visualization/car_fcd_sim.py` (self-tested).")
    para(pdf,
        "**The guarantee holds (Fig. 1).** With the schedule aligned to d*(σ) "
        "for a 5% target, the measured per-meeting false-alarm rate stays at the "
        "target across disturbance strengths from 1x to 4x the slack, while the "
        "*unaligned* schedule diverges from 7% toward 82% over the same range. "
        "The measured aligned rate tracks the analytic p(d*) exactly -- the "
        "realization-invariant bound of Sec. 3.2 is borne out, not merely "
        "asserted.")
    figure(pdf, os.path.join(FIGDIR, "guarantee.png"),
        "Fig. 1.  Realization-invariant false-alarm guarantee from (C4). The "
        "unaligned schedule (red) diverges as the current strengthens; the "
        "schedule aligned to d*(sigma) (green) holds the 5% target for every "
        "strength, and the measured rate matches the analytic p(d) (dashed).")
    para(pdf,
        "**Multi-witness is the wrong tool; FCST's edge is regime-limited "
        "(Fig. 2).** At a representative strength, single-miss and two-witness "
        "detection false-alarm on essentially the whole team (100% and 99.9% of "
        "alive robots flagged), plain liveness on 61%, and FCST on 9% -- "
        "confirming that under correlated disturbance witness-counting is "
        "counter-productive. But FCST's advantage over plain liveness is bounded: "
        "it reaches a low false-alarm operating point roughly 1.4x faster than "
        "liveness at *moderate* field persistence, ties it when the field "
        "decorrelates quickly, and *fails to reach that point at all* when the "
        "field is highly persistent -- because a failure inside a current that "
        "sits over a region for the whole horizon is, without communication, "
        "indistinguishable from disturbance. This is an information limit that "
        "bounds any comms-free detector and motivates a radio or relay fallback "
        "in that regime, rather than a defect specific to FCST.")
    figure(pdf, os.path.join(FIGDIR, "persistence_regime.png"),
        "Fig. 2.  FCST's advantage is regime-limited. Latency to reach a low "
        "false-alarm operating point vs. the disturbance's temporal persistence: "
        "FCST (green) beats liveness (blue) only in the shaded moderate-"
        "persistence band; at high persistence it cannot reach the target "
        "(no comms-free detector can).")

    # ================= 5 Limitations =================
    h1(pdf, "5", "Scope and Limitations")
    bullet(pdf,
        "**Novelty is not yet finalized.** The correlation-aligned design "
        "appears open, but the claim is pending a full reading of spatial "
        "quickest detection [6] and correlated-travel-time routing [7], the two "
        "nearest works.")
    bullet(pdf,
        "**The guarantee assumes the correlation length.** It holds for any field "
        "with correlation length at least ℓ; if ℓ is over-estimated the "
        "bound is optimistic, so a deployment must design against a conservative "
        "lower bound on ℓ.")
    bullet(pdf,
        "**Model idealization.** The Gaussian exponential-kernel field is a "
        "tractable stand-in; real current fields are anisotropic and "
        "non-stationary, and the offset model is signed and linear in the field.")
    bullet(pdf,
        "**FCST regime.** As above, FCST helps only at moderate disturbance "
        "persistence; the high-persistence regime needs a fallback.")
    bullet(pdf,
        "**Inherited assumptions.** Single failure at a time, deterministic scan "
        "times, and good local navigation, consistent with the staged plan of "
        "the broader project.")

    # ================= References =================
    h1(pdf, "6", "References")
    refs = [
        "[1] H.-L. Choi, L. Brunet, and J. P. How, \"Consensus-based "
        "decentralized auctions for robust task allocation,\" IEEE Trans. "
        "Robotics, vol. 25, no. 4, pp. 912--926, 2009.",
        "[2] N. Hayashibara, X. Défago, R. Yared, and T. Katayama, \"The "
        "phi accrual failure detector,\" in Proc. IEEE SRDS, 2004, pp. 66--78.",
        "[3] A. Das, I. Gupta, and A. Motivala, \"SWIM: Scalable weakly-"
        "consistent infection-style process group membership protocol,\" in "
        "Proc. IEEE/IFIP DSN, 2002, pp. 303--312.",
        "[4] Co-observation planning for multi-robot intrusion detection, "
        "arXiv:2403.13266, 2024. [author list and exact title to be verified]",
        "[5] Q. Zhong, F. Rossi, and D. A. Shell (to be verified), \"Planned "
        "synchronization for multi-robot systems with active observations,\" "
        "Autonomous Robots, 2025, doi:10.1007/s10514-025-10225-4.",
        "[6] Stochastic surveillance strategies for spatial quickest detection, "
        "arXiv:1210.3438, 2012. [author list to be verified]",
        "[7] Reliable vehicle routing problem in stochastic networks with "
        "correlated travel times, Operational Research, 2019, "
        "doi:10.1007/s12351-019-00452-w.",
        "[8] Data gathering with a team of robots under intermittent "
        "communication, arXiv:1706.02092. [to be verified]",
        "[9] Multi-robot rendezvous in an unknown environment with limited "
        "communication, arXiv:2405.08345. [to be verified]",
        "[10] S. Ropke and D. Pisinger, \"An adaptive large neighborhood search "
        "heuristic for the pickup and delivery problem with time windows,\" "
        "Transportation Science, vol. 40, no. 4, pp. 455--472, 2006.",
        "[11] M. Drexl, \"Synchronization in vehicle routing -- a survey of VRPs "
        "with multiple synchronization constraints,\" Transportation Science, "
        "vol. 46, no. 3, pp. 297--316, 2012.",
        "[12] A. Jadhav, \"A behavior-tree framework for channel-aware failure "
        "recovery in decentralized multi-robot task allocation under bursty "
        "communication,\" manuscript (our prior work), 2026.",
    ]
    pdf.set_font("serif", "", 9.2)
    for r in refs:
        pdf.mc(0, 4.4, r, align="J")
        pdf.ln(0.9)

    out = os.path.join(HERE, "car_fcd_method.pdf")
    pdf.output(out)
    print("wrote", out)


if __name__ == "__main__":
    build()
