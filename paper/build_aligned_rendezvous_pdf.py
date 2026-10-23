#!/usr/bin/env python3
"""Build paper/aligned_rendezvous.pdf.

No LaTeX toolchain in this environment, so the PDF is assembled with fpdf2:
DejaVu Serif body (Unicode, so Greek renders inline), display equations rendered
via matplotlib mathtext, and the generated result figures embedded.

Run:  python3 paper/make_paper_figs.py && python3 paper/build_aligned_rendezvous_pdf.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "results", "taskalloc")
EQDIR = os.path.join(HERE, "_eq_cache2")
os.makedirs(EQDIR, exist_ok=True)


def _find(*names):
    for n in names:
        p = os.path.join("/usr/share/fonts/truetype/dejavu", n)
        if os.path.exists(p):
            return p
    hits = glob.glob(f"/usr/share/fonts/**/{names[0]}", recursive=True)
    return hits[0] if hits else None


SERIF = _find("DejaVuSerif.ttf")
SERIF_B = _find("DejaVuSerif-Bold.ttf")
SERIF_I = _find("DejaVuSerif-Italic.ttf")
SANS = _find("DejaVuSans.ttf")
SANS_B = _find("DejaVuSans-Bold.ttf")
MONO = _find("DejaVuSansMono.ttf")

_eqn = [0]


def eq_png(tex, fontsize=19):
    _eqn[0] += 1
    path = os.path.join(EQDIR, f"eq{_eqn[0]}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${tex}$", fontsize=fontsize)
    fig.savefig(path, dpi=240, transparent=True, bbox_inches="tight",
                pad_inches=0.02)
    bb = t.get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    return path, bb.width / 240 * 25.4, bb.height / 240 * 25.4


LH = 4.9
BODY = 9.9
TITLE = "Multi-Robot Coverage with Aligned Rendezvous"


class Doc(FPDF):
    def mc(self, w, h, txt, **kw):
        kw.setdefault("new_x", XPos.LMARGIN)
        kw.setdefault("new_y", YPos.NEXT)
        return self.multi_cell(w, h, txt, **kw)

    def footer(self):
        self.set_y(-12)
        self.set_font("serif", "I", 7.6)
        self.set_text_color(120)
        self.cell(0, 6, f"{TITLE}  —  preprint draft  —  page {self.page_no()}",
                  align="C")
        self.set_text_color(0)


def setup(pdf):
    pdf.add_font("serif", "", SERIF)
    pdf.add_font("serif", "B", SERIF_B)
    pdf.add_font("serif", "I", SERIF_I)
    pdf.add_font("sans", "B", SANS_B)
    pdf.add_font("sans", "", SANS)
    pdf.add_font("mono", "", MONO)
    pdf.set_margins(20, 18, 20)
    pdf.set_auto_page_break(True, margin=16)


def h1(pdf, n, title):
    pdf.ln(2.6)
    if pdf.get_y() > pdf.h - 45:
        pdf.add_page()
    pdf.set_font("sans", "B", 11.6)
    pdf.mc(0, 5.8, f"{n}.  {title.upper()}")
    pdf.ln(1.0)
    pdf.set_font("serif", "", BODY)


def h2(pdf, title):
    pdf.ln(1.0)
    pdf.set_font("serif", "B", 10.2)
    pdf.mc(0, 5.2, title)
    pdf.ln(0.4)
    pdf.set_font("serif", "", BODY)


def para(pdf, text, gap=1.5):
    pdf.set_font("serif", "", BODY)
    pdf.mc(0, LH, text, align="J", markdown=True)
    pdf.ln(gap)


def bullet(pdf, text, marker="•"):
    pdf.set_font("serif", "", BODY)
    x = pdf.get_x()
    pdf.set_x(x + 4)
    pdf.mc(0, LH, f"{marker}  " + text, align="J", markdown=True)
    pdf.set_x(x)
    pdf.ln(0.7)


def equation(pdf, tex, fontsize=19, gap=2.0, tag=None):
    path, w, h = eq_png(tex, fontsize)
    pdf.ln(1.2)
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    maxw = page_w * 0.84
    if w > maxw:
        h, w = h * maxw / w, maxw
    if pdf.get_y() + h + 6 > pdf.h - pdf.b_margin:
        pdf.add_page()
    y = pdf.get_y()
    pdf.image(path, x=pdf.l_margin + (page_w - w) / 2, y=y, w=w, h=h)
    if tag:
        pdf.set_font("serif", "", BODY)
        pdf.set_xy(pdf.l_margin, y + h / 2 - 2.4)
        pdf.cell(page_w, 4.8, tag, align="R")
    pdf.set_y(y + h + gap)


def claim(pdf, label, text, proof=None):
    """A Proposition / Theorem block, with an optional proof ending in a square."""
    pdf.ln(0.8)
    pdf.set_font("serif", "B", BODY)
    pdf.set_x(pdf.l_margin)
    pdf.mc(0, LH, label, align="L")
    pdf.set_font("serif", "I", BODY)
    pdf.mc(0, LH, text, align="J", markdown=True)
    pdf.set_font("serif", "", BODY)
    if proof:
        pdf.ln(0.5)
        pdf.set_font("serif", "", BODY)
        pdf.mc(0, LH, "Proof:  " + proof, align="J", markdown=True)
        # the QED square (not in the font, so draw it)
        y = pdf.get_y() - LH
        pdf.rect(pdf.w - pdf.r_margin - 2.6, y + 1.2, 2.2, 2.2, style="F")
    pdf.ln(1.6)


def table(pdf, headers, rows, widths, caption=None, fs=8.4, notes=None):
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    scale = page_w / sum(widths)
    widths = [w * scale for w in widths]
    need = 6 + LH * (len(rows) + 2)
    if pdf.get_y() + need > pdf.h - pdf.b_margin:
        pdf.add_page()
    if caption:
        pdf.set_font("serif", "B", 8.8)
        pdf.mc(0, 4.6, caption, align="L")
        pdf.ln(0.6)
    pdf.set_font("sans", "B", fs)
    pdf.set_fill_color(232, 236, 241)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5.4, h, border="TB", align="C", fill=True)
    pdf.ln(5.4)
    pdf.set_font("serif", "", fs)
    for n, row in enumerate(rows):
        fill = n % 2 == 1
        pdf.set_fill_color(247, 248, 250)
        bold = str(row[0]).startswith("*")
        pdf.set_font("serif", "B" if bold else "", fs)
        for k, (c, w) in enumerate(zip(row, widths)):
            pdf.cell(w, 5.0, str(c).lstrip("*"), border=0,
                     align="L" if k == 0 else "C", fill=fill)
        pdf.ln(5.0)
    pdf.set_draw_color(120)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.set_draw_color(0)
    pdf.ln(1.2)
    if notes:
        pdf.set_font("serif", "I", 7.8)
        pdf.mc(0, 3.9, notes, align="L")
    pdf.ln(2.0)
    pdf.set_font("serif", "", BODY)


def figure(pdf, name, caption, frac=1.0):
    # a bare name resolves in the default figure dir; "sub/dir.png" resolves
    # against results/ so figures can live beside the experiment that made them
    path = (os.path.join(ROOT, "results", name) if "/" in name
            else os.path.join(FIG, name))
    if not os.path.exists(path):
        raise SystemExit("figure not found: %s" % path)
    if not os.path.exists(path):
        return
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    w = page_w * frac
    iw, ih = Image.open(path).size
    h = w * ih / iw
    if pdf.get_y() + h + 12 > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.ln(1.4)
    pdf.image(path, x=pdf.l_margin + (page_w - w) / 2, w=w, h=h)
    pdf.ln(1.4)
    pdf.set_font("serif", "", 8.3)
    pdf.mc(0, 4.0, caption, align="J", markdown=True)
    pdf.ln(2.2)
    pdf.set_font("serif", "", BODY)


def algo(pdf, title, lines):
    need = 8 + 4.3 * (len(lines) + 1)
    if pdf.get_y() + need > pdf.h - pdf.b_margin:
        pdf.add_page()
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.ln(1.2)
    y0 = pdf.get_y()
    pdf.set_draw_color(70)
    pdf.line(pdf.l_margin, y0, pdf.w - pdf.r_margin, y0)
    pdf.ln(0.8)
    pdf.set_font("sans", "B", 8.8)
    pdf.mc(0, 4.6, title)
    pdf.set_draw_color(150)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(0.8)
    pdf.set_font("mono", "", 7.9)
    for i, ln in enumerate(lines):
        if ln.startswith("%"):
            pdf.set_text_color(110)
            pdf.mc(0, 4.05, "    " + ln)
            pdf.set_text_color(0)
        else:
            pdf.mc(0, 4.05, ln)
    y = pdf.get_y() + 0.6
    pdf.set_draw_color(70)
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.set_draw_color(0)
    pdf.ln(3.0)
    pdf.set_font("serif", "", BODY)


def build():
    pdf = Doc(format="A4")
    setup(pdf)
    pdf.add_page()

    # ---------------- title block ----------------
    pdf.set_font("sans", "B", 16.5)
    pdf.mc(0, 7.6, "Multi-Robot Coverage with Aligned Rendezvous:\n"
                   "Comms-Free Failure Detection under Correlated Disturbance",
           align="C")
    pdf.ln(2.0)
    pdf.set_font("serif", "", 10.6)
    pdf.mc(0, 5.0, "Atharva Jadhav", align="C")
    pdf.set_font("serif", "I", 9.4)
    pdf.mc(0, 4.6, "Oregon State University", align="C")
    pdf.ln(3.0)

    pdf.set_font("serif", "B", 9.3)
    pdf.set_x(pdf.l_margin + 8)
    pdf.multi_cell(pdf.w - 2 * pdf.l_margin - 16, 4.6,
                   "Abstract", align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("serif", "", 9.3)
    pdf.set_x(pdf.l_margin + 8)
    pdf.multi_cell(
        pdf.w - 2 * pdf.l_margin - 16, 4.6,
        "We consider deadline-constrained multi-robot coverage by a team with no "
        "communication beyond physical proximity, so a scheduled rendezvous that "
        "fails to happen is the only failure signal available. The difficulty is "
        "that the disturbance which delays a robot — a current — is spatially "
        "correlated, so benign silence is regional and a missed rendezvous is "
        "ambiguous. Existing correlation-aligned rendezvous removes the ambiguity "
        "by holding the partners' approach corridors within a separation d*, "
        "which buys a false-alarm bound independent of the disturbance "
        "realization but prices it only in the abstract. We fold the constraint "
        "into a coverage-first allocation in which 100% coverage is hard, and "
        "show that on executable routes the requirement is a joint condition on "
        "approach DURATION and approach SEPARATION, of which the separation-only "
        "rule is a conservative corner. We prove that letting duration co-vary "
        "with separation — as it does when a rendezvous is sited on a pair of "
        "adjacent tasks — relaxes the admissible separation from Θ(B) to "
        "Θ(B^(1/3)) in the alignment budget, a factor that grows as the budget "
        "tightens and reaches 39× at our parameters. Because the accrual model "
        "behind that theory is our own, we test it against HoloOcean rigid-body "
        "hydrodynamics: its two structural claims hold, its linearity does not, "
        "and recomputing every false-alarm figure through the measured response "
        "reverses a conclusion. Simply arriving early, which the Gaussian model "
        "makes competitive with alignment, in fact has a false-alarm floor of "
        "0.80% that no time budget can beat, because past a point the reserved "
        "slack strands coverage work; alignment has no such floor. The "
        "consequence is a capability rather than an economy: over a grid of "
        "deadlines and disturbance strengths there is a region — deadline 1.5× "
        "the no-rendezvous makespan, disturbance 2× and above — in which our "
        "construction is the only policy that simultaneously covers every task, "
        "meets the detection cadence, finishes on time and holds benign false "
        "alarms below 1%. The region persists at team sizes 4, 6 and 8, and the "
        "construction costs no more than an unaligned mesh that offers no "
        "protection at all.",
        align="J", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1.2)
    pdf.set_font("serif", "I", 8.8)
    pdf.set_x(pdf.l_margin + 8)
    pdf.multi_cell(pdf.w - 2 * pdf.l_margin - 16, 4.4,
                   "Index terms — multi-robot systems, task allocation, rendezvous, "
                   "failure detection, intermittent communication, behaviour trees, "
                   "marine robotics.",
                   align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ---------------- I. Introduction ----------------
    h1(pdf, "I", "Introduction")
    para(pdf,
         "A team of autonomous underwater vehicles is asked to survey a field of "
         "points of interest. Radio does not propagate, and acoustic links are "
         "slow, short-ranged and unreliable, so for most of the mission the robots "
         "cannot talk to each other at all. If one of them floods, fouls a "
         "thruster, or simply stops, the others have no packet whose absence would "
         "tell them so. The only evidence they will ever get is physical: a "
         "rendezvous that was on the plan and did not happen.")
    para(pdf,
         "Planning for that evidence is not free. Meetings must be placed and timed "
         "in advance, and every meeting is a constraint on a route whose real job is "
         "coverage. Prior work on connectivity-constrained coordination relaxes the "
         "requirement that the network be connected at all times, replacing it with "
         "__periodic connectivity__ — the team must regain connectivity at a fixed "
         "interval [1], [2]. We take the same interval Δ but a stricter premise: the "
         "network is never connected except at a planned physical encounter, and the "
         "encounter is simultaneously the data exchange and the health check. We add "
         "one thing the connectivity literature usually leaves implicit: a "
         "**deadline**. A survey that must be finished before a tide, a weather "
         "window or a battery limit cannot pay for safety in time, and that single "
         "constraint turns out to separate the available policies completely.")
    para(pdf,
         "The premise creates a problem that the connectivity literature does not "
         "have. A robot is late because the current pushed it, and the current is "
         "spatially correlated: it delays a whole region at once. Two partners that "
         "were both in the current are both late, and still meet; a partner that was "
         "in it while the other was not misses. So a missed meeting is not evidence of "
         "death, it is evidence of __differential__ delay. Counting witnesses — the "
         "standard fix for an unreliable detector — is the wrong tool here, because "
         "correlation makes witnesses agree on a benign absence rather than "
         "independently confirm a real one.")
    para(pdf,
         "One known answer is to co-design the routing against the correlation: route "
         "both partners' approaches close enough together that their delays are drawn "
         "from essentially the same value of the disturbance field and cancel in the "
         "difference. Requiring the two approach locales to lie within a separation "
         "d*(σ, λ, α) yields a per-meeting false-alarm bound of α that holds for every "
         "realization of any field whose correlation length is at least λ — a guarantee "
         "obtained by routing design rather than by predicting the weather. What that "
         "line of work does not do is price the constraint against the mission. Its "
         "cost is reported as a fraction of an abstract 'meeting leg'; on a real "
         "allocation with a hard coverage requirement, the currency is makespan.")
    para(pdf,
         "This paper puts the constraint on an executable allocation and asks what it "
         "costs. The answer turned out to require revisiting the constraint itself. "
         "The quantity that decides whether a meeting is made is the variance of the "
         "difference of two arrival offsets, and that variance depends on how LONG each "
         "approach was as well as on how far apart the approaches were. Separation is "
         "expensive to buy on a coverage route — it means dragging two robots together "
         "and holding them there — whereas duration is nearly free, because it is set "
         "by where the route happens to end, which is a matter of ordering tasks the "
         "robot owed anyway. Exploiting that asymmetry is the content of the paper.")
    pdf.ln(0.6)
    para(pdf, "**The contributions of this paper are:**")
    bullet(pdf,
           "an **admissible-mission envelope** for comms-free coverage — the set "
           "of (deadline, disturbance) pairs a policy can fly while covering "
           "every task, meeting the detection cadence and holding benign false "
           "alarms below a target — and the identification of a region of it in "
           "which our construction is the **only** admissible policy, at team "
           "sizes 4, 6 and 8;", "1)")
    bullet(pdf,
           "the reason that region exists: absorbing delay by arriving early has "
           "a **false-alarm floor**, because the slack that buys the reduction "
           "eventually strands coverage work (Proposition 3), whereas alignment "
           "spends route order rather than time and has no floor;", "2)")
    bullet(pdf,
           "a joint duration–separation alignment condition (Proposition 1) that "
           "generalizes the separation-only constraint used in "
           "correlation-aligned rendezvous, of which that constraint is a "
           "conservative corner (Corollary 1);", "3)")
    bullet(pdf,
           "a proof (Theorem 1) that letting approach duration co-vary with "
           "separation relaxes the admissible separation from Θ(B) to Θ(B^(1/3)) "
           "in the alignment budget B, so the relaxation grows as B^(-2/3) — it "
           "is largest exactly where alignment is hardest;", "4)")
    bullet(pdf,
           "the **natural tour crossing** construction that realizes the cheap "
           "corner by siting each rendezvous on a reserved pair of adjacent "
           "cross-boundary tasks, with a rolling-reservation algorithm and a "
           "behaviour-tree execution layer whose tick priority is the "
           "lexicographic objective;", "5)")
    bullet(pdf,
           "a validation of the accrual model behind the theory against "
           "HoloOcean's rigid-body hydrodynamics rather than against our own "
           "kinematics, which confirms the two structural claims the theory "
           "needs, refutes its linearity, and — on recomputing every false-alarm "
           "figure through the measured response — **reverses** the conclusion "
           "that arriving early is competitive with alignment;", "6)")
    bullet(pdf,
           "a cadence-feasibility bound Δ ≥ 2·r_max/V + S for rendezvous meshes "
           "on compact regions, which the commonly used Δ violates on our "
           "instances and which is what puts the floor under absorption.", "7)")
    para(pdf,
         "**Why it matters.** Every alternative buys protection against being fooled "
         "by the weather with time, and time is exactly what a deadline denies. "
         "Our construction buys it with the order tasks are visited, which the "
         "deadline does not price. For a communication-denied deployment with a "
         "tide or a battery to beat, that is the difference between a failure "
         "detector one can afford to run and a mission one cannot fly at all.")

    # ---------------- II. Related work ----------------
    h1(pdf, "II", "Related Work")
    para(pdf,
         "**Connectivity-constrained coordination.** Hollinger and Singh introduced "
         "periodic connectivity, requiring a mobile network to regain connectivity every "
         "T_I steps, showed the resulting problem reduces to multi-robot informative "
         "path planning for given connected configurations, and gave a receding-horizon "
         "algorithm with submodular performance guarantees [1], [2]. Kantaros and "
         "Zavlanos treat intermittent connectivity as a temporal-logic constraint [3], "
         "and data-ferrying formulations schedule meetings to move information [4]. In "
         "all of these the meeting exists to carry data. Ours exists to carry evidence "
         "of life, which changes what a missed meeting means and therefore what the "
         "schedule must guarantee.")
    para(pdf,
         "**Rendezvous and co-observation as a detector.** Co-observation planning [5] "
         "is the closest prior mechanism: mutual observations are scheduled offline as "
         "common knowledge and a missed co-observation is the detectable event. Its "
         "problem is adversarial — a compromised robot cannot leave a region without "
         "missing one — whereas ours is stochastic failure plus comms-free recovery, "
         "and its disturbance model does not make benign misses correlated. Planned "
         "synchronization with active observations [6] likewise schedules meetings and "
         "decides when observing is worth its cost. We claim no novelty in the "
         "mechanism 'a missed planned meeting is the signal'; we claim the alignment "
         "condition that makes the signal trustworthy and cheap.")
    para(pdf,
         "**Failure detection.** The φ-accrual detector [7] is the standard adaptive "
         "detector and is per-node: it reasons about one peer's silence in isolation. "
         "Channel-aware and correlated-silence detectors [8] improve on this by gating "
         "a failure belief on how much of the rest of the team is audible, but every "
         "member of this family presumes a packet stream. Under the premise here there "
         "is none, and the analogue of 'channel health' has to be manufactured by the "
         "routing.")
    para(pdf,
         "**Task allocation and vehicle routing.** CBBA [9] and its communication-"
         "efficient descendants stop at the allocation and assume execution is "
         "reliable. Vehicle-routing formulations with synchronization constraints [10] "
         "are the closest classical relatives of a meeting mesh, and correlated "
         "travel-time VRPs [11] model exactly the disturbance structure we exploit, but "
         "neither treats a synchronization as an observation whose failure carries "
         "information.")
    para(pdf,
         "**Behaviour trees.** Reactive fault recovery is the founding use case of "
         "behaviour trees, and their modularity and preemption semantics are well "
         "characterized [12]. We use a behaviour tree as the execution layer for exactly "
         "that reason and make no novelty claim for it; its role here is to enforce the "
         "priority order that the optimization specifies, which we make explicit in "
         "Section V-E.")
    para(pdf,
         "Existing work therefore either schedules meetings to move data under a "
         "connectivity constraint, or uses a missed observation as a detection event "
         "without a disturbance model, or detects failures from packet statistics that "
         "are unavailable here. None co-designs the physical approach geometry against "
         "the disturbance's spatial correlation and prices the result against a hard "
         "coverage requirement. Our work fills that gap.")
    _part2(pdf)
    return pdf




def _part2(pdf):
    # ---------------- III. Problem setup ----------------
    h1(pdf, "III", "Problem Setup")
    para(pdf,
         "We are given a team of K robots and a set P of N points of interest in a "
         "planar workspace, all launched from a common dock. Each robot cruises at "
         "speed V and each point requires a scan of duration S. The team must visit "
         "every point. Robots exchange nothing except when co-located; the plan is "
         "computed offline and is common knowledge.")
    para(pdf,
         "A solution is a triple (x, π, M): an assignment x ∈ {0,1}^(N×K) with x_ik = 1 "
         "iff point i is served by robot k; a per-robot visiting order π = (π_1..π_K); "
         "and a rendezvous mesh M, a set of appointments (a, b, m, τ) requiring robots "
         "a and b to be at location m at time τ. Let T_k(x, π, M) be the time at which "
         "robot k completes its last scan. We seek")
    equation(pdf,
             r"(x^{*},\pi^{*},M^{*})\;\in\;\arg\min_{(x,\pi,M)\,\in\,\mathcal{F}}"
             r"\;\;\max_{k\in\{1,\dots ,K\}} T_k(x,\pi,M)",
             tag="(1)")
    para(pdf, "where the feasible set F is defined by four constraints:")
    bullet(pdf, "**C1 (coverage, hard).** Σ_k x_ik = 1 for every i ∈ P. Every point is "
                "served exactly once. This is not traded against anything.")
    bullet(pdf, "**C2 (recovery).** For every victim v there is a reassignment of v's "
                "remaining points to the survivors that completes.")
    bullet(pdf, "**C3 (detection cadence).** Every robot participates in at least one "
                "appointment in every window of length Δ.")
    bullet(pdf, "**C4 (alignment).** Every appointment satisfies the alignment "
                "condition of Section IV, so that a missed appointment is evidence of "
                "failure rather than of weather.")
    para(pdf,
         "In practice we solve (1) lexicographically, because the constraints are not "
         "equally negotiable:")
    equation(pdf,
             r"\mathrm{coverage}=100\%\;\;\succ\;\;\min\max_k T_k"
             r"\;\;\succ\;\;\min\max_v T^{\mathrm{post}}(v)"
             r"\;\;\succ\;\;\min\;\mathrm{meeting\;overhead}",
             tag="(2)", fontsize=16)
    para(pdf,
         "Problem (1) is NP-hard: setting K = 1 and dropping C2–C4 leaves the open "
         "travelling salesman problem. We therefore decompose it in Section V and make "
         "no optimality claim about the allocation; the claims of this paper are about "
         "the alignment constraint C4, which is where the new structure is.")
    h2(pdf, "A. Disturbance and meeting model")
    para(pdf,
         "The disturbance is a stationary Gaussian field u of unit variance with "
         "correlation ρ(d) = exp(−d/λ), where λ is the correlation length; this is the "
         "standard model for a current or wind field and is the one used by the "
         "correlation-aligned rendezvous literature. A current perturbs a vehicle's "
         "effective speed by a fraction κ of cruise, so the arrival offset a robot "
         "accumulates is proportional to how long it has been transiting through the "
         "field. A completed scan is a resynchronization point — the plan is replanned "
         "forward from the robot's actual clock whenever it finishes one — so the "
         "unabsorbed exposure that decides an appointment is the robot's final "
         "uninterrupted approach. Writing t_k for the duration of that approach and a_k "
         "for its locale (the travel-weighted mean position along it),")
    equation(pdf, r"\theta_k \;=\; \kappa\, t_k\, u(a_k),\qquad "
                  r"\theta_k\sim\mathcal{N}\!\left(0,\;\kappa^2 t_k^2\right)",
             tag="(3)")
    para(pdf,
         "Each partner waits ε past the appointment before abandoning it, so a delay "
         "common to both still produces a meeting and only the difference can destroy "
         "one. The appointment is therefore made iff")
    equation(pdf, r"\left|\,\theta_i-\theta_j\,\right| \;\leq\; \varepsilon", tag="(4)")
    para(pdf,
         "Setting t_i = t_j = T recovers the fixed-approach model used previously, in "
         "which the variance of the difference is 2σ²(1 − ρ(d)) with σ = κT; keeping t "
         "as a variable is the single modelling change that the rest of the paper "
         "rests on.")

    # ---------------- IV. Theory ----------------
    h1(pdf, "IV", "When is a Rendezvous Aligned?")
    claim(pdf, "Proposition 1 (alignment condition).",
          "Let an appointment have partners whose final approaches last t_i and t_j "
          "and whose approach locales are separated by d. Under (3)–(4) the benign "
          "miss probability is",
          None)
    equation(pdf,
             r"p(t_i,t_j,d)\;=\;2\,\Phi\!\left(\frac{-\varepsilon}"
             r"{\kappa\sqrt{\,t_i^{2}+t_j^{2}-2\,t_i t_j\,\rho(d)\,}}\right)",
             tag="(5)")
    para(pdf, "and p ≤ α holds if and only if  t_i² + t_j² − 2 t_i t_j ρ(d) ≤ B, where")
    equation(pdf, r"B\;:=\;\frac{\varepsilon^{2}}{\kappa^{2}\,z_{1-\alpha/2}^{2}}",
             tag="(6)")
    para(pdf,
         "**Proof.** θ_i and θ_j are jointly Gaussian, zero mean, with variances κ²t_i² "
         "and κ²t_j² and covariance κ²t_i t_j ρ(d), because u has unit variance and "
         "correlation ρ. Hence θ_i − θ_j is zero-mean Gaussian with variance "
         "κ²(t_i² + t_j² − 2 t_i t_j ρ(d)), and (4) fails with probability "
         "2Φ(−ε/sd), which is (5). The map sd ↦ 2Φ(−ε/sd) is strictly increasing, so "
         "p ≤ α iff sd ≤ ε/z_(1−α/2), i.e. iff the bracket is at most B.")
    y = pdf.get_y() - LH
    pdf.rect(pdf.w - pdf.r_margin - 2.6, y + 1.2, 2.2, 2.2, style="F")
    pdf.ln(1.4)
    para(pdf,
         "We call B the **alignment budget**: it is large when the meeting window ε is "
         "generous and small when the current κ is strong or the target α strict. The "
         "**admissible set** is A = {(t_i, t_j, d) : t_i² + t_j² − 2 t_i t_j ρ(d) ≤ B}. "
         "Proposition 1 says alignment is a property of a point in A, not of the "
         "separation alone. For the symmetric case t_i = t_j = t that both constructions "
         "below produce, membership reduces to 2t²(1 − ρ(d)) ≤ B, which is the curve "
         "drawn in Fig. 2(a).")
    claim(pdf, "Corollary 1 (the separation-only corner).",
          "Fix both approaches at a full-length transit t = T. Then (t,t,d) ∈ A iff",
          None)
    equation(pdf,
             r"d\;\leq\;d^{*}\;=\;-\lambda\ln\!\left(1-\frac{B}{2T^{2}}\right)"
             r"\;\;\approx\;\;\frac{\lambda B}{2T^{2}}\quad (B\ll T^{2})",
             tag="(7)")
    para(pdf,
         "This is exactly the constraint imposed by correlation-aligned rendezvous, and "
         "it is **linear in the budget**: halving the budget halves the admissible "
         "separation. It is what makes the constraint expensive. At the parameters of "
         "Section VI (κ = 0.4, ε = 30 s, α = 0.2, λ = 150 m, T = 300 s) it evaluates to "
         "d* = 2.9 m — a corridor narrower than the vehicles, which in practice can only "
         "be satisfied by bringing both robots to a standstill at the meeting point.")
    claim(pdf, "Corollary 2 (the crossing corner).",
          "Let the appointment be sited at the midpoint of two tasks separated by s, "
          "one owned by each partner, each partner ending its approach by scanning its "
          "own task. Then both approaches are the half-segment, t = s/2V, and the two "
          "locales are the two half-segment midpoints, d = s/2, so (t,t,d) ∈ A iff",
          None)
    equation(pdf,
             r"2\left(\frac{s}{2V}\right)^{2}\!\left(1-e^{-s/2\lambda}\right)\leq B"
             r"\qquad\Longrightarrow\qquad s\;\leq\;s^{*}=\left(4V^{2}\lambda B\right)^{1/3}",
             tag="(8)")
    para(pdf,
         "**Proof.** The first form is Proposition 1 with t = s/2V and d = s/2. For "
         "s much smaller than λ, 1 − e^(−s/2λ) = s/2λ + O(s²/λ²), so the left side is "
         "s³/(4V²λ) + O(s⁴), giving the stated bound; because the expansion "
         "over-estimates 1 − e^(−s/2λ), the closed form is a conservative "
         "(under-)estimate of the true s*.")
    y = pdf.get_y() - LH
    pdf.rect(pdf.w - pdf.r_margin - 2.6, y + 1.2, 2.2, 2.2, style="F")
    pdf.ln(1.4)
    para(pdf,
         "The admissible separation is now the **cube root** of the budget. The reason "
         "is structural: at a crossing the separation and the duration are the same "
         "physical quantity, so shrinking the geometry shrinks the exposure as well as "
         "decorrelating less, and the variance falls as s³ rather than as s.")
    claim(pdf, "Theorem 1 (relaxation factor).",
          "Let d_cross = s*/2 be the locale separation admitted at the crossing corner "
          "and d* that admitted at the separation-only corner. In the tight-budget "
          "regime B much smaller than T²,", None)
    equation(pdf,
             r"\frac{d_{\mathrm{cross}}}{d^{*}}\;=\;"
             r"4^{1/3}\,V^{2/3}\,T^{2}\,\lambda^{-2/3}\,B^{-2/3}"
             r"\;=\;\Theta\!\left(B^{-2/3}\right)",
             tag="(9)")
    para(pdf,
         "**Proof.** Substitute (8) and the small-budget form of (7): "
         "d_cross/d* = [(4V²λB)^(1/3)/2]·[2T²/(λB)] = 4^(1/3)V^(2/3)λ^(1/3)B^(1/3)T²/(λB), "
         "which simplifies to (9). The exponent is negative, so the factor diverges as "
         "the budget tightens.")
    y = pdf.get_y() - LH
    pdf.rect(pdf.w - pdf.r_margin - 2.6, y + 1.2, 2.2, 2.2, style="F")
    pdf.ln(1.4)
    para(pdf,
         "Theorem 1 is the paper's main analytical point, and its direction is worth "
         "stating plainly: **the advantage of letting duration co-vary with separation "
         "is largest exactly where alignment is hardest.** A stronger current or a "
         "stricter false-alarm target shrinks B, which shrinks the separation-only "
         "corridor linearly but the crossing separation only as a cube root.")
    h2(pdf, "A. Numerical verification")
    para(pdf,
         "Table I checks Proposition 1 against Monte Carlo sampling of the correlated "
         "field (4×10⁵ draws per row) and evaluates both corners exactly, without the "
         "small-argument expansion. The closed form of Proposition 1 is accurate to "
         "6×10⁻⁴ in probability. The exact relaxation factor rises from 2.8× in the "
         "loosest setting to 67.8× in the tightest, and Fig. 2(b) confirms the B^(−2/3) "
         "slope over two decades of budget. At the nominal parameters used in Section "
         "VI the exact factor is 39.4× while the closed form (9) gives 35.4×, "
         "conservative as expected.")
    table(pdf,
          ["σ/ε", "α", "B (s²)", "d* (m), Eq. 7", "s* exact (m)", "s* by Eq. 8",
           "d_cross/d*"],
          [["1×", "0.40", "127060", "183.6", "1025.2", "673.1", "2.8×"],
           ["1×", "0.20", "54799", "54.5", "697.1", "508.5", "6.4×"],
           ["1×", "0.05", "23429", "20.9", "483.8", "383.1", "11.6×"],
           ["2×", "0.40", "31765", "29.1", "550.0", "424.0", "9.4×"],
           ["2×", "0.20", "13700", "11.9", "388.5", "320.4", "16.4×"],
           ["2×", "0.05", "5857", "5.0", "278.4", "241.3", "28.1×"],
           ["4×", "0.40", "7941", "6.8", "313.1", "267.1", "23.1×"],
           ["*4×", "*0.20", "*3425", "*2.9", "*227.2", "*201.8", "*39.4×"],
           ["4×", "0.05", "1464", "1.2", "166.0", "152.0", "67.8×"]],
          [0.9, 0.8, 1.2, 1.5, 1.4, 1.3, 1.2],
          caption="TABLE I — The two corners of the admissible set, evaluated exactly "
                  "(λ = 150 m, V = 2 m/s, ε = 30 s, T = 300 s).",
          notes="Bold row = the operating point of Section VI. Proposition 1 agrees "
                "with Monte Carlo to within 6×10⁻⁴ over the four (t_i,t_j,d) settings "
                "tested. Eq. 8 is conservative against the exact root in every row.")
    figure(pdf, "fig_alignment.png",
           "**Fig. 2.** (a) The admissible alignment set. The separation-only corner "
           "(red) sits at a full-length approach and is admitted only within d* = 2.9 m; "
           "the crossing locus (blue dashed), along which duration and separation "
           "co-vary, enters the set 39× further out. (b) The relaxation factor against "
           "the alignment budget, with the B^(−2/3) reference slope of Theorem 1.")
    h2(pdf, "B. Detection and cadence")
    claim(pdf, "Proposition 2 (detection latency).",
          "If every robot participates in an appointment in every round of the mesh "
          "(C3), then a robot that fails at time t_f is missed by a scheduled partner "
          "at the first appointment after t_f, so the detection latency is at most "
          "Δ + ε.",
          "A failed robot attends no appointment. By C3 its next appointment occurs "
          "at some τ ≤ t_f + Δ, and its partner abandons the appointment at τ + ε "
          "having observed the absence. The partner is alive by assumption of a single "
          "failure, so the observation is made.")
    claim(pdf, "Proposition 3 (cadence feasibility).",
          "Let r_max(k) be the distance from robot k's worst-case meeting site to its "
          "farthest assigned point. A schedule in which k must be at that site at both "
          "ends of a window can serve that point only if Δ ≥ 2·r_max(k)/V + S.",
          "Within such a window the robot must leave the site, reach the point, scan "
          "it, and return, costing at least 2 r_max/V + S. If the window is shorter the "
          "point is infeasible in every window bounded by that site, and the schedule "
          "strands it.")
    para(pdf,
         "Proposition 3 is elementary but it bites. On the instances of Section VI it "
         "requires Δ ≥ 380 s, so the Δ = 300 s cadence commonly used with banded or "
         "grid-cell decompositions is **infeasible** on compact Voronoi regions: the "
         "executor deadlocks, coverage silently falls short, and — worse for an "
         "experimenter — the stranded robots sit at their meeting points and report "
         "perfect alignment. Our implementation tests the bound and refuses to report "
         "such a run as a result.")
    _part3(pdf)


def _part3(pdf):
    # ---------------- V. Algorithm ----------------
    h1(pdf, "V", "Algorithm Design")
    para(pdf,
         "We solve (2) in layers, each layer taking the previous one as fixed. Only "
         "Layer 2 is new; Layers 0 and 1 are included because the alignment cost is "
         "meaningless except relative to the coverage plan it perturbs.")
    h2(pdf, "A. Layer 0 — coverage-first allocation")
    para(pdf,
         "Points are partitioned by constrained k-means in which the assignment step "
         "is an exact balanced Hungarian assignment: each robot receives a prescribed "
         "count, so every point is assigned exactly once and C1 holds by construction "
         "rather than by penalty. Each robot's route is an open TSP from the dock "
         "(nearest neighbour, then 2-opt), and a local search moves points off the "
         "bottleneck robot while the makespan falls. This yields compact, balanced "
         "regions: on the reference instance, 23/22/23/22 points and a makespan of "
         "33.6 min, which is the no-rendezvous lower bound against which every later "
         "cost is quoted.")
    figure(pdf, "coverage_first.png",
           "**Fig. 1.** The coverage-first base the rendezvous mesh is added to. Left: "
           "the balanced partition and per-robot open-TSP routes; 100% of the 90 points "
           "are covered by construction and the makespan is 33.6 min. Right: the "
           "worst-case makespan after each single failure, the resilience term of the "
           "lexicographic objective (2). Adding appointments never changes the "
           "assignment, so this layer is unaffected by anything in Section V-C onward.")
    h2(pdf, "B. Layer 1 — recovery")
    para(pdf,
         "Region adjacency is the Delaunay graph of the region centroids. For each "
         "single failure the victim's points are reassigned to the nearest surviving "
         "centroid and the makespan recomputed, giving the worst-case post-failure "
         "makespan of (2); on the reference instance this is 55.6 min, +66% over "
         "nominal. Because the mesh of Layer 2 adds appointments but never changes the "
         "assignment, C2 is unaffected by anything that follows.")
    h2(pdf, "C. Layer 2 — the rendezvous mesh")
    para(pdf,
         "Each round of the mesh is a **maximum** matching on the adjacency graph, "
         "tie-broken toward the partner each robot has gone longest without meeting. "
         "Maximality matters: a greedy maximal matching can strand two mutually "
         "non-adjacent robots and open a 2Δ hole in C3, and on a four-region graph it "
         "does. Rotating the partner rather than fixing a buddy removes the single "
         "point of monitoring failure, and because the graph is small the matching is "
         "computed exactly by subset dynamic programming. Routes are then executed "
         "forward with the appointments as hard time-and-place constraints under a "
         "meeting-wins rule: work that does not fit before an appointment is deferred, "
         "never the appointment.")
    h2(pdf, "D. Reserving the crossings")
    para(pdf,
         "Corollary 2 requires each appointment to sit on a pair of nearby tasks, one "
         "owned by each partner, both still unscanned when the round arrives. Neither "
         "half of that is automatic. Incidental crossings essentially do not exist: on "
         "our instances only one to three cross-boundary pairs in the entire field lie "
         "within a 20 m contact range, and their nominal visit times differ by 101–754 "
         "s, so a crossing has to be constructed rather than harvested. And if it is "
         "merely designated, ordinary coverage work scans the few close pairs long "
         "before the round that needed them. Algorithm 1 therefore reserves anchors on "
         "a rolling look-ahead of h rounds and forbids ordinary work from consuming "
         "them.")
    algo(pdf, "Algorithm 1  Rolling crossing reservation (called at each round r)",
         ["  1: input: mesh M, unscanned sets U_k, look-ahead h, budget B",
          "  2: R <- {} ;  used <- {}",
          "  3: for r' = r to r+h do",
          "  4:     for each scheduled pair (a,b) in M[r'] do",
          "  5:         C <- { (i,j) : i in U_a, j in U_b, i,j not in used }",
          "%          candidates are cross-boundary task pairs, cached per (a,b)",
          "  6:         if C = {} then continue                  % fall back to a dwell",
          "  7:         (i*,j*) <- argmin_{(i,j) in C}  || p_i - p_j ||",
          "%          separation sets the false-alarm rate: Cor. 2 wants it small",
          "  8:         m <- (p_i* + p_j*)/2 ;   s <- || p_i* - p_j* ||",
          "  9:         if 2 (s/2V)^2 (1 - exp(-s/2L)) > B then flag (r',a,b) unaligned",
          " 10:         if either partner cannot reach its task, scan it and close",
          "             the hop to m by tau_{r'} then continue   % an unreachable",
          "%                                                       crossing is none",
          " 11:         R <- R + {(r',a,b,i*,j*,m)} ;  used <- used + {i*,j*}",
          " 12: protect { i*,j* : r' > r } from ordinary coverage work",
          " 13: return R"])
    para(pdf,
         "The look-ahead h is the construction's only real knob and it is two-sided. "
         "With h = 0 the anchors are eaten and 45% of appointments degrade to an "
         "arbitrary midpoint; reserving the whole mission instead locks roughly half "
         "the points away from the routes and is catastrophic, costing +249% makespan. "
         "Section VI-F traces the frontier between these.")
    h2(pdf, "E. Execution: a behaviour tree")
    para(pdf,
         "The frozen mesh is executed by an identical behaviour tree on every robot "
         "(Fig. 3). We use a behaviour tree because reactive fault recovery is its "
         "founding use case and because its fallback semantics express the one thing "
         "the runtime must get right — priority. We claim no novelty for the tree; the "
         "observation worth recording is that its tick order **is** the lexicographic "
         "objective (2), so the architecture and the optimization agree by "
         "construction rather than by convention:")
    bullet(pdf, "the **contingency** branch ticks first, so a detected failure preempts "
                "everything; it is a pure lookup in the precomputed table f(v, c) "
                "indexed by victim and last checkpoint, which keeps survivors in "
                "agreement without negotiating;")
    bullet(pdf, "the **appointment** branch ticks next, which is meeting-wins: an "
                "appointment is never sacrificed to a scan;")
    bullet(pdf, "inside it, the fallback tries **ScanReservedAnchor** before "
                "**DwellInCorridor**, so alignment is paid in useful work whenever the "
                "reservation survived and in idle only when it did not — this single "
                "fallback is where the cost difference of Section VI is produced;")
    bullet(pdf, "the **coverage** branch ticks last and only with the time that remains.")
    figure(pdf, "fig_bt.png",
           "**Fig. 3.** The per-robot behaviour tree. Tick priority left to right "
           "reproduces the lexicographic objective (2). Every branch is a lookup on "
           "the frozen mesh, so no runtime negotiation is required and survivors that "
           "observe different things still apply the same plan.")
    h2(pdf, "F. Complexity")
    para(pdf,
         "Candidate pairs cost O(|P_a||P_b|) per adjacent pair and are cached once. "
         "The exact matching is O(2^K·K) by subset DP, acceptable for the team sizes of "
         "interest and replaceable by blossom matching at scale. Algorithm 1 costs "
         "O(h·|M_r|·|C|) per round with |C| capped at the closest 60 pairs. The forward "
         "executor is O(N²) in the worst case through the greedy insertion test. All "
         "of it is offline: at runtime each robot performs table lookups.")

    # ---------------- VI. Results ----------------
    h1(pdf, "VI", "Simulation Results")
    para(pdf,
         "All results use K = 4 robots, N = 90 points on a 600 m square, V = 2 m/s, "
         "S = 60 s, ε = 30 s and Δ = 600 s (Proposition 3 requires ≥ 380 s here). The "
         "disturbance has λ = 150 m and strength σ/ε = 4 unless stated, i.e. κ = 0.4, a "
         "current worth ±40% of cruise; α = 0.2. Every configuration is run on the same "
         "five random instances and every run is verified to achieve 100% coverage, to "
         "keep every appointment, and to satisfy the cadence bound. False alarms are "
         "reported two ways: the **realised** rate, Monte-Carlo over the correlated "
         "field with each partner's schedule slack credited against its lateness, and "
         "the **no-slack bound**, the same schedule with slack removed — the quantity "
         "Proposition 1 bounds and the one that survives when the mission is tight. "
         "These are simulation results only; no field trials are reported.")
    h2(pdf, "A. Does the accrual model hold? Validation against HoloOcean")
    para(pdf,
         "Equation (3) is ours, and Propositions 1–2 and Theorem 1 rest entirely "
         "on it, so we tested it against dynamics we did not write. A vehicle is "
         "commanded down a straight leg to a waypoint in HoloOcean; a genuine "
         "ocean-current vector is imposed through the simulator's own "
         "set_ocean_currents interface; its rigid-body hydrodynamics and its "
         "built-in PID waypoint controller then produce whatever arrival time "
         "they produce, measured at a 2 m capture radius. No effective-speed "
         "model is imposed anywhere, so drag, thrust saturation, heading "
         "dynamics and controller behaviour are all free to falsify Eq. (3).")
    para(pdf,
         "**Platform.** HoloOcean's SurfaceVessel turns out to be inert to ocean "
         "currents — a 3 m/s current displaces it 0.18 m in 20 s — so the surface "
         "vessel of our scenario cannot be used for this test, and we validate on "
         "the HoveringAUV, which responds physically (8.08 m of drift under the "
         "same current). The claim under test concerns a vehicle transiting a "
         "current field, which the AUV is; the substitution costs us free-surface "
         "effects and we carry it as a limitation. Nominal transit is 58 m in "
         "63.4 s, a cruise of 0.92 m/s.")
    para(pdf,
         "Equation (3) makes two structural claims — proportionality to exposure "
         "time, and inheritance of the field's spatial correlation — and one "
         "distributional claim, linearity in u (hence a Gaussian θ). The two "
         "structural claims hold and the distributional one does not.")
    bullet(pdf,
           "**The exposure-time factor holds.** Flying legs of 20–120 m under a "
           "fixed current, the fitted exponent of θ against t is 1.03–1.05 "
           "against Eq. (3)'s 1.00, and θ/t is constant to within 6% over that "
           "six-fold range (Fig. 5b). This is the claim Theorem 1 depends on: the "
           "cube-root scaling of Corollary 2 comes from t entering the variance "
           "quadratically.")
    bullet(pdf,
           "**The correlation factor holds.** Flying a rank of eight parallel "
           "legs through one realization of a genuine correlated field and "
           "repeating over 60 realizations, Corr(θ_i, θ_j) is statistically "
           "indistinguishable from the realized Corr(u_i, u_j) at every "
           "separation, at a small-signal amplitude and at one over four times "
           "larger (Fisher-z, maximum |z| = 1.47 and 0.89 respectively). From the "
           "measured response curve alone, and before running this experiment, we "
           "predicted that the nonlinearity would attenuate that correlation by "
           "up to a third at the larger amplitude; the prediction was wrong, and "
           "wrong in the model's favour — the measured ratio Corr(θ)/Corr(u) is "
           "1.01–1.20. An offset integrates the field along the whole leg, and "
           "that spatial averaging smooths the excursions which drive the "
           "point-wise nonlinearity. The comparison must be made against the "
           "realized field correlation rather than against exp(−d/λ): at "
           "d = 320 m the field itself sampled to −0.03 against an expected "
           "0.118, which looks like a failure of the model but is sampling error "
           "in the field (the standard error of a correlation at n = 60 is "
           "≈ 0.13). Against the realized correlation the combined variance of "
           "Proposition 1 is recovered at small signal with a median ratio of "
           "1.25 (range 0.53–1.58), against a ≈ 18% standard error on a variance "
           "at this sample size; at the larger amplitude the measured variance "
           "runs a median 2.4× above the prediction (range 0.83–5.63). That gap "
           "is the nonlinearity, not the correlation: the shape is right and the "
           "scale is not.")
    bullet(pdf,
           "**The linearity does not hold.** The true response is convex and "
           "asymmetric (Fig. 5a): at 2 m/s a head current costs +39.7 s while a "
           "tail current saves only −6.8 s, a factor of 5.8. The tail side "
           "saturates — the largest speed-up available at 3 m/s is 22% — while "
           "the head side is unbounded, reaching +210% at 3 m/s. A symmetric "
           "Gaussian θ is therefore unphysical on the tail side. The linear "
           "regime is roughly |c| ≤ 0.3 m/s; even restricted to |c| ≤ 1 the "
           "linear fit achieves only R² = 0.88. The measured skewness of θ rises "
           "from +0.78 at small signal to +2.00 at the larger amplitude, "
           "against the zero that a Gaussian θ would have.")
    figure(pdf, "holoocean_eq3/fig_eq3_validation.png",
           "**Fig. 5.** Equation (3) against HoloOcean rigid-body hydrodynamics. "
           "(a) The response is convex and asymmetric, so the linearity in u is a "
           "small-signal approximation. (b) θ is proportional to exposure time, "
           "as claimed — flat lines are constant θ/t across a six-fold range of "
           "leg duration. (c) Offsets inherit the field's spatial correlation; "
           "the dashed line is the realized field correlation in the same "
           "realizations, which is the correct comparison. (d) The variance "
           "Proposition 1 actually uses, measured against its prediction.",
           frac=1.0)
    para(pdf,
         "**What this costs the paper.** The load-bearing structure survives: "
         "Var(θ_i − θ_j) retains the form t²·(1 − ρ_eff(d)) that Corollaries 1–2 "
         "and Theorem 1 exploit, because a monotone nonlinearity in u rescales κ "
         "and attenuates ρ without changing how t and d enter. What is lost is "
         "the Gaussian tail: the real θ is right-skewed, so the miss "
         "probabilities of Proposition 1 — and therefore the false-alarm columns "
         "of Tables II and III — are **optimistic**, with the late tail heavier "
         "than predicted. At the larger amplitude the spread is understated by "
         "about a factor of 1.5 in standard deviation, which is not a "
         "rounding error. Two things limit the damage. The asymmetry is a "
         "property of the vehicle and the current, not of any meeting design, so "
         "it shifts every schedule in Table II together rather than reordering "
         "them; and Theorem 1 compares two corners of the same admissible set "
         "under the same monotone response, so its cube-root scaling is "
         "unaffected and only the calibration of B moves. We would not defend the "
         "absolute false-alarm figures to better than an order of magnitude; we "
         "would defend the comparison between schedules, and the scaling.")

    h2(pdf, "B. The cost of alignment")
    table(pdf,
          ["schedule", "makespan +%", "FA measured %", "FA Gaussian %",
           "idle (min)", "appr. (s)", "C4 %"],
          [["unaligned mesh", "36.6 ± 4.2", "7.60", "8.48", "29.7", "89.5", "0"],
           ["absorb +60 s", "58.3 ± 8.3", "3.15", "0.55", "55.9", "91.1", "0"],
           ["absorb +120 s", "93.4 ± 2.8", "1.84", "0.01", "91.3", "93.0", "0"],
           ["absorb +240 s", "214.0 ± 18.4", "0.80", "0.00", "211.0", "95.9", "0"],
           ["C4 corridor", "58.3 ± 8.3", "0.00", "0.00", "55.9", "0.7", "100"],
           ["*crossings h = 1", "*36.3 ± 4.1", "*0.65", "*0.01", "*31.8", "*34.3", "*0"],
           ["crossings h = 4", "62.4 ± 2.5", "0.10", "0.00", "64.1", "15.6", "0"]],
          [1.75, 1.3, 1.15, 1.15, 0.85, 0.8, 0.6],
          caption="TABLE II — Cost and protection of each schedule, mean ± s.d. "
                  "over five instances at σ/ε = 4. Makespan is relative to the "
                  "33.6 min no-rendezvous Layer-0 plan. All rows achieve 100% "
                  "coverage and miss no appointment.",
          notes="'FA measured' recomputes the false alarms through the response "
                "curve measured in Section VI-A; 'FA Gaussian' is Eq. (3) as "
                "written. Absorption is the only family the two columns disagree "
                "about, and they disagree by up to 80×. 'C4 %' is the fraction of "
                "appointments meeting the separation-only rule d ≤ d* = 2.9 m: "
                "the crossing schedules meet it on none, yet are protected — they "
                "satisfy Proposition 1 through the duration term instead.")
    para(pdf,
         "Three things stand out. First, **the mesh itself is the dominant "
         "cost**: +36.6% makespan before any alignment is attempted. Second, "
         "**the C4 corridor delivers its guarantee** — 0.00% under either error "
         "model, with the approach shortened from 89.5 s to 0.7 s — but it costs "
         "a further +21.7 points of makespan, and Table II shows where: team "
         "idle nearly doubles, from 29.7 to 55.9 min. The corridor is bought "
         "with waiting. Third, **the crossing construction obtains comparable "
         "protection for nothing**: 36.3 ± 4.1% against the unaligned mesh's "
         "36.6 ± 4.2%, a difference well inside the instance-to-instance spread, "
         "while measured false alarms fall from 7.60% to 0.65%. Its approach is "
         "34.3 s rather than 89.5 s, and by Proposition 1 that reduction in "
         "duration does the work the corridor does by separation — note it "
         "satisfies the separation-only rule on no appointment at all.")

    h2(pdf, "C. Absorption has a floor; alignment does not")
    para(pdf,
         "A reviewer's first question about any alignment scheme is why not "
         "simply leave earlier and wait, and to our knowledge the "
         "correlation-aligned rendezvous literature does not answer it. Rows 2–4 "
         "of Table II do, and the answer depends entirely on which error model "
         "is used — which is why Section VI-A had to come first.")
    para(pdf,
         "Under Eq. (3) as written, absorption looks like a complete substitute: "
         "60 s of slack costs exactly what the corridor costs and reaches 0.55%, "
         "and 120 s reaches 0.01%. We reported that as an honest negative for "
         "the separation-only formulation in an earlier draft of this work. It "
         "is an artefact. Recomputed through the measured response, the same "
         "schedules give 3.15% and 1.84% — worse by 6× and 80× — because a "
         "fixed slack budget absorbs a bounded delay and the true head-current "
         "tail is not bounded. Alignment is unaffected by the change of model "
         "(0.00% either way) because it removes the differential rather than "
         "absorbing it.")
    para(pdf,
         "Absorption cannot be rescued by spending more. Pushing the budget to "
         "240 s reaches 0.80% at +214% makespan, and beyond that the schedule "
         "**stops existing**: at Δ = 600 s a slack reservation of 300 s leaves "
         "less than the 380 s that Proposition 3 requires, so coverage work is "
         "stranded and the run is not a mission. The floor is therefore "
         "structural, not statistical — it is set by the cadence bound, and it "
         "sits at 0.80% while the corridor is at 0.00% and crossings at 0.10% "
         "(Fig. 6b). This is the same shape of result as the false-alarm floor "
         "of the φ-accrual detector under bursty loss: a family of methods that "
         "cannot reach an operating point no matter how conservatively it is "
         "tuned.")

    h2(pdf, "D. Behaviour as the disturbance strengthens")
    table(pdf,
          ["σ/ε", "unaligned", "absorb +60 s", "absorb +240 s", "C4 corridor",
           "crossings h=1", "crossings h=4"],
          [["1×", "0.57", "0.01", "0.00", "0.00", "0.00", "0.00"],
           ["2×", "3.03", "0.54", "0.06", "0.01", "0.03", "0.02"],
           ["3×", "5.47", "1.68", "0.43", "0.04", "0.19", "0.07"],
           ["4×", "8.05", "3.00", "0.86", "0.10", "0.45", "0.16"]],
          [0.55, 1.0, 1.15, 1.15, 1.05, 1.1, 1.1],
          caption="TABLE III — Benign false alarms (%), recomputed through the "
                  "measured response, against disturbance strength (five "
                  "instances). Makespan is unchanged along each column "
                  "(Table II).",
          notes="Absorption degrades with the disturbance at every budget; "
                "alignment barely moves. The 240 s column is the most absorption "
                "can buy before the schedule strands work.")
    para(pdf,
         "The unaligned mesh degrades from 0.57% to 8.05% as the current "
         "strengthens. Every absorption budget degrades with it — even the "
         "largest affordable one, 240 s, runs from 0.00% to 0.86% — because the "
         "slack was sized for a disturbance the mission then exceeds. The "
         "aligned schedules barely move: the corridor from 0.00% to 0.10% and "
         "crossings at h = 4 from 0.00% to 0.16%, across a sixteen-fold change "
         "in disturbance variance. This is realization-invariance made concrete, "
         "and it is a property of the constraint rather than of a tuned "
         "parameter, which is why it does not have to be re-estimated when the "
         "weather changes — and why, unlike absorption, it does not have to be "
         "sized against a σ nobody measured before the mission.")

    h2(pdf, "E. The admissible-mission envelope")
    para(pdf,
         "Everything so far is an economy: the same protection for less time. "
         "The question a mission planner actually asks is different — can this "
         "mission be flown at all? Call a mission **admissible** for a policy "
         "when that policy simultaneously covers every task, meets the detection "
         "cadence Δ, finishes within a deadline D, and holds benign false alarms "
         "at or below a target. Absorption buys false-alarm reduction with time "
         "and therefore trades directly against D, and by Section VI-C it cannot "
         "be pushed past a bounded budget in any case. Alignment by corridor "
         "also costs time. Alignment by crossing costs route order, which the "
         "deadline does not price. That difference turns the economy into a "
         "capability.")
    figure(pdf, "fig_envelope.png",
           "**Fig. 6.** (a) The admissible-mission envelope over deadline and "
           "disturbance, at a 1% false-alarm target. In the green cells the "
           "natural-crossing construction is the only policy that can fly the "
           "mission; above them several policies work, and below them the mesh "
           "itself does not fit the deadline. (b) Absorption's floor: past "
           "+240 s of reserved slack the schedule strands coverage work, so no "
           "budget reaches below 0.80%, while the aligned schedules sit an order "
           "of magnitude lower and cheaper. (c) The Gaussian form of Eq. (3) "
           "flattered absorption by up to 80× and left alignment essentially "
           "unchanged, which is why the model validation had to precede the "
           "comparison.", frac=1.0)
    para(pdf,
         "Fig. 6(a) maps it. At a generous deadline every policy is admissible "
         "and the choice is an economy. Tightening the deadline to 1.5× the "
         "no-rendezvous makespan removes absorption and the corridor together — "
         "both cost more than +50% — and at a disturbance of 2× or more it also "
         "removes the unaligned mesh, which breaches the false-alarm target. "
         "What is left is the crossing construction alone. Tightening further, "
         "to 1.3×, removes everything: the mesh has an irreducible cost of about "
         "+35% and no rendezvous schedule fits inside that, which is an honest "
         "boundary on the claim rather than a result. The exclusive region is "
         "therefore narrow but real, and it is exactly the operating point a "
         "deadline-driven survey sits at.")
    table(pdf,
          ["K, N", "unaligned", "absorb +60 s", "absorb +240 s", "C4 corridor",
           "crossings h=1"],
          [["4, 90", "38.7 / 9.92", "56.0 / 3.01", "224 / 0.84", "56.0 / 0.00",
            "*35.4 / 0.73 yes"],
           ["6, 120", "25.9 / 5.42", "52.9 / 1.89", "187 / 0.45", "52.9 / 0.00",
            "*35.5 / 0.42 yes"],
           ["8, 160", "28.6 / 4.81", "52.2 / 1.56", "192 / 0.28", "52.2 / 0.00",
            "*33.8 / 0.63 yes"]],
          [0.7, 1.05, 1.15, 1.1, 1.1, 1.3],
          caption="TABLE IV — The exclusive region persists with team size: "
                  "makespan overhead (%) / measured false alarms (%) at σ/ε = 4, "
                  "three instances each.",
          notes="\"yes\" marks the policies admissible at a deadline of 1.5× with a 1% "
                "false-alarm target. In every team size the crossing "
                "construction is the only one, and it is the only column whose "
                "cost does not grow with the protection it provides.")
    para(pdf,
         "Two features of Table IV are worth separating. The crossing "
         "construction is admissible at all three team sizes, and it is the only "
         "one; but note also that it is the only column whose overhead is flat "
         "in K (35.4, 35.5, 33.8) while remaining protected. Absorption's "
         "overhead is flat too, but its false alarms are an order of magnitude "
         "worse; the corridor matches the protection but never fits the "
         "deadline. We do not claim the exclusive region widens with K — on "
         "these three sizes it does not measurably — only that it does not close.")

    h2(pdf, "F. The look-ahead frontier")
    table(pdf,
          ["look-ahead h", "0", "1", "2", "3", "4", "6"],
          [["makespan +%", "38.4", "35.5", "42.1", "62.5", "63.2", "123.7"],
           ["FA realised %", "0.10", "0.02", "0.00", "0.00", "0.00", "0.00"],
           ["no-slack bound %", "26.3", "12.8", "10.1", "3.9", "0.2", "0.9"],
           ["approach (s)", "59", "37", "32", "22", "16", "20"],
           ["fell back to dwell %", "45", "20", "16", "6", "0", "2"]],
          [1.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
          caption="TABLE V — How far ahead anchors are reserved traces the whole "
                  "cost-versus-guarantee frontier (four instances, σ/ε = 4).")
    para(pdf,
         "Reserving one round ahead is the knee: it removes most of the fallbacks at no "
         "cost. Reserving four rounds ahead buys the full separation-only-grade bound "
         "(0.2%) at +63.2%, which is what the corridor costs — so the guarantee is "
         "available either way, and the crossing route to it is not cheaper. Reserving "
         "six rounds ahead is worse on **both** axes, because the locked anchors starve "
         "the routes. The knob is genuinely two-sided and the optimum is interior.")
    figure(pdf, "layer2_c4.png",
           "**Fig. 4.** (a) A woven mesh on the 100%-coverage base; each × is a "
           "reserved cross-boundary task pair that both partners owed. (b) Where the "
           "time goes: the corridor's cost is the red idle band. (c) False alarms "
           "against current strength, solid realised and dotted no-slack. (d) The "
           "cost-versus-guarantee plane, with the crossing frontier annotated by "
           "look-ahead depth. Legend keys map to Table II as: plain = unaligned mesh, "
           "slack eNN = arrive early +NN s, loiter = C4 corridor, coopt = C4 corridor "
           "task-sited, cross laH = crossings h = H.")
    h2(pdf, "G. Negative and null results")
    para(pdf,
         "We record four findings that did not go our way, because each of them would "
         "otherwise be re-attempted.")
    bullet(pdf,
           "**Task-siting the corridor does not pay.** Placing the corridor meeting on "
           "a task while retaining the dwell budget costs more than it saves "
           "(+63.6% against +58.3%): the route distortion exceeds the recovered idle. "
           "An early single-instance result suggested otherwise and did not survive "
           "five instances.")
    bullet(pdf,
           "**Tightening α is free, which contradicts the expected Pareto.** Sweeping "
           "α from 0.4 to 0.02 shrinks d* from 6.8 m to 0.9 m with the corridor's "
           "makespan flat at +58%. The binding cost is the stop, not the corridor "
           "radius, so the false-alarm-versus-travel trade-off that the stylized "
           "proxy predicts does not exist on real routes — the cost is a step, not a "
           "curve.")
    bullet(pdf,
           "**Crossings at h = 1 do not deliver the separation-only guarantee.** "
           "Their no-slack bound is 11.5% and they satisfy d ≤ d* on no "
           "appointment; their low measured rate leans partly on the schedule's "
           "own slack. Reserving four rounds ahead buys the strict bound, at the "
           "corridor's price. The honest product is the frontier of Table V, not "
           "a single winner.")
    bullet(pdf,
           "**Multi-witness confirmation is useless here**, as the correlated "
           "model predicts: agreeing silent neighbours are exculpatory rather "
           "than corroborating. We report it because it is the first thing an "
           "i.i.d. intuition suggests.")
    para(pdf,
         "One further negative is methodological and we think it is the most "
         "useful thing in this section. An earlier version of this work "
         "concluded, from Eq. (3) as written, that simply arriving early was "
         "competitive with alignment and that the separation-only formulation "
         "was therefore not obviously worth its cost. That conclusion was an "
         "artefact of a Gaussian assumption in a model we had authored "
         "ourselves, and it survived until the model was checked against an "
         "independent physics engine. It is worth stating plainly because the "
         "failure mode generalizes: a self-authored noise model will tend to be "
         "symmetric, symmetric noise flatters any policy that buys a margin, and "
         "buying a margin is what most engineering fixes do.")

    # ---------------- VII. Discussion ----------------
    h1(pdf, "VII", "Discussion and Limitations")
    para(pdf,
         "The practical reading of Theorem 1 is that alignment should be bought in the "
         "currency that is cheap. Separation is expensive on a coverage route because "
         "it is dragged out of the geometry; duration is cheap because it is a "
         "by-product of the order in which tasks are visited, and ordering is free. "
         "The crossing construction is simply the cheapest point of A that a coverage "
         "route can reach without leaving its own work.")
    para(pdf,
         "Several limitations bound the claims. The results are simulation only — no "
         "field trials — though Section VI-A tests the accrual model against an "
         "independent physics engine rather than against our own kinematics. That "
         "test ran on HoloOcean's HoveringAUV because its SurfaceVessel does not "
         "respond to ocean currents at all, so free-surface effects on a real USV "
         "remain unexamined, and it covers one vehicle and one controller. The "
         "disturbance is a stationary Gaussian field with a known correlation length, "
         "and robustness to a mis-estimated λ is untested — the safe direction is to "
         "use a lower bound on λ, which shrinks both corners. We model a single "
         "failure; the recovery table is indexed by one victim, and concurrent "
         "failures are outside it. The accrual model of (3) treats a completed scan as "
         "a perfect resynchronization point, which is the assumption that makes the "
         "final approach the whole exposure; a model in which delay accumulates across "
         "an entire window would move both corners and we would expect it to favour "
         "the crossing corner less. The measured response is convex, so the absolute "
         "false-alarm figures are optimistic (Section VI-A). We have not "
         "forward-executed a recovery **with** "
         "the mesh in place, so the interaction between a takeover and a reserved "
         "anchor whose owner has died is unmodelled. Finally, the alignment condition "
         "governs false alarms only; detection latency is governed by C3 and is "
         "independent of everything in Section IV.")

    # ---------------- VIII. Conclusion ----------------
    h1(pdf, "VIII", "Conclusion")
    para(pdf,
         "We studied comms-free failure detection for a deadline-constrained "
         "multi-robot coverage team whose only evidence is a rendezvous that did "
         "not happen, under a disturbance whose spatial correlation makes a "
         "missed rendezvous ambiguous. We showed that the alignment requirement "
         "is a joint condition on approach duration and approach separation, "
         "that the separation-only constraint used previously is a conservative "
         "corner of it, and that letting duration co-vary with separation "
         "relaxes the admissible separation from Θ(B) to Θ(B^(1/3)) — a factor "
         "that grows as the budget tightens and is 39× at our operating point. "
         "The construction that realizes this, siting each rendezvous on a "
         "reserved pair of adjacent tasks, costs no more than an unaligned mesh "
         "that offers no protection at all.")
    para(pdf,
         "Testing the accrual model against an independent physics engine "
         "changed what that buys. Proportionality to exposure time and "
         "inheritance of the field's correlation both hold; linearity does not, "
         "and recomputing through the measured response shows that absorbing "
         "delay by arriving early — the obvious alternative, and one the "
         "Gaussian model makes competitive — has a false-alarm floor of 0.80% "
         "set by the cadence bound, while alignment has none. The result is a "
         "capability rather than an economy: there is a region of the "
         "(deadline, disturbance) plane, at a deadline of 1.5× the "
         "no-rendezvous makespan and a disturbance of 2× and above, in which "
         "our construction is the only policy that covers every task, meets the "
         "cadence, finishes on time and stays under a 1% false-alarm target, at "
         "team sizes 4, 6 and 8. Future work is to retire the single-failure "
         "assumption, to test sensitivity to a mis-estimated correlation length, "
         "and to take the mesh to field trials.")

    # ---------------- references ----------------
    pdf.ln(2.6)
    if pdf.get_y() > pdf.h - 45:
        pdf.add_page()
    pdf.set_font("sans", "B", 11.6)
    pdf.mc(0, 5.8, "REFERENCES")
    pdf.ln(1.0)
    refs = [
        "G. A. Hollinger and S. Singh, \"Multi-robot coordination with periodic "
        "connectivity,\" in Proc. IEEE Int. Conf. Robotics and Automation (ICRA), "
        "Anchorage, AK, 2010, pp. 4457–4462.",
        "G. A. Hollinger and S. Singh, \"Multirobot coordination with periodic "
        "connectivity: Theory and experiments,\" IEEE Trans. Robotics, vol. 28, no. 4, "
        "pp. 967–973, 2012.",
        "Y. Kantaros and M. M. Zavlanos, \"Distributed intermittent connectivity "
        "control of mobile robot networks,\" IEEE Trans. Automatic Control, vol. 62, "
        "no. 7, pp. 3109–3121, 2017.",
        "J. Banfi, A. Q. Li, N. Basilico, I. Rekleitis and F. Amigoni, \"Multirobot "
        "online construction of communication maps,\" in Proc. ICRA, 2017. "
        "[data-ferrying / intermittent-communication family; entry to be finalised]",
        "Co-observation planning for multi-robot intrusion detection, arXiv:2403.13266, "
        "2024. [author list to be verified]",
        "Y. Zhong, F. Rossi and D. A. Shell, \"Planned synchronization for multi-robot "
        "systems with active observations,\" Autonomous Robots, 2025, "
        "doi:10.1007/s10514-025-10225-4.",
        "N. Hayashibara, X. Défago, R. Yared and T. Katayama, \"The φ accrual failure "
        "detector,\" in Proc. IEEE Symp. Reliable Distributed Systems (SRDS), 2004, "
        "pp. 66–78.",
        "A. Jadhav, \"Channel-aware failure recovery for decentralized multi-robot task "
        "allocation under bursty communication,\" manuscript (our prior work), 2026.",
        "H.-L. Choi, L. Brunet and J. P. How, \"Consensus-based decentralized auctions "
        "for robust task allocation,\" IEEE Trans. Robotics, vol. 25, no. 4, "
        "pp. 912–926, 2009.",
        "M. Drexl, \"Synchronization in vehicle routing — a survey of VRPs with "
        "multiple synchronization constraints,\" Transportation Science, vol. 46, "
        "no. 3, pp. 297–316, 2012.",
        "Reliable vehicle routing problem in stochastic networks with correlated travel "
        "times, Operational Research, 2019, doi:10.1007/s12351-019-00452-w. "
        "[author list to be verified]",
        "M. Colledanchise and P. Ögren, Behavior Trees in Robotics and AI: An "
        "Introductory Guide. Boca Raton, FL: CRC Press, 2018.",
    ]
    pdf.set_font("serif", "", 8.6)
    for k, r in enumerate(refs, 1):
        x = pdf.get_x()
        pdf.set_x(x + 5)
        pdf.mc(0, 4.2, f"[{k}]  {r}", align="J")
        pdf.set_x(x)
        pdf.ln(0.9)
    pdf.ln(2.0)
    pdf.set_font("serif", "I", 8.2)
    pdf.mc(0, 4.2,
           "Reproducibility. Every number in Tables I–IV is produced by "
           "visualization/taskalloc_coverage_first.py in the accompanying repository "
           "(--dry_run runs the self-tests, --l2 --sweep regenerates the tables and "
           "Fig. 4); Fig. 2 and Fig. 3 by paper/make_paper_figs.py. Section VI-A and "
           "Fig. 5 come from visualization/holoocean_eq3_validation.py, run inside "
           "a HoloOcean environment (--exp probe, --exp time, --exp field --amp A), "
           "with the figure assembled by paper/make_eq3_fig.py. The disturbance "
           "physics is imported from the earlier prototype rather than reimplemented, "
           "and the self-tests assert that the closed form of Proposition 1 reproduces "
           "it exactly.", align="J")


if __name__ == "__main__":
    doc = build()
    out = os.path.join(HERE, "aligned_rendezvous.pdf")
    doc.output(out)
    print("wrote", out)
