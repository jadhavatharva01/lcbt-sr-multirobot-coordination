#!/usr/bin/env python3
"""Build paper/rob538_project_proposal.pdf (fpdf2 + DejaVu; no LaTeX here)."""
import glob
import os

from fpdf import FPDF
from fpdf.enums import XPos, YPos

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(name):
    p = os.path.join("/usr/share/fonts/truetype/dejavu", name)
    if os.path.exists(p):
        return p
    hits = glob.glob(f"/usr/share/fonts/**/{name}", recursive=True)
    return hits[0] if hits else None


TITLE = "Credit Assignment or Observability?"
SUB = ("Separating two causes of coordination failure in decentralized "
       "multi-robot task allocation under bursty communication")
COURSE = "ROB 538 - Multiagent Systems  |  Prof. Kagan Tumer"
AUTHOR = "Atharva Jadhav"

BODY = 9.5
LH = 4.3


class Doc(FPDF):
    def mc(self, w, h, txt, **kw):
        kw.setdefault("new_x", XPos.LMARGIN)
        kw.setdefault("new_y", YPos.NEXT)
        return self.multi_cell(w, h, txt, **kw)


def head(pdf, t):
    pdf.ln(0.9)
    pdf.set_font("sans", "B", 10.1)
    pdf.mc(0, 5.1, t)
    pdf.ln(0.4)
    pdf.set_font("serif", "", BODY)


def para(pdf, t, gap=1.45):
    pdf.set_font("serif", "", BODY)
    pdf.mc(0, LH, t, align="J", markdown=True)
    pdf.ln(gap)


def bullet(pdf, t):
    x = pdf.get_x()
    pdf.set_x(x + 5)
    pdf.set_font("serif", "", BODY)
    pdf.mc(pdf.w - pdf.l_margin - pdf.r_margin - 5, LH, t, align="J",
           markdown=True)
    pdf.set_x(x)
    pdf.ln(1.0)


def main():
    pdf = Doc(format="A4")
    pdf.add_font("serif", "", _find("DejaVuSerif.ttf"))
    pdf.add_font("serif", "B", _find("DejaVuSerif-Bold.ttf"))
    pdf.add_font("serif", "I", _find("DejaVuSerif-Italic.ttf"))
    pdf.add_font("sans", "B", _find("DejaVuSans-Bold.ttf"))
    pdf.set_margins(20, 16, 20)
    pdf.set_auto_page_break(True, margin=9)
    pdf.add_page()

    pdf.set_font("sans", "B", 15.5)
    pdf.mc(0, 7.6, TITLE, align="C")
    pdf.ln(0.8)
    pdf.set_font("serif", "I", 9.9)
    pdf.mc(0, 4.9, SUB, align="C")
    pdf.ln(1.8)
    pdf.set_font("serif", "", 10.0)
    pdf.mc(0, 4.9, COURSE, align="C")
    pdf.mc(0, 4.9, AUTHOR, align="C")
    pdf.ln(2.0)
    pdf.set_draw_color(140)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.set_draw_color(0)
    pdf.ln(2.6)

    head(pdf, "The question")
    para(pdf,
         "When a multiagent system underperforms because its agents act on "
         "local information, two very different things can be wrong. An agent "
         "may be unable to tell **what it caused** - its contribution buried in "
         "a shared reward, the credit assignment problem that difference "
         "rewards are built to fix. Or it may be unable to tell **what is "
         "happening** - the information needed to act well simply absent from "
         "its observations, which no amount of reward shaping can recover. The "
         "two are easy to conflate because they produce the same symptom. I "
         "propose to study a domain where they can be varied independently and "
         "measured against each other.")

    head(pdf, "Domain")
    para(pdf,
         "Robots are assigned tasks and some fail mid-execution; the team must "
         "detect the failure and reallocate the task over a channel with "
         "correlated, bursty packet loss. Each agent sees only the heartbeats "
         "it has received, and decides independently whether to take over a "
         "teammate's task. System utility is tasks completed, less duplicated "
         "effort, less detection latency. The environment will be an "
         "event-driven simulator with a Gilbert-Elliott burst channel and "
         "teams of 6 to 32 agents, run for 10^2-10^3 trials per configuration.")

    head(pdf, "The coordination failure to be explained")
    para(pdf,
         "Silence from a teammate is ambiguous: it may mean the teammate has "
         "died, or that the channel is in an outage. Taking over a dead "
         "teammate's task raises system utility; taking over a live one lowers "
         "it, by putting two robots on one task. During a burst, every agent "
         "independently concludes that every teammate has died, and the team "
         "produces a storm of duplicate takeovers that grows with team size - a "
         "scaling problem of the kind this course treats. The question I want "
         "to answer is which of the two causes above produces it.")

    head(pdf, "Experiment: a 2x2 that separates the causes")
    para(pdf,
         "I will train agents to learn a takeover policy from local "
         "observations, varying two factors independently:")
    bullet(pdf,
           "**Reward signal** - global team utility G, versus a difference "
           "reward D_i = G(z) - G(z_-i + c_i), the counterfactual \"what would "
           "system utility have been had I not taken over.\" D_i is exactly "
           "computable in this domain, because a run can be replayed with one "
           "agent's action replaced by a default.")
    bullet(pdf,
           "**Observation** - the agent's own received heartbeats only, versus "
           "those plus a measure of how much of the __rest__ of the team it can "
           "currently hear.")
    para(pdf,
         "If the difference reward alone removes the storm, the problem was "
         "credit assignment. If the added observation alone removes it, reward "
         "shaping cannot substitute for a missing signal. If both are needed, "
         "the effects are separable, and the 2x2 says by how much.")

    head(pdf, "Hypothesis")
    para(pdf,
         "I expect observability to dominate, for a structural reason: during "
         "an outage the agent's observations genuinely do not distinguish a "
         "dead teammate from an unheard one, so the information required to "
         "choose correctly is absent rather than merely obscured. A difference "
         "reward can sharpen an agent's estimate of its own contribution, but "
         "it cannot add a distinction the observation does not carry. If that "
         "holds, the result is a clean negative for the difference-reward "
         "toolkit wherever the limiting factor is what an agent can see rather "
         "than what it can be credited for - worth diagnosing before choosing "
         "a method. If credit assignment instead carries most of the effect, "
         "that is the more surprising outcome and the more interesting one.")

    head(pdf, "Evaluation and baselines")
    para(pdf,
         "I will measure false takeovers, detection latency and wasted travel "
         "against team size from 6 to 32 agents, with learning curves for each "
         "cell of the 2x2. Baselines: a hand-designed rule gating its failure "
         "belief on team audibility, the phi-accrual detector from distributed "
         "systems, and no recovery. I also intend to implement a single-item "
         "auction among audible agents, testing whether explicit negotiation "
         "beats learned implicit coordination when the channel is the "
         "bottleneck.")

    out = os.path.join(HERE, "rob538_project_proposal.pdf")
    pdf.output(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
