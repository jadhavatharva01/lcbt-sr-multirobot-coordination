#!/usr/bin/env python3
"""Build paper/declaration_of_ai_use.pdf (no LaTeX toolchain here; fpdf2 + DejaVu)."""
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


PAPER = ("A Behavior-Tree Framework for Channel-Aware Failure Recovery in "
         "Decentralized Multi-Robot Task Allocation under Bursty Communication")
AUTHOR = "Atharva Jadhav"

BODY = 9.9
LH = 4.75


class Doc(FPDF):
    def mc(self, w, h, txt, **kw):
        kw.setdefault("new_x", XPos.LMARGIN)
        kw.setdefault("new_y", YPos.NEXT)
        return self.multi_cell(w, h, txt, **kw)


def item(pdf, text):
    x = pdf.get_x()
    pdf.set_x(x + 5)
    pdf.set_font("serif", "", BODY)
    pdf.mc(pdf.w - pdf.l_margin - pdf.r_margin - 5, LH, text, align="J",
           markdown=True)
    pdf.set_x(x)
    pdf.ln(1.9)


def main():
    pdf = Doc(format="A4")
    pdf.add_font("serif", "", _find("DejaVuSerif.ttf"))
    pdf.add_font("serif", "B", _find("DejaVuSerif-Bold.ttf"))
    pdf.add_font("serif", "I", _find("DejaVuSerif-Italic.ttf"))
    pdf.add_font("sans", "B", _find("DejaVuSans-Bold.ttf"))
    pdf.set_margins(21, 19, 21)
    pdf.set_auto_page_break(True, margin=15)
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin

    # ---- heading block -----------------------------------------------------
    pdf.set_font("sans", "B", 17.5)
    pdf.mc(0, 8.6, "Declaration of AI Use", align="C")
    pdf.ln(1.4)
    pdf.set_font("serif", "I", 10.0)
    pdf.mc(0, 5.0, PAPER, align="C")
    pdf.ln(1.8)
    pdf.set_font("serif", "", 10.6)
    pdf.mc(0, 5.0, AUTHOR, align="C")
    pdf.ln(3.0)
    pdf.set_draw_color(140)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.set_draw_color(0)
    pdf.ln(3.8)

    # ---- section A ---------------------------------------------------------
    pdf.set_font("sans", "B", 11.4)
    pdf.mc(0, 5.8, "A.  AI used in the project")
    pdf.ln(1.3)
    item(pdf,
         "**1. Permutation and combination analysis.** I used AI to set up and "
         "simplify the combinatorial parts of my analysis, which were tedious "
         "and error-prone by hand. The clearest example is working out how many "
         "of the candidate robot meeting points were actually usable: for each "
         "pair of adjacent regions there were several hundred possible pairings "
         "of nearby survey points, and I needed to know how many met my distance "
         "and timing conditions. AI let me enumerate and filter these quickly, "
         "which is how I found that only one to three pairs in an entire "
         "instance qualified - a result that changed my design.")
    item(pdf,
         "**2. Checking for overlapping existing methods.** I used AI to search "
         "the literature for methods similar to mine, specifically to establish "
         "whether my approach was already published and therefore not worth "
         "pursuing. This was decisive rather than a formality. It surfaced a "
         "2026 __Computer Communications__ paper covering substantially the same "
         "combination of ideas as my earlier method, which led me to set that "
         "method aside, and later a co-observation planning paper using the same "
         "missed-meeting detection signal I had proposed. In both cases I read "
         "the sources myself to confirm the overlap before acting on it.")
    item(pdf,
         "**3. Coding complex components of my simulation.** While building my "
         "own simulation framework for testing, I used AI mainly for the "
         "HoloOcean integration. HoloOcean is a niche research simulator with a "
         "small user community and limited documentation or worked examples "
         "online, so there was little to work from beyond the API itself. I used "
         "AI for the scenario configuration, the vehicle control and "
         "ocean-current interfaces, and the experiment harness - in one instance "
         "a vehicle type behaved contrary to what the interface suggested, which "
         "only exploratory test code could establish. I also used it for some of "
         "the more intricate routines elsewhere in the code. I specified the "
         "intended behaviour throughout and verified outputs against test cases.")
    item(pdf,
         "**4. Identifying loopholes and testing novelty.** Most importantly, I "
         "used AI as a critic of my own work - to look for weaknesses in my "
         "reasoning and to check whether my solution was genuinely novel or "
         "overlapped with existing work. This changed my conclusions more than "
         "once. In the most significant instance, a modelling assumption "
         "underpinning one of my main results was tested against independent "
         "physics simulation and shown to be false, and I withdrew the "
         "associated claims. In another, a comparison I had reported turned out "
         "to be an artefact of an assumption rather than a real effect, and I "
         "corrected it.")

    # ---- section B ---------------------------------------------------------
    pdf.ln(1.4)
    pdf.set_font("sans", "B", 11.4)
    pdf.mc(0, 5.8, "B.  AI used in the drafting of the document")
    pdf.ln(1.3)
    item(pdf, "**1.** I used **Grammarly** to correct grammar and spelling.")
    item(pdf,
         "**2.** I used **Claude** to refine my draft - improving word choice "
         "and sentence structure so the writing reads in an appropriately "
         "professional and technical register. The substance, structure and "
         "arguments are mine; this use was to improve how they were expressed.")
    item(pdf,
         "**3.** I used **Claude** to generate LaTeX code for the more complex "
         "mathematical expressions in the paper, such as the multi-term variance "
         "expressions and the constrained optimisation statements, which are "
         "laborious to typeset correctly by hand.")

    pdf.ln(2.4)
    pdf.set_font("serif", "", BODY)
    pdf.mc(0, LH,
           "I verified the outputs of all of the above against my own data and "
           "reasoning, and I take full responsibility for the content of this "
           "work.", align="J")

    out = os.path.join(HERE, "declaration_of_ai_use.pdf")
    pdf.output(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
