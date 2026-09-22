"""Build docs/OU_fits_to_gain_RNNs_round2.pdf (round-2 report for the advisor). Run with a python that has reportlab + PIL.
Numbers come from output/track_a and output/track_b tables; anything in the RESULTS dict is filled from those files by fill_numbers().
"""
import pathlib, datetime, json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from PIL import Image as PILImage

FD = "/opt/homebrew/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf"
for n, f in [("DV", "DejaVuSans"), ("DV-B", "DejaVuSans-Bold"), ("DV-I", "DejaVuSans-Oblique"), ("DV-BI", "DejaVuSans-BoldOblique"), ("DVM", "DejaVuSansMono")]:
    pdfmetrics.registerFont(TTFont(n, f"{FD}/{f}.ttf"))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")
ROOT = pathlib.Path(__file__).resolve().parents[2]; OUTPDF = ROOT / "docs" / "OU_fits_to_gain_RNNs_round2.pdf"
B = ParagraphStyle("B", fontName="DV", fontSize=9.8, leading=12.8, spaceAfter=6)
T = ParagraphStyle("T", parent=B, fontName="DV-B", fontSize=14, leading=18, spaceAfter=2)
H = ParagraphStyle("H", parent=B, fontName="DV-B", fontSize=11, spaceBefore=10, spaceAfter=4, keepWithNext=True)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=10)
SM = ParagraphStyle("SM", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
HDR = ParagraphStyle("HDR", parent=B, fontName="DV-B", fontSize=8.5, leading=10, spaceAfter=0)
TS = TableStyle([("FONT", (0, 0), (-1, -1), "DV", 8.5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)])
def P(t, st=B): return Paragraph(t, st)
def tab(rows, colw, caption):
    rows = [[Paragraph(str(c), HDR) for c in rows[0]]] + [[Paragraph(str(c), ParagraphStyle("c", parent=B, fontSize=8.5, leading=10, spaceAfter=0)) for c in r] for r in rows[1:]]
    t = Table(rows, colWidths=colw, hAlign="LEFT"); t.setStyle(TS); return KeepTogether([t, Paragraph(caption, CAP)])
def fig(path, width_in, caption):
    im = PILImage.open(path); w, h = im.size; wi = width_in * inch
    return KeepTogether([Image(str(path), width=wi, height=wi * h / w), Paragraph(caption, CAP)])

# ---------------------------------------------------------------- numbers (filled from the track outputs)
R = {}
def fill_numbers():
    pass  # TODO: read output/track_a/*.csv and output/track_b/*.csv into R

def build():
    fill_numbers(); s = []
    s.append(P("OU accumulator fits to the gain-modulated RNNs, round 2: a control that cannot see the transition and a fit that does", T))
    s.append(P(f"Ivan Grahek, {datetime.date.today():%d %B %Y}. Code and results: branches <font face='DVM'>track-a-hssm-ou-corrected</font>, <font face='DVM'>track-b-evidence-conditioned-fit</font> and <font face='DVM'>round2-report</font> of <font face='DVM'>rnn_hssm</font> (issues #3 and #4). Software: HSSM 0.2.4, ssm-simulators 0.8.3, gain-controller-rnn at 1ca084b.", SM))
    s.append(Spacer(1, 6))

    s.append(P("1. The question", H))
    s.append(P("A scalar gain on the transfer slope of the trained NXX1 units moves the network between integration regimes. At gain 0.8 the choice axis is a leaky integrator, at 1.0 it is close to a line attractor, and at 1.2 the undecided state repels and the network falls into one of two committed wells. The landscape analysis of 21 September measures the Langevin drift of the choice variable dv on noise-only trials: its slope at the undecided state is about -4.3, +1.6 and +6.2 per second at gains 0.8, 1.0 and 1.2 (three networks pooled; the twenty-network curvature crosses zero at gain 0.94 with s.d. 0.07). Behaviourally, the psychophysical kernel of Figure 1F sweeps from recency to primacy across gain, with its slope crossing zero at gain 1.016 over twenty networks."))
    s.append(P("The accumulator-model version of this claim is an Ornstein-Uhlenbeck (OU) process, dx = (v - g x) dt + dW, with the leak g going from positive through zero to negative as gain rises. Throughout this report g follows the convention of ssm-simulators and HSSM: positive g is leaky and produces recency, negative g is unstable and produces primacy. The landscape drift slope has the opposite sign. Stretching RTs by a factor k describes the same process with g/k, a√k and v/√k; both conventions were re-verified by simulation for this report (median first-passage times 0.59, 0.78 and 1.10 s at g = -1, 0, +1 with v = 0 and a = 1, and stretched RT quantiles within 4 percent of native ones at k = 10)."))
    s.append(P("Round 2 answers two questions. First, once every defect of the original HSSM fits is removed, does a fit to RT and choice distributions see the transition? Second, does a likelihood-based fit of the same accumulator, conditioned on each trial's evidence stream, see it with per-network uncertainty? The first is the control for the second."))
    s.append(P("2. The original fits and why they failed", H))
    s.append(P("The original three fits (one network, seed 42, the pretrained HSSM OU likelihood) diverged on nearly every draw, with g at the unstable edge of the likelihood's range, drift at its upper edge and the fitted non-decision time absorbing the 0.3 s offset that had been added to the RTs. Four separate problems were identified in the first session and each is documented in HANDOFF.md on branch ou-round2-base."))
    s.append(P("The data were read out through a Weibull collapsing bound, which a constant-bound OU can only imitate with unstable dynamics. RT and choice were therefore regenerated from the same dv traces with a constant bound of 1.5, at the cost of 3.4, 0.3 and 0.0 percent omitted trials at gains 0.8, 1.0 and 1.2. The networks decide in 75 to 145 ms with a leak of several per second, outside the pretrained likelihood's box of |g| at most 1 per second and decision times of order one second; a time stretch of 8 to 10 is needed and none was used. Non-decision time was fitted, and it absorbed the offset. The 750 ms horizon truncates the data, and the recovery study showed that fitting truncated data with an untruncated likelihood pushes g toward unstable."))
    s.append(P("The first session also showed that these are not the whole story. An exact-simulator search with no parameter box reproduces the constant-bound RT and choice distributions at gains 1.0 and 1.2 almost perfectly, at a ≈ 0.3, drift ≈ 4 and g between +4 and +9 per second: leaky at every gain and most leaky at 1.2. A low bound is crossed within 50 to 100 ms by the fast mean-reverting fluctuations of the readout, and the OU's single leak term reports their decay rather than the slow choice mode the landscape measures. Track A tests whether that conclusion survives proper posteriors, twenty networks and posterior-predictive checks."))
    s.append(P("3. The control: corrected HSSM fits to RT and choice", H))
    s.append(P("TODO"))
    s.append(P("4. The result: the same accumulator fitted to choice given the evidence stream", H))
    s.append(P("TODO"))
    s.append(P("5. Leak versus bound", H))
    s.append(P("TODO"))
    s.append(P("6. What remains", H))
    s.append(P("TODO"))

    def footer(c, d):
        c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.grey); c.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(d.page)); c.restoreState()
    doc = SimpleDocTemplate(str(OUTPDF), pagesize=letter, leftMargin=0.9*inch, rightMargin=0.9*inch, topMargin=0.85*inch, bottomMargin=0.85*inch, title="OU fits to gain-modulated RNNs, round 2", author="Ivan Grahek")
    doc.build(s, onFirstPage=footer, onLaterPages=footer); print("built", OUTPDF)

if __name__ == "__main__":
    build()
