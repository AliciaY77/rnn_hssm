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
    s.append(P("TODO"))
    s.append(P("2. The original fits and why they failed", H))
    s.append(P("TODO"))
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
