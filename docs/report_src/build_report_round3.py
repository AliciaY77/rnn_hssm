"""Build docs/OU_fits_to_gain_RNNs_round3.pdf: two-page summary for the advisor. Figures from output/report, numbers from
output/track_a and output/track_b (see build_report_round2.py for the full write-up). Run with a python that has reportlab + PIL."""
import pathlib, datetime
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
ROOT = pathlib.Path(__file__).resolve().parents[2]; OUTPDF = ROOT / "docs" / "OU_fits_to_gain_RNNs_round3.pdf"; FIG = ROOT / "output/report"
B = ParagraphStyle("B", fontName="DV", fontSize=9.2, leading=11.6, spaceAfter=4)
T = ParagraphStyle("T", parent=B, fontName="DV-B", fontSize=13, leading=16, spaceAfter=2)
H = ParagraphStyle("H", parent=B, fontName="DV-B", fontSize=10.5, spaceBefore=6, spaceAfter=2, keepWithNext=True)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.3, leading=10.4, textColor=colors.HexColor("#333333"), spaceAfter=7)
SM = ParagraphStyle("SM", parent=B, fontSize=8.3, leading=10.4, textColor=colors.HexColor("#444444"))
HDR = ParagraphStyle("HDR", parent=B, fontName="DV-B", fontSize=8.5, leading=10, spaceAfter=0)
CELL = ParagraphStyle("CELL", parent=B, fontSize=8.5, leading=10, spaceAfter=0)
TS = TableStyle([("FONT", (0, 0), (-1, -1), "DV", 8.5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)])
def P(t, st=B): return Paragraph(t, st)
def tab(rows, colw, caption):
    rows = [[Paragraph(str(c), HDR) for c in rows[0]]] + [[Paragraph(str(c), CELL) for c in r] for r in rows[1:]]
    t = Table(rows, colWidths=colw, hAlign="LEFT"); t.setStyle(TS); return KeepTogether([t, Paragraph(caption, CAP)])
def fig(path, width_in, caption):
    im = PILImage.open(path); w, h = im.size; wi = width_in * inch
    return KeepTogether([Image(str(path), width=wi, height=wi * h / w), Paragraph(caption, CAP)])

s = []
s.append(P("OU fits to the gain-modulated RNNs: RT-and-choice fits stay leaky at every threshold; the same accumulator fitted to choice given the evidence recovers the transition", T))
s.append(P(f"Ivan Grahek, {datetime.date.today():%d %B %Y}. Round 3 summary; full write-up docs/OU_fits_to_gain_RNNs_round2.pdf, code and RESULTS.md on branch round2-report of rnn_hssm. Prediction: gain 0.8 / 1.0 / 1.2 = leaky / near-perfect / unstable integration, so the OU leak should go positive / zero / negative. Convention throughout: ssm-simulators' update is x += (v − g·x)·dt + noise, so <b>positive g is leaky</b>, g = −λ in Brunton's notation.", SM))
s.append(Spacer(1, 4))

s.append(P("1. Control: HSSM OU fitted to RT and choice, at three thresholds on dv", H))
s.append(P("The networks read out their choice at 750 ms and do not produce RTs. Every RT below is a threshold we place on the decision variable dv; trials that never reach it are dropped (HSSM has no censored likelihood). At a low threshold the crossing is driven by the fast fluctuations of the readout; near the committed attractors (|dv| ≈ 3) the slow choice mode governs it. Fit: pretrained OU likelihood, 20 networks pooled per gain, RTs × 10, non-decision time fixed at offset + minimum RT, no lapse, v ~ 1 + signed coherence, a, z and g shared across coherences, uniform priors over the likelihood's box. All 12 fits converged (R-hat 1.00, one divergence in 48 000 draws)."))
s.append(fig(FIG / "fig_r3_rt_ppc.png", 6.2, "Figure 1. The fitted model reproduces RT and choice at a low threshold but not at a high one for the leaky network. Rows: threshold 1.5, 2.5, 2.94; columns: gain. RT distributions over all coherences, correct up and error down (areas sum to one), network (filled) vs fitted OU at its posterior mean (line). Titles: fitted native g with 94 % HDI. Per-coherence versions: output/track_a/ppc_rt_*.png."))
s.append(fig(FIG / "fig_r3_threshold.png", 6.6, "Figure 2. How the fit changes with the threshold (all four fitted thresholds). Left: fitted g with 94 % HDI; it falls and orders by gain as the theory predicts, but never becomes negative, and gain 0.8 returns to the ceiling at 2.94. Middle: median absolute RT-quantile error (10–90 %, correct trials, per coherence). Right: omissions; the model predicts a fraction of the network's."))
rows = [("1.5", "0.8", "+10.0 [+9.9, +10.0]", "3.4 / 1.8", "6.5", "−0.041"), ("", "1.0", "+10.0 [+9.9, +10.0]", "0.3 / 0.2", "1.6", "−0.057"), ("", "1.2", "+9.2 [+8.7, +9.7]", "0.0 / 0.0", "0.8", "−0.064"),
        ("2.0", "0.8", "+10.0 [+9.9, +10.0]", "13.4 / 6.1", "20.9", "−0.024"), ("", "1.0", "+9.9 [+9.8, +10.0]", "1.7 / 1.2", "3.7", "−0.045"), ("", "1.2", "+4.9 [+4.3, +5.4]", "0.2 / 0.2", "2.4", "−0.057"),
        ("2.5", "0.8", "+6.7 [+6.2, +7.2]", "30.7 / 11.3", "33.7", "−0.011"), ("", "1.0", "+5.8 [+5.3, +6.2]", "10.1 / 2.9", "10.1", "−0.035"), ("", "1.2", "+4.7 [+4.2, +5.2]", "5.6 / 0.6", "3.5", "−0.052"),
        ("2.94", "0.8", "+9.9 [+9.6, +10.0]", "66.9 / 20.7", "37.8", "+0.006"), ("", "1.0", "+2.3 [+1.9, +2.6]", "30.4 / 6.3", "24.1", "−0.025"), ("", "1.2", "+0.8 [+0.5, +1.1]", "14.3 / 1.8", "4.3", "−0.043"),
        ("network", "all", "predicted + / 0 / −", "", "", "+.039 / +.002 / −.026")]
s.append(tab([["threshold", "gain", "g native (per s) [94 % HDI]", "omissions network / model (%)", "RT-quantile error (ms)", "kernel slope, fitted OU"]] + [list(r) for r in rows],
             [0.85*inch, 0.7*inch, 1.5*inch, 1.3*inch, 0.95*inch, 1.6*inch],
             "Table 1. Positive g is leaky and every HDI excludes zero. +10.0 is the likelihood's ceiling at k = 10: at threshold 1.5 g re-pins there for every stretch k = 8 … 24 (native value = k), so there only the sign is identified; 120 per-network fits at 1.5 give g > 0 in 120 of 120. Recovery with the theoretical g planted in simulated data returns g < 0 at gains 1.0 and 1.2 (sign 6 of 6), so the pipeline can see instability. Kernel slope: Figure 3."))
s.append(fig(FIG / "fig_r3_kernel.png", 5.5, "Figure 3. Kernel check: the fitted model (with its fitted bound) driven by the networks' own evidence streams. Top: kernels, network (solid) vs model (dashed). Bottom: kernel slope against gain. A higher threshold moves the model toward the network and brings out the ordering across gain, but the model still commits early and stays shifted toward primacy by 0.02–0.03."))
s.append(tab([["fit", "kernel slope, gain 0.8 / 1.0 / 1.2", "mean |slope error|", "bin-by-bin RMS error", "span 0.8 − 1.2"],
              ["network", "+0.039 / +0.002 / −0.026", "", "", "0.065"],
              ["HSSM, threshold 1.5", "−0.041 / −0.057 / −0.064", "0.059", "0.151", "0.023"],
              ["HSSM, threshold 2.0", "−0.024 / −0.045 / −0.057", "0.047", "0.115", "0.034"],
              ["HSSM, threshold 2.5", "−0.011 / −0.035 / −0.052", "0.038", "0.091", "0.040"],
              ["HSSM, threshold 2.94", "+0.006 / −0.025 / −0.043", "0.025", "0.065", "0.049"],
              ["Track B, choice given evidence (Section 2)", "+0.041 / +0.002 / −0.031", "0.002", "0.011", "0.072"]],
             [2.4*inch, 1.75*inch, 0.95*inch, 1.0*inch, 0.85*inch],
             "Table 2. Kernel error against the network. It falls steadily with the threshold, but even at 2.94 it is about ten times the evidence-conditioned fit's by slope and six times by bin, and the sign is still wrong at gain 1.0 — while at that threshold the RT fit has failed for gains 0.8 and 1.0 (Table 1)."))

s.append(P("2. Result: the same OU fitted to each trial's choice given its evidence stream (Brunton-style, no RTs)", H))
s.append(P("a<sub>t+1</sub> = a<sub>t</sub> + (v e<sub>t</sub> − g a<sub>t</sub>) dt + √dt ξ<sub>t</sub>, dt = 1 ms, choice = sign(a<sub>750</sub>); e<sub>t</sub> is the evidence the network received. Without a bound a<sub>T</sub> is Gaussian given the evidence, so each choice has an exact likelihood; fitted by maximum likelihood and NUTS per network and gain (60 fits) and pooled."))
s.append(fig(FIG / "fig_track_b_g.png", 4.3, "Figure 4. Left: fitted g per network with 94 % intervals (filled), pooled fit (bars), bounded fit with the deadline-commitment term (open squares), landscape drift coefficient (gray), kernel zero crossing of Fig 1F at gain 1.016 (dotted). Right: g against each network's own kernel slope, r = 0.996."))
s.append(tab([["gain", "median g across networks (per s) [IQR]", "intervals excluding 0 on the predicted side", "pooled g [94 % HDI]", "landscape coefficient (approx.)"],
              ["0.8", "+4.13 [+3.02, +4.63]", "20 of 20", "+3.95 [+3.83, +4.06]", "+4.3"],
              ["1.0", "+0.32 [−0.34, +0.79]", "10 positive, 5 negative", "+0.23 [+0.15, +0.32]", "−1.6"],
              ["1.2", "−2.63 [−3.56, −2.03]", "20 of 20", "−2.73 [−2.83, −2.62]", "−6.2"]],
             [0.5*inch, 2.0*inch, 1.7*inch, 1.6*inch, 1.2*inch],
             "Table 3. Leaky → near zero → unstable, in every network at the extreme gains. Zero crossing of g against gain: 1.016 (Fig 1F kernel: 1.016). Recovery on real evidence with fitted or theoretical g planted: 60 of 60 signs, 56 of 60 intervals cover the truth."))
s.append(fig(FIG / "fig_track_b_summary.png", 5.6, "Figure 5. Posterior predictive. Left: the fit reproduces the psychophysical kernel it was not fitted to (slopes +0.041 / +0.002 / −0.031 vs network +0.039 / +0.002 / −0.026). Middle: P(choice) by signed coherence, within 0.015 everywhere. Right: per-trial predicted probability against observed choice fraction."))

s.append(P("3. What we learned", H))
s.append(P("The RT here is a readout we impose on dv, not something the network does, and what the RT-and-choice fit measures depends on where we put it. At a low threshold it is crossed by the fast readout fluctuations: the OU fits those RTs almost perfectly by putting its one leak term on their decay, leaky and at the ceiling at every gain, with the wrong kernel. Raising the threshold lets the slow dynamics show through: g falls and orders by gain and the kernel ordering appears. But g never becomes negative, and at the thresholds where the ordering appears the one-timescale linear OU with a constant bound no longer fits the leaky network's RTs or omissions."))
s.append(P("Conditioning the same accumulator on each trial's evidence, and fitting only the choice the network actually makes, gives the predicted sign pattern per network with intervals and reproduces the kernel an order of magnitude better. A joint RT-and-choice model of these networks would need at least a second, fast timescale and a likelihood that handles non-crossers; the Brody lab's Fokker–Planck grid likelihood is the natural route for both. Caveats for the result: the magnitude at gain 0.8 depends on how a sticky bound is constrained (+4.1 free, +2.5 with the deadline commitment fraction; the sign does not change), and the linear OU under-reports the repulsion at gain 1.2 (−2.6 vs −6.2) because it cannot saturate into wells."))

def footer(c, d):
    c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.grey); c.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, str(d.page)); c.restoreState()
doc = SimpleDocTemplate(str(OUTPDF), pagesize=letter, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.65*inch, bottomMargin=0.6*inch, title="OU fits to gain-modulated RNNs, round 3 summary", author="Ivan Grahek")
doc.build(s, onFirstPage=footer, onLaterPages=footer); print("built", OUTPDF)
