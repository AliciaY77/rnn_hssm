import pathlib, datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
from PIL import Image as PILImage
FD = "/opt/homebrew/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf"
for n, f in [("DV", "DejaVuSans"), ("DV-B", "DejaVuSans-Bold"), ("DV-I", "DejaVuSans-Oblique"), ("DV-BI", "DejaVuSans-BoldOblique"), ("DVM", "DejaVuSansMono")]:
    pdfmetrics.registerFont(TTFont(n, f"{FD}/{f}.ttf"))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")
ROOT = pathlib.Path("/Users/igrahek/Library/CloudStorage/Dropbox-Brown/Ivan Grahek/Ivan/Studies/rnn_hssm"); OUTPDF = ROOT / "docs" / "OU_fits_to_gain_RNNs_2026-09-22.pdf"
B = ParagraphStyle("B", fontName="DV", fontSize=9.8, leading=12.8, spaceAfter=6)
T = ParagraphStyle("T", parent=B, fontName="DV-B", fontSize=14, leading=18, spaceAfter=2)
H = ParagraphStyle("H", parent=B, fontName="DV-B", fontSize=11, spaceBefore=10, spaceAfter=4)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=10)
SM = ParagraphStyle("SM", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"))
HDR = ParagraphStyle("HDR", parent=B, fontName="DV-B", fontSize=8.5, leading=10, spaceAfter=0)
TS = TableStyle([("FONT", (0, 0), (-1, -1), "DV", 8.5), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)])
def P(t, st=B): return Paragraph(t, st)
def tab(rows, colw, caption):
    rows = [[Paragraph(str(c), HDR) for c in rows[0]]] + rows[1:]
    t = Table(rows, colWidths=colw, hAlign="LEFT"); t.setStyle(TS); return KeepTogether([t, Paragraph(caption, CAP)])
def fig(path, width_in, caption):
    im = PILImage.open(path); w, h = im.size; wi = width_in * inch
    return KeepTogether([Image(str(path), width=wi, height=wi * h / w), Paragraph(caption, CAP)])

s = []
s.append(P("OU accumulator fits to the gain-modulated RNNs: what failed and what works", T))
s.append(P(f"Ivan Grahek, {datetime.date.today():%d %B %Y}. Code and results: branches <font face='DVM'>issue-1-ou-param-recovery</font> and <font face='DVM'>kernel-ou-fit</font> of <font face='DVM'>rnn_hssm</font>.", SM))
s.append(Spacer(1, 6))

s.append(P("The theory says gain moves the trained NXX1 network from a leaky integrator (gain 0.8) to near-perfect integration (1.0) to an unstable, attractor-like regime (1.2). We wanted an Ornstein-Uhlenbeck (OU) accumulator fitted to behaviour to show this: the OU leak parameter g should go from positive to zero to negative across gain. The HSSM OU fits did not converge. This memo explains why, and shows that a different fit of the same model recovers the transition."))

s.append(P("1. The original fits", H))
s.append(P("Data: one network (seed 42), gains 0.8, 1.0, 1.2, 2000 trials each. RT and choice come from the network's decision variable dv read out through a Weibull collapsing bound (height ln 19 = 2.94, shape 4, scale 469 ms, horizon 750 ms). Median RTs are 373, 258 and 183 ms. The fits used HSSM's pretrained OU likelihood network (LAN), drift regressed on coherence, a 0.3 s offset added to all RTs, and g, boundary a, start point z and non-decision time t free."))
s.append(tab([["gain", "divergent draws", "R-hat for g", "g", "drift at coherence 0.15", "t (s)"],
              ["0.8", "3997 of 4000", "4.1", "-0.79", "2.0", "0.39"], ["1.0", "3912 of 4000", "5.7", "-0.91", "2.0", "0.33"], ["1.2", "3997 of 4000", "2.0", "-0.91", "2.0", "0.32"]],
             [0.5*inch, 1.2*inch, 0.9*inch, 0.6*inch, 1.6*inch, 0.6*inch],
             "Table 1. The original OU fits. The LAN is valid for g in [-1, 1], drift in [-2, 2], a in [0.3, 3]. g sits on its lower edge and drift on its upper edge at every gain. t is the offset plus 20 to 90 ms."))

s.append(P("2. Why they failed", H))
s.append(P("Sign of g. In ssm-simulators, and therefore in the LAN, the OU update is dx = (v - g x) dt + dW. Positive g is leaky, negative g is unstable. The code's docstring had this reversed. Verified by simulation: median first-passage time is 0.61, 0.79 and 1.14 s for g = -1, 0, +1 at v = 0, a = 1. The fits above were therefore asking for maximal instability, not maximal leak."))
s.append(P("Timescale. In the network's own units the linear coefficient of the choice-axis dynamics at the undecided state is about +4 s<super>-1</super> at gain 0.8, near 0 at 1.0 and about -6 s<super>-1</super> at 1.2 (landscape analysis, RNN_Gain_Mod lab note of 21 Sep), and decisions take 100 to 400 ms. The LAN was trained on |g| at most 1 s<super>-1</super> and on decision times of order a second. Stretching RTs by a factor k rescales the same process as g to g/k, a to a times root k, v to v over root k, so a stretch of 6 to 10 would bring the networks inside the LAN's range. The original fits used no stretch."))
s.append(P("Collapsing bound. A constant-bound OU cannot represent the Weibull collapse. The collapse compresses RTs and forces late choices toward chance; the only way a constant-bound OU can imitate that is with unstable dynamics, which pushes g negative at every gain. To remove this, RT and choice were regenerated from the same dv traces with a constant bound. At bound 1.5 the cost is 3.4 percent omissions at gain 0.8 (8.6 percent at zero coherence); at the original height 2.94, two thirds of gain-0.8 trials never cross."))
s.append(P("Parameter recovery. Simulated OU data (g from -1 to +1, a = 1.0 or 1.5, 1000 or 3700 zero-coherence trials, time stretched by 6) were fitted with the same HSSM model. All 60 fits sampled cleanly. With HSSM defaults the sign of g was recovered in 7 of 16 non-zero cells; with no lapse mixture and no deadline in 12 of 16. Two effects account for this. Dropping trials past a deadline and fitting an untruncated likelihood turns leaky cells into confidently unstable ones (true g = +0.5 recovered as -0.72 with sd 0.02), and the RNN data have a hard 750 ms horizon. At short decision times, |g| T around 0.6, the posterior for g spans most of the range whatever the truth, even with 3700 trials."))
s.append(P("The deeper problem. With the constant-bound data in hand, the exact OU simulator was searched directly, with no parameter box, for the fit to the RT quantiles, accuracies and omission rates at each coherence. The OU reproduces the gain 1.0 and 1.2 data almost exactly, at a = 0.3, drift 4 and g between +4 and +9 s<super>-1</super>. So the model class is adequate, and this is what a perfectly specified likelihood would return. But g is positive at every gain and largest at gain 1.2, the opposite of the prediction. Raising the constant bound moves g toward zero, and the single-timescale OU stops fitting (Table 2). The reading is that a low bound is crossed within 50 to 100 ms by the fast, mean-reverting fluctuations of the readout, and the OU's one leak term reports their decay, which grows with gain. The slow choice mode only governs crossings near the attractors at |dv| of about 3, where a one-dimensional OU no longer describes the RT distribution and gain 0.8 loses most trials to omissions."))
s.append(tab([["constant bound on dv", "omissions", "fitted g (1/s)", "fit loss"], ["1.5", "0.0%", "+9.3", "0.14"], ["2.0", "0.2%", "+8.6", "0.62"], ["2.5", "5.6%", "+5.2", "2.2"], ["2.94", "14%", "+0.5", "10.3"]],
             [1.6*inch, 1.0*inch, 1.2*inch, 0.9*inch], "Table 2. Gain 1.2, unconstrained OU fitted to RT and choice distributions, 20 networks pooled. A loss below about 0.5 is a fit indistinguishable from the data by eye."))
s.append(P("Conclusion of this part: RT and choice distributions under a threshold readout do not carry the information that separates leak from instability in these networks. That information is when within the trial the decisive evidence arrived, and a fit that sees only each trial's coherence cannot use it."))

s.append(P("3. A fit that works", H))
s.append(P("Each trial's stimulus is a coherence plus independent noise at every millisecond, and we generated it, so the evidence sequence of every trial is known. A leaky accumulator's choice depends mainly on late evidence, an unstable one's on early evidence. The psychophysical kernel, a logistic regression of choice on the mean evidence in each of eight time bins, measures this from behaviour, and Figure 1F of the manuscript shows it sweeping from recency to primacy across gain."))
s.append(P("The same OU model, a<sub>t+1</sub> = a<sub>t</sub> + (v e<sub>t</sub> - g a<sub>t</sub>) dt + root(dt) noise, with a sticky bound B and choice = sign(a<sub>T</sub>), was driven by the networks' own evidence streams e<sub>t</sub>, and v, g, B were chosen so that its choices on those trials reproduce the network's kernel and psychometric curve. Choice is the network's end-of-trial readout, as in Figure 1F. Twenty networks, 2000 trials each, one fit per gain with all networks pooled and one fit per network."))
s.append(tab([["gain", "g (1/s)", "v", "B", "kernel slope: network, OU", "accuracy: network, OU", "landscape coefficient"],
              ["0.8", "+3.98", "46.5", "2.6", "+0.039, +0.041", "0.866, 0.865", "about +4.3"], ["1.0", "+0.24", "40.6", "2.1", "+0.002, +0.002", "0.887, 0.885", "about -1.6"], ["1.2", "-2.50", "40.5", "4.9", "-0.026, -0.028", "0.869, 0.871", "about -6.2"]],
             [0.55*inch, 0.7*inch, 0.5*inch, 0.45*inch, 1.55*inch, 1.4*inch, 1.2*inch],
             "Table 3. Pooled evidence-conditioned OU fits, 40,000 trials per gain. Last column: the landscape analysis's drift coefficient at the undecided state in the same sign convention. The OU is linear and the network saturates into wells at |dv| of about 3, so the fitted magnitude at gain 1.2 is smaller. v does not change with gain, consistent with the landscape finding that gain does not change the evidence tilt."))
s.append(fig(ROOT / "output/kernel_fit/ou_kernel_fits.png", 6.3, "Figure 1. Left: kernels of the networks (solid) and of the fitted OU driven by the same evidence (dashed). Middle: psychometric curves. Right: kernel slope as a function of g with v and B at their fitted values; the mapping is monotonic and the same at all gains, so the kernel is a direct readout of g."))
s.append(fig(ROOT / "output/kernel_fit/ou_kernel_fits_per_network.png", 4.6, "Figure 2. One fit per network. Left: g by gain, bars are mean and s.e.m. Right: g against each network's own kernel slope. All 20 networks are leaky at gain 0.8; 15 of 20 are unstable at gain 1.2, median g = -1.8."))
s.append(P("One caveat. Three of the 60 single-network fits landed on g of about +12 with a low sticky bound. Strong leak with very early commitment also gives a primacy kernel; this is the bound-versus-leak trade-off known from the Brunton model. It does not occur in the pooled fits, and it can be broken with the fraction of trials whose dv reaches the bound before the deadline, or with the RTs."))

s.append(P("4. Next steps", H))
s.append(P("The transition is recoverable from behaviour with an accumulator model when the fit is conditioned on each trial's evidence, and not when it is fitted to RT and choice distributions. The HSSM problems (sign convention, no time stretch, collapsing bound, free non-decision time) are all fixable, but fixing them would give converged fits that still say leaky at every gain. For a parametric confirmation the next step is a likelihood-based Brunton fit, choice given evidence stream with leak, bound and sensory noise free, with the bound-hit fraction added to break the degeneracy."))
s.append(P("Runs: HSSM 0.3 (original fits) and 0.2.4 (recovery, Oscar jobs 6604819, 6604891); ssm-simulators 0.8.3 (Oscar jobs 6606075, 6606166, 6606270, 6606439); gain-controller-rnn at 1ca084b.", SM))

def footer(c, d):
    c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.grey); c.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(d.page)); c.restoreState()
doc = SimpleDocTemplate(str(OUTPDF), pagesize=letter, leftMargin=0.9*inch, rightMargin=0.9*inch, topMargin=0.85*inch, bottomMargin=0.85*inch, title="OU fits to gain-modulated RNNs", author="Ivan Grahek")
doc.build(s, onFirstPage=footer, onLaterPages=footer); print("built", OUTPDF)
