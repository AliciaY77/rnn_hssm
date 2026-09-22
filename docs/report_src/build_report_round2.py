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
    s.append(P("Track A refits the constant-bound data (bound 1.5 on dv, twenty networks, 38 640 to 39 988 trials per gain) with HSSM's pretrained OU likelihood and every defect removed. RTs are stretched by k = 10 and offset by 0.3 s. Non-decision time is fixed at the offset plus the dataset's minimum RT, less one native millisecond so that the fastest trial keeps a positive decision time. The lapse mixture is off. Drift is regressed on signed coherence and choice is coded as the network's choice, not as accuracy. Priors are uniform over the likelihood's box. Three pooled fits per stretch, 120 per-network fits, three hierarchical fits, six recovery fits and two appendix fits were run; every one of them has R-hat at most 1.01, bulk effective sample size above 400 and zero divergent transitions."))
    s.append(tab([["gain", "g, stretched [94% HDI]", "g native (per s)", "a native", "drift native at coherence 0.15", "posterior mass at the +1 ceiling", "networks with g > 0"],
                  ["0.8", "+0.998 [+0.994, +1.000]", "+10.0", "0.38", "3.8", "1.00", "20 of 20"],
                  ["1.0", "+0.996 [+0.988, +1.000]", "+10.0", "0.32", "4.1", "0.99", "20 of 20"],
                  ["1.2", "+0.921 [+0.872, +0.968]", "+9.2", "0.29", "4.1", "0.01", "20 of 20"]],
                 [0.45*inch, 1.55*inch, 0.85*inch, 0.6*inch, 1.1*inch, 1.05*inch, 0.85*inch],
                 "Table 1. Pooled HSSM fits at k = 10 (Track A, output/track_a/headline_pooled.csv). Positive g is leaky. Native units follow g·k, a/√k and v·√k. The last column counts the 120 per-network fits at the same stretch; 118 of them have a 94 percent HDI that excludes zero on the leaky side."))
    s.append(P("The sign is unambiguous and the magnitude is not. At gains 0.8 and 1.0 the posterior for g sits on the +1 edge of the likelihood's range at every stretch from 8 to 24, so the native value simply equals k and is a floor rather than an estimate. Over the same three-fold range of k the native bound and native drift are stable to about ten percent, which is what the rescaling rule requires of identified parameters. At gain 1.2 the posterior comes off the edge from k = 10 on and its native value rises from +9.2 to +14.7 per s between k = 10 and 24, so it is also not pinned down. The exact-simulator search of the first session had found the same thing in a cruder way. The likelihood is monotone in g up to the box edge because the constant-bound RT distributions have a 6 ms minimum and a flat tail out to the 750 ms horizon, and strong mean reversion is the only way a one-boundary OU produces both."))
    s.append(P("The fitted model reproduces what it was fitted to. Simulating 20 000 trials per coherence from 200 posterior draws and applying the same horizon gives median absolute RT quantile errors of 6.5, 1.6 and 0.8 ms for correct trials at gains 0.8, 1.0 and 1.2, accuracy within 0.02 at every coherence, and omission rates within 1.6 percentage points (Figure 1, top). The posterior predictive used an Euler integrator of the same process rather than the ssm-simulators C code, because that code corrupts its heap when called repeatedly with changing parameters; on a small cross-check the two engines agree within Monte Carlo noise."))
    s.append(P("The decisive check drives the fitted model with the networks' own evidence streams and asks for the psychophysical kernel (Figure 1, bottom). With the fitted bound, which in native units is 0.29 to 0.38 and is reached within about 100 ms on more than 99.9 percent of trials, the model commits before most of the evidence arrives and shows primacy at every gain, with slopes of -0.041, -0.057 and -0.064 against the networks' +0.039, +0.002 and -0.026. What little ordering there is follows the falling bound, not the leak. With the bound removed, the fitted leak alone gives recency at every gain, slopes +0.068, +0.067 and +0.066, because the fitted g is the same large positive number at all three gains. Neither variant reproduces any single gain, and neither reproduces the ordering."))
    s.append(P("Two further checks bound the interpretation. First, the pipeline is not blind to instability: when data are simulated from the fitted parameters with g replaced by the theoretical values +0.43, -0.16 and -0.62 (stretched), the refit recovers the sign in all six datasets and at gain 1.2 the 94 percent HDI excludes zero on the unstable side. Magnitudes shrink toward zero by roughly a third, so only one of six HDIs covers the truth and the numbers in Table 1 should be read as signs. Second, the original sign flip is reproduced as a confound: the same network at the same gain fitted on the Weibull collapsing-bound readout gives g = -0.22 [-0.28, -0.15] at k = 10, the negative sign of the original three fits."))
    s.append(fig(ROOT / "output/report/fig_track_a.png", 6.5, "Figure 1. Track A. Top: RT distributions of the twenty networks (filled) and of the HSSM-fitted OU at its pooled posterior mean (line), all coherences, constant bound 1.5, at gains 0.8, 1.0 and 1.2. Bottom left: psychophysical kernels of the networks (solid) and of the fitted OU driven by the same evidence streams with its fitted bound (dashed). Bottom middle: the same with the bound removed. Bottom right: kernel slope against gain for the networks and both variants. Positive g is leaky."))
    s.append(P("4. The result: the same accumulator fitted to choice given the evidence stream", H))
    s.append(P("Track B fits the same OU process, in the network's own time with dt = 1 ms, to each trial's deadline choice given that trial's evidence sequence, in the manner of Brunton, Botvinick and Brody (2013). The process is a<sub>t+1</sub> = a<sub>t</sub> + (v e<sub>t</sub> - g a<sub>t</sub>) dt + √dt ξ<sub>t</sub> with a<sub>0</sub> = a<sub>bias</sub>, an optional sticky bound at ±B, and choice given by the sign of a<sub>T</sub>. The noise scale is fixed at one, so v and B are in noise units. Without the bound a<sub>T</sub> is Gaussian given the evidence, so the likelihood of each choice is a normal probability and fits take seconds. This route was fitted by maximum likelihood and by NUTS with weak priors for every network and gain, and pooled over the twenty networks. The analytic likelihood was checked against Monte Carlo on random trials to within 0.005, and the fitted code reproduces an independent implementation of the same fit to within 0.003 in g."))
    s.append(tab([["gain", "median g across networks (per s)", "interquartile range", "intervals excluding zero on the predicted side", "pooled g [94% HDI]", "landscape coefficient"],
                  ["0.8", "+4.13", "+3.02 to +4.63", "20 of 20 positive", "+3.95 [+3.83, +4.06]", "about +4.3"],
                  ["1.0", "+0.32", "-0.34 to +0.79", "10 positive, 5 negative", "+0.23 [+0.15, +0.32]", "about -1.6"],
                  ["1.2", "-2.63", "-3.56 to -2.03", "20 of 20 negative", "-2.73 [-2.83, -2.62]", "about -6.2"]],
                 [0.45*inch, 1.2*inch, 1.1*inch, 1.45*inch, 1.35*inch, 0.9*inch],
                 "Table 2. Track B, unbounded evidence-conditioned fit (output/track_b/analytic_per_network.csv, analytic_pooled.csv). Positive g is leaky. Intervals are 94 percent HDIs from NUTS; the maximum-likelihood estimates agree with the posterior means within 0.1 standard error. The landscape coefficient is the drift slope at the undecided state from three networks, sign-converted."))
    s.append(P("The sign pattern is the one the theory predicts, in every network at the two extreme gains. Twenty of twenty networks are leaky at gain 0.8 and twenty of twenty are unstable at gain 1.2, with intervals that exclude zero in every case. At gain 1.0 the networks straddle zero, with a pooled estimate slightly positive, which is consistent with the kernel slope crossing zero at gain 1.016 rather than exactly at 1.0. The per-network g follows each network's own kernel slope almost linearly (Figure 2, right). The pooled values match the first session's simulation-matching estimates (+3.98, +0.24 and -2.50) and the ordering and zero crossing of the landscape coefficients, with a smaller magnitude at gain 1.2 because the network saturates into wells at |dv| near 3 that a linear process does not have."))
    s.append(P("The fitted model reproduces the psychophysical kernel it was not fitted to. Simulating choices from the fitted parameters on the real evidence streams gives pooled kernel slopes of +0.041, +0.002 and -0.031 against the networks' +0.039, +0.002 and -0.026, and the per-network slopes fall on the identity line (Figure 3). The psychometric curves overlap. The one visible shortfall is at gain 1.2, where the network's kernel drops faster over the first 200 ms than a linear unstable process can, the signature of early commitment into a well. Recovery on real evidence streams with either the fitted parameters or the theoretical values planted returns the right sign in every case and covers the truth in most."))
    s.append(fig(ROOT / "output/report/fig_track_b_g.png", 6.5, "Figure 2. Track B. Left: fitted OU leak g per network with 94 percent intervals against gain, pooled fits as coloured bars, landscape coefficients as gray bars, and the kernel zero crossing of Figure 1F as the dotted line. Right: the same estimates against each network's own kernel slope. Positive g is leaky."))
    s.append(fig(ROOT / "output/report/fig_track_b_kernel.png", 6.5, "Figure 3. Track B. Left: psychophysical kernels of the networks (solid) and of the fitted OU driven by the same evidence streams (dashed), averaged over the twenty networks. Right: kernel slope of the fitted model against that of the network, one point per network and gain."))
    s.append(P("ROUTE2_PLACEHOLDER"))
    s.append(P("5. Leak versus bound", H))
    s.append(P("TODO"))
    s.append(P("6. What remains", H))
    s.append(P("The conclusion is that the leaky-to-unstable transition is in the networks' behaviour and an accumulator model recovers it, but only when the fit is conditioned on each trial's evidence. Fitting RT and choice distributions with a threshold readout, however well done, returns a leaky process at every gain, because under a low bound the crossing is governed by fast readout fluctuations rather than the slow choice mode. WHAT_REMAINS_PLACEHOLDER"))

    def footer(c, d):
        c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.grey); c.drawRightString(letter[0] - 0.9 * inch, 0.55 * inch, str(d.page)); c.restoreState()
    doc = SimpleDocTemplate(str(OUTPDF), pagesize=letter, leftMargin=0.9*inch, rightMargin=0.9*inch, topMargin=0.85*inch, bottomMargin=0.85*inch, title="OU fits to gain-modulated RNNs, round 2", author="Ivan Grahek")
    doc.build(s, onFirstPage=footer, onLaterPages=footer); print("built", OUTPDF)

if __name__ == "__main__":
    build()
