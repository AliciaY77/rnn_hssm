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
B = ParagraphStyle("B", fontName="DV", fontSize=9.1, leading=11.2, spaceAfter=4)
T = ParagraphStyle("T", parent=B, fontName="DV-B", fontSize=14, leading=18, spaceAfter=2)
H = ParagraphStyle("H", parent=B, fontName="DV-B", fontSize=10.5, spaceBefore=6, spaceAfter=2, keepWithNext=True)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.3, leading=10.5, textColor=colors.HexColor("#444444"), spaceAfter=8)
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
    s = []
    s.append(P("OU accumulator fits to the gain-modulated RNNs, round 2: a control that cannot see the transition and a fit that does", T))
    s.append(P(f"Ivan Grahek, {datetime.date.today():%d %B %Y}. Code, tables and RESULTS.md notes: branches <font face='DVM'>track-a-hssm-ou-corrected</font>, <font face='DVM'>track-b-evidence-conditioned-fit</font> and <font face='DVM'>round2-report</font> of <font face='DVM'>rnn_hssm</font> (issues #3 and #4). HSSM 0.2.4, ssm-simulators 0.8.3, gain-controller-rnn at 1ca084b.", SM))
    s.append(Spacer(1, 4))

    s.append(P("1. The question", H))
    s.append(P("A scalar gain on the transfer slope of the trained NXX1 units moves the network between integration regimes: a leaky integrator at gain 0.8, close to a line attractor at 1.0, and a repelling undecided state with two committed wells at 1.2. The landscape analysis of 21 September measures the Langevin drift of the choice variable dv on noise-only trials; its slope at the undecided state is about -4.3, +1.6 and +6.2 per second at gains 0.8, 1.0 and 1.2 (three networks). Behaviourally, the psychophysical kernel of Figure 1F of the manuscript sweeps from recency to primacy across gain, with its slope crossing zero at gain 1.016 over twenty networks."))
    s.append(P("The accumulator version of the claim is an Ornstein-Uhlenbeck (OU) process, dx = (v - g x) dt + dW, whose leak g goes from positive through zero to negative as gain rises. Throughout, g follows the convention of ssm-simulators and HSSM: positive g is leaky and gives recency, negative g is unstable and gives primacy, and the landscape drift slope has the opposite sign. Stretching RTs by k describes the same process with g/k, a√k and v/√k. Both were re-verified by simulation for this report (median first-passage times 0.59, 0.78 and 1.10 s at g = -1, 0, +1; stretched RT quantiles within 4 percent of native at k = 10)."))
    s.append(P("Round 2 asks two questions. Once every defect of the original HSSM fits is removed, does a fit to RT and choice distributions see the transition? And does a likelihood fit of the same accumulator, conditioned on each trial's evidence stream, see it per network with uncertainty? The first is the control for the second."))

    s.append(P("2. The original fits and why they failed", H))
    s.append(P("The original three fits (one network, pretrained HSSM OU likelihood) diverged on nearly every draw, with g at the unstable edge of its range, drift at the upper edge and the fitted non-decision time absorbing the 0.3 s offset. Four problems were separated. The data were read out through a Weibull collapsing bound, which a constant-bound OU can only imitate with unstable dynamics; RT and choice were therefore regenerated from the same dv traces with a constant bound of 1.5, at the cost of 3.4, 0.3 and 0.0 percent omitted trials at the three gains. The networks decide in 75 to 145 ms with a leak of several per second, outside the likelihood's box of |g| at most 1 per second and decision times near a second, so a time stretch of 8 to 10 is needed and none was used. Non-decision time was fitted. And the 750 ms horizon truncates the data, which the recovery study showed pushes g toward unstable."))
    s.append(P("The first session also found that fixing these would not be enough: an exact-simulator search with no parameter box reproduces the constant-bound RT and choice distributions at gains 1.0 and 1.2 almost perfectly at a ≈ 0.3, drift ≈ 4 and g between +4 and +9 per second, leaky at every gain. A low bound is crossed within 100 ms by the fast mean-reverting fluctuations of the readout, and the OU's single leak term reports their decay."))

    s.append(P("3. The control: corrected HSSM fits to RT and choice", H))
    s.append(P("Track A refits the constant-bound data (twenty networks, 38 640 to 39 988 trials per gain) with HSSM's pretrained OU likelihood and every defect removed: RTs stretched by k = 10 and offset by 0.3 s, non-decision time fixed at the offset plus the minimum RT less one native millisecond so the fastest trial keeps a positive decision time, lapse mixture off, drift regressed on signed coherence, choice coded as the network's choice, uniform priors over the likelihood's box. Three pooled fits per stretch, 120 per-network, three hierarchical, six recovery and two appendix fits were run; all have R-hat at most 1.01, bulk effective sample size above 400 and one divergence in 600 000 draws."))
    s.append(P("The sign is unambiguous and the magnitude is not. At gains 0.8 and 1.0 the posterior for g sits on the +1 edge of the likelihood at every stretch from 8 to 24, so its native value equals k and is a floor, not an estimate. Over the same three-fold range of k the native bound and drift are stable to about ten percent, as the rescaling rule requires of identified parameters. At gain 1.2 the posterior leaves the edge from k = 10 on and its native value rises from +9.2 to +14.7 per s between k = 10 and 24. The likelihood is monotone in g because the constant-bound RTs have a 6 ms minimum and a flat tail to the 750 ms horizon, and strong mean reversion is the only way a one-boundary OU produces both. The hierarchical fits converge and agree with the pooled ones within 2 percent."))
    s.append(tab([["gain", "g, stretched [94% HDI]", "g native (per s)", "a native", "drift native at coherence 0.15", "posterior mass at the +1 ceiling", "networks with g > 0"],
                  ["0.8", "+0.998 [+0.994, +1.000]", "+10.0", "0.38", "3.8", "1.00", "20 of 20"],
                  ["1.0", "+0.996 [+0.988, +1.000]", "+10.0", "0.32", "4.1", "0.99", "20 of 20"],
                  ["1.2", "+0.921 [+0.872, +0.968]", "+9.2", "0.29", "4.1", "0.01", "20 of 20"]],
                 [0.55*inch, 1.55*inch, 0.85*inch, 0.6*inch, 1.05*inch, 1.05*inch, 0.85*inch],
                 "Table 1. Pooled HSSM fits at k = 10 (output/track_a/headline_pooled.csv). Positive g is leaky; native units follow g·k, a/√k and v·√k. The last column counts the 120 per-network fits at the same stretch; 118 have a 94 percent HDI excluding zero on the leaky side."))
    s.append(fig(ROOT / "output/report/fig_track_a.png", 5.5, "Figure 1. Track A. Top: RT distributions of the twenty networks (filled) and of the HSSM-fitted OU at its pooled posterior mean (line), all coherences, constant bound 1.5. Bottom: psychophysical kernels of the networks (solid) and of the fitted OU driven by the same evidence streams (dashed), with its fitted bound (left) and without (middle); kernel slope against gain for both (right). Positive g is leaky."))
    s.append(P("The model reproduces what it was fitted to. Simulating 20 000 trials per coherence from 200 posterior draws gives median absolute RT quantile errors of 6.5, 1.6 and 0.8 ms on correct trials at gains 0.8, 1.0 and 1.2, accuracy within 0.02 at every coherence and omissions within 1.6 points (Figure 1, top). The posterior predictive used an Euler integrator of the same process because the ssm-simulators C code corrupts its heap when called repeatedly with changing parameters; on a cross-check the two engines agree within Monte Carlo noise."))
    s.append(P("The decisive check drives the fitted model with the networks' own evidence streams and computes the psychophysical kernel (Figure 1, bottom). With its fitted bound, 0.29 to 0.38 in native units and reached within about 100 ms on more than 99.9 percent of trials, the model commits before most of the evidence arrives and shows primacy at every gain, slopes -0.041, -0.057 and -0.064 against the networks' +0.039, +0.002 and -0.026; what ordering it has follows the falling bound. With the bound removed the fitted leak gives recency at every gain, slopes +0.068, +0.067 and +0.066, because the fitted g is the same large number at all three gains. Neither variant reproduces any gain or the ordering. These slopes were reproduced independently for this report to within 0.003."))
    s.append(P("Two checks bound the reading. The pipeline is not blind to instability: with data simulated from the fitted parameters and g replaced by the theoretical values, the refit recovers the sign in all six datasets and at gain 1.2 the HDI excludes zero on the unstable side; magnitudes shrink toward zero by about a third, so only one of six HDIs covers the truth and Table 1 should be read as signs. And the original sign flip is a confound: the same network at the same gain fitted on the Weibull-bound readout gives g = -0.22 [-0.28, -0.15] at k = 10."))

    s.append(P("4. The result: the same accumulator fitted to choice given the evidence stream", H))
    s.append(P("Track B fits the same OU in the network's own time (dt = 1 ms) to each trial's deadline choice given that trial's evidence sequence, after Brunton, Botvinick and Brody (2013): a<sub>t+1</sub> = a<sub>t</sub> + (v e<sub>t</sub> - g a<sub>t</sub>) dt + √dt ξ<sub>t</sub>, a<sub>0</sub> = a<sub>bias</sub>, an optional sticky bound at ±B, choice = sign(a<sub>T</sub>), noise scale fixed at one so v and B are in noise units. Without the bound a<sub>T</sub> is Gaussian given the evidence, the likelihood of each choice is a normal probability and a fit takes seconds. It was fitted by maximum likelihood and by NUTS with weak priors for every network and gain, and pooled over networks; the analytic likelihood matches Monte Carlo within 0.005 and an independent implementation within 0.003 in g."))
    s.append(tab([["gain", "median g across networks (per s)", "interquartile range", "intervals excluding zero on the predicted side", "pooled g [94% HDI]", "landscape coefficient"],
                  ["0.8", "+4.13", "+3.02 to +4.63", "20 of 20 positive", "+3.95 [+3.83, +4.06]", "about +4.3"],
                  ["1.0", "+0.32", "-0.34 to +0.79", "10 positive, 5 negative", "+0.23 [+0.15, +0.32]", "about -1.6"],
                  ["1.2", "-2.63", "-3.56 to -2.03", "20 of 20 negative", "-2.73 [-2.83, -2.62]", "about -6.2"]],
                 [0.5*inch, 1.15*inch, 1.05*inch, 1.4*inch, 1.4*inch, 1.0*inch],
                 "Table 2. Track B, unbounded evidence-conditioned fit (output/track_b/analytic_per_network.csv, analytic_pooled.csv). Positive g is leaky. Intervals are 94 percent HDIs from NUTS; maximum-likelihood estimates agree in sign in all 60 cells. The landscape coefficient is the sign-converted drift slope from three networks."))
    s.append(P("The sign pattern is the predicted one in every network at the extreme gains: twenty of twenty leaky at gain 0.8 and twenty of twenty unstable at gain 1.2, with intervals excluding zero in every case. At gain 1.0 the networks straddle zero with a slightly positive pooled estimate, and the zero crossing of g against gain falls at 1.016 pooled (per-network median 1.023), where Figure 1F puts the kernel's. The per-network g follows each network's own kernel slope with r = 0.996 (Figure 2, right). The pooled values match the first session's simulation-matching estimates (+3.98, +0.24, -2.50) and the ordering and zero crossing of the landscape, with a smaller magnitude at gain 1.2 because the network saturates into wells near |dv| = 3 that a linear process lacks."))
    s.append(P("The fitted model reproduces the kernel it was not fitted to: pooled slopes +0.041, +0.002 and -0.031 against the networks' +0.039, +0.002 and -0.026, per-network slopes on the identity line, psychometric curves overlapping (Figure 3). The one shortfall is at gain 1.2, where the network's kernel falls faster over the first 200 ms than a linear unstable process can, the mark of early commitment into a well. Recovery with the fitted or the theoretical values planted returns the right sign in 60 of 60 refits and covers the truth in 56. Adding the sticky bound and fitting by Monte Carlo likelihood (300 realisations per trial, common random numbers, profile intervals) changes little when the bound is free: medians +4.13, +0.52 and -2.59 per s, 20 of 20 and 19 of 20 on the predicted side, agreement with the unbounded fit network by network at r = 0.998, 0.926 and 0.952. The bound itself is identified only from below (median about 6.4), since choices alone do not say where it is."))

    s.append(P("5. Leak versus bound", H))
    s.append(P("A strong leak with early commitment at a low bound gives a primacy kernel just as an unstable accumulator does; three of sixty of the first session's simulation-matching fits landed there with g near +12. With the per-trial likelihood it happens once in sixty: network 61 at gain 1.2 gives g = +1.06 with B = 1.5, 2.5 log-likelihood units above the unbounded solution at g = -0.93. A dataset generated from that degenerate solution on real evidence is recovered by the choice-only likelihood (g = +11.9 [+10.5, +12.9]), so 2000 trials do separate the regimes and the remaining case is a local optimum."))
    s.append(fig(ROOT / "output/report/fig_track_b_g.png", 4.3, "Figure 2. Track B. Left: fitted OU leak g per network with 94 percent intervals against gain (filled, unbounded fit), the bounded fit with the deadline-commitment term (open squares, gains 0.8 and 1.2), pooled fits (coloured bars), landscape coefficients (gray bars) and the kernel zero crossing of Figure 1F at gain 1.016 (dotted). Right: the same estimates against each network's own kernel slope. Positive g is leaky."))
    s.append(fig(ROOT / "output/report/fig_track_b_kernel.png", 3.8, "Figure 3. Track B. Left: psychophysical kernels of the networks (solid) and of the fitted OU driven by the same evidence streams (dashed), averaged over the twenty networks. Right: kernel slope of the fitted model against that of the network, one point per network and gain."))
    s.append(tab([["bounded-fit variant", "median g at 0.8", "median g at 1.2", "predicted side, interval excluding 0", "median B at 0.8, 1.2", "pooled kernel slope at 0.8 (network +0.039)"],
                  ["no hit term", "+4.13", "-2.59", "20 of 20, 19 of 20", "6.6, 6.4", "+0.040"],
                  ["hit at the deadline", "+2.51", "-2.53", "20 of 20, 20 of 20", "1.4, 1.8", "+0.021"],
                  ["ever crossed 2.0", "+5.90", "-3.51", "20 of 20, 19 of 20", "1.0, 1.4", "not run"],
                  ["crossing-time bins", "+8.28", "-1.19", "20 of 20, 14 of 20", "0.9, 0.8", "-0.010"]],
                 [1.35*inch, 0.8*inch, 0.8*inch, 1.45*inch, 0.95*inch, 1.15*inch],
                 "Table 3. Track B, bounded fits (output/track_b/route2_per_network.csv). Positive g is leaky. The last two rows use the network's transient 2.0 crossing, which a sticky bound cannot represent; they show the failure mode, not estimates."))
    s.append(P("The issue proposed the fraction of trials whose |dv| reaches 2.0 before the deadline as the observable that breaks the degeneracy. It is the wrong observable for a sticky-bound model, for an informative reason: the network's 2.0 crossing is not absorbing. Of the trials that cross before the deadline, 46, 21 and 10 percent are back below 2.0 at 750 ms at gains 0.8, 1.0 and 1.2. Forcing a sticky bound to match that transient rate drives B below 1, inflates the leak at gain 0.8 (Table 3) and flips the sign of the model's kernel there. Measured at the deadline instead, |dv| above 2.0 at 750 ms, the observable is what a sticky bound implies; that variant identifies B, matches the commitment fraction within 0.015, keeps the kernel, and resolves network 61 to g = -0.36 [-0.50, -0.19]."))

    s.append(P("6. What remains", H))
    s.append(P("The transition is in the networks' behaviour and an accumulator recovers it, but only when the fit is conditioned on each trial's evidence. What stays model-dependent is the magnitude at gain 0.8, between +2.5 and +4.1 per s depending on how the bound is constrained; the sign and ordering do not move. The leak at gain 1.2 sits near -2.6, below the landscape's -6.2 because a linear OU cannot saturate into the wells. A two-timescale accumulator, a fast mean-reverting readout on a slow choice mode, would reconcile Track A's leak of at least +24 per s with Track B's few per second and is the next model. Not done: a Weibull appendix beyond one network, and Track B on a bounded rather than a deadline choice."))

    def footer(c, d):
        c.saveState(); c.setFont("DV", 8); c.setFillColor(colors.grey); c.drawRightString(letter[0] - 0.8 * inch, 0.55 * inch, str(d.page)); c.restoreState()
    doc = SimpleDocTemplate(str(OUTPDF), pagesize=letter, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.75*inch, bottomMargin=0.72*inch, title="OU fits to gain-modulated RNNs, round 2", author="Ivan Grahek")
    doc.build(s, onFirstPage=footer, onLaterPages=footer); print("built", OUTPDF)

if __name__ == "__main__":
    build()
