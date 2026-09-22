import pathlib, datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
FD = "/opt/homebrew/anaconda3/lib/python3.13/site-packages/matplotlib/mpl-data/fonts/ttf"
pdfmetrics.registerFont(TTFont("DV", FD + "/DejaVuSans.ttf")); pdfmetrics.registerFont(TTFont("DV-B", FD + "/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", FD + "/DejaVuSans-Oblique.ttf")); pdfmetrics.registerFont(TTFont("DV-BI", FD + "/DejaVuSans-BoldOblique.ttf"))
pdfmetrics.registerFont(TTFont("DVM", FD + "/DejaVuSansMono.ttf"))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-BI")

ROOT = pathlib.Path("/Users/igrahek/Library/CloudStorage/Dropbox-Brown/Ivan Grahek/Ivan/Studies/rnn_hssm")
OUTPDF = ROOT / "docs" / "OU_fits_to_gain_RNNs_2026-09-22.pdf"
ss = getSampleStyleSheet()
for k in ss.byName: ss.byName[k].fontName = "DV-B" if "Heading" in k or k == "Title" else "DV"
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="DV-B", fontSize=14, spaceBefore=14, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="DV-B", fontSize=11, spaceBefore=10, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["Normal"], fontName="DV", fontSize=9.5, leading=13, spaceAfter=6)
BL = ParagraphStyle("BL", parent=B, leftIndent=14, bulletIndent=4, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.5, leading=11, textColor=colors.HexColor("#444444"), spaceAfter=10)
SMALL = ParagraphStyle("S", parent=B, fontSize=8.5, leading=11)
TS = TableStyle([("FONT", (0, 0), (-1, -1), "DV", 8), ("FONT", (0, 0), (-1, 0), "DV-B", 8),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f3f3")]),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)])

def P(t, st=B): return Paragraph(t, st)
def bullets(items): return [Paragraph(i, BL, bulletText="•") for i in items]
def fig(path, width_in, caption):
    im = PILImage.open(path); w, h = im.size; wi = width_in * inch; hi = wi * h / w
    return KeepTogether([Image(str(path), width=wi, height=hi), Paragraph(caption, CAP)])
HDR = ParagraphStyle("HDR", parent=B, fontName="DV-B", fontSize=8, leading=9.5, spaceAfter=0)
def table(rows, colw=None):
    rows = [[Paragraph(str(c), HDR) for c in rows[0]]] + rows[1:]
    t = Table(rows, colWidths=colw, hAlign="LEFT", repeatRows=1); t.setStyle(TS); return t
def tcap(rows, colw, caption): return KeepTogether([table(rows, colw), Paragraph(caption, CAP)])

story = []
story.append(P("Fitting an Ornstein–Uhlenbeck accumulator to gain-modulated RNN behaviour", ss["Title"]))
story.append(P("Why the HSSM OU fits failed, what the behaviour can and cannot tell us about leak, and a fit that recovers the predicted leaky → perfect → attractive transition", ParagraphStyle("sub", parent=B, fontSize=11, leading=14, textColor=colors.HexColor("#333333"))))
story.append(P(f"Ivan Grahek — draft for discussion, {datetime.date.today():%d %B %Y}. Analyses on branches <font face='DVM'>issue-1-ou-param-recovery</font> and <font face='DVM'>kernel-ou-fit</font> of <font face='DVM'>rnn_hssm</font>; every number below is reproducible from those branches.", SMALL))
story.append(Spacer(1, 8))

story.append(P("Summary", H1))
story += bullets([
 "<b>Goal.</b> Use an Ornstein–Uhlenbeck (OU) accumulator, fitted to the networks' RTs and choices, to show that gain moves the trained NXX1 network from a leaky integrator (gain 0.8) through near-perfect integration (1.0) into an attractive / unstable regime (1.2), as the fixed-point and landscape analyses predict.",
 "<b>What went wrong.</b> The three HSSM OU fits (one network, gains 0.8 / 1.0 / 1.2) did not converge: ~98–100% of draws were divergences and the leak parameter g was pinned at the edge of its allowed range at every gain. Four separate causes were identified and tested: (i) the sign convention of g in HSSM is the reverse of what the code assumed; (ii) the networks' decision timescale is 5–10× faster than the range the pretrained likelihood network (LAN) was trained on; (iii) the data were generated with a collapsing (Weibull) bound that a constant-bound OU cannot represent; (iv) more fundamentally, RT and choice <i>distributions</i> do not carry the information that distinguishes leak from instability in these networks.",
 "<b>Evidence for (iv).</b> A parameter-recovery study shows the LAN cannot recover g reliably in this regime, and a direct simulator search shows that even an unconstrained, perfectly specified OU fitted to RT/choice distributions returns 'leaky' at every gain, most strongly at gain 1.2 — the opposite of the prediction. The single leak term ends up describing the fast fluctuations of the readout that cross a low threshold, not the slow choice mode.",
 "<b>Solution.</b> Fit the same OU model to <i>each trial's choice given that trial's actual evidence sequence</i> (the Brunton et al. 2013 logic), matched through the psychophysical kernel and psychometric curve rather than RTs. On 20 networks this returns g = +4.0 / +0.2 / −2.5 s<super>−1</super> at gains 0.8 / 1.0 / 1.2 (positive = leaky), reproducing the network kernels bin by bin. The regime transition is recoverable from behaviour — but from the temporal weighting of evidence, not from RT distributions.",
])

story.append(P("1. The data and the original fits", H1))
story.append(P("Twenty trained NXX1 networks (seeds 42–61) perform a Mante-style context-dependent random-dot task with a 750 ms trial, 1 ms per step, at gains 0.8, 1.0 and 1.2 (2000 trials per cell). RT and choice are read out post hoc from the network's output decision variable dv by a <b>Weibull collapsing bound</b>: initial height ln 19 ≈ 2.94, shape 4.0, scale 469 ms, decaying to ~0 by 750 ms so every trial terminates. Median RTs are 373 / 258 / 183 ms at the three gains; nothing exceeds 620 ms."))
story.append(P("The OU fits (<font face='DVM'>model/fit_ornstein.py</font>) used HSSM's pretrained <font face='DVM'>ornstein</font> LAN with drift regressed on coherence, a 0.3 s offset added to every RT, and the leak g, boundary a, start point z and non-decision time t free. Sampling: NUTS, 4 chains × 1000 draws."))
story.append(tcap([["gain", "divergent draws", "R-hat (g)", "posterior g", "implied drift at coh 0.15", "t"],
                    ["0.8", "3997 / 4000", "4.1", "−0.79 (edge −1)", "2.0 (edge 2)", "0.39"],
                    ["1.0", "3912 / 4000", "5.7", "−0.91 (edge −1)", "2.0 (edge 2)", "0.33"],
                    ["1.2", "3997 / 4000", "2.0", "−0.91 (edge −1)", "2.0 (edge 2)", "0.32"]],
                   [0.5*inch, 1.1*inch, 0.8*inch, 1.3*inch, 1.6*inch, 0.5*inch], "Table 1. The three original OU fits. Two parameters sit on the boundary of the LAN's training range simultaneously, and t simply absorbs the 0.3 s offset."))

story.append(P("2. Diagnosis", H1))
story.append(P("2.1 Sign convention", H2))
story.append(P("In ssm-simulators and hence in the LAN, the OU update is d<i>x</i> = (<i>v</i> − g·<i>x</i>)d<i>t</i> + d<i>W</i>: <b>g &gt; 0 is leaky, g &lt; 0 is unstable / attractive</b>. This was verified by simulation (median first-passage time 0.61 / 0.79 / 1.14 s for g = −1 / 0 / +1 at v = 0, a = 1). The script's docstring had it the other way round, so the fits pinned at g ≈ −0.9 were asking for maximal <i>instability</i>, not maximal leak. All values in this report use the simulator's convention."))
story.append(P("2.2 Timescales: where the networks sit relative to the LAN's range", H2))
story.append(P("A LAN is a neural-network approximation of the likelihood, valid only inside the parameter box it was trained on: a ∈ [0.3, 3], |v| ≤ 2, |g| ≤ 1 s<super>−1</super>, t ∈ [1 ms, 2 s], in units where the diffusion is 1 per √s. The direct measurement of the networks' choice-axis dynamics (Langevin drift on noise-only trials, <font face='DVM'>RNN_Gain_Mod/lab/2026-09-21_gain-leaky-to-bistable</font>) gives a linear coefficient at the undecided state of about +4 s<super>−1</super> (leaky) at gain 0.8, ≈ 0 at 1.0 and about −6 s<super>−1</super> (repelling) at 1.2, with decisions taking 100–400 ms. Both the leak rates and the decision times are 5–10× outside what the LAN saw in training."))
story.append(P("Stretching time (RT × k) rescales the same process as g → g/k, a → a√k, v → v/√k (verified numerically). A stretch of k ≈ 6–10 does bring the networks' OU description inside the box, so the LAN is not unusable in principle — but the original fits used k = 1, and, as §2.4–2.5 show, a correctly boxed RT fit would still not answer the question."))
story.append(P("2.3 The collapsing bound", H2))
story.append(P("A constant-bound OU cannot represent a Weibull collapse. The collapse compresses RTs and forces late, near-chance choices; a constant-bound OU can only imitate that with unstable dynamics, which pushes g negative at every gain — the direction the original fits went. To remove this confound the RT/choice data were <b>regenerated from the same decision-variable traces with a constant bound</b>. The price is omissions (trials that never cross) at low gain: at bound 1.5, 3.4% of gain-0.8 trials (8.6% at zero coherence) do not cross within 750 ms; at the original height 2.94 two thirds do not. Side result: with a constant bound every RT distribution is unimodal, so the bimodal RT histograms in the original data are produced by the collapse."))
story.append(P("2.4 Parameter recovery with the pretrained LAN", H2))
story.append(P("Simulated OU data (g ∈ {−1, −0.5, 0, +0.5, +1}, a ∈ {1.0, 1.5}, n ∈ {1000, 3700}, zero coherence, time stretched ×6) were fitted with the same HSSM model, in three variants: HSSM defaults (5% lapse mixture + 4.5 s deadline truncation), no lapse, and no lapse with no deadline. All 60 fits sampled cleanly (0 divergences), so what follows is about the likelihood, not the sampler."))
story.append(tcap([["variant", "sign of g correct (g ≠ 0)", "94% HDI covers true g"],
                    ["defaults (lapse + deadline)", "7 / 16", "4 / 12"], ["no lapse, deadline", "5 / 16", "3 / 12"], ["no lapse, no deadline", "12 / 16", "7 / 12"]],
                   [2.2*inch, 1.9*inch, 1.7*inch], "Table 2. Recovery summary. Three findings: (1) <b>deadline truncation flips the sign of g</b> — dropping trials past a horizon and fitting with an untruncated likelihood turns every leaky cell into a confidently unstable one (true +0.5 → −0.72, sd 0.02) with a inflated 1.5 → 2.2 and t → 0; the effect is present with 12% of trials dropped, and the RNN data have a hard 750 ms horizon. (2) <b>g is unidentifiable when decisions are short relative to 1/|g|</b>: at |g|·T ≈ 0.6 the posterior spans most of the range regardless of the truth, even with 3700 trials. (3) Even with the right sign, magnitude is biased toward the edges. The lapse mixture is not the culprit."))
story.append(fig(ROOT / "docs/report_src/fig1_compact.png", 4.2, "Figure 1. Recovery of g (left column) and a (right column) in the three variants (rows: HSSM defaults; no lapse; no lapse and no deadline): posterior mean and 94% HDI against true g. Blue: a = 1.0, red: a = 1.5; squares n = 1000, circles n = 3700. The dashed identity line is where a working recovery would sit; dashed horizontals are the true a. Full five-parameter version in output/recovery/recovery_g.png."))

story.append(P("2.5 Does the OU family describe the fixed-bound behaviour at all?", H2))
story.append(P("The exact simulator was searched directly (random search + Nelder–Mead on per-coherence RT quantiles, accuracy and omission rate) — first inside the LAN box at several time stretches, then with the box removed. With the box removed the OU reproduces the gain-1.0 and gain-1.2 fixed-bound data essentially exactly (Figure 2), at a ≈ 0.3, v ≈ 4 and g ≈ +4 to +9 s<super>−1</super> in native time. So the model class is adequate. But the fitted leak is <b>positive (leaky) at every gain and largest at gain 1.2</b>, where the network is most strongly repelling — and this is the answer from a perfectly specified likelihood, so retraining the LAN would return the same thing with narrower error bars."))
story.append(fig(ROOT / "docs/report_src/fig2_compact.png", 6.4, "Figure 2. Gain 1.2, constant bound 1.5, no parameter box: network RT histograms (blue) and the best OU (red) for correct (top) and error (bottom) trials at coherences 0, 0.06 and 0.15 (the other three coherences look the same). Best parameters: a = 0.28, drift 4.2 at coherence 0.15, t = 6 ms, g = +9.3 s<super>−1</super>, i.e. strongly leaky."))
story.append(tcap([["bound on dv", "omissions", "fitted g (1/s)", "fit loss"], ["1.5", "0.0%", "+9.3", "0.14"], ["2.0", "0.2%", "+8.6", "0.62"], ["2.5", "5.6%", "+5.2", "2.2"], ["2.94 (original height)", "14%", "+0.5", "10.3"]],
                   [1.6*inch, 1.0*inch, 1.2*inch, 0.9*inch], "Table 3. Gain 1.2: as the constant bound is raised toward the committed attractors (|dv| ≈ 3), the fitted leak falls toward zero but the single-timescale OU stops fitting. Reading: a low bound is crossed within 50–100 ms by the fast, strongly mean-reverting fluctuations of the readout (10 ms unit time constant), and the OU's one leak term reports <i>their</i> decay, which grows with gain. Only a bound near the attractors is governed by the slow choice mode that the landscape analysis measures — and there a one-dimensional OU no longer describes the mixture of fast and slow crossings, while gain 0.8 loses most trials to omissions."))
story.append(P("Taken together: RT and choice distributions under a threshold readout do not expose the regime. The information that distinguishes leak from instability is <i>when</i> within the trial the evidence that drove the choice arrived, and that is invisible to a fit that only sees each trial's coherence."))

story.append(P("3. Solution: condition the fit on the trial-by-trial evidence", H1))
story.append(P("Each trial's stimulus is a coherence plus fresh noise at every millisecond, and we generated it, so the evidence sequence of every trial is known exactly. A leaky accumulator's choice reflects mainly the last few hundred milliseconds (recency); an unstable one amplifies what it saw first (primacy); a perfect integrator weights all of it equally. The <b>psychophysical kernel</b> — a logistic regression of choice on the mean evidence in each of 8 time bins — measures this weighting from behaviour alone, and the RNN paper's Figure 1F already shows it sweeping recency → flat → primacy across gain (slope zero-crossing at gain 1.016 over 20 networks)."))
story.append(P("The fit here is the cheapest version of the Brunton et al. (2013) approach. The same OU model, a<sub>t+1</sub> = a<sub>t</sub> + (v·e<sub>t</sub> − g·a<sub>t</sub>)Δt + √Δt·ξ<sub>t</sub>, with a sticky bound ±B and choice = sign(a<sub>T</sub>), is driven by the networks' <i>own</i> per-trial evidence streams e<sub>t</sub>, and (v, g, B) are chosen so that its choices on those same trials reproduce the network's kernel and psychometric curve. Choice is the network's end-of-trial readout, the same definition as Figure 1F. One fit per gain pooling all 20 networks (40,000 trials), plus one fit per network."))
story.append(tcap([["gain", "fitted g (1/s)", "v", "B", "kernel slope network / OU", "accuracy network / OU", "landscape drift slope (same sign)"],
                    ["0.8", "+3.98 (leaky)", "46.5", "2.6", "+0.039 / +0.041", "0.866 / 0.865", "≈ +4.3"],
                    ["1.0", "+0.24 (≈ perfect)", "40.6", "2.1", "+0.002 / +0.002", "0.887 / 0.885", "≈ −1.6"],
                    ["1.2", "−2.50 (unstable)", "40.5", "4.9", "−0.026 / −0.028", "0.869 / 0.871", "≈ −6.2"]],
                   [0.45*inch, 1.2*inch, 0.45*inch, 0.45*inch, 1.35*inch, 1.25*inch, 1.5*inch], "Table 4. Pooled stimulus-conditioned OU fits. The ordering and the zero crossing match the theory; magnitudes agree with the landscape measurement to within a factor of ~2 at the extremes (the OU is linear, whereas the network's repelling regime saturates into wells at |dv| ≈ 3, which a linear fit under-reports). The evidence scaling v is the same at every gain, matching the landscape finding that gain does not change the evidence tilt."))
story.append(fig(ROOT / "output/kernel_fit/ou_kernel_fits.png", 7.0, "Figure 3. Left: psychophysical kernels of the networks (solid) and of the fitted OU driven by the same evidence streams (dashed), 20 networks pooled. Middle: psychometric curves. Right: kernel slope as a function of g with v and B held at their fitted values — the mapping is monotonic and identical across gains, so the kernel is a clean readout of g; dotted lines are the networks' slopes."))
story.append(fig(ROOT / "output/kernel_fit/ou_kernel_fits_per_network.png", 5.8, "Figure 4. Per-network fits (one per network and gain, 2000 trials each). Left: fitted g by gain, black bars = mean ± s.e.m. Right: fitted g against each network's own kernel slope. All 20 networks are leaky at gain 0.8; 15 of 20 are unstable at gain 1.2 (median g = −1.8)."))
story.append(P("<b>Caveat: a known degeneracy.</b> Three of the 60 single-network fits (two at gain 1.2, one at 1.0) landed on g ≈ +12 with a low sticky bound: strong leak plus very early commitment also produces a primacy kernel. This is the λ-versus-bound trade-off familiar from the Brunton model. It does not occur in the pooled fits and it can be broken with information we have — the fraction of trials whose decision variable reaches the bound before the deadline, or the RTs themselves — or with a weak prior that B exceeds the typical end-of-trial |a|. Until then the per-network median is the robust summary."))

story.append(P("4. What this settles and what comes next", H1))
story += bullets([
 "The predicted leaky → perfect → attractive transition <b>is</b> recoverable from behaviour with an accumulator model, provided the fit is conditioned on the trial-by-trial evidence. It is <b>not</b> recoverable from RT/choice marginals under a threshold readout, whatever likelihood is used: the same OU family on the same networks gives the opposite conclusion when fitted that way.",
 "The HSSM OU route as run had four independent problems (sign convention, timescale range, collapsing bound, and the missing information above). The first three are fixable — fixed non-decision time at the RT floor, a time stretch of ~8–10, a constant or matched bound — but fixing them would produce converged fits that still say 'leaky everywhere'.",
 "Next step, if a parametric confirmation is wanted: a proper likelihood-based Brunton fit (choice given evidence stream; λ, bound, sensory noise free), which is the formal version of §3; adding the bound-hit fraction breaks the remaining degeneracy. If HSSM is to stay in the loop, its likelihood would need the evidence stream as an input, which the pretrained models do not take.",
 "All datasets (constant-bound RT/choice at five bound heights, evidence streams and deadline choices for 20 networks × 3 gains), scripts and result tables are on the two branches named above; the pooled kernel fit runs in about two minutes on a laptop.",
])
story.append(Spacer(1, 6))
story.append(P("Reproducibility. Original fits: Oscar, HSSM 0.3.x (numpyro 0.21). Recovery study: 60 HSSM 0.2.4 fits, Oscar jobs 6604819 / 6604891. Simulator matching: ssm-simulators 0.8.3, Oscar jobs 6606075 / 6606166 / 6606270 / 6606439. Evidence export and kernel fits: local, gainrnn code at gain-controller-rnn @ 1ca084b. Prepared with Claude Code.", SMALL))

doc = SimpleDocTemplate(str(OUTPDF), pagesize=letter, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.8*inch, bottomMargin=0.8*inch,
                        title="OU fits to gain-modulated RNNs", author="Ivan Grahek")
def _footer(canvas, doc):
    canvas.saveState(); canvas.setFont("DV", 8); canvas.setFillColor(colors.grey)
    canvas.drawRightString(letter[0] - 0.8 * inch, 0.5 * inch, f"OU fits to gain-modulated RNNs — page {doc.page}"); canvas.restoreState()
doc.build(story, onFirstPage=_footer, onLaterPages=_footer); print("built", OUTPDF)
