"""Round-3 kernel figure: the HSSM-fitted OU (fitted bound, matched diffusion) driven by the networks' own evidence streams, at
thresholds 1.5 / 2.5 / 2.94 (columns). Top: kernels at the posterior mean, network (solid) vs model (dashed). Bottom: kernel
slope against gain: network with a 95 % bootstrap interval over trials (network_kernel_boot.csv), model with its 95 % posterior
interval over 40 posterior draws (common random numbers, so the interval is posterior uncertainty only), and the
evidence-conditioned fit of Section 2 (pooled, point estimate) for reference."""
import pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = pathlib.Path(__file__).resolve().parents[2]; TA = ROOT / "output/track_a"; TB = ROOT / "output/track_b"; OUT = ROOT / "output/report"
GAINS = [0.8, 1.0, 1.2]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; SHOW = ["1.5", "2.5", "2.94"]; V = "a_fitted_matched"
nb = pd.read_csv(OUT / "network_kernel_boot.csv").set_index("gain"); tb = pd.read_csv(TB / "kernel_reproduction_pooled.csv").set_index("gain")
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
fig, ax = plt.subplots(2, 3, figsize=(11, 6.4), gridspec_kw=dict(height_ratios=[1.2, 1])); xb = (np.arange(8) + 0.5) * 750 / 8; rows = []
for j, b in enumerate(SHOW):
    k = pd.read_csv(TA / f"kernel_ppc_b{b}_d40.csv"); k = k[(k.which == "posterior_mean") & (k.variant == V)].set_index("gain")
    dr = pd.read_csv(TA / f"kernel_ppc_draws_b{b}_d40.csv"); dr = dr[dr.variant == V]
    a = ax[0, j]
    for gn in GAINS:
        r = k.loc[gn]; a.plot(xb, [r[f"wnet{i}"] for i in range(8)], "-o", color=COL[gn], ms=4); a.plot(xb, [r[f"wsim{i}"] for i in range(8)], "--s", color=COL[gn], ms=4, alpha=0.85, mfc="w")
    a.axhline(0, color="gray", lw=0.5); a.set_ylim(-0.05, 0.62); a.set_xlabel("time within trial (ms)")
    a.set_title(f"threshold {b}: kernel\n(model hits its bound on {', '.join(f'{100*k.loc[g].frac_bounded:.0f}' for g in GAINS)} % of trials)")
    if j == 0:
        a.set_ylabel("weight on choice (L1-normalised)")
        a.legend([Line2D([], [], color="gray", marker="o", ms=4), Line2D([], [], color="gray", ls="--", marker="s", ms=4, mfc="w")] + [Line2D([], [], color=COL[g], lw=3) for g in GAINS],
                 ["network", "fitted OU"] + [f"gain {g}" for g in GAINS], frameon=False, loc="upper center", ncol=2, fontsize=8)
    a = ax[1, j]; m_mid, m_lo, m_hi = [], [], []
    for gn in GAINS:
        s = dr[dr.gain == gn].slope_sim.values; m_mid.append(np.median(s)); m_lo.append(np.quantile(s, 0.025)); m_hi.append(np.quantile(s, 0.975))
        rows.append(dict(threshold=b, gain=gn, net=nb.loc[gn].slope, net_lo=nb.loc[gn].lo, net_hi=nb.loc[gn].hi, model=m_mid[-1], model_lo=m_lo[-1], model_hi=m_hi[-1], n_draws=len(s)))
    net = nb.loc[GAINS]
    a.errorbar(GAINS, net.slope, yerr=[net.slope - net.lo, net.hi - net.slope], fmt="-o", color="C2", capsize=3, label="network (95 % bootstrap CI)")
    a.errorbar(GAINS, m_mid, yerr=[np.array(m_mid) - m_lo, np.array(m_hi) - m_mid], fmt="--s", color="k", capsize=3, label="HSSM OU (95 % posterior CI)")
    a.plot(GAINS, tb.loc[GAINS].s_sim, ":^", color="C4", ms=5, label="choice-given-evidence fit (Sec. 2)")
    a.axhline(0, color="gray", lw=0.5); a.set_ylim(-0.08, 0.06); a.set_xticks(GAINS); a.set_xlabel("gain"); a.set_title(f"threshold {b}: kernel slope against gain")
    if j == 0: a.set_ylabel("kernel slope"); a.legend(frameon=False, loc="upper right", fontsize=7.5)
plt.tight_layout(); fig.savefig(OUT / "fig_r3_kernel.png", dpi=160)
t = pd.DataFrame(rows); t.to_csv(OUT / "r3_kernel_ci.csv", index=False); print(t.round(4).to_string(index=False))
