"""Round-3 kernel figure: the HSSM-fitted OU (with its fitted bound, matched diffusion, posterior mean) driven by the networks'
own evidence streams, at thresholds 1.5 / 2.5 / 2.94 (columns). Top: kernels, network (solid) vs model (dashed). Bottom:
kernel slope against gain, network vs model. Also writes r3_kernel_error.csv with Track B for reference."""
import pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = pathlib.Path(__file__).resolve().parents[2]; TA = ROOT / "output/track_a"; TB = ROOT / "output/track_b"; OUT = ROOT / "output/report"
GAINS = [0.8, 1.0, 1.2]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; SHOW = ["1.5", "2.5", "2.94"]; ALL = ["1.5", "2.0", "2.5", "2.94"]
def load(b):
    d = pd.read_csv(TA / ("kernel_ppc.csv" if b == "1.5" else f"kernel_ppc_b{b}.csv"))
    return d[(d.which == "posterior_mean") & (d.variant == "a_fitted_matched")].set_index("gain")
rows = []
for b in ALL:
    d = load(b)
    for gn in GAINS:
        r = d.loc[gn]; wn = np.array([r[f"wnet{i}"] for i in range(8)]); ws = np.array([r[f"wsim{i}"] for i in range(8)])
        rows.append(dict(fit=f"HSSM, threshold {b}", gain=gn, slope_net=r.slope_net, slope_model=r.slope_sim, rms_bins=float(np.sqrt(np.mean((ws - wn) ** 2))), frac_bounded=r.frac_bounded))
kn = pd.read_csv(TB / "kernel_reproduction_per_network.csv"); kp = pd.read_csv(TB / "kernel_reproduction_pooled.csv").set_index("gain")
for gn, d in kn.groupby("gain"):
    wn = d[[f"wnet{i}" for i in range(8)]].mean().values; ws = d[[f"w1_{i}" for i in range(8)]].mean().values
    rows.append(dict(fit="Track B, choice given evidence", gain=gn, slope_net=kp.loc[gn, "s_net"], slope_model=kp.loc[gn, "s_sim"], rms_bins=float(np.sqrt(np.mean((ws - wn) ** 2))), frac_bounded=np.nan))
t = pd.DataFrame(rows); t.to_csv(OUT / "r3_kernel_error.csv", index=False)
summ = t.groupby("fit", sort=False).apply(lambda x: pd.Series(dict(s08=x.slope_model.iloc[0], s10=x.slope_model.iloc[1], s12=x.slope_model.iloc[2],
        slope_err=np.abs(x.slope_model - x.slope_net).mean(), rms=x.rms_bins.mean(), span=x.slope_model.iloc[0] - x.slope_model.iloc[2]))).reset_index()
summ.to_csv(OUT / "r3_kernel_error_summary.csv", index=False); print(summ.round(4).to_string(index=False))
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
fig, ax = plt.subplots(2, 3, figsize=(11, 6.2), gridspec_kw=dict(height_ratios=[1.25, 1])); xb = (np.arange(8) + 0.5) * 750 / 8
for j, b in enumerate(SHOW):
    d = load(b); a = ax[0, j]
    for gn in GAINS:
        r = d.loc[gn]; a.plot(xb, [r[f"wnet{i}"] for i in range(8)], "-o", color=COL[gn], ms=4); a.plot(xb, [r[f"wsim{i}"] for i in range(8)], "--s", color=COL[gn], ms=4, alpha=0.85, mfc="w")
    a.axhline(0, color="gray", lw=0.5); a.set_ylim(-0.05, 0.62); a.set_xlabel("time within trial (ms)")
    fb = ", ".join(f"{100*d.loc[g].frac_bounded:.0f}" for g in GAINS)
    a.set_title(f"threshold {b}: kernel\n(model hits its bound on {fb} % of trials)")
    if j == 0:
        a.set_ylabel("weight on choice (L1-normalised)")
        a.legend([Line2D([], [], color="gray", marker="o", ms=4), Line2D([], [], color="gray", ls="--", marker="s", ms=4, mfc="w")] + [Line2D([], [], color=COL[g], lw=3) for g in GAINS],
                 ["network", "fitted OU"] + [f"gain {g}" for g in GAINS], frameon=False, loc="upper center", ncol=2, fontsize=8)
    a = ax[1, j]
    a.plot(GAINS, [d.loc[g].slope_net for g in GAINS], "-o", color="C2", label="network"); a.plot(GAINS, [d.loc[g].slope_sim for g in GAINS], "--s", color="k", label="fitted OU")
    a.axhline(0, color="gray", lw=0.5); a.set_ylim(-0.08, 0.06); a.set_xticks(GAINS); a.set_xlabel("gain"); a.set_title(f"threshold {b}: kernel slope against gain")
    if j == 0: a.set_ylabel("kernel slope"); a.legend(frameon=False, loc="lower left")
plt.tight_layout(); fig.savefig(OUT / "fig_r3_kernel.png", dpi=160); print("saved")
