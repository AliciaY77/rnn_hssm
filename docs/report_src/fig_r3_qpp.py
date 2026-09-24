"""Round-3 quantile-probability figure, drawn the way HSSM's plot_quantile_probability draws it with predictive_style="ellipse"
(the call in model/fit_ornstein.py): rows = threshold, columns = gain, all |coherence| levels in each panel. x = proportion of
trials in that condition that are correct (right) or errors (left); y = RT quantiles 0.25 / 0.5 / 0.75 (HSSM default q = 5).
Network: X markers joined per quantile in order of x (seaborn lineplot). Model: 95 % ellipse per (quantile, |coherence|,
correct/error) over the 80 posterior-predictive datasets, computed as HSSM's _compute_ellipse_params (mean, covariance,
n_std = sqrt(chi2.ppf(0.95, 2)), >= 5 points). Top axis: the |coherence| of each network point. Data: track_a/ppc_qpp.py."""
import pathlib, numpy as np, pandas as pd, seaborn as sns
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse; from matplotlib.lines import Line2D
from scipy.stats import chi2
ROOT = pathlib.Path(__file__).resolve().parents[2]; TA = ROOT / "output/track_a"; OUT = ROOT / "output/report"
GAINS = [0.8, 1.0, 1.2]; BOUNDS = [1.5, 2.5, 2.94]; QS = [0.25, 0.5, 0.75]
cmap = sns.cubehelix_palette(as_cmap=True); QCOL = {q: cmap((q - QS[0]) / (QS[-1] - QS[0])) for q in QS}   # seaborn's default numeric hue map
N_STD = np.sqrt(chi2.ppf(0.95, df=2))
def ellipse(pts):
    mean = pts.mean(0); cov = np.cov(pts.T); cov = (cov + cov.T) / 2 + 1e-10 * np.eye(2); ev, evec = np.linalg.eigh(cov)
    return mean, 2 * N_STD * np.sqrt(np.clip(ev, 0, None)), np.degrees(np.arctan2(evec[1, 0], evec[0, 0]))
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, axes = plt.subplots(len(BOUNDS), 3, figsize=(8.2, 8.6), sharex=True, sharey=True); YMAX = 0; rows = []
for i, b in enumerate(BOUNDS):
    for j, gn in enumerate(GAINS):
        ax = axes[i, j]; d = pd.read_csv(TA / f"ppc_qpp_g{gn}_k10_b{b}.csv"); obs = d[d.draw == -1]; pred = d[d.draw >= 0]
        for q in QS:
            o = obs[np.isclose(obs["quantile"], q)].sort_values("proportion")
            ax.plot(o.proportion, o.rt, "-X", color=QCOL[q], ms=5, lw=1.2, zorder=3)
            YMAX = max(YMAX, o.rt.max())
            for (ac, ic), g in pred[np.isclose(pred["quantile"], q)].groupby(["acoh", "is_correct"]):
                if len(g) < 5: continue
                (cx, cy), (w, h), ang = ellipse(g[["proportion", "rt"]].values)
                ax.add_patch(Ellipse((cx, cy), w * 1.0, h * 1.0, angle=ang, facecolor="none", edgecolor=QCOL[q], lw=1.0, alpha=0.6, zorder=2))
                oo = obs[np.isclose(obs["quantile"], q) & (obs.acoh == ac) & (obs.is_correct == ic)]
                rows.append(dict(bound=b, gain=gn, quantile=q, acoh=ac, correct=ic, n_obs=int(oo.n.iloc[0]) if len(oo) else 0, rt_obs=float(oo.rt.iloc[0]) if len(oo) else np.nan,
                                 p_obs=float(oo.proportion.iloc[0]) if len(oo) else np.nan, rt_pred=cy, p_pred=cx, n_datasets=len(g)))
        top = obs[np.isclose(obs["quantile"], 0.5)].groupby("proportion")["acoh"].first().reset_index()
        sec = ax.twiny(); sec.set_xlim(-0.02, 1.02); sec.set_xticks(top.proportion); sec.set_xticklabels([f"{c:g}" for c in top.acoh], fontsize=5.5, rotation=90)
        sec.tick_params(length=2, pad=1)
        ax.set_title(f"threshold {b}, gain {gn}", pad=18 if i == 0 else 16)
        if i == len(BOUNDS) - 1: ax.set_xlabel("proportion (errors left, correct right)")
        if j == 0: ax.set_ylabel("RT quantile (ms)")
axes[0, 0].set_xlim(-0.02, 1.02); axes[0, 0].set_ylim(0, 60 * np.ceil(YMAX * 1.08 / 60))
fig.legend([Line2D([], [], color=QCOL[q], marker="X", lw=1.2) for q in QS] + [Line2D([], [], color="gray", marker="o", ls="", mfc="none", ms=9, alpha=0.6)],
           [f"network, {q:g} quantile" for q in QS] + ["fitted OU, 95 % ellipse (80 datasets)"], loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.005))
plt.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT / "fig_r3_qpp.png", dpi=160)
pd.DataFrame(rows).to_csv(OUT / "r3_qpp_points.csv", index=False); print("saved; y max", round(YMAX))
