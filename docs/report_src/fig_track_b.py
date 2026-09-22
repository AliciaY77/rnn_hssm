"""Report figures for Track B.
fig_track_b_g.png : per-network OU leak g with 94 % intervals against gain (route 1 Bayesian; route 2 with the crossing-time
                    term where available), landscape coefficients marked, Fig 1F zero crossing marked.
fig_track_b_kernel.png : network kernels vs fitted-OU kernels per gain (pooled), plus per-network slope agreement.
Reads output/track_b/*.csv from the Track B worktree until merged. g > 0 leaky."""
import pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]
TB = ROOT / "output" / "track_b"
OUT = ROOT / "output" / "report"; OUT.mkdir(parents=True, exist_ok=True)
GAINS = [0.8, 1.0, 1.2]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; LAND = {0.8: 4.3, 1.0: -1.6, 1.2: -6.2}

r1 = pd.read_csv(TB / "analytic_per_network.csv")
kn = pd.read_csv(TB / "kernel_reproduction_per_network.csv")
r1 = r1.merge(kn[["seed", "gain", "slope_net", "slope_route1"]], on=["seed", "gain"], how="left")
pooled = pd.read_csv(TB / "analytic_pooled.csv").set_index("gain")
r2 = pd.read_csv(TB / "route2_per_network.csv"); r2 = r2[r2.hit_mode == "term"]

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]; rng = np.random.default_rng(3)
for gn in GAINS:
    d = r1[r1.gain == gn].sort_values("seed"); x = gn + np.linspace(-0.035, 0.035, len(d))
    ax.errorbar(x, d.g_mean, yerr=[d.g_mean - d.g_hdi_lo, d.g_hdi_hi - d.g_mean], fmt="o", color=COL[gn], ms=3.5, lw=0.9, capsize=0, label=f"gain {gn}: {len(d)} networks")
    d2 = r2[r2.gain == gn].sort_values("seed")
    if len(d2): ax.errorbar(gn + 0.06 + np.linspace(-0.02, 0.02, len(d2)), d2.g, yerr=[d2.g - d2.g_lo, d2.g_hi - d2.g], fmt="s", mfc="w", color=COL[gn], ms=3.5, lw=0.8, capsize=0, alpha=0.9)
    p = pooled.loc[gn]; ax.plot([gn - 0.045, gn + 0.045], [p.g_mean] * 2, color=COL[gn], lw=2.5)
    ax.plot([gn - 0.045, gn + 0.045], [LAND[gn]] * 2, color="gray", lw=4, alpha=0.5, zorder=0)
ax.axhline(0, color="gray", lw=0.6); ax.axvline(1.016, color="gray", ls=":", lw=1)
ax.set_xticks(GAINS); ax.set_xlabel("gain"); ax.set_ylabel("OU leak g (per s; > 0 leaky)")
ax.set_title("per-network OU leak g with intervals against gain")
from matplotlib.lines import Line2D
h, l = ax.get_legend_handles_labels()
ax.legend(h + [Line2D([], [], marker="s", mfc="w", color="gray", ls="", ms=4), Line2D([], [], color="gray", lw=4, alpha=0.5)], l + ["bounded fit with deadline-commitment term", "landscape coefficient (approx.)"], frameon=False, loc="lower left")
ax = axes[1]
for gn in GAINS:
    d = r1[r1.gain == gn]
    ax.errorbar(d.slope_net if "slope_net" in d else np.nan, d.g_mean, yerr=[d.g_mean - d.g_hdi_lo, d.g_hdi_hi - d.g_mean], fmt="o", color=COL[gn], ms=3.5, lw=0.9, capsize=0)
ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6); ax.set_xlabel("network kernel slope (Fig 1F measure)"); ax.set_ylabel("fitted g (per s)")
ax.set_title("fitted g against each network's own kernel slope")
plt.tight_layout(); fig.savefig(OUT / "fig_track_b_g.png", dpi=160); print("saved", OUT / "fig_track_b_g.png")

# kernel reproduction: mean of the per-network L1-normalised kernels (network vs fitted OU, route 1) + slope agreement
fig, axes = plt.subplots(1, 2, figsize=(10, 4.0)); xb = (np.arange(8) + 0.5) * 750 / 8
ax = axes[0]
for gn in GAINS:
    d = kn[kn.gain == gn]
    ax.plot(xb, d[[f"wnet{i}" for i in range(8)]].mean(), "-o", color=COL[gn], ms=4)
    ax.plot(xb, d[[f"w1_{i}" for i in range(8)]].mean(), "--s", color=COL[gn], ms=4, alpha=0.85)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color="gray", marker="o", ms=4), Line2D([], [], color="gray", ls="--", marker="s", ms=4)] + [Line2D([], [], color=COL[g], lw=3) for g in GAINS],
          ["network", "fitted OU"] + [f"gain {g}" for g in GAINS], frameon=False, loc="upper center", ncol=5, columnspacing=0.8, handlelength=1.5)
ax.set_xlabel("time within trial (ms)"); ax.set_ylabel("weight on choice (L1-normalised)"); ax.set_ylim(0.0, 0.37); ax.set_title("psychophysical kernel, mean over 20 networks")
ax = axes[1]
for gn in GAINS:
    d = kn[kn.gain == gn]; ax.plot(d.slope_net, d.slope_route1, "o", color=COL[gn], ms=4, label=f"gain {gn}")
lim = [kn[["slope_net", "slope_route1"]].min().min() - 0.005, kn[["slope_net", "slope_route1"]].max().max() + 0.005]
ax.plot(lim, lim, color="gray", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("network kernel slope"); ax.set_ylabel("fitted-OU kernel slope"); ax.set_title("per-network kernel slope, fitted OU vs network"); ax.legend(frameon=False)
plt.tight_layout(); fig.savefig(OUT / "fig_track_b_kernel.png", dpi=160); print("saved", OUT / "fig_track_b_kernel.png")
