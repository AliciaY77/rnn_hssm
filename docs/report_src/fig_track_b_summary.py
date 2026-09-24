"""Round-3 summary figure for Track B: kernel reproduction | psychometric | calibration, one row."""
import glob, pathlib, numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "output/report"; GAINS = [0.8, 1.0, 1.2]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; DT = 1e-3; T = 750; tt = np.arange(T)
kn = pd.read_csv(ROOT / "output/track_b/kernel_reproduction_per_network.csv"); tab = pd.read_csv(OUT / "track_b_ppc_psychometric.csv"); fits = pd.read_csv(ROOT / "output/track_b/analytic_per_network.csv")
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5)); xb = (np.arange(8) + 0.5) * 750 / 8
ax = axes[0]
for gn in GAINS:
    d = kn[kn.gain == gn]; ax.plot(xb, d[[f"wnet{i}" for i in range(8)]].mean(), "-o", color=COL[gn], ms=4); ax.plot(xb, d[[f"w1_{i}" for i in range(8)]].mean(), "--s", color=COL[gn], ms=4, alpha=0.85, mfc="w")
ax.legend([Line2D([], [], color="gray", marker="o", ms=4), Line2D([], [], color="gray", ls="--", marker="s", ms=4, mfc="w")] + [Line2D([], [], color=COL[g], lw=3) for g in GAINS], ["network", "fitted OU"] + [f"gain {g}" for g in GAINS], frameon=False, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.4, fontsize=7.5)
ax.set_ylim(0, 0.44); ax.set_xlabel("time within trial (ms)"); ax.set_ylabel("weight on choice (L1-normalised)"); ax.set_title("psychophysical kernel (not fitted)")
ax = axes[1]
for gn in GAINS:
    t = tab[tab.gain == gn]; g = t.groupby("coh").agg(p_net=("p_net", "mean"), p_mod=("p_mod", "mean")).reset_index()
    ax.plot(g.coh, g.p_net, "-o", color=COL[gn], ms=4); ax.plot(g.coh, g.p_mod, "--s", color=COL[gn], ms=3.5, alpha=0.85, mfc="w")
ax.set_xlabel("signed coherence"); ax.set_ylabel("P(choice = +)"); ax.set_title("psychometric, mean over 20 networks")
ax = axes[2]
for gn in GAINS:
    ps, chs = [], []
    for f in sorted(glob.glob(str(ROOT / "data/processed/kernel/seed*.npz"))):
        d = np.load(f); seed = int(pathlib.Path(f).stem[4:]); rel = d["rel"].astype(np.float32); r = fits[(fits.seed == seed) & (fits.gain == gn)].iloc[0]
        rho = 1 - r.g * DT; w = rho ** (T - 1 - tt); mu = r.a_bias * rho ** T + r.v * (rel @ w) * DT; s = np.sqrt(np.sum(rho ** (2 * (T - 1 - tt))) * DT)
        ps.append(norm.cdf(mu / s)); chs.append(d[f"choice_g{gn}"])
    p = np.concatenate(ps); ch = np.concatenate(chs); idx = np.clip(np.digitize(p, np.linspace(0, 1, 11)) - 1, 0, 9)
    xs = [p[idx == k].mean() for k in range(10)]; ys = [ch[idx == k].mean() for k in range(10)]
    ax.plot(xs, ys, "o", color=COL[gn], ms=4, label=f"gain {gn}: mean |dev| {np.mean(np.abs(np.array(xs) - np.array(ys))):.3f}")
ax.plot([0, 1], [0, 1], color="gray", lw=0.8); ax.set_xlabel("model P(choice = +) per trial"); ax.set_ylabel("observed fraction of + choices"); ax.set_title("calibration, 40 000 trials per gain"); ax.legend(frameon=False, loc="upper left")
plt.tight_layout(); fig.savefig(OUT / "fig_track_b_summary.png", dpi=160); print("saved")
