"""Track A RT posterior predictive by coherence: 3 gains x 6 |coherence| levels. Network RT histograms (filled: correct;
thin step: error) vs the HSSM-fitted OU at its pooled posterior mean (solid line: correct; dashed: error), simulated with
the same Euler integrator as fig_track_a.py, 7.8 s stretched horizon, non-crossers dropped. At coherence 0, "+" / "-"
choices take the place of correct / error."""
import pathlib, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "output" / "report"
GAINS = [0.8, 1.0, 1.2]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; K = 10.0; OFF = 0.3; COHS = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]

def simulate(v0, v1, a, z, g, t, coh, n, rng, dt=1e-3, horizon=OFF + K * 0.75):
    x = np.full(n, a * (2 * z - 1)); alive = np.ones(n, bool); rt = np.full(n, np.nan); ch = np.zeros(n); v = v0 + v1 * coh
    for i in range(1, int(round((horizon - t) / dt)) + 1):
        idx = np.flatnonzero(alive)
        if not len(idx): break
        x[idx] += (v - g * x[idx]) * dt + np.sqrt(dt) * rng.standard_normal(len(idx))
        hit = np.abs(x[idx]) >= a; done = idx[hit]; rt[done] = t + i * dt; ch[done] = np.sign(x[done]); alive[done] = False
    ok = ~np.isnan(rt); return (rt[ok] - OFF) / K * 1000, ch[ok]

hp = pd.read_csv(ROOT / "output/track_a/headline_pooled.csv"); hp = hp[hp.k == K].set_index("gain"); rng = np.random.default_rng(1)
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5})
fig, axes = plt.subplots(3, 6, figsize=(12, 6.2), sharex=True); bins = np.linspace(0, 750, 38)
for i, gn in enumerate(GAINS):
    d = pd.read_csv(ROOT / "data/processed/fixed_bound" / f"hssm_ready_nxx1_fixed_b1.5_g{gn}.csv"); r = hp.loc[gn]
    for j, c in enumerate(COHS):
        ax = axes[i, j]; dd = d[d.coherence.round(2) == c]
        # network: correct (or "+" at coh 0) filled, error (or "-") step
        if c == 0: sel_c, lab_c, lab_e = dd.response_choice == 1, '"+"', '"-"'
        else: sel_c, lab_c, lab_e = dd.response == 1, "correct", "error"
        # simulate both signs of coherence with equal weight, pooled, so that error/correct definitions match the data
        rts, chs = [], []
        for sgn in ([1] if c == 0 else [1, -1]):
            rt, ch = simulate(r.v_Intercept_mean, r.v_coherence_signed_mean, r.a_mean, r.z_mean, r.g_mean, r.t_fixed, sgn * c, 10000, rng)
            rts.append(rt); chs.append(ch * sgn if c != 0 else ch)
        rt = np.concatenate(rts); ch = np.concatenate(chs); sim_c = ch > 0
        # one normalisation per panel: correct (up) and error (down, negative weights) masses sum to 1 for network and for model
        wc = np.ones(sel_c.sum()) / len(dd); we = -np.ones((~sel_c).sum()) / len(dd)
        ax.hist(dd.rt[sel_c] * 1000, bins=bins, weights=wc, color=COL[gn], alpha=0.35)
        ax.hist(dd.rt[~sel_c] * 1000, bins=bins, weights=we, color=COL[gn], alpha=0.35)
        ax.hist(rt[sim_c], bins=bins, weights=np.ones(sim_c.sum()) / len(rt), histtype="step", color="k", lw=1.2)
        ax.hist(rt[~sim_c], bins=bins, weights=-np.ones((~sim_c).sum()) / len(rt), histtype="step", color="k", lw=1.2)
        ax.axhline(0, color="k", lw=0.5)
        acc_n = sel_c.mean(); acc_m = sim_c.mean()
        ax.set_title(f"gain {gn}, |coh| {c:.2f}\nP({lab_c}) net {acc_n:.2f} / OU {acc_m:.2f}", fontsize=7.5)
        ax.set_yticks([]); ax.set_xlim(0, 750); yl = ax.get_ylim(); ax.set_ylim(-max(abs(yl[0]), 0.35 * yl[1]), yl[1])
        if i == 2: ax.set_xlabel("RT (ms)")
from matplotlib.patches import Patch; from matplotlib.lines import Line2D
fig.legend([Patch(color="gray", alpha=0.35), Line2D([], [], color="k", lw=1.2)],
           ["network (correct up, error down)", "fitted OU (correct up, error down)"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "fig_track_a_rt_ppc.png", dpi=150); print("saved")
