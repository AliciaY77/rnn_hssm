"""Bound sweep for Track A: HSSM OU fitted to RT and choice read out from dv with constant bounds 1.5, 2.0, 2.5, 2.94.
Rows = bound, columns = gain. Each panel: RT distribution over all coherences (the data's own coherence mix), correct up and
error down, network (filled) vs the fitted OU at its posterior mean (line), both normalised over crossing trials so that up +
down = 1. The model is simulated in the stretched frame exactly as fitted (dt = 1 ms stretched, bounds +/- a, start a(2z - 1),
fixed t, 0.3 + 10 * 0.75 s horizon, non-crossers dropped and counted as omissions). Also writes bound_sweep_table.csv.
Positive g is leaky."""
import pathlib, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch; from matplotlib.lines import Line2D
ROOT = pathlib.Path(__file__).resolve().parents[2]
TA = ROOT / "output" / "track_a"
OUT = ROOT / "output" / "report"; OUT.mkdir(parents=True, exist_ok=True)
GAINS = [0.8, 1.0, 1.2]; BOUNDS = [1.5, 2.0, 2.5, 2.94]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; K = 10.0; OFF = 0.3; H = OFF + K * 0.75

def simulate(p, cohs, counts, rng, dt=1e-3):
    rts, cor = [], []; om_n = 0; tot = 0
    for c, n in zip(cohs, counts):
        n = int(n); x = np.full(n, p["a"] * (2 * p["z"] - 1)); alive = np.ones(n, bool); rt = np.full(n, np.nan); ch = np.zeros(n)
        v = p["v_Intercept"] + p["v_coherence_signed"] * c
        for i in range(1, int(round((H - p["t"]) / dt)) + 1):
            idx = np.flatnonzero(alive)
            if not len(idx): break
            x[idx] += (v - p["g"] * x[idx]) * dt + np.sqrt(dt) * rng.standard_normal(len(idx))
            hit = np.abs(x[idx]) >= p["a"]; d = idx[hit]; rt[d] = p["t"] + i * dt; ch[d] = np.sign(x[d]); alive[d] = False
        ok = ~np.isnan(rt); om_n += (~ok).sum(); tot += n
        rts.append((rt[ok] - OFF) / K * 1000)
        cor.append(ch[ok] > 0 if c > 0 else (ch[ok] < 0 if c < 0 else ch[ok] > 0))   # coh 0: "+" counts as "correct" side
    return np.concatenate(rts), np.concatenate(cor), om_n / tot

omis = pd.read_csv(ROOT / "output/fixed_bound/omissions.csv"); omis = omis[omis.coherence == "all"]
rng = np.random.default_rng(3); rows = []
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
fig, axes = plt.subplots(len(BOUNDS), 3, figsize=(9.5, 9.2), sharex=True); bins = np.linspace(0, 750, 51)
for i, b in enumerate(BOUNDS):
    for j, gn in enumerate(GAINS):
        ax = axes[i, j]; tag = f"g{gn}_k10_b{b}_pooled"
        dfile, sfile, mfile = TA / f"{tag}_draws.csv", TA / f"{tag}_summary.csv", TA / f"{tag}_meta.json"
        d = pd.read_csv(ROOT / "data/processed/fixed_bound" / f"hssm_ready_nxx1_fixed_b{b}_g{gn}.csv")
        om_net = float(omis[(omis.bound == b) & (omis.gain == gn)].omit.iloc[0])
        cor_net = np.where(d.coherence_signed.round(2) == 0, d.response_choice == 1, d.response == 1)
        ax.hist(d.rt[cor_net] * 1000, bins=bins, weights=np.ones(cor_net.sum()) / len(d), color=COL[gn], alpha=0.35)
        ax.hist(d.rt[~cor_net] * 1000, bins=bins, weights=-np.ones((~cor_net).sum()) / len(d), color=COL[gn], alpha=0.35)
        if not dfile.exists():
            ax.text(0.5, 0.5, "fit not available", transform=ax.transAxes, ha="center"); continue
        dr = pd.read_csv(dfile); p = dr.mean(numeric_only=True).to_dict(); meta = json.loads(mfile.read_text()); summ = pd.read_csv(sfile).set_index("param")
        cc = d.coherence_signed.round(2).value_counts().sort_index(); scale = 25000 / cc.sum()
        rt, cor, om_mod = simulate(p, cc.index.values, np.maximum((cc.values * scale).round(), 50), rng)
        ax.hist(rt[cor], bins=bins, weights=np.ones(cor.sum()) / len(rt), histtype="step", color="k", lw=1.1)
        ax.hist(rt[~cor], bins=bins, weights=-np.ones((~cor).sum()) / len(rt), histtype="step", color="k", lw=1.1)
        ax.axhline(0, color="k", lw=0.4); ax.set_yticks([])
        gq = dr["g"] * K; lo, hi = np.quantile(gq, [0.03, 0.97]); edge = float((dr["g"] > 1 - 0.02).mean())
        ax.set_title(f"bound {b}, gain {gn}: g = {gq.mean():+.1f} [{lo:+.1f}, {hi:+.1f}] /s", fontsize=8)
        ax.text(0.97, 0.93, f"median {np.median(d.rt)*1000:.0f} / {np.median(rt):.0f} ms\nomit {100*om_net:.1f} / {100*om_mod:.1f} %\nP(correct) {cor_net.mean():.2f} / {cor.mean():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
        rows.append(dict(bound=b, gain=gn, n=meta.get("n"), omit_net=om_net, omit_model=om_mod, g_native=gq.mean(), g_lo=lo, g_hi=hi, edge_mass_g=edge,
                         a_native=float(dr["a"].mean() / np.sqrt(K)), v_native_015=float((dr["v_Intercept"] + 0.15 * dr["v_coherence_signed"]).mean() * np.sqrt(K)),
                         r_hat_max=meta.get("r_hat_max"), ess_bulk_min=meta.get("ess_bulk_min"), divergences=meta.get("n_divergences"),
                         median_rt_net=float(np.median(d.rt) * 1000), median_rt_model=float(np.median(rt)), pc_net=float(cor_net.mean()), pc_model=float(cor.mean())))
    axes[i, 0].set_ylabel(f"bound {b}\ncorrect up / error down")
for ax in axes[-1]: ax.set_xlabel("RT (ms)")
fig.legend([Patch(color="gray", alpha=0.35), Line2D([], [], color="k", lw=1.1)], ["network", "fitted OU (posterior mean)"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005))
plt.tight_layout(rect=(0, 0.025, 1, 1)); fig.savefig(OUT / "fig_bound_sweep.png", dpi=150)
tab = pd.DataFrame(rows); tab.to_csv(OUT / "bound_sweep_table.csv", index=False); print(tab.round(3).to_string(index=False))
