"""Round-3 figure: how the HSSM OU fit changes with the threshold placed on dv (bounds 1.5, 2.0, 2.5, 2.94; k = 10, pooled).
(a) native leak g with 94 % HDI (positive = leaky; +10 = the likelihood's ceiling at k = 10); (b) median absolute RT-quantile
error on correct trials (10/30/50/70/90 %, over all |coherence|), from the per-coherence posterior-predictive tables;
(c) omission fraction, network vs model (200-draw posterior predictive, averaged over the 11 signed coherences). Writes output/report/r3_threshold_table.csv."""
import pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = pathlib.Path(__file__).resolve().parents[2]; TA = ROOT / "output/track_a"; OUT = ROOT / "output/report"
GAINS = [0.8, 1.0, 1.2]; BOUNDS = [1.5, 2.0, 2.5, 2.94]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}
sw = pd.read_csv(OUT / "bound_sweep_table.csv")
rows = []
for b in BOUNDS:
    for gn in GAINS:
        f = TA / (f"ppc_rt_g{gn}_k10_b{b}_pooled.csv" if b == 1.5 else f"ppc_rt_g{gn}_k10_b{b}_pooled_b{b}.csv")
        d = pd.read_csv(f); qc = d[[c for c in d.columns if c.startswith("qc") and c.endswith("_err")]].abs().values.ravel()
        qe = d[[c for c in d.columns if c.startswith("qe") and c.endswith("_err")]].abs().values.ravel()
        r = sw[(sw.bound == b) & (sw.gain == gn)].iloc[0]
        w = (d.abs_coherence > 0).map({True: 2.0, False: 1.0})            # 11 signed coherences: 0 once, each |coh| > 0 twice
        omit_model = float((w * d.omit_sim).sum() / w.sum())              # = ppc_rt_summary omit_sim (200 posterior draws)
        rows.append(dict(bound=b, gain=gn, g=r.g_native, g_lo=r.g_lo, g_hi=r.g_hi, omit_net=r.omit_net, omit_model=omit_model,
                         qerr_correct=float(np.nanmedian(qc)), qerr_error=float(np.nanmedian(qe))))
t = pd.DataFrame(rows); t.to_csv(OUT / "r3_threshold_table.csv", index=False); print(t.round(3).to_string(index=False))
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4)); off = {0.8: -0.03, 1.0: 0.0, 1.2: 0.03}
for gn in GAINS:
    d = t[t.gain == gn]; x = d.bound + off[gn]
    ax[0].errorbar(x, d.g, yerr=[d.g - d.g_lo, d.g_hi - d.g], fmt="-o", color=COL[gn], ms=5, capsize=2, label=f"gain {gn}")
    ax[1].plot(d.bound, d.qerr_correct, "-o", color=COL[gn], ms=5)
    ax[2].plot(d.bound, 100 * d.omit_net, "-o", color=COL[gn], ms=5); ax[2].plot(d.bound, 100 * d.omit_model, "--s", color=COL[gn], ms=4, mfc="w")
ax[0].axhline(10, color="gray", ls=":", lw=1); ax[0].text(2.05, 10.3, "likelihood ceiling (k = 10)", fontsize=7.5, color="gray")
ax[0].axhline(0, color="gray", lw=0.7); ax[0].set_ylim(-1.5, 11.5)
ax[0].set_ylabel("fitted g (per s; > 0 leaky)"); ax[0].set_title("fitted leak: positive everywhere,\nfalls and orders by gain"); ax[0].legend(frameon=False, loc="lower left")
ax[1].set_ylabel("RT-quantile error, correct (ms)"); ax[1].set_title("RT misfit grows for the leaky network")
ax[2].set_ylabel("omissions (%)"); ax[2].set_title("omissions: network (filled) vs model (open)")
ax[2].legend([Line2D([], [], color="gray", marker="o"), Line2D([], [], color="gray", ls="--", marker="s", mfc="w")], ["network", "fitted OU"], frameon=False, loc="upper left")
for a in ax: a.set_xlabel("threshold on dv"); a.set_xticks(BOUNDS)
plt.tight_layout(); fig.savefig(OUT / "fig_r3_threshold.png", dpi=160); print("saved")
