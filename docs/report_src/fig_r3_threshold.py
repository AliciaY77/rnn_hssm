"""Round-3 Figure 1: how the pooled HSSM OU fit changes with the threshold placed on dv (1.5, 2.0, 2.5, 2.94; k = 10).
(a) native leak g, posterior median with 95 % interval (2.5 / 97.5 % quantiles of the 400 saved draws); positive = leaky,
+10 = the likelihood's ceiling at k = 10. (b) median RT of crossing trials and (c) omission fraction: network (filled) vs the
model's 95 % posterior-predictive interval (open; 40 posterior draws, one replicate dataset of the data's size and coherence mix
per draw, track_a/ppc_replicates.py). Writes output/report/r3_threshold_table.csv."""
import pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = pathlib.Path(__file__).resolve().parents[2]; TA = ROOT / "output/track_a"; OUT = ROOT / "output/report"
GAINS = [0.8, 1.0, 1.2]; BOUNDS = [1.5, 2.0, 2.5, 2.94]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; K = 10.0
rows = []
for b in BOUNDS:
    for gn in GAINS:
        g = pd.read_csv(TA / f"g{gn}_k10_b{b}_pooled_draws.csv").g * K
        r = pd.read_csv(TA / f"ppc_replicates_g{gn}_k10_b{b}.csv")
        q = lambda x, p: float(np.quantile(x, p))
        rows.append(dict(bound=b, gain=gn, g=float(np.median(g)), g_lo=q(g, .025), g_hi=q(g, .975),
                         med_net=r.net_median_rt.iloc[0], med_mod=float(r.median_rt.median()), med_lo=q(r.median_rt, .025), med_hi=q(r.median_rt, .975),
                         om_net=100 * r.net_omit.iloc[0], om_mod=100 * float(r.omit.median()), om_lo=100 * q(r.omit, .025), om_hi=100 * q(r.omit, .975),
                         acc_net=r.net_acc.iloc[0], acc_mod=float(r.acc.median()), n_rep=len(r)))
t = pd.DataFrame(rows); t.to_csv(OUT / "r3_threshold_table.csv", index=False); print(t.round(2).to_string(index=False))
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8})
fig, ax = plt.subplots(1, 3, figsize=(11, 3.5)); off = {0.8: -0.035, 1.0: 0.0, 1.2: 0.035}
for gn in GAINS:
    d = t[t.gain == gn]; x = d.bound + off[gn]
    ax[0].errorbar(x, d.g, yerr=[d.g - d.g_lo, d.g_hi - d.g], fmt="-o", color=COL[gn], ms=5, capsize=2.5, label=f"gain {gn}")
    ax[1].plot(x, d.med_net, "-o", color=COL[gn], ms=5)
    ax[1].errorbar(x, d.med_mod, yerr=[d.med_mod - d.med_lo, d.med_hi - d.med_mod], fmt="--s", color=COL[gn], ms=4.5, mfc="w", capsize=2.5)
    ax[2].plot(x, d.om_net, "-o", color=COL[gn], ms=5)
    ax[2].errorbar(x, d.om_mod, yerr=[d.om_mod - d.om_lo, d.om_hi - d.om_mod], fmt="--s", color=COL[gn], ms=4.5, mfc="w", capsize=2.5)
ax[0].axhline(10, color="gray", ls=":", lw=1); ax[0].text(2.05, 10.3, "likelihood ceiling (k = 10)", fontsize=7.5, color="gray")
ax[0].axhline(0, color="gray", lw=0.7); ax[0].set_ylim(-1.5, 11.5); ax[0].set_ylabel("fitted g (per s; > 0 leaky)")
ax[0].set_title("fitted leak (95 % posterior CI)"); ax[0].legend(frameon=False, loc="lower left")
ax[1].set_ylabel("median RT (ms)"); ax[1].set_title("median RT: network vs model (95 % PPI)")
ax[2].set_ylabel("omissions (%)"); ax[2].set_title("omissions: network vs model (95 % PPI)")
for a in (ax[1], ax[2]):
    a.legend([Line2D([], [], color="gray", marker="o"), Line2D([], [], color="gray", ls="--", marker="s", mfc="w")], ["network", "fitted OU"], frameon=False, loc="upper left")
for a in ax: a.set_xlabel("threshold on dv"); a.set_xticks(BOUNDS)
plt.tight_layout(); fig.savefig(OUT / "fig_r3_threshold.png", dpi=160); print("saved")
