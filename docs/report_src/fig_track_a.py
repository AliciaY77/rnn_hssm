"""Report figure for Track A: RT distributions reproduced (top), psychophysical kernels not (bottom).
Top: per gain, network RT histogram (constant bound 1.5, all coherences, crossers only) vs the HSSM-fitted OU
(pooled posterior mean at k = 10) simulated with an Euler-Maruyama integrator in the stretched frame (dt = 1 ms),
same 750 ms native horizon, non-crossers dropped. Bottom: network kernels vs the fitted OU driven by the real
evidence streams, from output/track_a/kernel_ppc.csv (variant a = as fitted with bound, matched diffusion;
variant b = leak only, no bound). g > 0 leaky."""
import pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[2]
TA = ROOT.parent / "rnn_hssm_track_a" / "output" / "track_a"     # track worktree until merged
if not TA.exists(): TA = ROOT / "output" / "track_a"
OUT = ROOT / "output" / "report"; OUT.mkdir(parents=True, exist_ok=True)
GAINS = [0.8, 1.0, 1.2]; COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}; K = 10.0; OFF = 0.3

def simulate(v0, v1, a, z, g, t, cohs, n_per, rng, dt=1e-3, horizon=OFF + K * 0.75):
    """Stretched-frame OU with absorbing bounds at +/-a, start a(2z-1); returns native RT (s) and choice(+1/-1)."""
    rts, chs = [], []
    for c in cohs:
        x = np.full(n_per, a * (2 * z - 1), np.float64); alive = np.ones(n_per, bool); rt = np.full(n_per, np.nan); ch = np.zeros(n_per)
        v = v0 + v1 * c; nsteps = int(round((horizon - t) / dt))
        for i in range(1, nsteps + 1):
            idx = np.flatnonzero(alive)
            if not len(idx): break
            x[idx] += (v - g * x[idx]) * dt + np.sqrt(dt) * rng.standard_normal(len(idx))
            hit = np.abs(x[idx]) >= a
            done = idx[hit]; rt[done] = t + i * dt; ch[done] = np.sign(x[done]); alive[done] = False
        ok = ~np.isnan(rt); rts.append((rt[ok] - OFF) / K); chs.append(ch[ok] * np.sign(c) if c != 0 else ch[ok])
    return np.concatenate(rts), np.concatenate(chs)

hp = pd.read_csv(TA / "headline_pooled.csv"); hp = hp[hp.k == K].set_index("gain")
kp = pd.read_csv(TA / "kernel_ppc.csv"); kp = kp[kp.which == "posterior_mean"]
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
bins = np.linspace(0, 750, 51)
for j, gn in enumerate(GAINS):
    d = pd.read_csv(ROOT / "data/processed/fixed_bound" / f"hssm_ready_nxx1_fixed_b1.5_g{gn}.csv")
    cohs = sorted(d.coherence_signed.round(2).unique()); r = hp.loc[gn]
    rt_sim, _ = simulate(r.v_Intercept_mean, r.v_coherence_signed_mean, r.a_mean, r.z_mean, r.g_mean, r.t_fixed, cohs, 4000, rng)
    ax = axes[0, j]
    ax.hist(d.rt * 1000, bins=bins, density=True, color=COL[gn], alpha=0.35, label="network")
    ax.hist(rt_sim * 1000, bins=bins, density=True, histtype="step", color="k", lw=1.3, label="fitted OU")
    ax.set_title(f"gain {gn}:  g = {r.g_nat_mean:+.1f} /s, a = {r.a_nat_mean:.2f} (native)", fontsize=9)
    ax.set_xlabel("RT (ms)"); ax.set_xlim(0, 750)
    if j == 0: ax.set_ylabel("density"); ax.legend(fontsize=8, frameon=False)
    ax.text(0.97, 0.9, f"median {np.median(d.rt)*1000:.0f} vs {np.median(rt_sim)*1000:.0f} ms", transform=ax.transAxes, ha="right", fontsize=8)
xb = (np.arange(8) + 0.5) * 750 / 8
for j, (var, title) in enumerate([("a_fitted_matched", "fitted OU with its bound (a ≈ 0.3): primacy at every gain"), ("b_leakonly_matched", "fitted leak alone, no bound: recency at every gain")]):
    ax = axes[1, j]
    for gn in GAINS:
        row = kp[(kp.gain == gn) & (kp.variant == var)].iloc[0]
        ax.plot(xb, [row[f"wnet{i}"] for i in range(8)], "-o", color=COL[gn], ms=4)
        ax.plot(xb, [row[f"wsim{i}"] for i in range(8)], "--s", color=COL[gn], ms=4, alpha=0.8)
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel("time within trial (ms)"); ax.set_title(title, fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color="gray", marker="o", ms=4), Line2D([], [], color="gray", ls="--", marker="s", ms=4)] + [Line2D([], [], color=COL[g], lw=3) for g in GAINS],
              ["network", "fitted OU"] + [f"gain {g}" for g in GAINS], fontsize=7, frameon=False, loc="upper center", ncol=5, columnspacing=0.8, handlelength=1.5)
    if j == 0: ax.set_ylabel("weight on choice (L1-normalised)")
ax = axes[1, 2]
net = [kp[(kp.gain == gn) & (kp.variant == "a_fitted_matched")].slope_net.iloc[0] for gn in GAINS]
for var, lab, mk in [("a_fitted_matched", "fitted OU with bound", "s"), ("b_leakonly_matched", "leak only", "^")]:
    ax.plot(GAINS, [kp[(kp.gain == gn) & (kp.variant == var)].slope_sim.iloc[0] for gn in GAINS], "--" + mk, color="k", mfc="w" if mk == "^" else "k", label=lab)
ax.plot(GAINS, net, "-o", color="C2", label="network"); ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("gain"); ax.set_ylabel("kernel slope"); ax.set_title("kernel slope against gain", fontsize=9); ax.legend(fontsize=8, frameon=False, loc="center right"); ax.set_xticks(GAINS)
plt.tight_layout(); fig.savefig(OUT / "fig_track_a.png", dpi=160); print("saved", OUT / "fig_track_a.png")
