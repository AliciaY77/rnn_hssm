"""
Checks and figures for Track B (issue #4, step 6).

  --what kernels   simulate choices from the fitted parameters on the REAL evidence streams (fresh noise)
                   and compare the psychophysical kernel and psychometric curve to the network's, per gain
                   (pooled-fit parameters) and per network.  `kernel` is imported verbatim from
                   kernel_fit/fit_ou_kernel.py so the weights are comparable with Fig 1F.
  --what gfig      per-network g with intervals against gain, for every route/variant, with the landscape
                   values (+4.3 / -1.6 / -6.2 per s, approximate, 3 networks) and the Fig 1F zero crossing
                   (f = 1.016) marked;  and g vs the network's own kernel slope with error bars.
  --what recovery  recovery figure (route 1 and route 2).
  --what tables    the summary tables that go into RESULTS.md and the issue comment.
  --what all

Outputs: output/track_b/*.png, *.csv
"""
import argparse, json, pathlib, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernel_fit")); sys.path.insert(0, str(ROOT / "track_b"))
from fit_ou_kernel import kernel, psychometric, N_BINS  # noqa: E402  (verbatim, as required)
from ou_lik import GAINS, load_network, simulate_bounded, analytic_p  # noqa: E402

OUT = ROOT / "output" / "track_b"
SEEDS = list(range(42, 62))
G_LANDSCAPE = {0.8: 4.3, 1.0: -1.6, 1.2: -6.2}
COL = {0.8: "C0", 1.0: "k", 1.2: "C3"}


def sim_choice(rel, g, v, B, a_bias, seed):
    aT, _ = simulate_bounded(rel, g, v, B, a_bias, M=1, seed=seed)
    return (aT[0] > 0).astype(int)


def load_bounded(dirname="bounded"):
    rows = []
    for f in sorted((OUT / dirname).glob("*.json")):
        j = json.loads(f.read_text())
        j.pop("profile", None)
        rows.append(j)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ------------------------------------------------------------------ kernel / psychometric
def what_kernels(n_trials=None):
    mle = pd.read_csv(OUT / "analytic_per_network_mle.csv")
    pooled = pd.read_csv(OUT / "analytic_pooled_mle.csv")
    bnd = load_bounded()
    rows, fits = [], {}
    for gain in GAINS:
        rel_all, ch_all, coh_all, ch_r1, ch_r2 = [], [], [], [], []
        pr = pooled[pooled.gain == gain].iloc[0]
        for seed in SEEDS:
            d = load_network(seed, gain=gain, n_trials=n_trials)
            rel, y, coh = d["rel"], d[f"choice_g{gain}"], d["coh"]
            f1 = mle[(mle.seed == seed) & (mle.gain == gain)].iloc[0]
            c1 = sim_choice(rel, f1.g, f1.v, np.inf, f1.a_bias, seed=50_000 + seed)
            w_net, s_net = kernel(rel, y); w1, s1 = kernel(rel, c1)
            row = dict(seed=seed, gain=gain, slope_net=s_net, slope_route1=s1,
                       acc_net=float((y == (coh > 0)).mean()), acc_route1=float((c1 == (coh > 0)).mean()),
                       **{f"wnet{i}": w_net[i] for i in range(N_BINS)},
                       **{f"w1_{i}": w1[i] for i in range(N_BINS)})
            b = bnd[(bnd.seed == seed) & (bnd.gain == gain) & (bnd.hit_mode == "cross")] if len(bnd) else []
            if len(b):
                b = b.iloc[0]
                c2 = sim_choice(rel, b.g, b.v, b.B, b.a_bias, seed=60_000 + seed)
                w2, s2 = kernel(rel, c2)
                row.update(slope_route2=s2, acc_route2=float((c2 == (coh > 0)).mean()),
                           **{f"w2_{i}": w2[i] for i in range(N_BINS)})
                ch_r2.append(c2)
            rows.append(row)
            rel_all.append(rel); ch_all.append(y); coh_all.append(coh); ch_r1.append(c1)
        rel_all = np.concatenate(rel_all); ch_all = np.concatenate(ch_all)
        coh_all = np.concatenate(coh_all); ch_r1 = np.concatenate(ch_r1)
        # pooled-fit parameters on all 40 000 trials
        ch_p = sim_choice(rel_all, pr.g, pr.v, np.inf, pr.a_bias, seed=70_000 + int(gain * 10))
        w_net, s_net = kernel(rel_all, ch_all)
        w_p, s_p = kernel(rel_all, ch_p)
        w_pn, s_pn = kernel(rel_all, ch_r1)
        lv, P_net = psychometric(coh_all, ch_all); _, P_p = psychometric(coh_all, ch_p)
        fits[gain] = dict(w_net=w_net, w_sim=w_p, w_pernet=w_pn, lv=lv, P_net=P_net, P_sim=P_p,
                          s_net=s_net, s_sim=s_p, s_pernet=s_pn,
                          acc_net=float((ch_all == (coh_all > 0)).mean()),
                          acc_sim=float((ch_p == (coh_all > 0)).mean()))
        if ch_r2:
            ch_r2 = np.concatenate(ch_r2)
            w2, s2 = kernel(rel_all, ch_r2); _, P2 = psychometric(coh_all, ch_r2)
            fits[gain].update(w_r2=w2, s_r2=s2, P_r2=P2,
                              acc_r2=float((ch_r2 == (coh_all > 0)).mean()))
        print(f"gain {gain}: pooled kernel slope network {s_net:+.4f} | route-1 pooled fit {s_p:+.4f} | "
              f"route-1 per-network {s_pn:+.4f}" + (f" | route-2 per-network {fits[gain]['s_r2']:+.4f}"
                                                   if "s_r2" in fits[gain] else "") +
              f" | acc network {fits[gain]['acc_net']:.3f} model {fits[gain]['acc_sim']:.3f}", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "kernel_reproduction_per_network.csv", index=False)
    pd.DataFrame([dict(gain=g, **{k: v for k, v in f.items() if np.ndim(v) == 0}) for g, f in fits.items()]
                 ).to_csv(OUT / "kernel_reproduction_pooled.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    xb = (np.arange(N_BINS) + 0.5) * 750 / N_BINS
    for gain, f in fits.items():
        axes[0].plot(xb, f["w_net"], "-o", color=COL[gain], label=f"network gain {gain}")
        axes[0].plot(xb, f["w_sim"], "--s", color=COL[gain], alpha=.7, label=f"OU route 1 gain {gain}")
        if "w_r2" in f:
            axes[0].plot(xb, f["w_r2"], ":^", color=COL[gain], alpha=.7, label=f"OU route 2 gain {gain}")
        axes[1].plot(f["lv"], f["P_net"], "-o", color=COL[gain])
        axes[1].plot(f["lv"], f["P_sim"], "--s", color=COL[gain], alpha=.7)
    axes[0].set_xlabel("time within trial (ms)"); axes[0].set_ylabel("influence on choice (L1-normalised)")
    axes[0].legend(fontsize=6); axes[0].set_title("kernel: network (solid) vs fitted OU (dashed)")
    axes[1].set_xlabel("signed relevant coherence"); axes[1].set_ylabel("P(choice = +)")
    axes[1].set_title("psychometric (40 000 trials per gain)")
    for gain, d in tab.groupby("gain"):
        axes[2].plot(d.slope_net, d.slope_route1, "o", color=COL[gain], ms=4, alpha=.7,
                     label=f"gain {gain} (route 1)")
        if "slope_route2" in d:
            axes[2].plot(d.slope_net, d.slope_route2, "^", color=COL[gain], ms=4, alpha=.4, mfc="none",
                         label=f"gain {gain} (route 2)")
    lim = [tab.slope_net.min() - .01, tab.slope_net.max() + .01]
    axes[2].plot(lim, lim, "-", color="gray", lw=.8); axes[2].set_xlabel("network kernel slope")
    axes[2].set_ylabel("model kernel slope"); axes[2].legend(fontsize=6)
    axes[2].set_title("kernel slope, per network")
    plt.tight_layout(); fig.savefig(OUT / "kernel_reproduction.png", dpi=140)
    print("saved", OUT / "kernel_reproduction.png")
    for gain, d in tab.groupby("gain"):
        cols = ["slope_route1"] + (["slope_route2"] if "slope_route2" in d else [])
        for c in cols:
            ok = d[[c, "slope_net"]].dropna()
            sl, ic = np.polyfit(ok.slope_net, ok[c], 1)
            r = np.corrcoef(ok.slope_net, ok[c])[0, 1]
            print(f"gain {gain} {c}: model = {sl:.3f} x network {ic:+.4f}, r = {r:.3f}, "
                  f"mean model {ok[c].mean():+.4f} vs network {ok.slope_net.mean():+.4f} (n = {len(ok)})")
    return tab, fits


# ------------------------------------------------------------------------------ g figures
def what_gfig():
    m = pd.read_csv(OUT / "analytic_per_network_mle.csv")
    try:
        b = pd.read_csv(OUT / "analytic_per_network_bayes.csv")
    except FileNotFoundError:
        b = None
    bnd = load_bounded()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    off = {"route 1 MLE": -0.035, "route 1 Bayes": -0.012, "route 2 (no hit term)": 0.012,
           "route 2 (hit term)": 0.035}
    series = [("route 1 MLE", m, "g", "g_lo", "g_hi", "C0")]
    if b is not None:
        series.append(("route 1 Bayes", b, "g_mean", "g_hdi_lo", "g_hdi_hi", "C1"))
    if len(bnd):
        for hm, lab, c in (("none", "route 2 (no hit term)", "C2"), ("cross", "route 2 (hit term)", "C3")):
            d = bnd[bnd.hit_mode == hm]
            if len(d):
                series.append((lab, d, "g", "g_lo", "g_hi", c))
    for lab, d, gc, lo, hi, c in series:
        for gain, dd in d.groupby("gain"):
            x = np.full(len(dd), gain) + off[lab] + np.linspace(-0.008, 0.008, len(dd))
            ax.errorbar(x, dd[gc], yerr=[np.maximum(dd[gc] - dd[lo], 0), np.maximum(dd[hi] - dd[gc], 0)],
                        fmt="o", ms=3, lw=.7, color=c, alpha=.7,
                        label=lab if gain == 0.8 else None)
    for gain, gl in G_LANDSCAPE.items():
        ax.plot([gain - 0.05, gain + 0.05], [gl, gl], "-", color="gray", lw=2, alpha=.8,
                label="landscape (approx.)" if gain == 0.8 else None)
    ax.axhline(0, color="gray", lw=.5); ax.axvline(1.016, color="gray", ls=":", lw=1)
    ax.text(1.016, ax.get_ylim()[1], " Fig 1F zero crossing f = 1.016", fontsize=6, va="top")
    ax.set_xlabel("gain f"); ax.set_ylabel("OU leak g (per s; > 0 leaky / recency)")
    ax.set_title("per-network g with intervals"); ax.legend(fontsize=6)

    ax = axes[1]
    for gain, dd in m.groupby("gain"):
        ax.errorbar(dd.g, np.arange(len(dd)) + {0.8: 0, 1.0: 0.25, 1.2: 0.5}[gain],
                    xerr=1.96 * dd.g_se, fmt="o", ms=3, lw=.7, color=COL[gain], label=f"gain {gain}")
    ax.axvline(0, color="gray", lw=.5); ax.set_yticks(range(len(m[m.gain == 0.8])))
    ax.set_yticklabels(m[m.gain == 0.8].seed.values, fontsize=6)
    ax.set_xlabel("g (per s)"); ax.set_ylabel("network seed"); ax.legend(fontsize=6)
    ax.set_title("route 1 MLE, 95 % Wald")

    ax = axes[2]
    kn = pd.read_csv(ROOT / "output" / "kernel_fit" / "network_kernels.csv")
    j = m.merge(kn[["seed", "gain", "slope"]], on=["seed", "gain"])
    for gain, dd in j.groupby("gain"):
        ax.errorbar(dd.slope, dd.g, yerr=1.96 * dd.g_se, fmt="o", ms=3, lw=.7, color=COL[gain],
                    label=f"gain {gain}")
    sl, ic = np.polyfit(j.slope, j.g, 1)
    xs = np.linspace(j.slope.min(), j.slope.max(), 10)
    ax.plot(xs, sl * xs + ic, "-", color="gray", lw=1)
    r = np.corrcoef(j.slope, j.g)[0, 1]
    ax.axhline(0, color="gray", lw=.5); ax.axvline(0, color="gray", lw=.5)
    ax.set_xlabel("network kernel slope (Fig 1F measure)"); ax.set_ylabel("fitted g (per s)")
    ax.set_title(f"g vs kernel slope: g = {sl:.1f} x slope {ic:+.2f}, r = {r:.3f}")
    ax.legend(fontsize=6)
    plt.tight_layout(); fig.savefig(OUT / "g_by_gain.png", dpi=140)
    print("saved", OUT / "g_by_gain.png")
    print(f"g vs network kernel slope (route 1 MLE, all 60 fits): slope {sl:.2f}, intercept {ic:+.3f}, r = {r:.3f}")
    for gain, dd in j.groupby("gain"):
        rr = np.corrcoef(dd.slope, dd.g)[0, 1]
        print(f"  gain {gain}: r = {rr:.3f} (n = {len(dd)})")
    return j


# ------------------------------------------------------------------------------- recovery
def what_recovery():
    r1 = pd.read_csv(OUT / "recovery_route1.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, kind in zip(axes, ["fitted", "theory"]):
        d = r1[r1.kind == kind]
        for gain, dd in d.groupby("gain"):
            ax.errorbar(dd.g_true, dd.g, yerr=1.96 * dd.g_se, fmt="o", ms=4, lw=.8, color=COL[gain],
                        alpha=.8, label=f"gain {gain}")
        lim = [min(d.g_true.min(), d.g.min()) - 1, max(d.g_true.max(), d.g.max()) + 1]
        ax.plot(lim, lim, "-", color="gray", lw=.8); ax.axhline(0, color="gray", lw=.4)
        ax.axvline(0, color="gray", lw=.4)
        ax.set_xlabel("true g (per s)"); ax.set_ylabel("recovered g (per s)")
        ax.set_title(f"route 1, truth = {kind} ({int(d.sign_ok.sum())}/{len(d)} signs, "
                     f"{int(d.covered.sum())}/{len(d)} covered)")
        ax.legend(fontsize=6)
    plt.tight_layout(); fig.savefig(OUT / "recovery.png", dpi=140)
    print("saved", OUT / "recovery.png")
    print(r1.groupby(["kind", "gain"]).agg(n=("g", "size"), g_true=("g_true", "mean"),
                                           g_mean=("g", "mean"), bias=("g", "mean"),
                                           sign_ok=("sign_ok", "mean"),
                                           covered=("covered", "mean")).round(3).to_string())
    f2 = OUT / "recovery_route2.csv"
    if f2.exists():
        print("\nroute-2 recovery:")
        print(pd.read_csv(f2).round(3).to_string(index=False))


# --------------------------------------------------------------------------------- tables
def _summary(d, gc, lo, hi, label, rows):
    for gain, dd in d.groupby("gain"):
        pred = "g > 0" if gain < 0.9 else ("g < 0" if gain > 1.1 else "either")
        q1, med, q3 = np.percentile(dd[gc], [25, 50, 75])
        excl = ((dd[lo] > 0) if gain < 0.9 else (dd[hi] < 0) if gain > 1.1
                else ((dd[lo] > 0) | (dd[hi] < 0)))
        rows.append(dict(route=label, gain=gain, n=len(dd), median_g=med, iqr_lo=q1, iqr_hi=q3,
                         frac_sign=float((dd[gc] > 0).mean() if gain < 0.9 else (dd[gc] < 0).mean()),
                         n_excl_zero=int(excl.sum()), predicted=pred,
                         median_B=float(dd.B.median()) if "B" in dd else np.nan,
                         median_v=float(dd[gc.replace("g", "v")].median()) if gc.replace("g", "v") in dd else np.nan))


def what_tables():
    rows = []
    m = pd.read_csv(OUT / "analytic_per_network_mle.csv")
    _summary(m, "g", "g_lo", "g_hi", "route 1 MLE", rows)
    try:
        b = pd.read_csv(OUT / "analytic_per_network_bayes.csv")
        _summary(b, "g_mean", "g_hdi_lo", "g_hdi_hi", "route 1 Bayes", rows)
    except FileNotFoundError:
        pass
    bnd = load_bounded()
    for hm, lab in (("none", "route 2 (no hit term)"), ("cross", "route 2 (hit term)"),
                    ("bern", "route 2 (Bernoulli hit)")):
        d = bnd[bnd.hit_mode == hm] if len(bnd) else []
        if len(d):
            _summary(d, "g", "g_lo", "g_hi", lab, rows)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "summary_by_route.csv", index=False)
    print(tab.round(3).to_string(index=False))
    if len(bnd):
        print("\ndegenerate solutions (g >= 8 and B < 2):")
        deg = bnd[(bnd.g >= 8) & (bnd.B < 2)]
        print(deg[["seed", "gain", "hit_mode", "g", "B", "v", "nll", "model_hit_frac",
                   "net_hit_frac"]].round(3).to_string(index=False) if len(deg) else "  none")
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["kernels", "gfig", "recovery", "tables", "all"], default="all")
    ap.add_argument("--n-trials", type=int, default=None)
    a = ap.parse_args()
    if a.what in ("kernels", "all"):
        what_kernels(a.n_trials)
    if a.what in ("gfig", "all"):
        what_gfig()
    if a.what in ("recovery", "all"):
        what_recovery()
    if a.what in ("tables", "all"):
        what_tables()


if __name__ == "__main__":
    main()
