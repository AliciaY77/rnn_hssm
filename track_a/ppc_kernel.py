"""
Track A, the decisive posterior-predictive check (issue #3): does the HSSM-fitted OU reproduce the
networks' psychophysical kernel, and in particular its ORDERING across gain (recency -> flat -> primacy)?

The HSSM fit is on RT/choice marginals only. Here the fitted parameters are converted to native time and
the SAME OU is driven by the networks' own per-trial evidence streams (data/processed/kernel/seed*.npz,
field `rel`: (2000, 750) per network, per-step mean = signed coherence, per-step sd = 1), exactly as
kernel_fit/fit_ou_kernel.py::simulate does. The kernel is computed with kernel_fit/fit_ou_kernel.py::kernel
verbatim (8 bins, logistic regression on z-scored bin means, L1-normalised weights, slope = linear fit).

Native-unit conversion of a fit at stretch k (RT x k <=> g -> g/k, a -> a*sqrt(k), v -> v/sqrt(k)):
    g_nat  = g * k
    a_nat  = a / sqrt(k)
    x0_nat = a_nat * (2 z - 1)                     (HSSM z is the RELATIVE start point; z = 0.5 unbiased)
    v_nat(coh) = (v_Intercept + v_coherence_signed * coh) * sqrt(k)
    t_nat  = (t - 0.3) / k                         (accumulation starts after t_nat)

EVIDENCE MAPPING (documented as the issue requires). The fit's drift is linear in signed coherence:
v_nat(coh) = v0_nat + v1_nat * coh with v0_nat = v_Intercept*sqrt(k), v1_nat = v_coherence_signed*sqrt(k).
The evidence stream has E[e_t] = coh and sd(e_t) = 1, so substituting e_t for its mean reproduces the
fitted mean drift exactly:
    dx = (v0_nat + v1_nat * e_t - g_nat * x) dt + sigma * dW.
This adds evidence-driven noise of per-step variance (v1_nat * dt)^2 on top of the fitted intrinsic
diffusion. Two treatments are reported:
    dW1  sigma = 1                          (the fitted diffusion; total per-step variance > dt)
    dWm  sigma^2 = 1 - v1_nat^2 * dt        (matched: total per-step variance = dt exactly)

Variants:
    (a) "as fitted"  : sticky bound at +/- a_nat, start x0_nat, choice = sign(x_T)   [sigma = 1 and matched]
    (b) "leak only"  : no bound, same g_nat / v_nat, choice = sign(x_T)              [sigma = 1 and matched]

Outputs -> output/track_a/kernel_ppc.csv, kernel_ppc_draws.csv, kernel_ppc.png
Usage:  /opt/homebrew/anaconda3/bin/python track_a/ppc_kernel.py --k 10 --n-draws 20
"""
import argparse
import glob
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernel_fit"))
from fit_ou_kernel import kernel, psychometric, N_BINS, DT  # verbatim kernel definition

KDATA = ROOT / "data" / "processed" / "kernel"
OUT = ROOT / "output" / "track_a"
NETK = ROOT / "output" / "kernel_fit" / "network_kernels.csv"


def simulate_evidence_ou(rel, noise, v0, v1, g, a=None, x0=0.0, sigma=1.0, t0_steps=0):
    """OU driven by the real evidence stream. Returns (choice 0/1, fraction that hit the bound).

    dx = (v0 + v1*e_t - g*x) dt + sigma*sqrt(dt)*xi ;  sticky bound at +/- a (a=None -> no bound).
    choice = 1 iff x_T > 0 (the networks' deadline rule).
    """
    n, T = rel.shape
    x = np.full(n, np.float32(x0))
    frozen = np.zeros(n, bool)
    sq = np.float32(sigma * np.sqrt(DT))
    v0 = np.float32(v0); v1 = np.float32(v1); g = np.float32(g); dt = np.float32(DT)
    for t in range(t0_steps, T):
        dx = (v0 + v1 * rel[:, t] - g * x) * dt + sq * noise[:, t]
        x = np.where(frozen, x, x + dx)
        if a is not None:
            frozen |= np.abs(x) >= a
    return (x > 0).astype(int), float(frozen.mean()) if a is not None else 0.0


def load_gain(gain):
    files = sorted(glob.glob(str(KDATA / "seed*.npz")))
    rel = np.concatenate([np.load(f)["rel"].astype(np.float32) for f in files])
    coh = np.concatenate([np.load(f)["coh_signed"] for f in files])
    ch = np.concatenate([np.load(f)[f"choice_g{gain}"] for f in files]).astype(int)
    seeds = np.concatenate([np.full(len(np.load(f)["coh_signed"]), int(pathlib.Path(f).stem[4:]))
                            for f in files])
    return rel, coh, ch, seeds, len(files)


def native(row, k):
    """(v0_nat, v1_nat, g_nat, a_nat, x0_nat, t_nat) from one posterior draw at stretch k."""
    sk = np.sqrt(k)
    v0 = float(row["v_Intercept"]) * sk
    v1 = float(row["v_coherence_signed"]) * sk
    g = float(row["g"]) * k
    a = float(row["a"]) / sk
    x0 = a * (2 * float(row["z"]) - 1)
    t_nat = (float(row["t"]) - 0.3) / k
    return v0, v1, g, a, x0, t_nat


VARIANTS = [
    ("a_fitted_sigma1", True, False),
    ("a_fitted_matched", True, True),
    ("b_leakonly_sigma1", False, False),
    ("b_leakonly_matched", False, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gains", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--draws-glob", type=str, default="g{gain}_k{k:g}_b1.5_pooled*_draws.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-tag", type=str, default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_{a.out_tag}" if a.out_tag else ""
    netk = pd.read_csv(NETK)
    rows, draw_rows, curves = [], [], {}

    for gn in a.gains:
        t0 = time.time()
        pat = a.draws_glob.format(gain=gn, k=a.k)
        hits = sorted(OUT.glob(pat))
        if not hits:
            print(f"gain {gn}: no draws file matching {pat}; skipping", flush=True)
            continue
        draws = pd.read_csv(hits[-1])
        print(f"gain {gn}: {hits[-1].name}, {len(draws)} draws", flush=True)

        rel, coh, ch_net, seeds, n_net = load_gain(gn)
        w_net, s_net = kernel(rel, ch_net)
        lv, P_net = psychometric(coh, ch_net)
        acc_net = float((ch_net == (coh > 0)).mean())
        print(f"  network: {len(rel)} trials, {n_net} nets, kernel slope {s_net:+.4f}, acc {acc_net:.3f}",
              flush=True)
        rng = np.random.default_rng(a.seed)
        noise = rng.standard_normal(rel.shape, dtype=np.float32)   # common random numbers

        # posterior mean draw + n_draws thinned draws
        mean_row = draws.mean(numeric_only=True)
        idx = np.linspace(0, len(draws) - 1, min(a.n_draws, len(draws))).astype(int)
        jobs = [("mean", mean_row)] + [(f"draw{i}", draws.iloc[j]) for i, j in enumerate(idx)]

        for vname, use_bound, matched in VARIANTS:
            per_draw_w, per_draw_s, per_draw_acc, per_draw_fb = [], [], [], []
            for jname, row in jobs:
                v0, v1, g, a_nat, x0, t_nat = native(row, a.k)
                var_ev = (v1 ** 2) * DT
                if matched:
                    sigma = float(np.sqrt(max(1.0 - var_ev, 1e-6)))
                else:
                    sigma = 1.0
                t0_steps = int(round(max(t_nat, 0.0) / DT))
                ch, fb = simulate_evidence_ou(rel, noise, v0, v1, g,
                                              a=(a_nat if use_bound else None),
                                              x0=x0, sigma=sigma, t0_steps=t0_steps)
                w, s = kernel(rel, ch)
                acc = float((ch == (coh > 0)).mean())
                if jname == "mean":
                    _, P_sim = psychometric(coh, ch)
                    curves[(gn, vname)] = dict(w_net=w_net, w_sim=w, lv=lv, P_net=P_net, P_sim=P_sim)
                    rows.append(dict(gain=gn, variant=vname, which="posterior_mean",
                                     v0_nat=v0, v1_nat=v1, g_nat=g, a_nat=(a_nat if use_bound else np.nan),
                                     x0_nat=x0, sigma=sigma, evidence_var_frac=var_ev, t_nat=t_nat,
                                     slope_net=s_net, slope_sim=s, acc_net=acc_net, acc_sim=acc,
                                     frac_bounded=fb, n_trials=len(rel),
                                     **{f"wnet{i}": w_net[i] for i in range(N_BINS)},
                                     **{f"wsim{i}": w[i] for i in range(N_BINS)}))
                else:
                    per_draw_w.append(w); per_draw_s.append(s); per_draw_acc.append(acc); per_draw_fb.append(fb)
                    draw_rows.append(dict(gain=gn, variant=vname, draw=jname, g_nat=g,
                                          a_nat=(a_nat if use_bound else np.nan), v1_nat=v1,
                                          slope_sim=s, acc_sim=acc, frac_bounded=fb))
            W = np.array(per_draw_w); S = np.array(per_draw_s)
            rows.append(dict(gain=gn, variant=vname, which=f"draws_mean(n={len(S)})",
                             v0_nat=np.nan, v1_nat=np.nan, g_nat=np.nan, a_nat=np.nan, x0_nat=np.nan,
                             sigma=np.nan, evidence_var_frac=np.nan, t_nat=np.nan,
                             slope_net=s_net, slope_sim=float(S.mean()), slope_sim_sd=float(S.std()),
                             slope_sim_lo=float(np.quantile(S, 0.03)), slope_sim_hi=float(np.quantile(S, 0.97)),
                             acc_net=acc_net, acc_sim=float(np.mean(per_draw_acc)),
                             frac_bounded=float(np.mean(per_draw_fb)), n_trials=len(rel),
                             **{f"wnet{i}": w_net[i] for i in range(N_BINS)},
                             **{f"wsim{i}": W[:, i].mean() for i in range(N_BINS)}))
            print(f"  {vname}: slope sim {S.mean():+.4f} (sd {S.std():.4f}) vs net {s_net:+.4f}"
                  f" | acc {np.mean(per_draw_acc):.3f} vs {acc_net:.3f} | bounded {np.mean(per_draw_fb):.3f}",
                  flush=True)
        print(f"  gain {gn} done in {(time.time()-t0)/60:.1f} min", flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"kernel_ppc{suffix}.csv", index=False)
    pd.DataFrame(draw_rows).to_csv(OUT / f"kernel_ppc_draws{suffix}.csv", index=False)
    print("\n" + tab[["gain", "variant", "which", "g_nat", "a_nat", "slope_net", "slope_sim",
                      "acc_net", "acc_sim", "frac_bounded"]].round(4).to_string(index=False))

    # figure: one panel per variant, network (solid) vs model (dashed), three gains
    vlist = [v[0] for v in VARIANTS]
    fig, axes = plt.subplots(1, len(vlist) + 1, figsize=(4.2 * (len(vlist) + 1), 3.8))
    cols = {0.8: "C0", 1.0: "k", 1.2: "C3"}
    xb = (np.arange(N_BINS) + 0.5) * 750 / N_BINS
    for ax, vname in zip(axes, vlist):
        for gn in a.gains:
            c = curves.get((gn, vname))
            if c is None:
                continue
            ax.plot(xb, c["w_net"], "-o", color=cols.get(gn, "C2"), label=f"network f={gn}")
            ax.plot(xb, c["w_sim"], "--s", color=cols.get(gn, "C2"), alpha=.7, label=f"HSSM OU f={gn}")
        ax.axhline(1 / N_BINS, color="gray", lw=.5)
        ax.set_xlabel("time within trial (ms)")
        ax.set_ylabel("influence on choice (L1-normalised)")
        ax.set_title(vname, fontsize=9)
    axes[0].legend(fontsize=6)
    ax = axes[-1]
    sub = tab[tab.which.str.startswith("draws_mean")]
    for i, vname in enumerate(vlist):
        d = sub[sub.variant == vname]
        ax.plot(d.gain, d.slope_sim, "-s", label=vname)
    d0 = tab[tab.variant == vlist[0]].drop_duplicates("gain")
    ax.plot(d0.gain, d0.slope_net, "-o", color="k", lw=2, label="network")
    ax.axhline(0, color="gray", lw=.5)
    ax.set_xlabel("gain f"); ax.set_ylabel("kernel slope"); ax.set_title("kernel slope vs gain", fontsize=9)
    ax.legend(fontsize=6)
    plt.tight_layout()
    fig.savefig(OUT / f"kernel_ppc{suffix}.png", dpi=140)
    print("saved", OUT / f"kernel_ppc{suffix}.png")


if __name__ == "__main__":
    main()
