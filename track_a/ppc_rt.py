"""
Track A posterior-predictive check on RT and choice (issue #3).

For a pooled fit <tag>, draw N_DRAWS parameter vectors from <tag>_draws.csv, simulate with the exact
ssm-simulators `ornstein` simulator (what the LAN approximates), apply the same horizon as the data
(native 0.75 s -> stretched 0.3 + k*0.75 s; trials that do not cross are omissions, as in the data),
and compare with the network data per |coherence|:
  * RT quantiles 10/30/50/70/90 for correct and error trials, reported in NATIVE ms,
  * accuracy (P(choice matches sign(coherence))), P(choice = +) at coherence 0,
  * omission fraction.

ssm-simulators segfaults for max_t ~ 0.75 s, so we always simulate to at least 2 s and truncate afterwards.
Sign convention: dx = (v - g x) dt + dW; g > 0 leaky, g < 0 unstable.

Run on Oscar (needs ssms):
  /users/igrahek/.conda/envs/pyHSSM_New_Nov24/bin/python track_a/ppc_rt.py --tag g1.0_k10_b1.5_pooled
  ... --all-pooled --k 10
Outputs -> output/track_a/ppc_rt_<tag>.csv, ppc_rt_<tag>.png, ppc_rt_summary.csv
"""
import argparse
import glob
import json
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from ssms.basic_simulators.simulator import simulator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "fixed_bound"
OUT = ROOT / "output" / "track_a"
OFFSET = 0.3
HORIZON_NATIVE = 0.75
Q = [0.1, 0.3, 0.5, 0.7, 0.9]


def obs_table(gain, bound, k):
    df = pd.read_csv(DATA / f"hssm_ready_nxx1_fixed_b{bound}_g{gain}.csv")
    df["acoh"] = df.coherence.round(2)
    df["correct"] = np.where(df.acoh == 0, np.nan,
                             (df.response_choice.values * np.sign(df.coherence_signed.values)) > 0)
    n_seeds = df.seed.nunique()
    # per-|coherence| omission fraction: 2000 trials per network per gain were simulated
    return df, n_seeds


def sim_one_draw(row, cohs, n_per_coh, horizon, rng_seed):
    """Simulate n_per_coh trials at each signed coherence. Returns dict coh -> (rt, choice, omit)."""
    out = {}
    max_t = max(2.0, horizon + 0.01)
    for i, c in enumerate(cohs):
        v = float(row["v_Intercept"]) + float(row["v_coherence_signed"]) * c
        s = simulator(theta=dict(v=v, a=float(row["a"]), z=float(row["z"]), g=float(row["g"]),
                                 t=float(row["t"])),
                      model="ornstein", n_samples=n_per_coh, delta_t=0.001, max_t=max_t,
                      random_state=rng_seed + i)
        rt = np.asarray(s["rts"]).ravel()
        ch = np.asarray(s["choices"]).ravel()
        ok = (rt > 0) & (rt <= horizon)
        out[c] = (rt[ok], np.where(ch[ok] > 0, 1.0, -1.0), float(1 - ok.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--all-pooled", action="store_true", help="every g*_k<k>_b*_pooled fit at --k")
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--n-draws", type=int, default=200)
    ap.add_argument("--n-per-coh", type=int, default=100, help="trials per coherence PER DRAW")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.all_pooled:
        tags = [pathlib.Path(f).name.replace("_draws.csv", "")
                for f in sorted(glob.glob(str(OUT / f"g*_k{a.k:g}_b{a.bound}_pooled_draws.csv")))]
    else:
        tags = [a.tag]
    print("tags:", tags, flush=True)

    all_rows = []
    for tag in tags:
        t0 = time.time()
        meta = json.loads((OUT / f"{tag}_meta.json").read_text())
        gain, k, bound = meta["gain"], meta["k"], meta["bound"]
        horizon = OFFSET + k * HORIZON_NATIVE
        draws = pd.read_csv(OUT / f"{tag}_draws.csv")
        idx = np.linspace(0, len(draws) - 1, min(a.n_draws, len(draws))).astype(int)
        df, n_seeds = obs_table(gain, bound, k)
        cohs = sorted(df.coherence_signed.round(2).unique())
        print(f"\n=== {tag}: gain {gain}, k {k}, horizon {horizon:.2f} s stretched, "
              f"{len(idx)} draws x {a.n_per_coh} trials x {len(cohs)} coherences", flush=True)

        sim_rt, sim_ch, sim_om = {c: [] for c in cohs}, {c: [] for c in cohs}, {c: [] for c in cohs}
        for j, ii in enumerate(idx):
            res = sim_one_draw(draws.iloc[ii], cohs, a.n_per_coh, horizon, a.seed + 1000 * j)
            for c in cohs:
                rt, ch, om = res[c]
                sim_rt[c].append(rt); sim_ch[c].append(ch); sim_om[c].append(om)
            if (j + 1) % 50 == 0:
                print(f"  draw {j+1}/{len(idx)} ({(time.time()-t0)/60:.1f} min)", flush=True)
        for c in cohs:
            sim_rt[c] = np.concatenate(sim_rt[c]); sim_ch[c] = np.concatenate(sim_ch[c])

        rows = []
        for ac in sorted(df.acoh.unique()):
            sel = df.acoh == ac
            o_rt = df.rt[sel].values * 1000.0                       # native ms
            o_ch = df.response_choice[sel].values
            o_cs = df.coherence_signed[sel].values
            o_corr = (o_ch * np.sign(o_cs)) > 0 if ac > 0 else None
            o_n_expected = 2000 * n_seeds * (sel.sum() / len(df)) if False else None
            cs = [c for c in cohs if abs(round(c, 2)) == ac]
            s_rt = np.concatenate([sim_rt[c] for c in cs])
            s_ch = np.concatenate([sim_ch[c] for c in cs])
            s_cs = np.concatenate([np.full(len(sim_rt[c]), c) for c in cs])
            s_rt_nat = (s_rt - OFFSET) / k * 1000.0                 # native ms
            s_corr = (s_ch * np.sign(s_cs)) > 0 if ac > 0 else None
            s_om = float(np.mean([np.mean(sim_om[c]) for c in cs]))
            r = dict(tag=tag, gain=gain, k=k, abs_coherence=ac, n_obs=int(sel.sum()),
                     n_sim=int(len(s_rt)),
                     acc_obs=(float(o_corr.mean()) if ac > 0 else float((o_ch == 1).mean())),
                     acc_sim=(float(s_corr.mean()) if ac > 0 else float((s_ch == 1).mean())),
                     omit_sim=s_om)
            if ac > 0:
                for lab, om, sm in [("c", o_corr, s_corr), ("e", ~o_corr, ~s_corr)]:
                    for q in Q:
                        oq = float(np.quantile(o_rt[om], q)) if om.sum() >= 30 else np.nan
                        sq = float(np.quantile(s_rt_nat[sm], q)) if sm.sum() >= 30 else np.nan
                        r[f"q{lab}{int(q*100)}_obs"] = oq
                        r[f"q{lab}{int(q*100)}_sim"] = sq
                        r[f"q{lab}{int(q*100)}_err"] = sq - oq
            else:
                for q in Q:
                    oq = float(np.quantile(o_rt, q)); sq = float(np.quantile(s_rt_nat, q))
                    r[f"qc{int(q*100)}_obs"] = oq; r[f"qc{int(q*100)}_sim"] = sq
                    r[f"qc{int(q*100)}_err"] = sq - oq
            rows.append(r)
        tab = pd.DataFrame(rows)
        tab.to_csv(OUT / f"ppc_rt_{tag}.csv", index=False)
        errc = tab[[c for c in tab.columns if c.startswith("qc") and c.endswith("_err")]].values.ravel()
        erre = tab[[c for c in tab.columns if c.startswith("qe") and c.endswith("_err")]].values.ravel()
        print(tab[["abs_coherence", "n_obs", "acc_obs", "acc_sim", "omit_sim",
                   "qc50_obs", "qc50_sim", "qc50_err"]].round(3).to_string(index=False), flush=True)
        print(f"  |quantile error| correct: median {np.nanmedian(np.abs(errc)):.1f} ms, "
              f"max {np.nanmax(np.abs(errc)):.1f} ms; error trials: median "
              f"{np.nanmedian(np.abs(erre)):.1f} ms, max {np.nanmax(np.abs(erre)):.1f} ms", flush=True)
        all_rows.append(dict(tag=tag, gain=gain, k=k,
                             n_draws=len(idx), n_sim_per_coh=len(sim_rt[cohs[0]]),
                             omit_obs=meta["omission_frac"],
                             omit_sim=float(np.mean([np.mean(sim_om[c]) for c in cohs])),
                             med_abs_qerr_correct_ms=float(np.nanmedian(np.abs(errc))),
                             max_abs_qerr_correct_ms=float(np.nanmax(np.abs(errc))),
                             med_abs_qerr_error_ms=float(np.nanmedian(np.abs(erre))),
                             max_abs_qerr_error_ms=float(np.nanmax(np.abs(erre))),
                             max_abs_acc_err=float(np.abs(tab.acc_sim - tab.acc_obs).max()),
                             minutes=(time.time() - t0) / 60))

        # figure: RT histograms per |coherence|, correct (top) and error (bottom)
        acs = sorted(df.acoh.unique())
        fig, axes = plt.subplots(2, len(acs), figsize=(2.6 * len(acs), 5.2), sharex=True)
        bins = np.linspace(0, 750, 40)
        for jc, ac in enumerate(acs):
            sel = df.acoh == ac
            o_rt = df.rt[sel].values * 1000.0
            o_ch = df.response_choice[sel].values; o_cs = df.coherence_signed[sel].values
            cs = [c for c in cohs if abs(round(c, 2)) == ac]
            s_rt_nat = (np.concatenate([sim_rt[c] for c in cs]) - OFFSET) / k * 1000.0
            s_ch = np.concatenate([sim_ch[c] for c in cs])
            s_cs = np.concatenate([np.full(len(sim_rt[c]), c) for c in cs])
            if ac > 0:
                masks = [((o_ch * np.sign(o_cs)) > 0, (s_ch * np.sign(s_cs)) > 0),
                         ((o_ch * np.sign(o_cs)) < 0, (s_ch * np.sign(s_cs)) < 0)]
                labs = ["correct", "error"]
            else:
                masks = [(o_ch == 1, s_ch == 1), (o_ch == -1, s_ch == -1)]
                labs = ['choice "+"', 'choice "-"']
            for ri, ((om, sm), lab) in enumerate(zip(masks, labs)):
                ax = axes[ri, jc]
                ax.hist(o_rt[om], bins=bins, density=True, alpha=.5, color="C0", label="network")
                if sm.sum() > 20:
                    ax.hist(s_rt_nat[sm], bins=bins, density=True, histtype="step", color="C3", lw=1.4,
                            label="HSSM OU")
                ax.set_title(f"|coh| {ac} {lab}", fontsize=8)
            axes[1, jc].set_xlabel("RT (ms, native)")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{tag}: RT posterior predictive (native ms); g = {draws.g.mean():.3f} "
                     f"(g_native {draws.g.mean()*k:+.2f}/s, >0 = leaky)", fontsize=10)
        plt.tight_layout()
        fig.savefig(OUT / f"ppc_rt_{tag}.png", dpi=130)
        print("saved", OUT / f"ppc_rt_{tag}.png", flush=True)

    summ = pd.DataFrame(all_rows)
    out = OUT / "ppc_rt_summary.csv"
    if out.exists():
        old = pd.read_csv(out)
        summ = pd.concat([old[~old.tag.isin(summ.tag)], summ], ignore_index=True)
    summ.to_csv(out, index=False)
    print("\n" + summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
