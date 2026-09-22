"""
Parameter recovery for the evidence-conditioned OU (issue #4, Track B, step 5).

Choices are simulated on the REAL evidence streams of 5 networks (42, 46, 51, 58, 61) x 3 gains with
  (i)  the fitted route-1 parameters, and
  (ii) the theoretical leak from the landscape analysis, g = +4.3 / -1.6 / -6.2 per s at gains
       0.8 / 1.0 / 1.2 (the drift slope with the sign flipped; g > 0 leaky), with the fitted v and a_bias,
and refitted with route 1.  Recovery is judged by the sign of g and by whether the truth is inside the
95 % Wald interval.  `--reps` repeats each cell with different simulation noise.

`--stage make-r2` writes the synthetic datasets for the route-2 recovery runs (fit_bounded.py --synthetic):
  - (ii) at gains 0.8 and 1.2 with the fitted B (bounded simulation, so the bound-hit observable is the
    model's own first-crossing time -- the hit term is correctly specified there, unlike in the real-data
    fits where the model bound is identified with the network's |dv| = 2.0 level),
  - one degenerate dataset (g = +12, B = 1.2, v/a_bias from the gain-1.2 fit, gain-1.2 evidence) to test
    whether the bound-hit term resolves the leak-vs-bound degeneracy of kernel_fit/RESULTS.md.

Usage:
    python track_b/recover.py --stage route1          # -> output/track_b/recovery_route1.csv
    python track_b/recover.py --stage make-r2         # -> output/track_b/recovery/*.npz + tasks file
    python track_b/recover.py --stage collect-r2      # -> output/track_b/recovery_route2.csv
"""
import argparse, json, pathlib, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ou_lik import DT, GAINS, load_network, analytic_moments, fit_analytic, simulate_bounded  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "track_b"
REC = OUT / "recovery"
SEEDS = [42, 46, 51, 58, 61]
G_THEORY = {0.8: 4.3, 1.0: -1.6, 1.2: -6.2}          # landscape drift slope, sign-flipped to the OU g
DEGEN = dict(g=12.0, B=1.2)


def simulate_choices_analytic(rel, g, v, a_bias, rng):
    mu, s = analytic_moments(rel, g, v, a_bias)
    from scipy.stats import norm
    p = norm.cdf(mu / s)
    return (rng.random(len(p)) < p).astype(int)


def _recover_one(args):
    seed, gain, kind, rep, g_true, v_true, a_true, n_trials, g0s = args
    d = load_network(seed, gain=gain, n_trials=n_trials)
    rel = d["rel"]
    rng = np.random.default_rng(abs(hash((seed, gain, kind, rep))) % 2**31)
    y = simulate_choices_analytic(rel, g_true, v_true, a_true, rng)
    t0 = time.time()
    r = fit_analytic(rel, y, g0s=g0s)
    lo, hi = r["g"] - 1.96 * r["g_se"], r["g"] + 1.96 * r["g_se"]
    row = dict(seed=seed, gain=gain, kind=kind, rep=rep, g_true=g_true, v_true=v_true,
               a_bias_true=a_true, g=r["g"], g_se=r["g_se"], g_lo=lo, g_hi=hi, v=r["v"],
               a_bias=r["a_bias"], nll=r["nll"], n_trials=len(y),
               sign_ok=bool(np.sign(r["g"]) == np.sign(g_true)),
               covered=bool(lo <= g_true <= hi), secs=time.time() - t0)
    print(f"seed {seed} gain {gain} [{kind} rep {rep}]: true g = {g_true:+.3f} -> "
          f"{r['g']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
          f"(sign {'ok' if row['sign_ok'] else 'WRONG'}, "
          f"{'covered' if row['covered'] else 'NOT covered'}), v {v_true:.1f} -> {r['v']:.1f}", flush=True)
    return row


def stage_route1(reps, n_trials, mle_csv, procs=7, g0s=(0.0,)):
    import multiprocessing as mp
    mle = pd.read_csv(mle_csv)
    jobs = []
    for seed in SEEDS:
        for gain in GAINS:
            f = mle[(mle.seed == seed) & (mle.gain == gain)].iloc[0]
            for kind, g_true in (("fitted", float(f.g)), ("theory", G_THEORY[gain])):
                for rep in range(reps):
                    jobs.append((seed, gain, kind, rep, g_true, float(f.v), float(f.a_bias),
                                 n_trials, tuple(g0s)))
    with mp.get_context("fork").Pool(procs) as pool:
        rows = pool.map(_recover_one, jobs)
    tab = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "recovery_route1.csv", index=False)
    print("\n--- route-1 recovery ---")
    print(tab.groupby(["kind", "gain"]).agg(n=("g", "size"), g_true=("g_true", "mean"),
                                            g_mean=("g", "mean"), g_sd=("g", "std"),
                                            sign_ok=("sign_ok", "mean"),
                                            covered=("covered", "mean")).round(3).to_string())
    return tab


def stage_make_r2(n_trials, mle_csv, bounded_dir):
    """Synthetic bounded datasets: theory-g at gains 0.8/1.2 with the fitted B, plus one degenerate case."""
    mle = pd.read_csv(mle_csv)
    REC.mkdir(parents=True, exist_ok=True)
    tasks = []
    B_fit = {}
    bd = pathlib.Path(bounded_dir)
    for f in sorted(bd.glob("seed*_none.json")):
        j = json.loads(f.read_text())
        B_fit[(j["seed"], j["gain"])] = j["B"]
    print(f"route-2 bounds available for {len(B_fit)} (seed, gain) cells")
    for seed in SEEDS:
        d = load_network(seed, n_trials=n_trials)
        rel = d["rel"]
        for gain in (0.8, 1.2):
            f = mle[(mle.seed == seed) & (mle.gain == gain)].iloc[0]
            B = B_fit.get((seed, gain), 3.0)
            g_true = G_THEORY[gain]
            y, tc = _sim_bounded_dataset(rel, g_true, float(f.v), B, float(f.a_bias),
                                         seed=770_000 + seed * 10 + int(gain * 10))
            name = f"rec_seed{seed}_g{gain}_theory"
            np.savez(REC / f"{name}.npz", y=y, tcross=tc, g_true=g_true, v_true=float(f.v),
                     B_true=B, a_bias_true=float(f.a_bias))
            tasks += [f"{seed} {gain} none {name}", f"{seed} {gain} cross {name}"]
            print(f"{name}: g_true {g_true:+.2f} B_true {B:.2f} v {float(f.v):.1f} -> "
                  f"acc {np.mean(y == (d['coh'] > 0)):.3f}, hit frac {np.mean(~np.isnan(tc)):.3f}")
    # degenerate case: strong leak + low sticky bound on gain-1.2 evidence of seed 42
    seed, gain = 42, 1.2
    d = load_network(seed, n_trials=n_trials)
    f = mle[(mle.seed == seed) & (mle.gain == gain)].iloc[0]
    y, tc = _sim_bounded_dataset(d["rel"], DEGEN["g"], float(f.v), DEGEN["B"], 0.0, seed=999_001)
    name = f"rec_degenerate_seed{seed}_g{gain}"
    np.savez(REC / f"{name}.npz", y=y, tcross=tc, g_true=DEGEN["g"], v_true=float(f.v),
             B_true=DEGEN["B"], a_bias_true=0.0)
    tasks += [f"{seed} {gain} none {name}", f"{seed} {gain} cross {name}", f"{seed} {gain} bern {name}"]
    print(f"{name}: g_true {DEGEN['g']:+.2f} B_true {DEGEN['B']:.2f} -> "
          f"acc {np.mean(y == (d['coh'] > 0)):.3f}, hit frac {np.mean(~np.isnan(tc)):.3f}")
    (ROOT / "track_b" / "bash" / "tasks_recovery.txt").write_text("\n".join(tasks) + "\n")
    print(f"\n{len(tasks)} route-2 recovery tasks -> track_b/bash/tasks_recovery.txt")


def _sim_bounded_dataset(rel, g, v, B, a_bias, seed):
    """One realisation per trial; returns the choice and the first-crossing time (ms, NaN if never)."""
    aT, tcross = simulate_bounded(rel, g, v, B, a_bias, M=1, seed=seed)
    y = (aT[0] > 0).astype(int)
    tc = tcross[0].astype(float)
    tc[tc >= rel.shape[1]] = np.nan
    return y, tc


def stage_collect_r2(rec_out):
    rows = []
    for f in sorted(pathlib.Path(rec_out).glob("*.json")):
        j = json.loads(f.read_text())
        syn = np.load(j["synthetic"]) if j.get("synthetic") else None
        if syn is None:
            continue
        gt, Bt = float(syn["g_true"]), float(syn["B_true"])
        rows.append(dict(dataset=pathlib.Path(j["synthetic"]).stem, seed=j["seed"], gain=j["gain"],
                         hit_mode=j["hit_mode"], g_true=gt, B_true=Bt, v_true=float(syn["v_true"]),
                         g=j["g"], g_lo=j["g_lo"], g_hi=j["g_hi"], B=j["B"], v=j["v"],
                         a_bias=j["a_bias"], nll=j["nll"], model_hit=j["model_hit_frac"],
                         data_hit=j["net_hit_frac"],
                         sign_ok=bool(np.sign(j["g"]) == np.sign(gt)),
                         covered=bool(j["g_lo"] <= gt <= j["g_hi"]) if np.isfinite(j["g_lo"]) else None))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "recovery_route2.csv", index=False)
    print(tab.round(3).to_string(index=False))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["route1", "make-r2", "collect-r2"], default="route1")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--mle-csv", type=pathlib.Path, default=OUT / "analytic_per_network_mle.csv")
    ap.add_argument("--bounded-dir", type=pathlib.Path, default=OUT / "bounded")
    ap.add_argument("--rec-out", type=pathlib.Path, default=OUT / "bounded_recovery")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--g0s", type=float, nargs="+", default=[0.0])
    a = ap.parse_args()
    if a.stage == "route1":
        stage_route1(a.reps, a.n_trials, a.mle_csv, a.procs, a.g0s)
    elif a.stage == "make-r2":
        stage_make_r2(a.n_trials, a.mle_csv, a.bounded_dir)
    else:
        stage_collect_r2(a.rec_out)


if __name__ == "__main__":
    main()
