"""
Route 1 (issue #4, Track B): unbounded evidence-conditioned OU fitted to each network's deadline
choices by maximum likelihood (numerical-Hessian SEs) and by NUTS (numpyro), per (seed, gain) and
pooled over the 20 networks per gain.

    P(choice_i = 1 | e_i, theta) = Phi(mu_i / s),  mu_i = a_bias*rho^T + v*dt*sum_t rho^(T-1-t) e_it,
    s^2 = dt*sum_t rho^(2(T-1-t)),  rho = 1 - g*dt,  dt = 1 ms, T = 750, sigma = 1 fixed.
    g > 0 leaky (recency), g < 0 unstable (primacy).

Priors for the Bayesian version (as specified in the issue): g ~ N(0, 10), v ~ HalfNormal(100),
a_bias ~ N(0, 1); 4 chains x 1000 warmup / 1000 draws.

Usage
    python track_b/fit_analytic.py --mode mle                     # 60 fits + 3 pooled, ~2 min
    <venv with numpyro>/python track_b/fit_analytic.py --mode bayes
    python track_b/fit_analytic.py --mode merge                   # -> analytic_per_network.csv
    python track_b/fit_analytic.py --mode mle --n-trials 1000 --tag n1000     # sensitivity
Outputs in output/track_b/: analytic_per_network{,_mle,_bayes}.csv, analytic_pooled{,_mle,_bayes}.csv
"""
import argparse, itertools, pathlib, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ou_lik import DT, GAINS, load_network, fit_analytic, nll_analytic, analytic_p  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "track_b"
SEEDS = list(range(42, 62))


# ------------------------------------------------------------------------------- MLE
def _mle_one(args):
    seed, gain, n_trials, data_dir = args
    d = load_network(seed, gain=gain, n_trials=n_trials, data_dir=data_dir)
    t0 = time.time()
    r = fit_analytic(d["rel"], d[f"choice_g{gain}"])
    r.update(seed=seed, gain=gain, n_trials=len(d["rel"]), secs=time.time() - t0)
    r["g_lo"] = r["g"] - 1.96 * r["g_se"]
    r["g_hi"] = r["g"] + 1.96 * r["g_se"]
    print(f"seed {seed} gain {gain}: g = {r['g']:+.3f} +/- {r['g_se']:.3f} "
          f"[{r['g_lo']:+.3f}, {r['g_hi']:+.3f}], v = {r['v']:.2f} +/- {r['v_se']:.2f}, "
          f"a_bias = {r['a_bias']:+.3f} +/- {r['a_bias_se']:.3f}, nll = {r['nll']:.2f} ({r['secs']:.0f}s)",
          flush=True)
    return r


def run_mle(seeds, gains, n_trials, data_dir, pooled=True, procs=8):
    import multiprocessing as mp
    jobs = [(s, g, n_trials, data_dir) for s, g in itertools.product(seeds, gains)]
    with mp.get_context("fork").Pool(procs) as pool:
        rows = pool.map(_mle_one, jobs)
    tab = pd.DataFrame(rows).sort_values(["gain", "seed"])
    pooled_rows = []
    if pooled:
        for gain in gains:
            ds = [load_network(s, gain=gain, n_trials=n_trials, data_dir=data_dir) for s in seeds]
            rel = np.concatenate([d["rel"] for d in ds])
            y = np.concatenate([d[f"choice_g{gain}"] for d in ds])
            t0 = time.time()
            r = fit_analytic(rel, y)
            r.update(gain=gain, n_networks=len(seeds), n_trials=len(y), secs=time.time() - t0,
                     g_lo=r["g"] - 1.96 * r["g_se"], g_hi=r["g"] + 1.96 * r["g_se"])
            pooled_rows.append(r)
            print(f"POOLED gain {gain} ({len(y)} trials): g = {r['g']:+.3f} +/- {r['g_se']:.3f} "
                  f"[{r['g_lo']:+.3f}, {r['g_hi']:+.3f}], v = {r['v']:.2f}, a_bias = {r['a_bias']:+.3f} "
                  f"({r['secs']:.0f}s)", flush=True)
    return tab, pd.DataFrame(pooled_rows)


# -------------------------------------------------------------------------- Bayesian
def _numpyro_model(rel, y, T):
    import jax.numpy as jnp, numpyro
    import numpyro.distributions as dist
    from jax.scipy.special import log_ndtr
    g = numpyro.sample("g", dist.Normal(0.0, 10.0))
    v = numpyro.sample("v", dist.HalfNormal(100.0))
    a_bias = numpyro.sample("a_bias", dist.Normal(0.0, 1.0))
    rho = 1.0 - g * DT
    k = jnp.arange(T - 1, -1, -1.0)
    w = jnp.exp(k * jnp.log(rho))
    mu = v * DT * (rel @ w) + a_bias * rho ** T
    s = jnp.sqrt(DT * jnp.sum(w ** 2))
    z = mu / s
    numpyro.factor("ll", jnp.sum(jnp.where(y == 1, log_ndtr(z), log_ndtr(-z))))


def hdi(x, prob=0.94):
    x = np.sort(np.asarray(x))
    n = len(x); k = int(np.floor(prob * n))
    lo = x[:n - k]; hi = x[k:]
    i = np.argmin(hi - lo)
    return float(lo[i]), float(hi[i])


def run_nuts(rel, y, warmup=1000, draws=1000, chains=4, seed=0):
    import jax, numpyro
    from numpyro.infer import MCMC, NUTS
    numpyro.set_host_device_count(chains)
    import jax.numpy as jnp
    relj = jnp.asarray(rel, dtype=jnp.float32); yj = jnp.asarray(y)
    mcmc = MCMC(NUTS(_numpyro_model), num_warmup=warmup, num_samples=draws, num_chains=chains,
                chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), relj, yj, rel.shape[1])
    s = mcmc.get_samples(group_by_chain=True)
    summ = numpyro.diagnostics.summary(s, prob=0.94)
    flat = {k: np.asarray(v).reshape(-1) for k, v in s.items()}
    out = {}
    for p in ("g", "v", "a_bias"):
        lo, hi = hdi(flat[p])
        out.update({f"{p}_mean": float(flat[p].mean()), f"{p}_sd": float(flat[p].std()),
                    f"{p}_hdi_lo": lo, f"{p}_hdi_hi": hi,
                    f"{p}_rhat": float(summ[p]["r_hat"]), f"{p}_ess": float(summ[p]["n_eff"])})
    out["p_g_gt0"] = float((flat["g"] > 0).mean())
    return out


def run_bayes(seeds, gains, n_trials, data_dir, pooled=True, draws=1000, warmup=1000, chains=4):
    rows, pooled_rows = [], []
    for gain in gains:
        for seed in seeds:
            d = load_network(seed, gain=gain, n_trials=n_trials, data_dir=data_dir)
            t0 = time.time()
            r = run_nuts(d["rel"], d[f"choice_g{gain}"], warmup, draws, chains, seed=seed)
            r.update(seed=seed, gain=gain, n_trials=len(d["rel"]), secs=time.time() - t0)
            rows.append(r)
            print(f"seed {seed} gain {gain}: g = {r['g_mean']:+.3f} +/- {r['g_sd']:.3f} "
                  f"HDI94 [{r['g_hdi_lo']:+.3f}, {r['g_hdi_hi']:+.3f}] rhat {r['g_rhat']:.3f} "
                  f"ess {r['g_ess']:.0f} | v {r['v_mean']:.2f} a_bias {r['a_bias_mean']:+.3f} "
                  f"({r['secs']:.0f}s)", flush=True)
        if pooled:
            ds = [load_network(s, gain=gain, n_trials=n_trials, data_dir=data_dir) for s in seeds]
            rel = np.concatenate([d["rel"] for d in ds]); y = np.concatenate([d[f"choice_g{gain}"] for d in ds])
            t0 = time.time()
            r = run_nuts(rel, y, warmup, draws, chains, seed=1000 + int(gain * 10))
            r.update(gain=gain, n_networks=len(seeds), n_trials=len(y), secs=time.time() - t0)
            pooled_rows.append(r)
            print(f"POOLED gain {gain} ({len(y)} trials): g = {r['g_mean']:+.3f} "
                  f"HDI94 [{r['g_hdi_lo']:+.3f}, {r['g_hdi_hi']:+.3f}] rhat {r['g_rhat']:.3f} "
                  f"({r['secs']:.0f}s)", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(pooled_rows)


# ----------------------------------------------------------------------------- main
def summarise(tab, col, lo, hi, label):
    print(f"\n--- {label} ---")
    for gain, d in tab.groupby("gain"):
        pred = "positive" if gain < 0.9 else ("negative" if gain > 1.1 else "either")
        q1, med, q3 = np.percentile(d[col], [25, 50, 75])
        excl = ((d[lo] > 0) if pred == "positive" else (d[hi] < 0) if pred == "negative"
                else ((d[lo] > 0) | (d[hi] < 0)))
        print(f"gain {gain}: median g = {med:+.3f} [IQR {q1:+.3f}, {q3:+.3f}], "
              f"g > 0 in {int((d[col] > 0).sum())}/{len(d)}, "
              f"interval excludes 0 on the predicted side ({pred}) in {int(excl.sum())}/{len(d)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mle", "bayes", "merge"], default="mle")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--gains", type=float, nargs="+", default=list(GAINS))
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--data-dir", type=pathlib.Path, default=None)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--no-pooled", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = a.data_dir or (ROOT / "data" / "processed" / "kernel")
    tag = ("_" + a.tag) if a.tag else ""

    if a.mode == "mle":
        tab, pool = run_mle(a.seeds, a.gains, a.n_trials, data_dir, not a.no_pooled, a.procs)
        tab.to_csv(OUT / f"analytic_per_network_mle{tag}.csv", index=False)
        if len(pool):
            pool.to_csv(OUT / f"analytic_pooled_mle{tag}.csv", index=False)
        summarise(tab, "g", "g_lo", "g_hi", "route 1 MLE (Wald 95 % interval)")
    elif a.mode == "bayes":
        tab, pool = run_bayes(a.seeds, a.gains, a.n_trials, data_dir, not a.no_pooled,
                              a.draws, a.warmup, a.chains)
        tab.to_csv(OUT / f"analytic_per_network_bayes{tag}.csv", index=False)
        if len(pool):
            pool.to_csv(OUT / f"analytic_pooled_bayes{tag}.csv", index=False)
        summarise(tab, "g_mean", "g_hdi_lo", "g_hdi_hi", "route 1 Bayesian (94 % HDI)")
    else:
        m = pd.read_csv(OUT / f"analytic_per_network_mle{tag}.csv")
        b = pd.read_csv(OUT / f"analytic_per_network_bayes{tag}.csv")
        j = m.merge(b, on=["seed", "gain", "n_trials"], suffixes=("", "_b"))
        j["g_mle_minus_post"] = j["g"] - j["g_mean"]
        j["g_diff_in_se"] = j["g_mle_minus_post"] / j["g_se"]
        j.to_csv(OUT / f"analytic_per_network{tag}.csv", index=False)
        mp_, bp = (OUT / f"analytic_pooled_mle{tag}.csv"), (OUT / f"analytic_pooled_bayes{tag}.csv")
        if mp_.exists() and bp.exists():
            jp = pd.read_csv(mp_).merge(pd.read_csv(bp), on=["gain", "n_trials"], suffixes=("", "_b"))
            jp.to_csv(OUT / f"analytic_pooled{tag}.csv", index=False)
            print(jp[["gain", "g", "g_se", "g_lo", "g_hi", "g_mean", "g_hdi_lo", "g_hdi_hi",
                      "g_rhat", "v", "v_mean", "a_bias", "a_bias_mean"]].round(3).to_string(index=False))
        print(f"\nmax |MLE - posterior mean| / SE = {j['g_diff_in_se'].abs().max():.3f} "
              f"(check: < 1); max R-hat = {j[['g_rhat', 'v_rhat', 'a_bias_rhat']].max().max():.4f}")
        summarise(j, "g", "g_lo", "g_hi", "route 1 MLE (Wald 95 %)")
        summarise(j, "g_mean", "g_hdi_lo", "g_hdi_hi", "route 1 Bayesian (94 % HDI)")


if __name__ == "__main__":
    main()
