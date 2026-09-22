"""
Likelihoods for the evidence-conditioned OU accumulator (issue #4, Track B).

Model (network's own time, dt = 1 ms, T = 750, sigma = 1 FIXED, so v and B are in noise units):

    a_0   = a_bias
    a_t+1 = a_t + (v * e_t - g * a_t) * dt + sqrt(dt) * xi_t ,   xi_t ~ N(0, 1)
    sticky bound: once |a| >= B the accumulator freezes for the rest of the trial
    choice = 1 ("+") iff a_T > 0                                 (the network's DEADLINE choice)

Sign convention as everywhere in this repo: **g > 0 leaky (recency), g < 0 unstable (primacy)**.
The landscape "drift slope" has the opposite sign.

Route 1 (B = infinity): a_T is Gaussian given the evidence stream,
    rho   = 1 - g*dt
    mu_i  = a_bias * rho^T + v * dt * sum_t rho^(T-1-t) e_it
    s^2   =            dt  * sum_t rho^(2(T-1-t))          (trial-independent)
    P(choice_i = 1) = Phi(mu_i / s)
computed with scipy's log-cdf for stability.

Route 2 (sticky bound): Monte-Carlo likelihood with common random numbers -- M realisations per trial
driven by the SAME noise block for every parameter evaluation (the noise depends only on (seed, M, N, T)).
P(choice = 1) is smoothed with a logistic kernel on a_T (scale TAU) so the objective is continuous.
The model's bound-hit observables (fraction and first-crossing time) come out of the same simulation.

Unit tests (`python track_b/ou_lik.py --test`):
  (i)  analytic P(choice) vs Monte Carlo (20 000 realisations, 20 random trials): max |delta| < 0.01
  (ii) route-1 MLEs on seed 42 reproduce the coordinator's smoke numbers
       (g = +4.55 / +0.75 / -2.04 per s at gains 0.8 / 1.0 / 1.2) within 0.05
  (iii) MC log-likelihood with a huge bound matches the analytic log-likelihood within 1 %
  (iv) the MC simulator reproduces kernel_fit/fit_ou_kernel.py::simulate exactly (same choices)
"""
import argparse, pathlib, sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "kernel"
DT = 0.001
TAU = 0.10          # logistic smoothing scale for the MC choice probability (test iii: +0.18 % on the nll)
GAINS = (0.8, 1.0, 1.2)


# ----------------------------------------------------------------------------- data
def load_network(seed, gain=None, n_trials=None, data_dir=DATA):
    """rel as float32 (float16 on disk), the deadline choices and the bound-hit observables."""
    d = np.load(pathlib.Path(data_dir) / f"seed{seed}.npz")
    out = dict(rel=d["rel"].astype(np.float32), coh=np.asarray(d["coh_signed"], float),
               labels=np.asarray(d["labels"]))
    for gn in (GAINS if gain is None else [gain]):
        out[f"choice_g{gn}"] = np.asarray(d[f"choice_g{gn}"], int)
        out[f"dvT_g{gn}"] = np.asarray(d[f"dvT_g{gn}"], float)
        for nm in ("tcross2", "tcross15"):
            if f"{nm}_g{gn}" in d.files:
                out[f"{nm}_g{gn}"] = np.asarray(d[f"{nm}_g{gn}"], float)
    if n_trials is not None:
        for k, val in out.items():
            out[k] = val[:n_trials]
    return out


# ----------------------------------------------------------------- route 1: analytic
def _rho(g):
    return 1.0 - g * DT


def filter_weights(g, T):
    """w_k = rho^(T-1-t) for t = 0..T-1 (so w[0] is the oldest sample's weight)."""
    k = np.arange(T - 1, -1, -1, dtype=np.float64)
    r = _rho(g)
    return np.exp(k * np.log(r)) if r > 0 else r ** k


def analytic_moments(rel, g, v, a_bias):
    """mu (per trial) and s (scalar) of a_T for the unbounded OU driven by rel."""
    T = rel.shape[1]
    w = filter_weights(g, T)
    mu = v * DT * (rel @ w) + a_bias * _rho(g) ** T
    s = np.sqrt(DT * np.sum(w ** 2))
    return mu, s


def analytic_p(rel, g, v, a_bias):
    mu, s = analytic_moments(rel, g, v, a_bias)
    return norm.cdf(mu / s)


def nll_analytic(theta, rel, y, ridge=0.0):
    """-log P(choices | evidence) for the unbounded OU.  theta = (g, v, a_bias)."""
    g, v, a_bias = float(theta[0]), float(theta[1]), float(theta[2])
    if not (-40 <= g <= 40 and 0 < v <= 1e4 and abs(a_bias) <= 20):
        return 1e12
    mu, s = analytic_moments(rel, g, v, a_bias)
    z = mu / s
    ll = np.where(y == 1, norm.logcdf(z), norm.logcdf(-z))
    out = -float(np.sum(ll))
    return out + ridge * (g ** 2) if np.isfinite(out) else 1e12


def fit_analytic(rel, y, g0s=(-5.0, 0.0, 5.0), v0=50.0, a0=0.0, verbose=False):
    """MLE of (g, v, a_bias) by Nelder-Mead from several g starts + a Powell polish.
    Returns dict with the estimates, SEs from a numerical Hessian, nll and n evaluations."""
    best = None
    for g0 in g0s:
        x0 = np.array([g0, v0, a0], float)
        simplex = np.vstack([x0, x0 + [1.0, 0, 0], x0 + [0, 10.0, 0], x0 + [0, 0, 0.2]])
        r = minimize(nll_analytic, x0, args=(rel, y), method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxiter=4000, maxfev=4000,
                                  xatol=1e-6, fatol=1e-8))
        r = minimize(nll_analytic, r.x, args=(rel, y), method="Powell",
                     options=dict(maxiter=4000, xtol=1e-7, ftol=1e-10))
        if verbose:
            print(f"    g0={g0:+.1f} -> nll {r.fun:.4f} g={r.x[0]:+.4f} v={r.x[1]:.2f} a={r.x[2]:+.4f}")
        if best is None or r.fun < best.fun:
            best = r
    se = hessian_se(lambda th: nll_analytic(th, rel, y), best.x)
    return dict(g=float(best.x[0]), v=float(best.x[1]), a_bias=float(best.x[2]),
                g_se=se[0], v_se=se[1], a_bias_se=se[2], nll=float(best.fun), n=len(y))


def hessian_se(f, x, rel_step=1e-4):
    """SEs from a central-difference Hessian of the negative log-likelihood."""
    x = np.asarray(x, float)
    h = np.maximum(np.abs(x) * rel_step, 1e-5)
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp, xm = x.copy(), x.copy()
                xp[i] += h[i]; xm[i] -= h[i]
                H[i, i] = (f(xp) - 2 * f0 + f(xm)) / h[i] ** 2
            else:
                a, b, c, d = x.copy(), x.copy(), x.copy(), x.copy()
                a[i] += h[i]; a[j] += h[j]
                b[i] += h[i]; b[j] -= h[j]
                c[i] -= h[i]; c[j] += h[j]
                d[i] -= h[i]; d[j] -= h[j]
                H[i, j] = H[j, i] = (f(a) - f(b) - f(c) + f(d)) / (4 * h[i] * h[j])
    try:
        cov = np.linalg.inv(H)
        return np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        return np.full(n, np.nan)


# --------------------------------------------------------- route 2: bounded simulator
def simulate_bounded(rel, g, v, B, a_bias, M=300, seed=0, return_paths=False):
    """M realisations per trial with common random numbers (the noise block depends only on
    (seed, M, N, T), never on the parameters).  Returns (a_T, tcross) both (M, N) float32;
    tcross = first step index with |a| >= B, or T if the trial never hits."""
    N, T = rel.shape
    rng = np.random.default_rng(seed)
    a = np.full((M, N), np.float32(a_bias))
    frozen = np.zeros((M, N), bool)
    tcross = np.full((M, N), T, np.int32)
    sq = np.float32(np.sqrt(DT))
    dt = np.float32(DT)
    gv, vv = np.float32(g), np.float32(v)
    for t in range(T):
        xi = rng.standard_normal((M, N), dtype=np.float32)
        if frozen.all():
            continue                                   # keep drawing: common random numbers stay aligned
        da = (vv * rel[:, t] - gv * a) * dt + sq * xi
        a = np.where(frozen, a, a + da)
        newly = (~frozen) & (np.abs(a) >= B)
        if newly.any():
            tcross[newly] = t
            frozen |= newly
    return a, tcross


def mc_probs(rel, g, v, B, a_bias, M=300, seed=0, tau=TAU, T=None):
    """P(choice = 1) per trial (logistic-smoothed) and the model's first-crossing times."""
    aT, tcross = simulate_bounded(rel, g, v, B, a_bias, M=M, seed=seed)
    p = np.mean(1.0 / (1.0 + np.exp(-np.clip(aT.astype(np.float64) / tau, -40, 40))), axis=0)
    return p, tcross


def crossing_bins(T=750, edges=(0, 50, 100, 150, 250, 400)):
    """Categorical bins for the first-crossing time: 6 time bins + 'never' (index 6)."""
    return np.array(list(edges) + [T], float)


def categorise_cross(tc, T=750, edges=(0, 50, 100, 150, 250, 400)):
    """tc in ms (NaN or >= T = never) -> category 0..len(edges) (last = never)."""
    tc = np.asarray(tc, float)
    cat = np.digitize(np.nan_to_num(tc, nan=T + 1.0), np.array(edges[1:], float), right=False)
    cat = np.where(np.isnan(tc) | (tc >= T), len(edges), cat)
    return cat.astype(int)


def nll_bounded(theta, rel, y, hit=None, M=300, seed=0, tau=TAU, hit_mode="none",
                cross_cat=None, eps=1e-4):
    """-log P(choices [, bound-hit observable] | evidence) for the bounded OU.
    theta = (g, v, B, a_bias).
      hit_mode 'none'  : choices only
      hit_mode 'bern'  : + Bernoulli log-likelihood of 1[the network's |dv| reached 2.0 before T]
      hit_mode 'cross' : + categorical log-likelihood of the network's first-crossing-time bin
                         (6 time bins + 'never'); subsumes 'bern'.
    Both hit modes identify the model's sticky bound B with the network's |dv| = 2.0 level."""
    g, v, B, a_bias = (float(x) for x in theta)
    if not (-40 <= g <= 40 and 0 < v <= 1e4 and 0.05 <= B <= 200 and abs(a_bias) <= 20):
        return 1e12
    N, T = rel.shape
    aT, tcross = simulate_bounded(rel, g, v, B, a_bias, M=M, seed=seed)
    p = np.mean(1.0 / (1.0 + np.exp(-np.clip(aT.astype(np.float64) / tau, -40, 40))), axis=0)
    p = np.clip(p, eps, 1 - eps)
    nll = -float(np.sum(np.where(y == 1, np.log(p), np.log1p(-p))))
    if hit_mode == "bern":
        ph = np.clip((tcross < T).mean(0), eps, 1 - eps)
        nll -= float(np.sum(np.where(hit == 1, np.log(ph), np.log1p(-ph))))
    elif hit_mode == "cross":
        K = cross_cat.max() + 1 if cross_cat is not None else 7
        mc = categorise_cross(np.where(tcross >= T, np.nan, tcross), T=T)      # (M, N)
        counts = np.stack([(mc == k).sum(0) for k in range(7)], 0).astype(float)   # (7, N)
        pk = np.clip(counts / mc.shape[0], eps, None)
        pk /= pk.sum(0, keepdims=True)
        nll -= float(np.sum(np.log(pk[cross_cat, np.arange(N)])))
        _ = K
    elif hit_mode != "none":
        raise ValueError(hit_mode)
    return nll if np.isfinite(nll) else 1e12


# ----------------------------------------------------------------------------- tests
def _test_analytic_vs_mc(verbose=True):
    d = load_network(42)
    rel = d["rel"]
    rng = np.random.default_rng(7)
    idx = rng.choice(len(rel), 20, replace=False)
    sub = rel[idx]
    worst = 0.0
    rows = []
    for (g, v, a_bias) in [(4.5, 55.0, 0.0), (0.0, 50.0, 0.2), (-2.0, 50.0, -0.1), (10.0, 80.0, 0.5)]:
        pa = analytic_p(sub, g, v, a_bias)
        aT, _ = simulate_bounded(sub, g, v, np.inf, a_bias, M=20000, seed=11)
        pm = (aT > 0).mean(0)
        dmax = float(np.max(np.abs(pa - pm)))
        rows.append((g, v, a_bias, dmax))
        worst = max(worst, dmax)
        if verbose:
            print(f"  g={g:+.1f} v={v:.0f} a_bias={a_bias:+.1f}: max|P_analytic - P_MC(20000)| = {dmax:.4f}")
    ok = worst < 0.01
    print(f"  TEST (i) analytic vs Monte Carlo: max |delta| = {worst:.4f} over 4x20 trials -> {'PASS' if ok else 'FAIL'}")
    return ok, rows


def _test_smoke_mles(verbose=True):
    target = {0.8: 4.55, 1.0: 0.75, 1.2: -2.04}
    d = load_network(42)
    ok = True
    rows = []
    for gn in GAINS:
        t0 = time.time()
        r = fit_analytic(d["rel"], d[f"choice_g{gn}"], verbose=verbose)
        dg = abs(r["g"] - target[gn])
        ok &= dg < 0.05
        rows.append((gn, r["g"], r["g_se"], r["v"], r["a_bias"], target[gn], dg))
        print(f"  gain {gn}: g = {r['g']:+.4f} +/- {r['g_se']:.4f} (smoke {target[gn]:+.2f}, |d| {dg:.4f}), "
              f"v = {r['v']:.2f}, a_bias = {r['a_bias']:+.4f}, nll = {r['nll']:.2f}  [{time.time()-t0:.1f}s]")
    print(f"  TEST (ii) seed-42 route-1 MLEs vs smoke numbers -> {'PASS' if ok else 'FAIL'}")
    return ok, rows


def _test_mc_matches_analytic_ll(taus=(0.2, 0.1, 0.05, 0.02), M=300):
    """MC log-likelihood with a huge bound vs the analytic log-likelihood, as a function of tau."""
    d = load_network(42)
    rel, y = d["rel"], d["choice_g1.2"]
    r = fit_analytic(rel, y)
    th = (r["g"], r["v"], r["a_bias"])
    ana = nll_analytic(th, rel, y)
    aT, tcross = simulate_bounded(rel, th[0], th[1], 50.0, th[2], M=M, seed=3)
    print(f"  analytic nll = {ana:.2f} at g={th[0]:+.3f} v={th[1]:.2f}; MC with B = 50, M = {M}:")
    out = []
    for tau in taus:
        p = np.clip(np.mean(1 / (1 + np.exp(-np.clip(aT.astype(float) / tau, -40, 40))), 0), 1e-4, 1 - 1e-4)
        nll = -float(np.sum(np.where(y == 1, np.log(p), np.log1p(-p))))
        out.append((tau, nll, 100 * (nll - ana) / ana))
        print(f"    tau = {tau:.2f}: MC nll = {nll:.2f}  ({100*(nll-ana)/ana:+.2f} %)")
    ph = (tcross < rel.shape[1]).mean()
    print(f"    (bound never hit at B = 50: model hit fraction {ph:.4f})")
    ok = any(abs(o[2]) < 1.0 for o in out)
    print(f"  TEST (iii) MC vs analytic log-likelihood within 1 % for some tau -> {'PASS' if ok else 'FAIL'}")
    return ok, out


def _test_simulator_matches_kernel_fit():
    """The bounded simulator must reproduce kernel_fit/fit_ou_kernel.py::simulate exactly."""
    sys.path.insert(0, str(ROOT / "kernel_fit"))
    from fit_ou_kernel import simulate as kf_simulate
    d = load_network(42)
    rel = d["rel"][:200]
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(rel.shape, dtype=np.float32)
    v, g, B = 46.5, 3.98, 2.6
    ch_kf = kf_simulate(rel, v, g, B, noise)
    # same recursion, one realisation, same noise block
    N, T = rel.shape
    a = np.zeros(N, np.float32); frozen = np.zeros(N, bool); sq = np.float32(np.sqrt(DT))
    for t in range(T):
        da = (np.float32(v) * rel[:, t] - np.float32(g) * a) * np.float32(DT) + sq * noise[:, t]
        a = np.where(frozen, a, a + da)
        frozen |= np.abs(a) >= B
    ch_mine = (a > 0).astype(int)
    ok = bool(np.array_equal(ch_kf, ch_mine))
    print(f"  TEST (iv) recursion identical to kernel_fit::simulate on 200 trials -> {'PASS' if ok else 'FAIL'}")
    return ok, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        print("TEST (i) analytic P(choice) vs Monte Carlo")
        ok1, _ = _test_analytic_vs_mc()
        print("\nTEST (iv) simulator vs kernel_fit")
        ok4, _ = _test_simulator_matches_kernel_fit()
        print("\nTEST (iii) MC log-likelihood vs analytic (tau sweep)")
        ok3, _ = _test_mc_matches_analytic_ll()
        print("\nTEST (ii) seed-42 route-1 MLEs")
        ok2, _ = _test_smoke_mles()
        print(f"\nALL TESTS {'PASS' if all([ok1, ok2, ok3, ok4]) else 'FAIL'}")
        sys.exit(0 if all([ok1, ok2, ok3, ok4]) else 1)


if __name__ == "__main__":
    main()
