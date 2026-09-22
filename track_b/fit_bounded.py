"""
Route 2 (issue #4, Track B): the Brunton-style bounded OU driven by each trial's own evidence stream,
fitted to the network's deadline choices by Monte-Carlo maximum likelihood with common random numbers.

    a_0 = a_bias;  a_t+1 = a_t + (v*e_t - g*a_t)*dt + sqrt(dt)*xi_t;  sticky bound |a| >= B freezes a;
    choice = 1 iff a_T > 0.     dt = 1 ms, T = 750, sigma = 1 fixed (v and B in noise units).
    g > 0 leaky (recency), g < 0 unstable (primacy).

P(choice_i = 1) = mean over M realisations of logistic(a_T / tau), tau = 0.1 (ou_lik.TAU; with B = 50 this
reproduces the analytic route-1 log-likelihood to +0.18 %).  The M x N noise block depends only on
(noise_seed, M, N, T), never on the parameters, so the objective is deterministic (common random numbers).

Bound-hit term (the observable that breaks the leak-vs-bound degeneracy).  The network's observable is
tcross2 = the first ms at which |dv| >= 2.0 (NaN if never).  We identify the model's sticky bound B with
the network's |dv| = 2.0 level -- that is the judgment call, recorded in track_b/RESULTS.md -- and add
    --hit-mode bern   : the Bernoulli log-likelihood of 1[the trial ever crossed]
    --hit-mode cross  : the categorical log-likelihood of the first-crossing-time bin
                        (0-50, 50-100, 100-150, 150-250, 250-400, 400-750 ms, never); subsumes 'bern'
    --hit-mode none   : choices only
to the choice log-likelihood.  Fits are run with and without it.

Optimiser: Nelder-Mead on (g, v, log B, a_bias) from the route-1 MLE with B in {1, 3, 10} (three starts,
to probe the leak/bound degeneracy), then a profile likelihood on g (11 points, re-optimising the other
three, warm-started) with the 1.92-unit drop as the 95 % interval.

Usage (one task):
    python track_b/fit_bounded.py --seed 42 --gain 1.2 --hit-mode none --M 300
    ... --no-profile --max-iter 120        (smoke)
Writes output/track_b/bounded/seed<S>_g<gain>_<hit-mode>.json
"""
import argparse, json, os, pathlib, sys, time
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ou_lik  # noqa: E402
from ou_lik import DT, TAU, load_network, fit_analytic, categorise_cross  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "track_b" / "bounded"
CROSS_EDGES = (0, 50, 100, 150, 250, 400)
N_CAT = len(CROSS_EDGES) + 1


# ------------------------------------------------------------------------- backends
def make_simulator(rel, M, noise_seed, backend="auto"):
    """Returns sim(g, v, B, a_bias) -> (p_choice1 (N,), cat_counts (N_CAT, N) or None).
    Common random numbers: the noise block is fixed at construction."""
    N, T = rel.shape
    if backend == "auto":
        try:
            import jax  # noqa: F401
            backend = "jax"
        except ImportError:
            backend = "numpy"
    if backend == "jax":
        return _make_jax(rel, M, noise_seed), backend
    return _make_numpy(rel, M, noise_seed), backend


def _make_numpy(rel, M, noise_seed):
    N, T = rel.shape

    def sim(g, v, B, a_bias):
        aT, tcross = ou_lik.simulate_bounded(rel, g, v, B, a_bias, M=M, seed=noise_seed)
        p = np.mean(1.0 / (1.0 + np.exp(-np.clip(aT.astype(np.float64) / TAU, -40, 40))), axis=0)
        cat = categorise_cross(np.where(tcross >= T, np.nan, tcross), T=T, edges=CROSS_EDGES)
        counts = np.stack([(cat == k).sum(0) for k in range(N_CAT)], 0).astype(np.float64)
        return p, counts
    return sim


def _make_jax(rel, M, noise_seed):
    import jax, jax.numpy as jnp
    from jax import lax
    N, T = rel.shape
    rng = np.random.default_rng(noise_seed)
    # noise block (T, M, N) float16, generated once: the common random numbers
    noise = np.empty((T, M, N), np.float16)
    for t in range(T):
        noise[t] = rng.standard_normal((M, N), dtype=np.float32).astype(np.float16)
    noise = jnp.asarray(noise)
    relj = jnp.asarray(rel.T)                                   # (T, N)
    edges = jnp.asarray(CROSS_EDGES[1:], jnp.float32)
    sq = np.float32(np.sqrt(DT)); dt = np.float32(DT)

    @jax.jit
    def _run(g, v, B, a_bias):
        def step(carry, inp):
            a, frozen, tcross = carry
            e_t, xi, t = inp
            da = (v * e_t[None, :] - g * a) * dt + sq * xi.astype(jnp.float32)
            a = jnp.where(frozen, a, a + da)
            newly = (~frozen) & (jnp.abs(a) >= B)
            tcross = jnp.where(newly, t.astype(jnp.float32), tcross)
            return (a, frozen | newly, tcross), None
        a0 = jnp.full((M, N), a_bias, jnp.float32)
        f0 = jnp.zeros((M, N), bool)
        t0 = jnp.full((M, N), np.float32(T))
        (aT, _, tcross), _ = lax.scan(step, (a0, f0, t0),
                                      (relj, noise, jnp.arange(T, dtype=jnp.float32)))
        p = jnp.mean(jax.nn.sigmoid(aT / TAU), axis=0)
        cat = jnp.where(tcross >= T, N_CAT - 1, jnp.searchsorted(edges, tcross, side="right"))
        counts = jnp.stack([(cat == k).sum(0) for k in range(N_CAT)], 0)
        return p, counts.astype(jnp.float32)

    def sim(g, v, B, a_bias):
        p, c = _run(np.float32(g), np.float32(v), np.float32(B), np.float32(a_bias))
        return np.asarray(p, np.float64), np.asarray(c, np.float64)
    return sim


# ------------------------------------------------------------------------ objective
def make_objective(sim, y, hit=None, cross_cat=None, hit_mode="none", eps=1e-4, M=300):
    N = len(y)
    idx = np.arange(N)

    def nll(theta):
        g, v, logB, a_bias = (float(x) for x in theta)
        B = float(np.exp(logB))
        if not (-40 <= g <= 40 and 0 < v <= 1e4 and 0.05 <= B <= 500 and abs(a_bias) <= 20):
            return 1e9
        p, counts = sim(g, v, B, a_bias)
        p = np.clip(p, eps, 1 - eps)
        out = -float(np.sum(np.where(y == 1, np.log(p), np.log1p(-p))))
        if hit_mode != "none":
            pk = np.clip(counts / counts.sum(0, keepdims=True), eps, None)
            pk /= pk.sum(0, keepdims=True)
            if hit_mode == "bern":
                ph = np.clip(1.0 - pk[N_CAT - 1], eps, 1 - eps)
                out -= float(np.sum(np.where(hit == 1, np.log(ph), np.log1p(-ph))))
            elif hit_mode == "cross":
                out -= float(np.sum(np.log(pk[cross_cat, idx])))
            else:
                raise ValueError(hit_mode)
        return out if np.isfinite(out) else 1e9
    return nll


def nm(nll, x0, steps, maxiter):
    from scipy.optimize import minimize
    x0 = np.asarray(x0, float)
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
    return minimize(nll, x0, method="Nelder-Mead",
                    options=dict(initial_simplex=simplex, maxiter=maxiter, maxfev=maxiter,
                                 xatol=1e-3, fatol=1e-3))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gain", type=float, required=True)
    ap.add_argument("--hit-mode", choices=["none", "bern", "cross"], default="none")
    ap.add_argument("--M", type=int, default=300)
    ap.add_argument("--n-trials", type=int, default=None)
    ap.add_argument("--data-dir", type=pathlib.Path, default=None)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--profile-iter", type=int, default=80)
    ap.add_argument("--b-starts", type=float, nargs="+", default=[1.0, 3.0, 10.0])
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--n-profile", type=int, default=11)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--tag", default="")
    # for recovery: fit synthetic choices instead of the network's
    ap.add_argument("--synthetic", type=pathlib.Path, default=None,
                    help="npz with y (and optionally hit/cross_cat) to fit instead of the network data")
    a = ap.parse_args()
    out_dir = a.out or OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    d = load_network(a.seed, gain=a.gain, n_trials=a.n_trials,
                     data_dir=a.data_dir or (ROOT / "data" / "processed" / "kernel"))
    rel = d["rel"]
    N, T = rel.shape
    y = d[f"choice_g{a.gain}"]
    tc = d.get(f"tcross2_g{a.gain}")
    if a.synthetic is not None:
        syn = np.load(a.synthetic)
        y = np.asarray(syn["y"], int)[:N]
        tc = np.asarray(syn["tcross"], float)[:N] if "tcross" in syn.files else tc
    if tc is None and a.hit_mode != "none":
        raise SystemExit("tcross2 not in the npz -- re-run kernel_fit/export_evidence.py")
    hit = (~np.isnan(tc)).astype(int) if tc is not None else None
    cross_cat = categorise_cross(tc, T=T, edges=CROSS_EDGES) if tc is not None else None

    noise_seed = 10_000 + a.seed * 10 + int(round(a.gain * 10))
    sim, backend = make_simulator(rel, a.M, noise_seed, a.backend)
    nll = make_objective(sim, y, hit, cross_cat, a.hit_mode, M=a.M)

    r1 = fit_analytic(rel, y)
    print(f"route-1 MLE: g = {r1['g']:+.3f} +/- {r1['g_se']:.3f}, v = {r1['v']:.2f}, "
          f"a_bias = {r1['a_bias']:+.3f}, nll = {r1['nll']:.2f}", flush=True)
    t0 = time.time(); f_probe = nll([r1["g"], r1["v"], np.log(50.0), r1["a_bias"]])
    print(f"backend {backend}, M = {a.M}: first evaluation (incl. compile) {time.time()-t0:.1f}s, "
          f"nll(B = 50) = {f_probe:.2f}", flush=True)
    t0 = time.time(); nll([r1["g"], r1["v"], np.log(3.0), r1["a_bias"]])
    per_eval = time.time() - t0
    print(f"per evaluation: {per_eval:.2f}s", flush=True)

    best = None
    for B0 in a.b_starts:
        x0 = [r1["g"], r1["v"], np.log(B0), r1["a_bias"]]
        r = nm(nll, x0, steps=[0.8, 5.0, 0.4, 0.15], maxiter=a.max_iter)
        r = nm(nll, r.x, steps=[0.3, 2.0, 0.15, 0.05], maxiter=a.max_iter // 2)
        print(f"  B0 = {B0}: nll {r.fun:.3f} g = {r.x[0]:+.3f} v = {r.x[1]:.2f} "
              f"B = {np.exp(r.x[2]):.3f} a_bias = {r.x[3]:+.3f} ({r.nit} it)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    g_hat, v_hat, B_hat, a_hat = best.x[0], best.x[1], float(np.exp(best.x[2])), best.x[3]
    nll_hat = float(best.fun)

    # diagnostics at the optimum
    p, counts = sim(g_hat, v_hat, B_hat, a_hat)
    pk = counts / counts.sum(0, keepdims=True)
    model_hit = float(np.mean(1 - pk[N_CAT - 1]))
    net_hit = float(np.mean(hit)) if hit is not None else float("nan")
    acc_model = float(np.mean((p > 0.5) == (d["coh"] > 0)))

    prof = []
    if not a.no_profile:
        delta = max(1.0, 3 * r1["g_se"])
        grid = g_hat + delta * np.linspace(-1, 1, a.n_profile)
        warm = [v_hat, np.log(B_hat), a_hat]
        for direction in (grid[grid >= g_hat], grid[grid < g_hat][::-1]):
            w = list(warm)
            for gg in direction:
                def nll_g(th, gg=gg):
                    return nll([gg, th[0], th[1], th[2]])
                r = nm(nll_g, w, steps=[3.0, 0.2, 0.08], maxiter=a.profile_iter)
                w = list(r.x)
                prof.append(dict(g=float(gg), nll=float(r.fun), v=float(r.x[0]),
                                 B=float(np.exp(r.x[1])), a_bias=float(r.x[2])))
                print(f"  profile g = {gg:+.3f}: nll {r.fun:.3f} (drop {r.fun - nll_hat:+.3f}) "
                      f"B = {np.exp(r.x[1]):.3f}", flush=True)
        prof.sort(key=lambda z: z["g"])
    g_lo, g_hi = profile_interval(prof, g_hat, nll_hat) if prof else (np.nan, np.nan)

    res = dict(seed=a.seed, gain=a.gain, hit_mode=a.hit_mode, M=a.M, tau=TAU, n_trials=N,
               backend=backend, g=float(g_hat), v=float(v_hat), B=float(B_hat), a_bias=float(a_hat),
               nll=nll_hat, g_lo=float(g_lo), g_hi=float(g_hi),
               r1_g=r1["g"], r1_g_se=r1["g_se"], r1_v=r1["v"], r1_a_bias=r1["a_bias"], r1_nll=r1["nll"],
               model_hit_frac=model_hit, net_hit_frac=net_hit, acc_model=acc_model,
               acc_net=float(np.mean(y == (d["coh"] > 0))), per_eval_s=per_eval,
               minutes=(time.time() - t_start) / 60, profile=prof,
               synthetic=str(a.synthetic) if a.synthetic else None)
    name = a.tag or f"seed{a.seed}_g{a.gain}_{a.hit_mode}"
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"\nRESULT seed {a.seed} gain {a.gain} [{a.hit_mode}]: g = {g_hat:+.3f} "
          f"[{g_lo:+.3f}, {g_hi:+.3f}], B = {B_hat:.3f}, v = {v_hat:.2f}, a_bias = {a_hat:+.3f}, "
          f"nll = {nll_hat:.2f}, model hit {model_hit:.3f} vs network {net_hit:.3f}, "
          f"{res['minutes']:.1f} min -> {out_dir / (name + '.json')}", flush=True)


def profile_interval(prof, g_hat, nll_hat, drop=1.92):
    """Linear interpolation of the 1.92-unit drop on each side of the profile."""
    g = np.array([p["g"] for p in prof]); f = np.array([p["nll"] for p in prof]) - nll_hat
    lo, hi = np.nan, np.nan
    left = (g <= g_hat)
    if left.sum() > 1:
        gl, fl = g[left][::-1], f[left][::-1]
        for i in range(1, len(gl)):
            if fl[i] >= drop:
                lo = gl[i - 1] + (gl[i] - gl[i - 1]) * (drop - fl[i - 1]) / (fl[i] - fl[i - 1]); break
        else:
            lo = gl[-1]
    right = (g >= g_hat)
    if right.sum() > 1:
        gr, fr = g[right], f[right]
        for i in range(1, len(gr)):
            if fr[i] >= drop:
                hi = gr[i - 1] + (gr[i] - gr[i - 1]) * (drop - fr[i - 1]) / (fr[i] - fr[i - 1]); break
        else:
            hi = gr[-1]
    return lo, hi


if __name__ == "__main__":
    main()
