"""
Track A (issue #3): corrected HSSM OU fits to the constant-bound (b = 1.5) RNN behaviour.

What is corrected relative to the original `model/fit_ornstein.py` fits (issue #1):
  * constant-bound data (issue #2), 20 networks, not the Weibull-collapse data, not seed 42 only;
  * time stretch k (default 10) so the networks' OU description falls INSIDE the pretrained LAN box
    (RT x k  <=>  g -> g/k, a -> a*sqrt(k), v -> v/sqrt(k));
  * non-decision time FIXED (never fitted) at t = c + k * min(rt_native) with c = 0.3 s, passed as a float;
  * no lapse mixture (p_outlier=None, lapse=None) -- the default 5 % Uniform(0, 20 s) lapse conflicts with
    the 750 ms horizon;
  * choice coding: `response_choice` (+1 = choice "+"), drift regressed on SIGNED coherence, because
    accuracy coding is arbitrary at zero coherence and those trials carry most of the information about g.

SIGN CONVENTION (ssm-simulators / HSSM): dx = (v - g*x) dt + dW, so **g > 0 = LEAKY (recency)** and
**g < 0 = UNSTABLE / attractive (primacy)**. Boundaries at -a and +a; z = relative start point.

Native-unit translation of a fit at stretch k:
    g_nat = g * k        a_nat = a / sqrt(k)      v_nat(coh) = v(coh) * sqrt(k)      t_nat = (t - 0.3) / k

Outputs (output/track_a/):
    <tag>_summary.csv   one row per parameter: mean, sd, hdi_3%, hdi_97%, r_hat, ess_bulk, ess_tail
                        (+ rows for the derived quantities v(0.15), v(0), v(-0.15) and the native-unit values)
    <tag>_meta.json     n, k, t, omission fraction, divergences, minutes, edge mass, priors, data facts
    <tag>.nc            full InferenceData (gitignored; keep on Oscar)

Usage:
    python track_a/fit_hssm_ou.py --gain 1.0 --k 10 --n-sub 2000 --smoke
    python track_a/fit_hssm_ou.py --gain 1.0 --k 10 --n-sub 20000
    python track_a/fit_hssm_ou.py --gain 1.0 --k 10 --seed-filter 42
"""
import numpyro

numpyro.set_host_device_count(4)

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd

import pytensor

pytensor.config.floatX = "float32"
from jax import config as jax_config

jax_config.update("jax_enable_x64", False)

import arviz as az
import hssm

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "fixed_bound"
OUT = ROOT / "output" / "track_a"

OFFSET = 0.3  # s, added to the stretched RTs so that t stays inside the LAN box interior
COH_REF = 0.15  # reference (max) signed coherence for the derived drift v(0.15)
TRIALS_PER_SEED = 2000  # trials simulated per network per gain before omissions were dropped

# Pretrained ssm-simulators `ornstein` LAN box
BOX = dict(v=(-2.0, 2.0), a=(0.3, 3.0), z=(0.1, 0.9), g=(-1.0, 1.0), t=(1e-3, 2.0))
EDGE_TOL = 0.02


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gain", type=float, required=True, choices=[0.8, 1.0, 1.2])
    p.add_argument("--k", type=float, default=10.0, help="time stretch: rt_fit = k * rt_native + 0.3")
    p.add_argument("--bound", type=float, default=1.5)
    p.add_argument("--n-sub", type=int, default=0, help="stratified subsample size (0 = all trials)")
    p.add_argument("--seed-filter", type=int, default=0, help="fit one network only (its RNN seed)")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.95)
    p.add_argument("--smoke", action="store_true", help="1 chain, 20 tune / 20 draws; pipeline check only")
    p.add_argument("--rng", type=int, default=0, help="seed for the subsample and the sampler")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--n-draws-csv", type=int, default=400,
                   help="thinned posterior draws written to <tag>_draws.csv for the PPC scripts")
    p.add_argument("--data-csv", type=str, default="", help="override the input CSV (appendix fits)")
    p.add_argument("--default-priors", action="store_true",
                   help="use HSSM's own default priors instead of the explicit box-uniform ones")
    return p.parse_args()


def box_uniform_priors():
    """Uniform priors spanning exactly the LAN box.

    For the drift regression the box constrains v(coh) = b0 + b1*coh, not the coefficients themselves;
    we give b0 the full v box and b1 the range that keeps |b1 * 0.15| <= 2, and let the likelihood
    (which is -inf outside the box in HSSM) handle the joint constraint.
    """
    return {
        "v": {
            "Intercept": {"name": "Uniform", "lower": BOX["v"][0], "upper": BOX["v"][1]},
            "coherence_signed": {"name": "Uniform",
                                 "lower": BOX["v"][0] / COH_REF, "upper": BOX["v"][1] / COH_REF},
        },
        "a": {"name": "Uniform", "lower": BOX["a"][0], "upper": BOX["a"][1]},
        "z": {"name": "Uniform", "lower": BOX["z"][0], "upper": BOX["z"][1]},
        "g": {"name": "Uniform", "lower": BOX["g"][0], "upper": BOX["g"][1]},
    }


def load_data(args):
    if args.data_csv:
        path = pathlib.Path(args.data_csv)
    else:
        path = DATA / f"hssm_ready_nxx1_fixed_b{args.bound}_g{args.gain}.csv"
    raw = pd.read_csv(path)
    n_all = len(raw)
    n_seeds_all = int(raw.seed.nunique())
    if args.seed_filter:
        raw = raw[raw.seed == args.seed_filter].reset_index(drop=True)
        if not len(raw):
            raise SystemExit(f"no trials for seed {args.seed_filter} in {path}")
    n_kept_full = len(raw)
    n_seeds = int(raw.seed.nunique())
    # omissions were dropped when the data were made: TRIALS_PER_SEED per network per gain were simulated
    omission_frac = 1.0 - n_kept_full / (TRIALS_PER_SEED * n_seeds)

    rt_min_native = float(raw.rt.min())  # min RT of the FULL dataset of that gain (before subsampling)
    sub_note = "all"
    if args.n_sub and args.n_sub < len(raw):
        rng = np.random.default_rng(args.rng + 7)
        per = int(np.ceil(args.n_sub / n_seeds))
        parts = []
        for s, d in raw.groupby("seed"):
            take = min(per, len(d))
            parts.append(d.iloc[rng.choice(len(d), take, replace=False)])
        sub = pd.concat(parts)
        if len(sub) > args.n_sub:
            sub = sub.iloc[rng.choice(len(sub), args.n_sub, replace=False)]
        raw = sub.sort_index().reset_index(drop=True)
        sub_note = f"stratified {per}/seed -> {len(raw)}"

    df = pd.DataFrame({
        "rt": args.k * raw.rt.values + OFFSET,
        "response": raw.response_choice.values.astype(float),
        "coherence_signed": raw.coherence_signed.values.astype(float),
    })
    meta = dict(
        data_csv=str(path), n_all_rows=n_all, n_seeds_all=n_seeds_all, n_seeds=n_seeds,
        n_rows_after_seed_filter=n_kept_full, n_fitted=int(len(df)), subsample=sub_note,
        omission_frac=float(omission_frac), rt_min_native=rt_min_native,
        rt_median_native=float(raw.rt.median()), rt_max_native=float(raw.rt.max()),
        p_choice_plus=float((raw.response_choice == 1).mean()),
        accuracy=float((raw.response == 1).mean()),
    )
    return df, meta, rt_min_native


def edge_mass(samples, lo, hi, tol=EDGE_TOL):
    s = np.asarray(samples).ravel()
    return float(np.mean((s <= lo + tol) | (s >= hi - tol)))


def main():
    args = parse_args()
    k = args.k
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (
        f"g{args.gain}_k{k:g}_b{args.bound}"
        + (f"_s{args.seed_filter}" if args.seed_filter else "_pooled")
        + (f"_n{args.n_sub}" if args.n_sub else "")
        + ("_smoke" if args.smoke else "")
    )
    print(f"=== {tag} ===", flush=True)

    df, dmeta, rt_min_native = load_data(args)
    t_fixed = round(OFFSET + k * rt_min_native, 6)
    below = int((df.rt <= t_fixed).sum())
    print(f"n = {len(df)}  k = {k}  t_fixed = {t_fixed:.4f} s  (min stretched RT {df.rt.min():.4f})", flush=True)
    print(f"RT stretched: min {df.rt.min():.3f} med {df.rt.median():.3f} max {df.rt.max():.3f} s;"
          f" P(response=+1) = {(df.response == 1).mean():.3f}; omissions {dmeta['omission_frac']*100:.2f} %",
          flush=True)
    if below:
        raise SystemExit(f"{below} RTs are <= the fixed t ({t_fixed}); refusing to fit")
    if not (BOX["t"][0] <= t_fixed <= BOX["t"][1]):
        raise SystemExit(f"fixed t {t_fixed} outside the LAN box {BOX['t']}")

    priors = None if args.default_priors else box_uniform_priors()
    include = [dict(name="v", formula="v ~ 1 + coherence_signed", link="identity")]
    if priors is not None:
        include[0]["prior"] = priors["v"]
    kw = {} if priors is None else dict(a=priors["a"], z=priors["z"], g=priors["g"])

    model = hssm.HSSM(
        data=df,
        model="ornstein",
        loglik_kind="approx_differentiable",
        p_outlier=None,
        lapse=None,
        t=t_fixed,            # FIXED, passed as a float -- never fitted
        include=include,
        **kw,
    )
    print(model, flush=True)

    draws, tune, chains = (20, 20, 1) if args.smoke else (args.draws, args.tune, args.chains)
    sampler = "numpyro" if hssm.__version__ >= "0.3" else "nuts_numpyro"
    t0 = time.time()
    idata = model.sample(sampler=sampler, chains=chains, cores=chains, draws=draws, tune=tune,
                         target_accept=args.target_accept, random_seed=1000 + args.rng,
                         idata_kwargs=dict(log_likelihood=False))
    minutes = (time.time() - t0) / 60
    print(f"sampling took {minutes:.1f} min", flush=True)

    post = idata.posterior
    params = [p for p in ["v_Intercept", "v_coherence_signed", "a", "z", "g"] if p in post]
    summ = az.summary(idata, var_names=params, hdi_prob=0.94).reset_index().rename(columns={"index": "param"})

    # derived drifts and native-unit versions, with the same summary columns
    b0 = post["v_Intercept"].values
    b1 = post["v_coherence_signed"].values if "v_coherence_signed" in post else np.zeros_like(b0)
    derived = {
        "v(0.15)": b0 + COH_REF * b1,
        "v(0)": b0,
        "v(-0.15)": b0 - COH_REF * b1,
        "g_native": post["g"].values * k,
        "a_native": post["a"].values / np.sqrt(k),
        "v_native(0.15)": (b0 + COH_REF * b1) * np.sqrt(k),
    }
    der = az.summary(az.convert_to_dataset({n: v for n, v in derived.items()}), hdi_prob=0.94)
    der = der.reset_index().rename(columns={"index": "param"})
    summ = pd.concat([summ, der], ignore_index=True)

    edges = dict(
        a=edge_mass(post["a"].values, *BOX["a"]),
        g=edge_mass(post["g"].values, *BOX["g"]),
        v_at_0p15=edge_mass(derived["v(0.15)"], *BOX["v"]),
        v_at_0=edge_mass(derived["v(0)"], *BOX["v"]),
        z=edge_mass(post["z"].values, *BOX["z"]),
    )
    n_div = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else -1
    total = int(draws * chains)

    for col, val in dict(gain=args.gain, k=k, bound=args.bound, seed_filter=args.seed_filter,
                         n=len(df), t_fixed=t_fixed, tag=tag).items():
        summ[col] = val
    summ.to_csv(OUT / f"{tag}_summary.csv", index=False)
    print(summ[["param", "mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"]].to_string(index=False),
          flush=True)

    meta = dict(
        tag=tag, gain=args.gain, k=k, bound=args.bound, seed_filter=args.seed_filter,
        offset_s=OFFSET, t_fixed=t_fixed, t_native=(t_fixed - OFFSET) / k,
        draws=draws, tune=tune, chains=chains, total_draws=total, target_accept=args.target_accept,
        n_divergences=n_div, divergence_frac=(n_div / total if total else np.nan),
        r_hat_max=float(np.nanmax(summ.r_hat.values)), ess_bulk_min=float(np.nanmin(summ.ess_bulk.values)),
        sampling_minutes=minutes, edge_mass=edges, edge_tol=EDGE_TOL, box=BOX,
        priors="box uniform (explicit)" if priors is not None else "hssm defaults",
        prior_spec=priors, smoke=args.smoke, hssm_version=hssm.__version__, sampler=sampler,
        posterior_mean={p: float(post[p].values.mean()) for p in params},
        derived_mean={n: float(v.mean()) for n, v in derived.items()},
        **dmeta,
    )
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print("divergences:", n_div, "of", total, "| edge mass:", {kk: round(vv, 4) for kk, vv in edges.items()},
          flush=True)

    # thinned posterior draws of the free parameters, so the PPC scripts can run without the netcdf
    flat = {p: post[p].values.reshape(-1) for p in params}
    n_flat = len(next(iter(flat.values())))
    take = np.linspace(0, n_flat - 1, min(args.n_draws_csv, n_flat)).astype(int)
    draws_df = pd.DataFrame({p: v[take] for p, v in flat.items()})
    draws_df["t"] = t_fixed
    draws_df["k"] = k
    draws_df["gain"] = args.gain
    draws_df["seed_filter"] = args.seed_filter
    draws_df.to_csv(OUT / f"{tag}_draws.csv", index=False)
    print("saved", OUT / f"{tag}_draws.csv", f"({len(draws_df)} draws)", flush=True)

    idata.to_netcdf(str(OUT / f"{tag}.nc"))
    print("saved", OUT / f"{tag}.nc", flush=True)


if __name__ == "__main__":
    main()
