"""
Parameter recovery for the pretrained HSSM Ornstein-Uhlenbeck LAN (issue #1).

Simulates one cell of the recovery grid with ssm-simulators (constant-bound OU),
applies a deadline (drops non-crossers), fits the same HSSM `ornstein` model used in
model/fit_ornstein.py (no coherence regression), and writes:

  output/recovery/cell<ID>_<tag>.nc          full InferenceData (gitignored)
  output/recovery/cell<ID>_<tag>_summary.csv one row per parameter (true, mean, sd, hdi, r_hat, ess)
  output/recovery/cell<ID>_<tag>_meta.json   grid values, n simulated, n dropped, divergences, timing

Grid (20 cells, issue #1): g in {-1, -0.5, 0, 0.5, 1} x a in {1.0, 1.5} x n in {1000, 3700}.
Fixed: v = 0.0, z = 0.5, t = 0.1 s (stretched native non-decision time; no 0.3 s offset).
Deadline: 4.5 s = 750 ms x 6 (time stretch k = 6).

Usage:
  python recovery/simulate_and_fit_ou.py --cell 7
  python recovery/simulate_and_fit_ou.py --cell 7 --smoke     # tiny sampler run, checks the pipeline
"""
import numpyro
numpyro.set_host_device_count(4)

import argparse
import itertools
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
from ssms.basic_simulators.simulator import simulator

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "recovery"

G_GRID = [-1.0, -0.5, 0.0, 0.5, 1.0]
A_GRID = [1.0, 1.5]
N_GRID = [1000, 3700]
GRID = list(itertools.product(G_GRID, A_GRID, N_GRID))  # 20 cells

FIXED = dict(v=0.0, z=0.5, t=0.1)
DEADLINE = 4.5     # s; 750 ms horizon x stretch 6
MAX_T = 20.0       # simulate long, then censor at DEADLINE
DELTA_T = 0.001


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", type=int, required=True, help=f"0..{len(GRID)-1}")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--tune", type=int, default=1000)
    p.add_argument("--smoke", action="store_true", help="tune=draws=20, 1 chain; pipeline check only")
    p.add_argument("--tag", type=str, default="")
    return p.parse_args()


def simulate(g, a, n, seed):
    theta = dict(FIXED, g=g, a=a)
    sim = simulator(theta=theta, model="ornstein", n_samples=n,
                    delta_t=DELTA_T, max_t=MAX_T, random_state=seed)
    rts = np.asarray(sim["rts"]).reshape(-1)
    choices = np.asarray(sim["choices"]).reshape(-1)
    df = pd.DataFrame({"rt": rts, "response": choices})
    # ssms marks non-terminated trials with negative rt (-999) or rt >= max_t; treat both as non-crossers
    n_noterm = int(((df.rt < 0) | (df.rt >= MAX_T)).sum())
    df = df[(df.rt > 0) & (df.rt < MAX_T)]
    n_late = int((df.rt > DEADLINE).sum())
    df = df[df.rt <= DEADLINE].reset_index(drop=True)
    df["response"] = np.where(df["response"] > 0, 1.0, -1.0)
    return df, dict(n_requested=n, n_not_terminated=n_noterm, n_past_deadline=n_late,
                    n_kept=int(len(df)), frac_dropped=float(1 - len(df) / n))


def main():
    args = parse_args()
    g, a, n = GRID[args.cell]
    seed = 1000 + args.cell
    tag = args.tag or (f"g{g:+.1f}_a{a:.1f}_n{n}" + ("_smoke" if args.smoke else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"cell {args.cell}: g={g} a={a} n={n} seed={seed} tag={tag}", flush=True)

    df, sim_meta = simulate(g, a, n, seed)
    print("simulated:", sim_meta, flush=True)
    print(f"  RT median {df.rt.median():.3f} s, 90th {df.rt.quantile(.9):.3f}, "
          f"P(response=+1) {(df.response == 1).mean():.3f}", flush=True)

    model = hssm.HSSM(data=df, model="ornstein", loglik_kind="approx_differentiable")
    print(model, flush=True)

    draws, tune, chains = (20, 20, 1) if args.smoke else (args.draws, args.tune, 4)
    t0 = time.time()
    idata = model.sample(sampler="numpyro", chains=chains, cores=chains, draws=draws, tune=tune,
                         target_accept=0.95, random_seed=seed)
    elapsed = time.time() - t0
    print(f"sampling took {elapsed/60:.1f} min", flush=True)

    truth = dict(FIXED, g=g, a=a)
    params = [p for p in ["v", "a", "z", "g", "t"] if p in idata.posterior]
    summ = az.summary(idata, var_names=params, hdi_prob=0.94)
    summ.insert(0, "true", [truth[p] for p in summ.index])
    summ.insert(0, "param", summ.index)
    for k, v in dict(cell=args.cell, g_true=g, a_true=a, n=n).items():
        summ[k] = v
    summ.to_csv(OUT / f"cell{args.cell:02d}_{tag}_summary.csv", index=False)
    print(summ[["param", "true", "mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string(), flush=True)

    n_div = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else -1
    meta = dict(cell=args.cell, g_true=g, a_true=a, n=n, seed=seed, tag=tag, smoke=args.smoke,
                draws=draws, tune=tune, chains=chains, n_divergences=n_div,
                total_draws=int(draws * chains), sampling_minutes=elapsed / 60,
                deadline_s=DEADLINE, fixed=FIXED, hssm_version=hssm.__version__, **sim_meta)
    (OUT / f"cell{args.cell:02d}_{tag}_meta.json").write_text(json.dumps(meta, indent=2))
    print("divergences:", n_div, "of", draws * chains, flush=True)

    idata.to_netcdf(str(OUT / f"cell{args.cell:02d}_{tag}.nc"))
    print("saved", OUT / f"cell{args.cell:02d}_{tag}.nc", flush=True)


if __name__ == "__main__":
    main()
