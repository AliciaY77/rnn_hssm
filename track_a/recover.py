"""
Track A step 7 (issue #3): could this pipeline have seen the regime transition if it had been in the data?

Simulate datasets from the pooled fit's own posterior-mean parameters, replacing g by
  --g-source theory : the landscape's drift slope at the undecided state, in the HSSM sign convention
                      (g_nat = +4.3 / -1.6 / -6.2 per s at gains 0.8 / 1.0 / 1.2; the landscape reports
                      -4.3 / +1.6 / +6.2 with the opposite sign, negative = restoring = leaky),
  --g-source fitted : the fitted posterior mean (a self-consistency check),
apply the same 750 ms native horizon (drop non-crossers, report the fraction), and refit with exactly the
same HSSM pipeline (same fixed t, same box-uniform priors, same sampler settings).

Sign convention: dx = (v - g x) dt + dW, so g > 0 = leaky, g < 0 = unstable.

Run on Oscar:
  python track_a/recover.py --gain 1.0 --k 10 --g-source theory
Outputs -> output/track_a/recover_<tag>_{summary.csv,meta.json,draws.csv,nc}
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fit_hssm_ou import BOX, COH_REF, OFFSET, OUT, box_uniform_priors, edge_mass  # noqa: E402

import arviz as az  # noqa: E402
import hssm  # noqa: E402
from ssms.basic_simulators.simulator import simulator  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "fixed_bound"
HORIZON_NATIVE = 0.75
# landscape drift slope at the undecided state, converted to the HSSM sign convention (g > 0 leaky)
G_THEORY_NATIVE = {0.8: +4.3, 1.0: -1.6, 1.2: -6.2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, required=True, choices=[0.8, 1.0, 1.2])
    ap.add_argument("--k", type=float, default=10.0)
    ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--g-source", choices=["theory", "fitted"], required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    k = a.k
    tag = f"recover_g{a.gain}_k{k:g}_{a.g_source}" + ("_smoke" if a.smoke else "")
    print(f"=== {tag} ===", flush=True)

    src = OUT / f"g{a.gain}_k{k:g}_b{a.bound}_pooled_meta.json"
    meta_src = json.loads(src.read_text())
    pm = meta_src["posterior_mean"]
    t_fixed = meta_src["t_fixed"]
    g_true = (G_THEORY_NATIVE[a.gain] / k) if a.g_source == "theory" else pm["g"]
    theta0 = dict(v_Intercept=pm["v_Intercept"], v_coherence_signed=pm["v_coherence_signed"],
                  a=pm["a"], z=pm["z"], g=g_true, t=t_fixed)
    print("generating parameters:", {kk: round(vv, 4) for kk, vv in theta0.items()},
          f"| g_native {g_true*k:+.2f}/s", flush=True)
    if not (BOX["g"][0] <= g_true <= BOX["g"][1]):
        print(f"WARNING: generating g {g_true:.3f} is outside the LAN box {BOX['g']}", flush=True)

    # coherence design copied from the real dataset, so the information about g is the same
    real = pd.read_csv(DATA / f"hssm_ready_nxx1_fixed_b{a.bound}_g{a.gain}.csv")
    cohs, counts = np.unique(real.coherence_signed.round(2).values, return_counts=True)
    n_per = np.round(counts / counts.sum() * a.n).astype(int)
    horizon = OFFSET + k * HORIZON_NATIVE
    max_t = max(2.0, horizon + 0.01)

    parts, n_req, n_drop = [], 0, 0
    for i, (c, npc) in enumerate(zip(cohs, n_per)):
        if npc <= 0:
            continue
        v = theta0["v_Intercept"] + theta0["v_coherence_signed"] * c
        s = simulator(theta=dict(v=v, a=theta0["a"], z=theta0["z"], g=theta0["g"], t=theta0["t"]),
                      model="ornstein", n_samples=int(npc), delta_t=0.001, max_t=max_t,
                      random_state=a.rng * 100 + i)
        rt = np.asarray(s["rts"]).ravel(); ch = np.asarray(s["choices"]).ravel()
        ok = (rt > 0) & (rt <= horizon)
        n_req += int(npc); n_drop += int((~ok).sum())
        parts.append(pd.DataFrame({"rt": rt[ok], "response": np.where(ch[ok] > 0, 1.0, -1.0),
                                   "coherence_signed": float(c)}))
    df = pd.concat(parts, ignore_index=True)
    omit = n_drop / n_req
    print(f"simulated {n_req}, kept {len(df)} (omissions {omit*100:.2f} %), "
          f"RT stretched min {df.rt.min():.3f} med {df.rt.median():.3f} max {df.rt.max():.3f}", flush=True)
    if (df.rt <= t_fixed).any():
        raise SystemExit("simulated RTs at or below the fixed t")

    priors = box_uniform_priors()
    model = hssm.HSSM(data=df, model="ornstein", loglik_kind="approx_differentiable",
                      p_outlier=None, lapse=None, t=t_fixed,
                      include=[dict(name="v", formula="v ~ 1 + coherence_signed", link="identity",
                                    prior=priors["v"])],
                      a=priors["a"], z=priors["z"], g=priors["g"])
    draws, tune, chains = (20, 20, 1) if a.smoke else (a.draws, a.tune, a.chains)
    sampler = "numpyro" if hssm.__version__ >= "0.3" else "nuts_numpyro"
    t0 = time.time()
    idata = model.sample(sampler=sampler, chains=chains, cores=chains, draws=draws, tune=tune,
                         target_accept=0.95, random_seed=2000 + a.rng,
                         idata_kwargs=dict(log_likelihood=False))
    minutes = (time.time() - t0) / 60

    post = idata.posterior
    params = [p for p in ["v_Intercept", "v_coherence_signed", "a", "z", "g"] if p in post]
    summ = az.summary(idata, var_names=params, hdi_prob=0.94).reset_index().rename(columns={"index": "param"})
    summ["true"] = [theta0[p] for p in summ.param]
    summ["true_native"] = [theta0[p] * k if p == "g" else
                           (theta0[p] / np.sqrt(k) if p == "a" else
                            (theta0[p] * np.sqrt(k) if p.startswith("v_") else theta0[p]))
                           for p in summ.param]
    g_post = post["g"].values.ravel()
    gm, glo, ghi = float(g_post.mean()), float(np.quantile(g_post, 0.03)), float(np.quantile(g_post, 0.97))
    n_div = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else -1
    for kk, vv in dict(gain=a.gain, k=k, g_source=a.g_source, tag=tag).items():
        summ[kk] = vv
    summ.to_csv(OUT / f"{tag}_summary.csv", index=False)
    print(summ[["param", "true", "mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "r_hat"]].to_string(index=False),
          flush=True)

    meta = dict(tag=tag, gain=a.gain, k=k, g_source=a.g_source, generating=theta0,
                g_true=g_true, g_true_native=g_true * k, g_mean=gm, g_native_mean=gm * k,
                g_hdi3=glo, g_hdi97=ghi, g_native_hdi3=glo * k, g_native_hdi97=ghi * k,
                sign_recovered=bool(np.sign(gm) == np.sign(g_true)),
                hdi_covers_truth=bool(glo <= g_true <= ghi),
                n_requested=n_req, n_fitted=int(len(df)), omission_frac=float(omit),
                t_fixed=t_fixed, horizon_stretched=horizon,
                n_divergences=n_div, total_draws=int(draws * chains),
                r_hat_max=float(np.nanmax(summ.r_hat.values)),
                ess_bulk_min=float(np.nanmin(summ.ess_bulk.values)),
                sampling_minutes=minutes,
                edge_mass=dict(a=edge_mass(post["a"].values, *BOX["a"]),
                               g=edge_mass(post["g"].values, *BOX["g"]),
                               v_at_0p15=edge_mass(post["v_Intercept"].values
                                                   + COH_REF * post["v_coherence_signed"].values, *BOX["v"])),
                hssm_version=hssm.__version__)
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"g true {g_true:+.4f} ({g_true*k:+.2f}/s native) -> recovered {gm:+.4f} "
          f"[{glo:+.4f}, {ghi:+.4f}] ({gm*k:+.2f}/s); sign {'OK' if meta['sign_recovered'] else 'WRONG'}; "
          f"HDI covers truth: {meta['hdi_covers_truth']}; {minutes:.1f} min, {n_div} divergences", flush=True)
    idata.to_netcdf(str(OUT / f"{tag}.nc"))


if __name__ == "__main__":
    main()
