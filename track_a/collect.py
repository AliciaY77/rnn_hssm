"""
Collect every Track A fit in output/track_a into three tables (issue #3).

  all_fits.csv            one row per fit: identifiers, n, k, fixed t, omission fraction, R-hat max,
                          ESS_bulk min, divergences, minutes, edge mass, and mean/sd/94 % HDI for
                          v_Intercept, v_coherence_signed, a, z, g plus the native-unit versions.
  headline_pooled.csv     the pooled fits, one row per (gain, k), sorted for the headline table.
  per_network.csv         the 60-per-k per-network fits summarised per (gain, k): fraction of networks
                          with g > 0, median / IQR of g_native, convergence counts.

Sign convention: dx = (v - g x) dt + dW, so **g > 0 = leaky, g < 0 = unstable**.
Native units at stretch k: g_nat = g*k, a_nat = a/sqrt(k), v_nat(coh) = v(coh)*sqrt(k), t_nat = (t-0.3)/k.

Usage: /opt/homebrew/anaconda3/bin/python track_a/collect.py
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "track_a"
PARAMS = ["v_Intercept", "v_coherence_signed", "a", "z", "g",
          "v(0.15)", "v(0)", "v(-0.15)", "g_native", "a_native", "v_native(0.15)"]
COLS = ["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]
SAFE = {"v(0.15)": "v0p15", "v(0)": "v0", "v(-0.15)": "vm0p15",
        "g_native": "g_nat", "a_native": "a_nat", "v_native(0.15)": "v_nat0p15"}


def main():
    rows = []
    for meta_path in sorted(OUT.glob("*_meta.json")):
        tag = meta_path.name[:-len("_meta.json")]
        summ_path = OUT / f"{tag}_summary.csv"
        if not summ_path.exists():
            print("no summary for", tag)
            continue
        if tag.startswith("recover_"):
            continue            # recovery fits get their own table below (different meta schema)
        m = json.loads(meta_path.read_text())
        s = pd.read_csv(summ_path).set_index("param")
        if tag.startswith("appendix_"):
            kind = "appendix_weibull"
        elif tag.startswith("timing_"):
            kind = "timing"
        elif m.get("hierarchical"):
            kind = "hierarchical"
        elif m.get("smoke"):
            kind = "smoke"
        elif m.get("seed_filter"):
            kind = "per_network"
        else:
            kind = "pooled"
        r = dict(
            tag=tag, kind=kind,
            gain=m["gain"], k=m["k"], bound=m.get("bound"), seed=m.get("seed_filter") or np.nan,
            n=m["n_fitted"], n_seeds=m.get("n_seeds"), subsample=m.get("subsample"),
            t_fixed=m["t_fixed"], t_native_ms=1000 * m["t_native"],
            omission_frac=m["omission_frac"], rt_min_native_ms=1000 * m.get("rt_min_native", np.nan),
            rt_median_native_ms=1000 * m.get("rt_median_native", np.nan),
            accuracy=m.get("accuracy"), p_choice_plus=m.get("p_choice_plus"),
            draws=m["draws"], tune=m["tune"], chains=m["chains"], total_draws=m["total_draws"],
            divergences=m["n_divergences"], divergence_frac=m["divergence_frac"],
            r_hat_max=m["r_hat_max"], ess_bulk_min=m["ess_bulk_min"],
            minutes=m["sampling_minutes"], priors=m.get("priors"),
            edge_a=m["edge_mass"]["a"], edge_g=m["edge_mass"]["g"],
            edge_v0p15=m["edge_mass"]["v_at_0p15"], edge_z=m["edge_mass"].get("z", np.nan),
        )
        for p in PARAMS:
            if p not in s.index:
                continue
            key = SAFE.get(p, p)
            for c in COLS:
                if c in s.columns:
                    r[f"{key}_{c.replace('%','').replace('hdi_','hdi')}"] = s.loc[p, c]
        rows.append(r)

    if not rows:
        raise SystemExit(f"no fits found in {OUT}")
    tab = pd.DataFrame(rows).sort_values(["kind", "k", "gain", "seed"])
    tab.to_csv(OUT / "all_fits.csv", index=False)
    print(f"{len(tab)} fits -> {OUT/'all_fits.csv'}")

    pooled = tab[tab.kind == "pooled"].sort_values(["k", "gain"])
    extra = tab[tab.kind.isin(["appendix_weibull", "hierarchical", "timing"])].sort_values(["kind", "k", "gain"])
    show = ["gain", "k", "n", "t_fixed", "omission_frac", "g_mean", "g_hdi3", "g_hdi97",
            "g_nat_mean", "g_nat_hdi3", "g_nat_hdi97", "a_mean", "a_nat_mean",
            "v_Intercept_mean", "v_coherence_signed_mean", "v0p15_mean", "z_mean",
            "r_hat_max", "ess_bulk_min", "divergences", "edge_g", "edge_a", "edge_v0p15", "minutes"]
    show = [c for c in show if c in pooled.columns]
    pooled[show].to_csv(OUT / "headline_pooled.csv", index=False)
    print("\n=== pooled fits (g > 0 = leaky) ===")
    print(pooled[show].round(4).to_string(index=False))

    # native-unit translation table (the issue's deliverable), pooled fits only
    if len(pooled):
        nt = pooled.copy()
        sk = np.sqrt(nt.k.values)
        nt["g_native_mean"] = nt.g_mean * nt.k
        nt["g_native_hdi3"] = nt.g_hdi3 * nt.k
        nt["g_native_hdi97"] = nt.g_hdi97 * nt.k
        nt["a_native_mean"] = nt.a_mean / sk
        nt["v_native_at_0.15"] = nt.v0p15_mean * sk
        nt["v_native_at_0"] = nt.v0_mean * sk
        nt["v_native_per_unit_coherence"] = nt.v_coherence_signed_mean * sk
        nt["t_native_ms"] = (nt.t_fixed - 0.3) / nt.k * 1000
        cols = ["gain", "k", "n", "g_mean", "g_native_mean", "g_native_hdi3", "g_native_hdi97",
                "a_mean", "a_native_mean", "v0p15_mean", "v_native_at_0.15", "v_native_at_0",
                "v_native_per_unit_coherence", "z_mean", "t_fixed", "t_native_ms",
                "edge_g", "r_hat_max", "ess_bulk_min", "divergences", "omission_frac"]
        cols = [c for c in cols if c in nt.columns]
        nt[cols].to_csv(OUT / "native_units.csv", index=False)
        print("\n=== native-unit translation (g_nat = g*k, a_nat = a/sqrt(k), "
              "v_nat = v*sqrt(k), t_nat = (t-0.3)/k; g > 0 = leaky) ===")
        print(nt[cols].round(4).to_string(index=False))

    if len(extra):
        cols = [c for c in show if c in extra.columns]
        extra[["tag", "kind"] + cols].to_csv(OUT / "other_fits.csv", index=False)
        print("\n=== appendix / hierarchical / timing fits ===")
        print(extra[["tag", "kind"] + cols].round(4).to_string(index=False))

    # recovery table (step 7 deliverable), built from the recover_*_meta.json files
    rec = []
    for mp in sorted(OUT.glob("recover_*_meta.json")):
        m = json.loads(mp.read_text())
        rec.append(dict(gain=m["gain"], k=m["k"], g_source=m["g_source"],
                        g_true=m["g_true"], g_true_native=m["g_true_native"],
                        g_mean=m["g_mean"], g_hdi3=m["g_hdi3"], g_hdi97=m["g_hdi97"],
                        g_native_mean=m["g_native_mean"], g_native_hdi3=m["g_native_hdi3"],
                        g_native_hdi97=m["g_native_hdi97"],
                        sign_recovered=m["sign_recovered"], hdi_covers_truth=m["hdi_covers_truth"],
                        n_requested=m["n_requested"], n_fitted=m["n_fitted"],
                        omission_frac=m["omission_frac"], t_fixed=m["t_fixed"],
                        r_hat_max=m["r_hat_max"], ess_bulk_min=m["ess_bulk_min"],
                        divergences=m["n_divergences"], total_draws=m["total_draws"],
                        edge_g=m["edge_mass"]["g"], minutes=m["sampling_minutes"]))
    if rec:
        rt = pd.DataFrame(rec).sort_values(["g_source", "gain"])
        rt.to_csv(OUT / "recovery_table.csv", index=False)
        print("\n=== recovery at k = 10 (g > 0 = leaky) ===")
        print(rt.round(4).to_string(index=False))
        print(f"sign recovered {int(rt.sign_recovered.sum())}/{len(rt)}; "
              f"94 % HDI covers the truth {int(rt.hdi_covers_truth.sum())}/{len(rt)}")

    per = tab[tab.kind == "per_network"]
    if len(per):
        agg = per.groupby(["gain", "k"]).apply(lambda d: pd.Series(dict(
            n_networks=len(d),
            frac_g_positive=float((d.g_mean > 0).mean()),
            n_g_positive=int((d.g_mean > 0).sum()),
            frac_hdi_excludes_0_positive=float((d.g_hdi3 > 0).mean()),
            frac_hdi_excludes_0_negative=float((d.g_hdi97 < 0).mean()),
            g_nat_median=float(d.g_nat_mean.median()),
            g_nat_q25=float(d.g_nat_mean.quantile(.25)),
            g_nat_q75=float(d.g_nat_mean.quantile(.75)),
            g_nat_min=float(d.g_nat_mean.min()), g_nat_max=float(d.g_nat_mean.max()),
            a_nat_median=float(d.a_nat_mean.median()),
            v_nat0p15_median=float(d.v_nat0p15_mean.median()),
            n_rhat_ok=int((d.r_hat_max <= 1.01).sum()),
            n_ess_ok=int((d.ess_bulk_min >= 400).sum()),
            n_div_lt1pct=int((d.divergence_frac < 0.01).sum()),
            max_edge_g=float(d.edge_g.max()), median_edge_g=float(d.edge_g.median()),
            median_minutes=float(d.minutes.median()),
        )), include_groups=False).reset_index()
        agg.to_csv(OUT / "per_network.csv", index=False)
        print("\n=== per-network fits ===")
        print(agg.round(4).to_string(index=False))
    else:
        print("\nno per-network fits yet")


if __name__ == "__main__":
    main()
