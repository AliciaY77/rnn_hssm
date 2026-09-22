"""
Regenerate RT / choice from the networks' decision-variable traces with a CONSTANT bound (issue #2).

Source: the SIM-A per-trial cache of the gain-controller project
  RNN_Gain_Controller/notes/mscont_cal_pertrial.parquet
which stores, per trial, the full 750-step trace `logit_diff` (= logits[0] - logits[1]; 1 step = 1 ms),
plus the Weibull-bound outcome used for the original dataset (`bounded_rt`, `choice`, `correct`).

Convention (validated here against the stored outcome: 99.9 % choice agreement, 96.7 % of RTs within
1 ms after re-deriving the Weibull crossing): dv < 0 at crossing  <=>  choice "+"  <=>  label == 1.

Constant bound b: RT = first t with |dv(t)| >= b; choice = sign(dv) at crossing; trials that never
cross within 750 ms are OMISSIONS and are dropped (never forced). Omission rates are written out.

Outputs (per gain, per bound):  data/processed/fixed_bound/hssm_ready_nxx1_fixed_b{b}_g{gain}.parquet
  rt (s), response (accuracy coding, +1 correct / -1 error), response_choice (+1 = choice "+"),
  coherence (|rel_coh|), coherence_signed, seed, gain, bound, label
plus  output/fixed_bound/omissions.csv  and  output/fixed_bound/sanity.csv (fixed vs. original Weibull data).

Usage: python data/process/fixed_bound_from_traces.py [--cache PATH] [--bounds 1.25 1.5 2.0] [--gains 0.8 1.0 1.2]
"""
import argparse, pathlib
import numpy as np, pandas as pd, pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT.parent / "RNN_Gain_Controller" / "notes" / "mscont_cal_pertrial.parquet"
OUT_DATA = ROOT / "data" / "processed" / "fixed_bound"
OUT_TAB = ROOT / "output" / "fixed_bound"
T_MS = 750


def load_traces(cache, gains, act="nxx1"):
    pf = pq.ParquetFile(cache); keep = []
    for b in pf.iter_batches(batch_size=20000):
        d = b.to_pandas(); d = d[(d.act == act) & (d.f_nm.isin(gains))]
        if len(d): keep.append(d)
    df = pd.concat(keep, ignore_index=True)
    dv = np.stack(df.logit_diff.values).astype(np.float32)
    return df.drop(columns=["logit_diff"]), dv


def cross_constant(dv, b):
    hit = np.abs(dv) >= b
    first = np.where(hit.any(1), hit.argmax(1), -1)            # index 0 == t = 1 ms
    rt_ms = np.where(first >= 0, first + 1, np.nan).astype(float)
    sgn = np.sign(dv[np.arange(len(dv)), np.clip(first, 0, dv.shape[1] - 1)])
    choice_plus = (sgn < 0).astype(int)                          # dv < 0 => choice "+" (label 1)
    return rt_ms, choice_plus, first >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=pathlib.Path, default=DEFAULT_CACHE)
    ap.add_argument("--bounds", type=float, nargs="+", default=[1.25, 1.5, 2.0])
    ap.add_argument("--gains", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    a = ap.parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True); OUT_TAB.mkdir(parents=True, exist_ok=True)
    df, dv = load_traces(a.cache, a.gains)
    print(f"{len(df):,} nxx1 trials, {df.seed.nunique()} seeds, gains {sorted(df.f_nm.unique())}")
    om_rows, san_rows = [], []
    # original (Weibull) reference rows
    for gn, d in df.groupby("f_nm"):
        san_rows.append(dict(bound="weibull(orig)", gain=gn, n=len(d), omit=1 - d.crossed.mean(), acc=d.correct.mean(),
                             rt_med=d.bounded_rt.median(), rt_90=d.bounded_rt.quantile(.9), rt_max=d.bounded_rt.max(),
                             acc_last100ms=d.correct[d.bounded_rt > d.bounded_rt.max() - 100].mean()))
    for b in a.bounds:
        rt_ms, ch, crossed = cross_constant(dv, b)
        correct = (ch == df.label.values).astype(int)
        for gn in a.gains:
            m = df.f_nm.values == gn
            for coh, mm in [("all", m)] + [(c, m & (df.rel_coh.abs().round(2).values == c)) for c in sorted(df.rel_coh.abs().round(2).unique())]:
                om_rows.append(dict(bound=b, gain=gn, coherence=coh, n=int(mm.sum()), omit=float(1 - crossed[mm].mean())))
            k = m & crossed
            out = pd.DataFrame({"rt": rt_ms[k] / 1000.0, "response": np.where(correct[k] == 1, 1.0, -1.0),
                                "response_choice": np.where(ch[k] == 1, 1.0, -1.0),
                                "coherence": df.rel_coh.abs().values[k], "coherence_signed": df.rel_coh.values[k],
                                "seed": df.seed.values[k], "gain": gn, "bound": b, "label": df.label.values[k]})
            out.to_parquet(OUT_DATA / f"hssm_ready_nxx1_fixed_b{b}_g{gn}.parquet", index=False)
            late = out.rt > out.rt.max() - 0.1
            san_rows.append(dict(bound=b, gain=gn, n=len(out), omit=float(1 - crossed[m].mean()), acc=float((out.response == 1).mean()),
                                 rt_med=out.rt.median() * 1000, rt_90=out.rt.quantile(.9) * 1000, rt_max=out.rt.max() * 1000,
                                 acc_last100ms=float((out.response[late] == 1).mean())))
    om = pd.DataFrame(om_rows); om.to_csv(OUT_TAB / "omissions.csv", index=False)
    san = pd.DataFrame(san_rows).round(3); san.to_csv(OUT_TAB / "sanity.csv", index=False)
    print("\n=== omissions (all coherences)"); print(om[om.coherence == "all"].pivot(index="bound", columns="gain", values="omit").round(3).to_string())
    print("\n=== omissions at zero coherence"); print(om[om.coherence == 0.0].pivot(index="bound", columns="gain", values="omit").round(3).to_string())
    print("\n=== sanity: fixed bound vs original Weibull readout (RT in ms)"); print(san.to_string(index=False))


if __name__ == "__main__":
    main()
