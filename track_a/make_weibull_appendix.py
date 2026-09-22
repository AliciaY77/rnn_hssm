"""
Appendix data for Track A step 8: the ORIGINAL Weibull-collapsing-bound readout, in the same
HSSM-ready format as the constant-bound files, so the identical corrected pipeline can be run on it.

The Weibull outcome is stored in the SIM-A per-trial cache (`bounded_rt`, `choice`, `crossed`); this
script only reformats it, exactly as data/process/fixed_bound_from_traces.py reformats its own crossings.
Non-crossers are dropped (never forced) and the omission fraction is printed.

Usage (local, needs pyarrow):
  /opt/homebrew/anaconda3/bin/python track_a/make_weibull_appendix.py --gain 1.0 --seed 42
Writes data/processed/fixed_bound/hssm_ready_nxx1_weibull_g<gain>[_s<seed>].csv  (gitignored)
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "RNN_Gain_Controller" / "notes" / "mscont_cal_pertrial.parquet"
OUT = ROOT / "data" / "processed" / "fixed_bound"

ap = argparse.ArgumentParser()
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=42, help="0 = all 20 networks")
ap.add_argument("--cache", type=pathlib.Path, default=CACHE)
a = ap.parse_args()

cols = ["seed", "act", "f_nm", "label", "rel_coh", "choice", "correct", "crossed", "bounded_rt"]
pf = pq.ParquetFile(a.cache)
keep = []
for b in pf.iter_batches(batch_size=50000, columns=cols):
    d = b.to_pandas()
    d = d[(d.act == "nxx1") & (np.isclose(d.f_nm, a.gain))]
    if a.seed:
        d = d[d.seed == a.seed]
    if len(d):
        keep.append(d)
df = pd.concat(keep, ignore_index=True)
n_all = len(df)
agree = float((df.correct.values == (df.choice.values == df.label.values)).mean())
print(f"{n_all} trials, {df.seed.nunique()} seeds; correct == (choice == label) for {agree*100:.2f} %")
print(f"omissions (not crossed): {1 - df.crossed.mean():.4f}; "
      f"bounded_rt min {df.bounded_rt.min()} med {df.bounded_rt.median()} max {df.bounded_rt.max()} ms")

d = df[df.crossed.astype(bool)].copy()
out = pd.DataFrame({
    "rt": d.bounded_rt.values / 1000.0,
    "response": np.where(d.correct.values == 1, 1.0, -1.0),
    "response_choice": np.where(d.choice.values == 1, 1.0, -1.0),   # 1 = choice "+" = label 1
    "coherence": np.abs(d.rel_coh.values),
    "coherence_signed": d.rel_coh.values,
    "seed": d.seed.values, "gain": a.gain, "bound": "weibull", "label": d.label.values,
})
OUT.mkdir(parents=True, exist_ok=True)
name = f"hssm_ready_nxx1_weibull_g{a.gain}" + (f"_s{a.seed}" if a.seed else "") + ".csv"
out.to_csv(OUT / name, index=False)
print(f"wrote {OUT/name}: {len(out)} trials, accuracy {(out.response==1).mean():.3f}, "
      f"P(choice +) {(out.response_choice==1).mean():.3f}, "
      f"rt min {out.rt.min():.3f} med {out.rt.median():.3f} max {out.rt.max():.3f} s")
print(out.groupby(out.coherence_signed.round(2)).response_choice.apply(lambda s: (s == 1).mean()).round(3).to_string())
