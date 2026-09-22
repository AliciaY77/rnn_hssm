"""Isolate the ssm-simulators crash in track_a/ppc_rt.py (job 6614551: `double free or corruption (out)`).

Replays the exact call pattern of ppc_rt.py -- the pooled posterior draws x the 11 signed coherences --
printing the theta before every call, so the abort names the offending parameter vector.

  python track_a/diag_ssms.py --tag g0.8_k10_b1.5_pooled --n-draws 200 --n-per-coh 100
"""
import argparse, json, pathlib
import numpy as np, pandas as pd
from ssms.basic_simulators.simulator import simulator

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "track_a"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="g0.8_k10_b1.5_pooled")
ap.add_argument("--n-draws", type=int, default=200)
ap.add_argument("--n-per-coh", type=int, default=100)
ap.add_argument("--max-t", type=float, default=0.0, help="0 = horizon + 0.01 as in ppc_rt")
a = ap.parse_args()

meta = json.loads((OUT / f"{a.tag}_meta.json").read_text())
k = meta["k"]; horizon = 0.3 + k * 0.75
max_t = a.max_t if a.max_t > 0 else max(2.0, horizon + 0.01)
draws = pd.read_csv(OUT / f"{a.tag}_draws.csv")
idx = np.linspace(0, len(draws) - 1, min(a.n_draws, len(draws))).astype(int)
cohs = [round(-0.15 + 0.03 * i, 2) for i in range(11)]
print(f"{a.tag}: k={k} horizon={horizon} max_t={max_t} {len(idx)} draws x {len(cohs)} cohs "
      f"x {a.n_per_coh} trials", flush=True)
print("g range in draws:", draws.g.min(), draws.g.max(), "| a range:", draws.a.min(), draws.a.max(),
      flush=True)

calls = 0
for j, ii in enumerate(idx):
    row = draws.iloc[ii]
    for i, c in enumerate(cohs):
        th = dict(v=float(row["v_Intercept"]) + float(row["v_coherence_signed"]) * c,
                  a=float(row["a"]), z=float(row["z"]), g=float(row["g"]), t=float(row["t"]))
        print(f"  call {calls} draw {j} coh {c}: {th}", flush=True)
        s = simulator(theta=th, model="ornstein", n_samples=a.n_per_coh, delta_t=0.001,
                      max_t=max_t, random_state=j * 1000 + i)
        rt = np.asarray(s["rts"]).ravel()
        calls += 1
print(f"ALL OK after {calls} calls")
