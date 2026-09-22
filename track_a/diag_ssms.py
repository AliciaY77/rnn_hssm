"""Isolate the ssm-simulators crash seen in ta_ppcrt (job 6614551): `double free or corruption (out)`.
Runs one configuration per invocation so a crash identifies itself; --sweep runs them in order."""
import argparse, sys
import numpy as np
from ssms.basic_simulators.simulator import simulator

THETA = dict(v=-1.2624, a=1.2013, z=0.4953, g=0.998, t=0.35)   # pooled gain-0.8 k=10 posterior mean, coh -0.15

CASES = [
    ("n100_maxt7.81", 100, 7.81, 0.001),
    ("n100_maxt10", 100, 10.0, 0.001),
    ("n100_maxt20", 100, 20.0, 0.001),
    ("n1000_maxt7.81", 1000, 7.81, 0.001),
    ("n20000_maxt7.81", 20000, 7.81, 0.001),
    ("n100_maxt7.81_g0.5", 100, 7.81, 0.001),
    ("n100_maxt2", 100, 2.0, 0.001),
]

ap = argparse.ArgumentParser()
ap.add_argument("--case", type=int, default=-1)
ap.add_argument("--repeats", type=int, default=30)
a = ap.parse_args()
todo = CASES if a.case < 0 else [CASES[a.case]]
for name, n, max_t, dt in todo:
    th = dict(THETA)
    if name.endswith("_g0.5"):
        th["g"] = 0.5
    print(f"--- {name}: n={n} max_t={max_t} dt={dt} theta={th}", flush=True)
    for r in range(a.repeats):
        s = simulator(theta=th, model="ornstein", n_samples=n, delta_t=dt, max_t=max_t, random_state=r)
        rt = np.asarray(s["rts"]).ravel()
        if r == 0:
            print(f"    rep0 ok: n={len(rt)} frac_crossed={(rt > 0).mean():.3f} "
                  f"med={np.median(rt[rt > 0]) if (rt > 0).any() else float('nan'):.3f}", flush=True)
    print(f"    {name}: {a.repeats} repeats OK", flush=True)
print("ALL OK")
