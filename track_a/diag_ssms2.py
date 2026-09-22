"""One ssms configuration per process, so a C-level abort identifies exactly which one crashes.
Hypothesis: the ornstein simulator corrupts the heap when trials REACH max_t (fail to terminate)."""
import argparse
import numpy as np
from ssms.basic_simulators.simulator import simulator

ap = argparse.ArgumentParser()
ap.add_argument("--v", type=float, required=True)
ap.add_argument("--a", type=float, default=1.1983833)
ap.add_argument("--z", type=float, default=0.49490383)
ap.add_argument("--g", type=float, default=0.9945302)
ap.add_argument("--t", type=float, default=0.35)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max-t", type=float, required=True)
ap.add_argument("--dt", type=float, default=0.001)
ap.add_argument("--repeats", type=int, default=10)
a = ap.parse_args()
th = dict(v=a.v, a=a.a, z=a.z, g=a.g, t=a.t)
print(f"v={a.v} max_t={a.max_t} dt={a.dt} n={a.n}: ", end="", flush=True)
fr = []
for r in range(a.repeats):
    s = simulator(theta=th, model="ornstein", n_samples=a.n, delta_t=a.dt, max_t=a.max_t, random_state=r)
    rt = np.asarray(s["rts"]).ravel()
    fr.append(float(((rt > 0) & (rt < a.max_t)).mean()))
print(f"OK, frac_terminated {np.mean(fr):.4f} (min {np.min(fr):.4f})", flush=True)
