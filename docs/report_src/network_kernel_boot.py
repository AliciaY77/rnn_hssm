"""Bootstrap 95 % interval for the networks' pooled psychophysical-kernel slope per gain (40 000 trials, 20 networks).
Resamples trials with replacement; the kernel is computed exactly as kernel_fit/fit_ou_kernel.py::kernel (8 bin means of the
relevant evidence, z-scored within the sample, logistic regression, L1-normalised weights, slope of a linear fit over bins).
The bin means are computed once and resampled, which is equivalent and avoids copying the 40 000 x 750 evidence array.
Writes output/report/network_kernel_boot.csv."""
import glob, pathlib, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "kernel_fit"))
from fit_ou_kernel import kernel
files = sorted(glob.glob(str(ROOT / "data/processed/kernel/seed*.npz"))); rel = np.concatenate([np.load(f)["rel"].astype(np.float32) for f in files])
edges = np.linspace(0, rel.shape[1], 9).astype(int); Bm = np.stack([rel[:, edges[k]:edges[k + 1]].mean(1) for k in range(8)], 1)
def slope(X, y):
    X = (X - X.mean(0)) / (X.std(0) + 1e-9); W = LogisticRegression(max_iter=2000).fit(X, y).coef_[0]; Wn = W / (np.abs(W).sum() + 1e-9)
    return float(np.polyfit(np.arange(8), Wn, 1)[0])
rng = np.random.default_rng(5); rows = []
for gn in [0.8, 1.0, 1.2]:
    ch = np.concatenate([np.load(f)[f"choice_g{gn}"] for f in files])
    s_full = slope(Bm, ch); s_ref = kernel(rel, ch)[1]; assert abs(s_full - s_ref) < 1e-6, (s_full, s_ref)
    bs = [slope(Bm[i], ch[i]) for i in (rng.integers(0, len(ch), len(ch)) for _ in range(300))]
    lo, hi = np.quantile(bs, [0.025, 0.975]); rows.append(dict(gain=gn, slope=s_full, lo=lo, hi=hi, sd=float(np.std(bs)), n_boot=len(bs)))
    print(f"gain {gn}: slope {s_full:+.4f} [{lo:+.4f}, {hi:+.4f}] (matches kernel(): {s_ref:+.4f})", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "output/report/network_kernel_boot.csv", index=False)
