"""Collect output/recovery/*_summary.csv + *_meta.json into one table and a recovery figure (issue #1)."""
import glob
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "recovery"

rows = [pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "cell*_summary.csv"))) if "smoke" not in f]
metas = [json.loads(pathlib.Path(f).read_text()) for f in sorted(glob.glob(str(OUT / "cell*_meta.json"))) if "smoke" not in f]
if not rows:
    raise SystemExit("no summaries yet")
summ = pd.concat(rows, ignore_index=True)
meta = pd.DataFrame(metas)[["cell", "g_true", "a_true", "n", "n_divergences", "total_draws", "frac_dropped", "sampling_minutes"]]
meta["frac_divergent"] = meta.n_divergences / meta.total_draws
summ = summ.merge(meta, on=["cell", "g_true", "a_true", "n"], how="left")
summ["covered"] = (summ["hdi_3%"] <= summ["true"]) & (summ["true"] <= summ["hdi_97%"])
summ.to_csv(OUT / "recovery_table.csv", index=False)

g = summ[summ.param == "g"].copy()
print("=== g recovery")
print(g[["cell", "g_true", "a_true", "n", "mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk",
         "frac_divergent", "frac_dropped", "covered"]].round(3).to_string(index=False))
interior = g[g.g_true.abs() < 1]
print(f"\nsign correct (g_true != 0): {((g[g.g_true != 0]['mean'] * g[g.g_true != 0].g_true) > 0).mean():.2f}")
print(f"HDI coverage, interior g: {interior.covered.mean():.2f}   |   all params, interior cells: "
      f"{summ[summ.g_true.abs() < 1].covered.mean():.2f}")
print(f"max r_hat: {summ.r_hat.max():.3f}   |   max frac divergent: {summ.frac_divergent.max():.3f}")

fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
for ax, p in zip(axes, ["g", "a", "v", "z", "t"]):
    d = summ[summ.param == p]
    for (a_true, n), dd in d.groupby(["a_true", "n"]):
        x = dd.g_true + (0.02 if n == 3700 else -0.02) + (0.04 if a_true == 1.5 else 0)
        ax.errorbar(x, dd["mean"], yerr=[dd["mean"] - dd["hdi_3%"], dd["hdi_97%"] - dd["mean"]],
                    fmt="o" if n == 3700 else "s", ms=4, capsize=2, label=f"a={a_true}, n={n}",
                    color="C0" if a_true == 1.0 else "C3", alpha=1 if n == 3700 else 0.5)
    if p == "g":
        ax.plot([-1.1, 1.1], [-1.1, 1.1], "k--", lw=1)
    else:
        for tv in d.true.unique():
            ax.axhline(tv, color="k", ls="--", lw=1)
    ax.set_xlabel("true g"); ax.set_title(p)
axes[0].set_ylabel("posterior mean, 94% HDI"); axes[0].legend(fontsize=7)
plt.tight_layout(); fig.savefig(OUT / "recovery_g.png", dpi=150); print("saved", OUT / "recovery_g.png")
