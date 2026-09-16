"""
Fit Ornstein-Uhlenbeck (leaky/unstable DDM) to RNN behavioral data using HSSM.

The ornstein model has a 'g' parameter (leak/instability):
    g < 0: leaky integrator
    g = 0: perfect integrator (standard DDM)
    g > 0: unstable/attractor dynamics

This directly maps to the Brunton lambda parameter.

Model: v ~ 1 + coherence, g free (uniform -1 to 1)

Usage:
    python model/fit_ornstein.py --gain 1.0 --seed 42
"""
import argparse
import pathlib
import hssm
import pytensor
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pytensor.config.floatX = "float32"
from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR  = pathlib.Path(__file__).resolve().parents[1] / "output"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = DATA_DIR / f"hssm_ready_nxx1_s{args.seed}_g{args.gain}.parquet"
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} trials")
    print(f"  RT range: {df['rt'].min():.3f} - {df['rt'].max():.3f} s")
    print(f"  Accuracy: {(df['response']==1.0).mean():.3f}")

    # Add RT offset — ornstein model needs t > 0 and RTs are very short
    df['rt'] = df['rt'] + 0.3
    print(f"  RT range after offset: {df['rt'].min():.3f} - {df['rt'].max():.3f} s")

    # Bin coherence for QPP
    df['coh_abs'] = df['coherence'].abs().round(2)
    df['coh_bin'] = df['coh_abs'].astype(str)

    model = hssm.HSSM(
        data=df,
        model="ornstein",
        loglik_kind="approx_differentiable",
        include=[
            {
                "name": "v",
                "formula": "v ~ 1 + coherence",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 2.0},
                    "coherence": {"name": "Normal", "mu": 0.0, "sigma": 2.0},
                },
                "link": "identity",
            }
        ],
    )
    print(model)

    idata = model.sample(
        sampler="numpyro",
        chains=4,
        cores=4,
        draws=args.draws,
        tune=args.tune,
        target_accept=0.95,
        idata_kwargs=dict(log_likelihood=True),
        random_seed=args.seed,
    )

    print("Sampling posterior predictive...")
    model.sample_posterior_predictive(idata, inplace=True)

    import matplotlib.pyplot as plt
    print("Generating plots...")

    # PPC plot
    try:
        result = model.plot_predictive(step=True, bins=50)
        if hasattr(result, 'figure'):
            result.figure.savefig(str(OUT_DIR / f"ornstein_ppc_nxx1_s{args.seed}_g{args.gain}.png"), dpi=150, bbox_inches='tight')
        elif hasattr(result, 'savefig'):
            result.savefig(str(OUT_DIR / f"ornstein_ppc_nxx1_s{args.seed}_g{args.gain}.png"), dpi=150, bbox_inches='tight')
        plt.close('all')
        print("PPC saved.")
    except Exception as e:
        print(f"PPC failed: {e}")

    # QPP
    try:
        ax = hssm.plotting.plot_quantile_probability(
            model, cond="coh_bin",
            predictive_style="ellipse", ellipse_confidence=0.95
        )
        ax.set_ylim(0.0, 1.2)
        ax.figure.savefig(str(OUT_DIR / f"ornstein_qpp_nxx1_s{args.seed}_g{args.gain}.png"), dpi=150, bbox_inches='tight')
        plt.close('all')
        print("QPP saved.")
    except Exception as e:
        print(f"QPP failed: {e}")

    out_path = OUT_DIR / f"ornstein_nxx1_s{args.seed}_g{args.gain}"
    idata.to_netcdf(str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
