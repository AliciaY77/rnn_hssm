"""
Fit DDM with Weibull collapsing bound to RNN behavioral data.
Model: v ~ 1 + coherence (coherence effect on drift rate)
One network (nxx1, seed=42), gain passed via --gain (default 1.0).
"""
import argparse
import pathlib
import hssm
import pytensor
import pandas as pd
import matplotlib
matplotlib.use('Agg')

pytensor.config.floatX = "float32"
from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
OUT      = pathlib.Path(__file__).resolve().parents[1] / "output"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    gain = args.gain
    tag = f"nxx1_s42_g{gain}"

    OUT.mkdir(parents=True, exist_ok=True)
    print("Fitting DDM with Weibull collapsing bound")
    data_path = DATA_DIR / f"hssm_ready_{tag}.parquet"
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    print(f"  {len(df):,} trials")

    # Add RT offset to help sampler (no real non-decision time in RNN data)
    df['rt'] = df['rt'] + 0.3
    print(f"  RT range after offset: {df['rt'].min():.3f} - {df['rt'].max():.3f} s")

    # Bin coherence for quantile probability plot
    df['coh_abs'] = df['coherence'].abs().round(2)
    df['coh_bin'] = df['coh_abs'].astype(str)

    model = hssm.HSSM(
        data=df,
        model="weibull",
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
            },
        ],
    )
    print(model)

    idata = model.sample(
        sampler="numpyro",
        chains=4,
        cores=4,
        draws=1000,
        tune=1000,
        target_accept=0.95,
        idata_kwargs=dict(log_likelihood=True),
        random_seed=42,
    )

    print("Sampling posterior predictive...")
    model.sample_posterior_predictive(idata, inplace=True)

    import matplotlib.pyplot as plt

    print("Generating plots...")

    # Default PPC plot
    try:
        result = model.plot_predictive(step=True, bins=50)
        if hasattr(result, 'figure'):
            result.figure.savefig(str(OUT / f"ddm_weibull_{tag}_ppc.png"), dpi=150, bbox_inches='tight')
        elif hasattr(result, 'savefig'):
            result.savefig(str(OUT / f"ddm_weibull_{tag}_ppc.png"), dpi=150, bbox_inches='tight')
        plt.close('all')
        print("PPC plot saved.")
    except Exception as e:
        print(f"PPC plot failed: {e}")

    # Quantile probability plot
    try:
        ax = hssm.plotting.plot_quantile_probability(
            model,
            cond="coh_bin",
            predictive_style="ellipse",
            ellipse_confidence=0.95
        )
        # Fixed y-axis range so fixed/weibull plots are comparable
        ax.set_ylim(0.0, 1.2)
        ax.figure.savefig(str(OUT / f"ddm_weibull_{tag}_qpp.png"), dpi=150, bbox_inches='tight')
        plt.close('all')
        print("QPP saved.")
    except Exception as e:
        print(f"Quantile probability plot failed: {e}")

    out_path = OUT / f"ddm_weibull_{tag}"
    idata.to_netcdf(str(out_path))
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
