"""
Fit DDM with Weibull collapsing bound to RNN behavioral data.
Model: v ~ 1 + coherence (coherence effect on drift rate)
First pass: one network (nxx1, seed=42, gain=1.0)
"""
import pathlib
import hssm
import pytensor
import pandas as pd

pytensor.config.floatX = "float32"
from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "hssm_ready_nxx1_s42_g1.0.parquet"
OUT  = pathlib.Path(__file__).resolve().parents[1] / "output"

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Fitting DDM with Weibull collapsing bound")
    print(f"Loading data from {DATA}")
    df = pd.read_parquet(DATA)
    print(f"  {len(df):,} trials")

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
            {
                "name": "alpha",
                "prior": {"name": "Gamma", "mu": 1.5, "sigma": 0.5},
            },
            {
                "name": "beta",
                "prior": {"name": "Gamma", "mu": 3.0, "sigma": 1.0},
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
        mp_ctx="spawn",
        idata_kwargs=dict(log_likelihood=True),
        random_seed=42,
    )

    print("Sampling posterior predictive...")
    model.sample_posterior_predictive(idata, inplace=True)

    out_path = OUT / "ddm_weibull_nxx1_s42_g1.0"
    idata.to_netcdf(str(out_path))
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
