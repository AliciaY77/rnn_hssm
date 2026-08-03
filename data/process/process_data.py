"""
Process raw RNN DDM data for HSSM fitting.

Filters to:
  - activation: nxx1 only
  - seed: 42 (one network for first pass)
  - gain: passed via --gain (default 1.0)

Output format for HSSM (accuracy coding):
  - rt: reaction time in seconds
  - response: 1.0 (correct) or -1.0 (incorrect)
  - coherence: signed relevant coherence (covariate for drift rate)
"""
import argparse
import pathlib
import pandas as pd
import numpy as np

RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "data_raw"
OUT     = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed"

ACTIVATION = "nxx1"
SEED       = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gain", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    gain = args.gain

    raw_path = RAW_DIR / f"rnn_ddm_{ACTIVATION}_s{SEED}_g{gain}.parquet"

    print(f"Loading {raw_path} ...")
    df = pd.read_parquet(raw_path)
    print(f"  Full dataset: {len(df):,} rows")

    # Filter
    df = df[
        (df["activation"] == ACTIVATION) &
        (df["subject_id"] == SEED) &
        (df["gain"] == gain)
    ].copy()
    print(f"  After filter (nxx1, seed={SEED}, gain={gain}): {len(df):,} rows")

    # HSSM format — accuracy coding
    # response: 1.0 = correct, -1.0 = incorrect
    out = pd.DataFrame({
        "rt":         df["rt_sec"].astype(float),
        "response":   np.where(df["correct"] == 1, 1.0, -1.0).astype(float),
        "coherence":  df["coherence"].abs().astype(float),  # |coherence| as drift predictor
    })

    # Sanity checks
    assert out["rt"].min() > 0, "negative RTs found"
    assert set(out["response"].unique()).issubset({1.0, -1.0}), "bad response values"
    print(f"  RT range: {out['rt'].min():.3f} - {out['rt'].max():.3f} s")
    print(f"  Accuracy: {(out['response']==1.0).mean():.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"hssm_ready_{ACTIVATION}_s{SEED}_g{gain}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"  Saved to {out_path}")

if __name__ == "__main__":
    main()
