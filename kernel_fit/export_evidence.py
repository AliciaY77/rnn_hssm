"""
Export, per network and gain, the per-trial RELEVANT evidence stream and the network's deadline choice,
plus the network's psychophysical kernel — the inputs a stimulus-conditioned accumulator fit needs.

Reuses gain-controller-rnn's own functions so the choice and kernel definitions are identical to Fig 1F:
  make_dataset(n, seed=0)              same stimuli for every gain (as psychophysical_kernel does)
  rollout_dv(model, data, f_nm, seed)  decision variable dv = logits[0] - logits[1], (N, T)
  choices_from_dv(ld) = (dv[:, -1] < 0)  deadline choice; 1 <=> "+" <=> label 1
  psychophysical_kernel(...)           logistic regression of choice on 8 time bins of relevant evidence

Run with the rnn_gain env from anywhere:
  RNN_T=750 RNN_EPOCHS=100 python kernel_fit/export_evidence.py --repo ../RNN_Gain_Mod/gain-controller-rnn --n-trials 2000
Writes data/processed/kernel/seed<S>.npz  (rel evidence float16 (N,T), labels, goals, coh_signed, and per gain:
choice_g<gain>, dvT_g<gain>) and output/kernel_fit/network_kernels.csv.
"""
import argparse, os, pathlib, sys, time
os.environ.setdefault("RNN_T", "750"); os.environ.setdefault("RNN_EPOCHS", "100")
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "data" / "processed" / "kernel"; OUT_TAB = ROOT / "output" / "kernel_fit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=pathlib.Path, default=ROOT.parent / "RNN_Gain_Mod" / "gain-controller-rnn")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 62)))
    ap.add_argument("--gains", type=float, nargs="+", default=[0.8, 1.0, 1.2])
    ap.add_argument("--n-trials", type=int, default=2000)
    ap.add_argument("--data-seed", type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo)); os.chdir(a.repo)
    import gainrnn.regime_lib as L, gainrnn.eigcore as E
    OUT_DATA.mkdir(parents=True, exist_ok=True); OUT_TAB.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in a.seeds:
        t0 = time.time(); m = E.load_model("nxx1", s)
        ds = L.make_dataset(a.n_trials, seed=a.data_seed)
        goal = np.asarray(ds.goals_list); labels = np.asarray(ds.labels)
        dat = ds.data[:, :, 0, :].numpy(); rel = dat[np.arange(len(goal)), :, goal]        # (N, T)
        coh_signed = np.asarray(ds.coherence, dtype=float)[np.arange(len(goal)), goal]   # relevant stream's coherence
        out = dict(rel=rel.astype(np.float16), labels=labels, goals=goal, coh_signed=coh_signed)
        for f in a.gains:
            ld = L.rollout_dv(m, ds.data, f_nm=f, seed=a.data_seed + 1)
            ch = L.choices_from_dv(ld)
            k = L.psychophysical_kernel(m, f, n_trials=a.n_trials, seed=a.data_seed, ld=ld, ds=ds)
            out[f"choice_g{f}"] = ch; out[f"dvT_g{f}"] = ld[:, -1].numpy()
            rows.append(dict(seed=s, gain=f, n=len(ch), acc=k["acc"], slope=k["slope"], **{f"w{i}": w for i, w in enumerate(k["weights"])}))
        np.savez_compressed(OUT_DATA / f"seed{s}.npz", **out)
        print(f"seed {s}: {time.time()-t0:.0f}s | " + " ".join(f"g{r['gain']}: slope {r['slope']:+.3f} acc {r['acc']:.3f}" for r in rows[-len(a.gains):]), flush=True)
    tab = pd.DataFrame(rows); tab.to_csv(OUT_TAB / "network_kernels.csv", index=False)
    print("\nmean kernel slope by gain:"); print(tab.groupby("gain")[["slope", "acc"]].agg(["mean", "sem"]).round(4).to_string())


if __name__ == "__main__":
    main()
