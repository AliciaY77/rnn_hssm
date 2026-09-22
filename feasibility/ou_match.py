"""
Feasibility check: can a constant-bound OU process *inside the pretrained LAN's parameter box*
reproduce the networks' fixed-bound RT/choice distributions?  (follow-up to issues #1 / #2)

Uses the exact ssm-simulators `ornstein` simulator (what the LAN approximates), not the LAN.
Target: data/processed/fixed_bound/hssm_ready_nxx1_fixed_b{bound}_g{gain}.parquet (all 20 seeds pooled),
summarised per |coherence| as accuracy, RT quantiles (correct; error where n >= 30) and omission rate.
Data RTs are multiplied by the time stretch k; the simulator runs to max_t = 0.75 * k (the horizon) and
trials that do not cross are omissions, as in the data.

Parameters searched (accuracy coding, v = v0 + v1 * |coh|), constrained to the LAN box after stretching:
  v in [-2, 2] at every coherence, a in [0.3, 3], z in [0.1, 0.9], g in [-1, 1], t in [1e-3, 2].
Sign convention: dx = (v - g x) dt + dW  ->  g > 0 leaky, g < 0 unstable.  Native-time g = k * g.

Outputs -> output/feasibility/match_b{bound}_g{gain}_k{k}.{json,csv,png}
Usage: python feasibility/ou_match.py --gain 0.8 --bound 1.5 --stretch 6 [--n-random 300 --n-sim 2000]
"""
import argparse, json, pathlib, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from ssms.basic_simulators.simulator import simulator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "fixed_bound"
OMIT = ROOT / "output" / "fixed_bound" / "omissions.csv"
OUT = ROOT / "output" / "feasibility"
Q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
BOX = dict(v=(-2.0, 2.0), a=(0.3, 3.0), z=(0.1, 0.9), g=(-1.0, 1.0), t=(1e-3, 2.0))
NAMES = ["v0", "v1", "a", "z", "g", "t"]



def _no_parquet():
    try:
        import pyarrow  # noqa: F401
        return False
    except ImportError:
        return True


def target_from_data(df, omissions, gain, bound, k):
    cohs = sorted(df.coherence.round(2).unique()); tgt = {}
    for c in cohs:
        d = df[df.coherence.round(2) == c]; rt = d.rt.values * k; corr = d.response.values == 1
        om = omissions[(omissions.bound == bound) & (omissions.gain == gain) & (omissions.coherence.astype(str) == str(c))].omit
        tgt[c] = dict(n=len(d), acc=corr.mean(), q_c=np.quantile(rt[corr], Q), q_e=np.quantile(rt[~corr], Q) if (~corr).sum() >= 30 else None,
                      omit=float(om.iloc[0]) if len(om) else 0.0)
    scale = float(np.quantile(df.rt.values * k, 0.75) - np.quantile(df.rt.values * k, 0.25))
    return tgt, scale


def simulate(theta_vec, cohs, n, max_t, seed):
    v0, v1, a, z, g, t = theta_vec; out = {}
    for i, c in enumerate(cohs):
        # ssms segfaults for small max_t (observed at 0.75 s): simulate to >= 2 s and apply the deadline afterwards
        s = simulator(theta=dict(v=v0 + v1 * c, a=a, z=z, g=g, t=t), model="ornstein", n_samples=n,
                      delta_t=0.001, max_t=max(max_t, 2.0), random_state=seed + i)
        rt = np.asarray(s["rts"]).ravel(); ch = np.asarray(s["choices"]).ravel()
        ok = (rt > 0) & (rt <= max_t); omit = 1 - ok.mean(); rt, ch = rt[ok], ch[ok]
        corr = ch > 0
        out[c] = dict(acc=corr.mean() if len(corr) else np.nan, omit=omit,
                      q_c=np.quantile(rt[corr], Q) if corr.sum() >= 5 else None,
                      q_e=np.quantile(rt[~corr], Q) if (~corr).sum() >= 5 else None, rt=rt, corr=corr)
    return out


def objective(theta_vec, tgt, scale, cohs, n, max_t, seed=0, w_acc=10.0, w_om=10.0):
    v0, v1, a, z, g, t = theta_vec
    for val, (lo, hi) in [(v0, BOX["v"]), (v0 + v1 * max(cohs), BOX["v"]), (a, BOX["a"]), (z, BOX["z"]), (g, BOX["g"]), (t, BOX["t"])]:
        if not (lo <= val <= hi): return 1e6
    sim = simulate(theta_vec, cohs, n, max_t, seed); loss = 0.0
    for c in cohs:
        T, S = tgt[c], sim[c]
        if S["q_c"] is None: return 1e6
        loss += np.sum(((S["q_c"] - T["q_c"]) / scale) ** 2)
        if T["q_e"] is not None and S["q_e"] is not None: loss += 0.5 * np.sum(((S["q_e"] - T["q_e"]) / scale) ** 2)
        loss += w_acc * (S["acc"] - T["acc"]) ** 2 * 10 + w_om * (S["omit"] - T["omit"]) ** 2 * 10
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, required=True); ap.add_argument("--bound", type=float, default=1.5)
    ap.add_argument("--stretch", type=float, default=6.0); ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--n-sim", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--v-max", type=float, default=2.0, help="relax the LAN drift box (default 2 = LAN)")
    ap.add_argument("--g-max", type=float, default=1.0, help="relax the LAN leak box (default 1 = LAN)")
    ap.add_argument("--a-min", type=float, default=0.3, help="relax the LAN boundary floor (default 0.3 = LAN)")
    ap.add_argument("--t-min", type=float, default=1e-3)
    a = ap.parse_args(); k = a.stretch; OUT.mkdir(parents=True, exist_ok=True)
    BOX.update(v=(-a.v_max, a.v_max), g=(-a.g_max, a.g_max), a=(a.a_min, BOX["a"][1]), t=(a.t_min, BOX["t"][1]))
    box_tag = "" if (a.v_max, a.g_max, a.a_min) == (2.0, 1.0, 0.3) else f"_box_v{a.v_max:g}_g{a.g_max:g}_a{a.a_min:g}"
    tag = f"match_b{a.bound}_g{a.gain}_k{k:g}{box_tag}"; t0 = time.time()
    stem = DATA / f"hssm_ready_nxx1_fixed_b{a.bound}_g{a.gain}"
    df = pd.read_csv(f"{stem}.csv") if not stem.with_suffix(".parquet").exists() or _no_parquet() else pd.read_parquet(f"{stem}.parquet")
    omissions = pd.read_csv(OMIT); cohs = sorted(df.coherence.round(2).unique())
    tgt, scale = target_from_data(df, omissions, a.gain, a.bound, k); max_t = 0.75 * k
    print(f"{tag}: {len(df)} trials, cohs {cohs}, RT scale (IQR, stretched) {scale:.3f} s, horizon {max_t} s", flush=True)

    rng = np.random.default_rng(a.seed); best = []
    for i in range(a.n_random):                       # random search inside the box
        vmax, gmax, amin = BOX["v"][1], BOX["g"][1], BOX["a"][0]
        v1 = rng.uniform(0, vmax / max(cohs)); v0 = rng.uniform(-0.5, min(1.0, vmax - v1 * max(cohs)))
        th = [v0, v1, rng.uniform(amin, 3.0), rng.uniform(0.35, 0.65), rng.uniform(-gmax, gmax), rng.uniform(BOX["t"][0], min(0.5 * k, 2.0))]
        best.append((objective(th, tgt, scale, cohs, 400, max_t, a.seed), th))
        if (i + 1) % 50 == 0: print(f"  random {i+1}/{a.n_random}: best {min(b[0] for b in best):.3f}", flush=True)
    best.sort(key=lambda x: x[0]); refined = []
    for loss0, th in best[:5]:                        # local refinement of the top 5
        r = minimize(objective, th, args=(tgt, scale, cohs, a.n_sim, max_t, a.seed), method="Nelder-Mead",
                     options=dict(maxiter=250, xatol=1e-3, fatol=1e-3))
        refined.append((r.fun, list(r.x))); print(f"  refine: {loss0:.3f} -> {r.fun:.3f}  {np.round(r.x, 3)}", flush=True)
    refined.sort(key=lambda x: x[0]); loss, th = refined[0]
    sim = simulate(th, cohs, 20000, max_t, 999)

    rows = []
    for c in cohs:
        T, S = tgt[c], sim[c]
        rows.append(dict(coherence=c, n=T["n"], acc_obs=T["acc"], acc_sim=S["acc"], omit_obs=T["omit"], omit_sim=S["omit"],
                         **{f"qc{int(q*100)}_obs": T["q_c"][j] / k * 1000 for j, q in enumerate(Q)},
                         **{f"qc{int(q*100)}_sim": S["q_c"][j] / k * 1000 for j, q in enumerate(Q)},
                         qe50_obs=(T["q_e"][2] / k * 1000 if T["q_e"] is not None else np.nan),
                         qe50_sim=(S["q_e"][2] / k * 1000 if S["q_e"] is not None else np.nan)))
    tab = pd.DataFrame(rows).round(3); tab.to_csv(OUT / f"{tag}.csv", index=False)
    print("\n=== best match (RT columns in native ms)"); print(tab.to_string(index=False))

    # g-sensitivity at the best fit: how much do the distributions move if g is forced elsewhere?
    sens = {}
    for gg in [-BOX["g"][1], -BOX["g"][1] / 2, 0.0, BOX["g"][1] / 2, BOX["g"][1]]:
        th2 = list(th); th2[4] = gg
        sens[gg] = dict(loss=objective(th2, tgt, scale, cohs, a.n_sim, max_t, a.seed),
                        q50_coh0_ms=float(np.median(simulate(th2, [0.0], 5000, max_t, 7)[0.0]["rt"]) / k * 1000))
    print("\n=== g sensitivity (others fixed at best): loss and median RT at coh 0 (native ms)")
    for gg, s in sens.items(): print(f"  g={gg:+.1f}: loss {s['loss']:.3f}  median RT coh0 {s['q50_coh0_ms']:.0f} ms")

    res = dict(tag=tag, gain=a.gain, bound=a.bound, stretch=k, loss=loss, box=dict(v_max=a.v_max, g_max=a.g_max, a_min=a.a_min), params=dict(zip(NAMES, map(float, th))),
               g_native_per_s=float(th[4] * k), v_at_max_coh=float(th[0] + th[1] * max(cohs)),
               on_box_edge={p: bool(abs(val - lo) < 0.02 or abs(val - hi) < 0.02) for p, val, (lo, hi) in
                            [("a", th[2], BOX["a"]), ("g", th[4], BOX["g"]), ("t", th[5], BOX["t"]), ("v_max", th[0] + th[1] * max(cohs), BOX["v"])]},
               g_sensitivity=sens, minutes=(time.time() - t0) / 60, n_trials=int(len(df)), scale_s=scale)
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=2, default=float)); print(json.dumps(res["params"], indent=None), "| g_native", round(res["g_native_per_s"], 2), "| edge:", res["on_box_edge"])

    fig, axes = plt.subplots(2, len(cohs), figsize=(3 * len(cohs), 5.5), sharex=True)
    for j, c in enumerate(cohs):
        d = df[df.coherence.round(2) == c]; S = sim[c]; bins = np.linspace(0, 750, 40)
        for row, (sel_o, sel_s, lab) in enumerate([(d.response == 1, S["corr"], "correct"), (d.response == -1, ~S["corr"], "error")]):
            ax = axes[row, j]; ax.hist(d.rt[sel_o] * 1000, bins=bins, density=True, alpha=0.5, color="C0", label="network")
            if sel_s.sum() > 5: ax.hist(S["rt"][sel_s] / k * 1000, bins=bins, density=True, histtype="step", color="C3", lw=1.5, label="OU best")
            ax.set_title(f"coh {c}  {lab}\nacc obs {tgt[c]['acc']:.2f} / sim {S['acc']:.2f}", fontsize=8)
        axes[1, j].set_xlabel("RT (ms, native)")
    axes[0, 0].legend(fontsize=7); fig.suptitle(f"{tag}: g={th[4]:+.2f} (native {th[4]*k:+.1f}/s), a={th[2]:.2f}, v_max={th[0]+th[1]*max(cohs):.2f}, t={th[5]:.3f}  loss={loss:.2f}", fontsize=9)
    plt.tight_layout(); fig.savefig(OUT / f"{tag}.png", dpi=130); print("saved", OUT / f"{tag}.png", f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
