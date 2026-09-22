"""Collect output/feasibility/match_*.json into one table (best OU match per gain x bound x stretch)."""
import glob, json, pathlib
import pandas as pd
OUT = pathlib.Path(__file__).resolve().parents[1] / "output" / "feasibility"
rows = []
for f in sorted(glob.glob(str(OUT / "match_*.json"))):
    r = json.load(open(f)); p = r["params"]; s = r["g_sensitivity"]
    rows.append(dict(gain=r["gain"], bound=r["bound"], k=r["stretch"], loss=r["loss"], g=p["g"], g_native=r["g_native_per_s"],
                     a=p["a"], a_native=p["a"] / r["stretch"] ** 0.5, v_max=r["v_at_max_coh"], z=p["z"], t=p["t"], t_native_ms=p["t"] / r["stretch"] * 1000,
                     edge=",".join(k for k, v in r["on_box_edge"].items() if v) or "-",
                     loss_gm1=s["-1.0"]["loss"], loss_g0=s["0.0"]["loss"], loss_gp1=s["1.0"]["loss"], minutes=r["minutes"]))
t = pd.DataFrame(rows).sort_values(["bound", "gain", "k"]); t.to_csv(OUT / "matches_table.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
print("\nbest stretch per (bound, gain):")
print(t.loc[t.groupby(["bound", "gain"]).loss.idxmin(), ["bound", "gain", "k", "loss", "g", "g_native", "a_native", "v_max", "t_native_ms", "edge"]].round(3).to_string(index=False))
