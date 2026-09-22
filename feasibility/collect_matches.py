"""Collect output/feasibility/match_*.json into one table (best OU match per gain x bound x stretch)."""
import glob, json, pathlib
import pandas as pd
OUT = pathlib.Path(__file__).resolve().parents[1] / "output" / "feasibility"
rows = []
for f in sorted(glob.glob(str(OUT / "match_*.json"))):
    r = json.load(open(f)); p = r["params"]; s = r["g_sensitivity"]
    bx = r.get("box", dict(v_max=2.0, g_max=1.0, a_min=0.3)); gm = str(float(bx["g_max"]))
    rows.append(dict(box=f"v{bx['v_max']:g}_g{bx['g_max']:g}_a{bx['a_min']:g}", gain=r["gain"], bound=r["bound"], k=r["stretch"], loss=r["loss"], g=p["g"], g_native=r["g_native_per_s"],
                     a=p["a"], a_native=p["a"] / r["stretch"] ** 0.5, v_max=r["v_at_max_coh"], z=p["z"], t=p["t"], t_native_ms=p["t"] / r["stretch"] * 1000,
                     edge=",".join(k for k, v in r["on_box_edge"].items() if v) or "-",
                     loss_gneg=s["-" + gm]["loss"], loss_g0=s["0.0"]["loss"], loss_gpos=s[gm]["loss"], minutes=r["minutes"]))
t = pd.DataFrame(rows).sort_values(["box", "bound", "gain", "k"]); t.to_csv(OUT / "matches_table.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
print("\nbest stretch per (bound, gain):")
print(t.loc[t.groupby(["box", "bound", "gain"]).loss.idxmin(), ["box", "bound", "gain", "k", "loss", "g", "g_native", "a_native", "v_max", "t_native_ms", "edge"]].round(3).to_string(index=False))
