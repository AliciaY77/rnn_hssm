"""Write the Track A array task files (one argument line for track_a/fit_hssm_ou.py per array index).

  python track_a/bash/make_tasks.py
Writes track_a/bash/tasks_pooled.txt (12) and tasks_pernet.txt (120).
"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
GAINS = [0.8, 1.0, 1.2]
K_POOLED = [8, 10, 12, 16]     # k = 10 is the plan's headline; 8 is the sanity run;
K_PERNET = [10, 16]            # 12/16 added because g piles on the +1 ceiling at k = 10 (g_nat ~ 9.5)
SEEDS = list(range(42, 62))

pooled = [f"--gain {g} --k {k}" for k in K_POOLED for g in GAINS]
pernet = [f"--gain {g} --k {k} --seed-filter {s}" for k in K_PERNET for g in GAINS for s in SEEDS]

(HERE / "tasks_pooled.txt").write_text("\n".join(pooled) + "\n")
(HERE / "tasks_pernet.txt").write_text("\n".join(pernet) + "\n")
print(f"tasks_pooled.txt: {len(pooled)} tasks (array 0-{len(pooled)-1})")
print(f"tasks_pernet.txt: {len(pernet)} tasks (array 0-{len(pernet)-1})")
