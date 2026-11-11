"""New experiments after PI feedback.
X1  cost function comparison under missing / spurious detections
X2  minimum number of visible fiducials x number of views
X3  view-pair consistency as a ground-truth-free failure check
"""
import os, sys, time, itertools
import numpy as np, pandas as pd
from common import *

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
N = 15
os.makedirs(OUT, exist_ok=True)

# ---------------- X1
f1 = os.path.join(OUT, "x1_costs.csv")
if not os.path.exists(f1):
    conds = [("all 5 detected", {}), ("1 missing per view", dict(drop=1)),
             ("same ball missing in all views", dict(drop=1, same_missing=True)), ("2 missing per view", dict(drop=2)),
             ("2 spurious per view", dict(extra=2)), ("1 missing + 1 spurious", dict(drop=1, extra=1))]
    rows = []
    for (cn, ck), (fn, f) in itertools.product(conds, COSTS.items()):
        rng = np.random.default_rng(2024); t = time.time()
        for i in range(N):
            rows.append({**trial(rng, P3, f, **ck), "condition": cn, "cost": fn, "trial": i})
        print(f"X1 {cn:32s} {fn:24s} {time.time()-t:5.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(f1, index=False)

# ---------------- X2
f2 = os.path.join(OUT, "x2_min_fiducials.csv")
if not os.path.exists(f2):
    rows = []
    for (vn, ang), drop in itertools.product([("2 views (120°/180°)", (120, 180)), ("3 views (60° apart)", REAL3)], (0, 1, 2)):
        rng = np.random.default_rng(77); Pl = views(ang); t = time.time()
        for i in range(N):
            rows.append({**trial(rng, Pl, cost_reverse, drop=drop), "views": vn, "visible": 5 - drop, "trial": i})
        print(f"X2 {vn} visible={5-drop} {time.time()-t:5.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(f2, index=False)

# ---------------- X3
f3 = os.path.join(OUT, "x3_pair_consistency.csv")
if not os.path.exists(f3):
    rows = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for cn, ck in [("0.5 px", dict(noise_px=.5)), ("2 px", dict(noise_px=2.)), ("4 px", dict(noise_px=4.)),
                   ("1 missing per view, 0.5 px", dict(noise_px=.5, drop=1))]:
        rng = np.random.default_rng(31); t = time.time()
        for i in range(N):
            r = trial(rng, P3, cost_reverse, full=True, **ck)
            Tp = []
            for a, b in pairs:   # re-solve with each pair of views from the same 3-view start
                p, _, _ = solve_cmaes(cost_reverse, [P3[a], P3[b]], [r["Y"][a], r["Y"][b]], XYZ, r["p0"],
                                      seed=int(rng.integers(1, 2**31 - 1)))
                Tp.append(T_of(p))
            pair_tre = [rms(tre_per_point(T, r["T_gt"], GRID)) for T in Tp]
            spread = max(rms(tre_per_point(Tp[a], Tp[b], GRID)) for a, b in itertools.combinations(range(3), 2))
            rows.append(dict(condition=cn, trial=i, tre_3view=r["rms_tre_vol"], pair_spread=spread,
                             tre_pair_best=min(pair_tre), tre_pair_worst=max(pair_tre), final_cost=r["final_cost"],
                             **{f"tre_pair_{a}{b}": v for (a, b), v in zip(pairs, pair_tre)}))
        print(f"X3 {cn:30s} {time.time()-t:5.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(f3, index=False)
print("DONE", flush=True)
