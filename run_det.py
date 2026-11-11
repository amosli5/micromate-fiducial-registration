"""D1: detector recall / centre error vs dose and bone overlap.  D2: end-to-end image -> detector -> solver -> TRE."""
import os, sys, time, numpy as np, pandas as pd
from common import *
from detector import render, detect, match, ball_radii_px
import robust
OUT = sys.argv[1] if len(sys.argv) > 1 else "figs"
f1 = os.path.join(OUT, "d1_detector.csv")
if not os.path.exists(f1):
    rows = []
    for bones in (0, 2):
        for I0 in (1e3, 3e3, 1e4, 1e5):
            rng = np.random.default_rng(500); t = time.time()
            for tr in range(4):
                T, _, _ = make_random_displacement_pose(1.0, rng, max_rot_deg=20, max_trans_mm=15)
                for k, P in enumerate(P3):
                    uv = project_all(T, P, XYZ); r = ball_radii_px(T, P, XYZ)
                    img = render(uv, r, rng, I0=I0, n_bones=bones)
                    d, sc = detect(img, r.mean())
                    rec, fp, e = match(d, uv); rec5, fp5, _ = match(d[:5], uv)
                    rows.append(dict(bones=bones, I0=I0, trial=tr, view=k, recall=rec, fp=fp, recall_top5=rec5, n_det=len(d),
                                     err_median=np.median(e) if len(e) else np.nan, errs=";".join(f"{x:.3f}" for x in e)))
            print(f"D1 bones={bones} I0={I0:.0e} {time.time()-t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f1, index=False)
f2 = os.path.join(OUT, "d2_end_to_end.csv")
if not os.path.exists(f2):
    rows = []
    for bones in (0, 2):
        rng = np.random.default_rng(900); t = time.time()
        for tr in range(8):
            T_gt, rv, tv = make_random_displacement_pose(1.0, rng, max_rot_deg=20, max_trans_mm=15)
            p_gt = np.r_[rv.ravel(), tv.ravel()]
            Y = []
            for P in P3:
                uv = project_all(T_gt, P, XYZ); r = ball_radii_px(T_gt, P, XYZ)
                d, _ = detect(render(uv, r, rng, I0=1e4, n_bones=bones), r.mean()); Y.append(d)
            p0 = cmaes_centroid_aligned_initial_guess_from_gt_rotation(rng=rng, p_gt=p_gt, P_list=P3, uv_list=Y, xyz=XYZ,
                                                                      DISP_DEG=10.0, tvec_center=np.zeros(3))[0]
            for name, c in COSTS.items():
                p, f, _ = robust.solve_cmaes(c, P3, Y, XYZ, p0, seed=int(rng.integers(1, 2**31 - 1)))
                rows.append(dict(bones=bones, trial=tr, cost=name, n_det=";".join(str(len(y)) for y in Y),
                                 rms_tre_vol=rms(tre_per_point(T_of(p), T_gt, GRID)), final_cost=f))
        print(f"D2 bones={bones} {time.time()-t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f2, index=False)
print("DONE", flush=True)
