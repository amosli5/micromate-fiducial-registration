import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/experiments")
from harness import trial, views

N = int(os.environ.get("N_TRIALS", 12))
OUT = "/home/claude/experiments/results.csv"

REAL3 = (120, 180, 240)                      # spacing used in the real-data notebook
BASE = dict(angles=REAL3, noise_px=0.5, init_rot_err_deg=10.0, drop=0, extra=0)

configs = []
def add(exp, label, **kw):
    configs.append((exp, label, {**BASE, **kw}))

# E1 - imaging geometry
add("E1 views", "1 view (0°)", angles=(0,))
add("E1 views", "2 views, 15° apart", angles=(0, 15))
add("E1 views", "2 views, 30° apart", angles=(0, 30))
add("E1 views", "2 views, 60° apart", angles=(0, 60))
add("E1 views", "2 views, 90° apart", angles=(0, 90))
add("E1 views", "2 views, opposed (0/180°)", angles=(0, 180))
add("E1 views", "3 views, 60° apart", angles=REAL3)
add("E1 views", "4 views, 0/90/180/270°", angles=(0, 90, 180, 270))
# E2 - detection noise
for n in (0.0, 0.5, 1.0, 2.0, 4.0):
    add("E2 noise", f"{n:g} px", noise_px=n)
# E3 - missing / spurious detections
add("E3 detections", "all 5 detected", )
add("E3 detections", "1 missing per view", drop=1)
add("E3 detections", "2 missing per view", drop=2)
add("E3 detections", "1 spurious per view", extra=1)
add("E3 detections", "2 spurious per view", extra=2)
# E4 - initial rotation error (capture range)
for d in (0, 10, 20, 30, 45):
    add("E4 init", f"±{d}° per axis", init_rot_err_deg=float(d))

done = set()
if os.path.exists(OUT):
    d=pd.read_csv(OUT); done=set(zip(d.experiment,d.label))

cache = {}
for exp, label, c in configs:
    key = json.dumps(c, sort_keys=True)
    rows = []
    if (exp, label) in done:
        continue
    if key in cache:
        rows = [{**r, "experiment": exp, "label": label} for r in cache[key]]
    else:
        rng = np.random.default_rng(1234)          # same poses across configs
        P_list = views(c["angles"])
        t = time.time()
        for i in range(N):
            r = trial(rng, P_list, init_rot_err_deg=c["init_rot_err_deg"], noise_px=c["noise_px"],
                      drop_per_view=c["drop"], extra_per_view=c["extra"])
            rows.append({**r, "trial": i, "key": key, "experiment": exp, "label": label})
        cache[key] = rows
        print(f"{exp:14s} {label:28s} {time.time()-t:6.1f}s  "
              f"median RMS TRE {np.median([r['rms_tre_vol'] for r in rows]):8.3f} mm", flush=True)
    pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
print("DONE", flush=True)
