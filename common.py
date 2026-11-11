import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "micromate_robot_registration-main")))
from cmaes_package.io import get_robot_fiducials
from cmaes_package.transform import make_T_from_rvec_tvec, make_random_displacement_pose
from cmaes_package.synth_camera import make_intrinsics, make_orbit_projections
from cmaes_package.solver import cmaes_centroid_aligned_initial_guess_from_gt_rotation
from robust import T_of, project_all, solve_cmaes, cost_forward, cost_reverse, make_cost_trimmed


def apply_T(T, pts): return (T[:3, :3] @ pts.T).T + T[:3, 3]


T_SMALL = make_T_from_rvec_tvec(np.array([[np.pi/30], [np.pi/18], [np.pi/60]]), np.array([[0.5], [1.0], [2.0]]))
XYZ = apply_T(T_SMALL, get_robot_fiducials())
SAD, SDD, PIX, NPX = 600.0, 1200.0, 0.4, 976
K = make_intrinsics(SDD / PIX, SDD / PIX, NPX / 2, NPX / 2)
REAL3 = (120, 180, 240)
def views(angles): return make_orbit_projections(K=K, angles_deg=tuple(angles), radius_mm=SAD)[0]
P3 = views(REAL3)

_L = np.array([416, 424, 416]) * 0.718
GRID = np.stack(np.meshgrid(*[np.linspace(-l / 2, l / 2, 10) for l in _L], indexing="xy"), -1).reshape(-1, 3)
def tre_per_point(Te, Tg, pts): return np.linalg.norm(apply_T(Te, pts) - apply_T(Tg, pts), axis=1)
def rms(x): return float(np.sqrt(np.mean(np.square(x))))


def observe(P_list, T, rng, noise_px=0.0, drop=0, extra=0, same_missing=False):
    """Project, add noise, delete `drop` fiducials per view (same ones in every view if same_missing),
    add `extra` uniform clutter points, shuffle. Returns detections and the indices kept per view."""
    out, kept = [], []
    fixed = rng.choice(len(XYZ), drop, replace=False) if (same_missing and drop) else None
    for P in P_list:
        uv = project_all(T, P, XYZ) + rng.normal(scale=noise_px, size=(len(XYZ), 2))
        keep = np.arange(len(XYZ))
        if drop:
            gone = fixed if fixed is not None else rng.choice(len(XYZ), drop, replace=False)
            keep = np.setdiff1d(keep, gone)
        u = uv[keep]
        if extra:
            lo, hi = uv.min(0) - 40, uv.max(0) + 40
            u = np.vstack([u, rng.uniform(lo, hi, size=(extra, 2))])
        out.append(u[rng.permutation(len(u))]); kept.append(keep)
    return out, kept


COSTS = {"repo (all 5 fiducials)": cost_forward, "detections → fiducials": cost_reverse,
         "best 4 per view": make_cost_trimmed(4), "best 3 per view": make_cost_trimmed(3)}


def trial(rng, P_list, cost=cost_forward, noise_px=0.5, drop=0, extra=0, same_missing=False,
          init_rot_err_deg=10.0, start_at_truth=False, full=False):
    T_gt, rv, tv = make_random_displacement_pose(1.0, rng, max_rot_deg=20.0, max_trans_mm=15.0)
    p_gt = np.r_[rv.ravel(), tv.ravel()]
    Y, kept = observe(P_list, T_gt, rng, noise_px, drop, extra, same_missing)
    p0 = p_gt if start_at_truth else cmaes_centroid_aligned_initial_guess_from_gt_rotation(
        rng=rng, p_gt=p_gt, P_list=P_list, uv_list=Y, xyz=XYZ, DISP_DEG=init_rot_err_deg, tvec_center=np.zeros(3))[0]
    p, f, es = solve_cmaes(cost, P_list, Y, XYZ, p0, seed=int(rng.integers(1, 2**31 - 1)))
    Te = T_of(p)
    r = dict(rms_tre_vol=rms(tre_per_point(Te, T_gt, GRID)), rms_tre_fid=rms(tre_per_point(Te, T_gt, XYZ)),
             final_cost=f, cost_at_truth=cost(p_gt, P_list, Y, XYZ), evals=int(es.countevals))
    if full: r.update(p=p, p_gt=p_gt, p0=p0, T_est=Te, T_gt=T_gt, Y=Y, kept=kept)
    return r
