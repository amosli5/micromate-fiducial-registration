"""
Experiment harness around the UNMODIFIED GitLab cmaes_package solver.

Only the scenario is controlled here (camera geometry, detection noise,
missing / spurious detections, initial rotation error). Stage 1 and Stage 2
are the repo's own functions.
"""
import sys, time
import numpy as np

sys.path.insert(0, "/home/claude/repo/micromate_robot_registration-main")
from cmaes_package.io import get_robot_fiducials
from cmaes_package.transform import make_T_from_rvec_tvec, make_random_displacement_pose
from cmaes_package.projection import project_point
from cmaes_package.tre import make_volume_grid_points, tre_stats, apply_T_to_points
from cmaes_package.synth_camera import make_intrinsics, make_orbit_projections
from cmaes_package.solver import (
    cmaes_centroid_aligned_initial_guess_from_gt_rotation,
    solve_pose_CMAES_closest_point,
)

# Fiducials exactly as in the repo's final notebook (CAD points moved by T_small)
T_SMALL = make_T_from_rvec_tvec(np.array([[np.pi/30], [np.pi/18], [np.pi/60]]),
                                np.array([[0.5], [1.0], [2.0]]))
XYZ = apply_T_to_points(T_SMALL, get_robot_fiducials())

# TRE grid exactly as in the repo (~300 mm cube centred on the origin)
_, GRID = make_volume_grid_points(np.array([416, 424, 416.]), np.array([.718] * 3))

# Representative C-arm/CBCT geometry (assumed values, see summary):
#   source-to-isocentre 600 mm, source-to-detector 1200 mm,
#   976 x 976 detector at 0.4 mm/px  ->  f = 1200/0.4 = 3000 px, principal point at centre
def real_like_K(sdd_mm=1200.0, pix_mm=0.4, n_px=976):
    f = sdd_mm / pix_mm
    return make_intrinsics(f, f, n_px / 2, n_px / 2)

def views(angles, radius_mm=600.0, K=None):
    K = real_like_K() if K is None else K
    P_list, _ = make_orbit_projections(K=K, angles_deg=tuple(angles), radius_mm=radius_mm)
    return P_list


def observe(xyz, P_list, T, rng, noise_px=0.0, drop_per_view=0, extra_per_view=0):
    """Project, add pixel noise, drop and/or add spurious points, shuffle."""
    Y = []
    for P in P_list:
        uv = np.vstack([project_point(x, P, T) for x in xyz])
        uv = uv + rng.normal(scale=noise_px, size=uv.shape)
        if drop_per_view:
            keep = rng.choice(len(uv), size=len(uv) - drop_per_view, replace=False)
            uv = uv[keep]
        if extra_per_view:
            lo, hi = uv.min(0) - 40, uv.max(0) + 40           # clutter near the robot
            uv = np.vstack([uv, rng.uniform(lo, hi, size=(extra_per_view, 2))])
        Y.append(uv[rng.permutation(len(uv))])
    return Y


def trial(rng, P_list, rot_deg=20.0, trans_mm=15.0, init_rot_err_deg=10.0,
          noise_px=0.0, drop_per_view=0, extra_per_view=0):
    # Ground-truth pose: random axis/direction with fixed magnitudes (repo helper, s=1)
    T_gt, rvec, tvec = make_random_displacement_pose(1.0, rng, max_rot_deg=rot_deg, max_trans_mm=trans_mm)
    p_gt = np.r_[rvec.ravel(), tvec.ravel()]
    Y = observe(XYZ, P_list, T_gt, rng, noise_px, drop_per_view, extra_per_view)

    t0 = time.perf_counter()
    # Stage 1 (repo): GT rotation perturbed by +/- init_rot_err_deg per component, translation by centroid CMA-ES
    p0, info = cmaes_centroid_aligned_initial_guess_from_gt_rotation(
        rng=rng, p_gt=p_gt, P_list=P_list, uv_list=Y, xyz=XYZ,
        DISP_DEG=init_rot_err_deg, tvec_center=np.zeros(3))
    # Stage 2 (repo): full 6-DOF closest-point CMA-ES, default notebook settings
    p, rmse, es, _ = solve_pose_CMAES_closest_point(
        P_list=P_list, uv_list=Y, xyz_list=XYZ, p0=p0,
        rot_step_deg=15.0, trans_step_mm=5.0, sigma0=1.0, popsize=100, maxiter=500, verbose=False)
    rt = time.perf_counter() - t0

    T_est = make_T_from_rvec_tvec(p[:3], p[3:])
    vol = tre_stats(T_est, T_gt, GRID)
    fid = tre_stats(T_est, T_gt, XYZ)          # error at the fiducials themselves
    return dict(rms_tre_vol=vol["rms_tre"], max_tre_vol=vol["max_tre"],
                rms_tre_fid=fid["rms_tre"], reproj_rmse_px=rmse, runtime_s=rt,
                evals=int(es.countevals),
                init_rot_err_deg=float(np.rad2deg(np.linalg.norm(p0[:3] - p_gt[:3]))))
