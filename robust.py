"""
Alternative Stage 2 cost functions, plugged into the same CMA-ES setup as the repo
(z-space search, 15 deg / 5 mm per unit, sigma0 = 1, popsize 100, maxiter 500).

All costs are RMS per coordinate in pixels, like the repo's closest_point_rmse_objective.
"""
import numpy as np, cma
from cmaes_package.transform import make_T_from_rvec_tvec


def T_of(p):
    return make_T_from_rvec_tvec(np.asarray(p[:3], float), np.asarray(p[3:], float))


def project_all(T, P, xyz):
    """Vectorised version of the repo's project_point for all fiducials."""
    X = np.c_[xyz, np.ones(len(xyz))] @ T.T @ P.T
    return X[:, :2] / X[:, 2:3]


def dist_matrix(p, P, Y, xyz):
    uv = project_all(T_of(p), P, xyz)
    return np.linalg.norm(uv[:, None, :] - np.asarray(Y)[None, :, :], axis=2)   # (N fiducials, M detections)


def _rms(d2_list):
    d2 = np.concatenate(d2_list) if d2_list else np.array([1e18])
    return float(np.sqrt(d2.sum() / (2 * len(d2))))          # per-coordinate RMS, like the repo


def cost_forward(p, P_list, Y_list, xyz):
    """Repo cost: every fiducial -> its nearest detection (loops over the 5 model points)."""
    return _rms([dist_matrix(p, P, Y, xyz).min(1) ** 2 for P, Y in zip(P_list, Y_list) if len(Y)])


def cost_reverse(p, P_list, Y_list, xyz):
    """Every detection -> its nearest projected fiducial (loops over what was detected)."""
    return _rms([dist_matrix(p, P, Y, xyz).min(0) ** 2 for P, Y in zip(P_list, Y_list) if len(Y)])


def make_cost_trimmed(k):
    """Every fiducial -> nearest detection, but keep only the k best fiducials per view
    (loops over the minimum number of fiducials, so k=5 is the repo cost)."""
    def cost(p, P_list, Y_list, xyz):
        out = []
        for P, Y in zip(P_list, Y_list):
            if len(Y):
                d = np.sort(dist_matrix(p, P, Y, xyz).min(1))
                out.append(d[:min(k, len(Y))] ** 2)
        return _rms(out)
    return cost


def solve_cmaes(cost, P_list, Y_list, xyz, p0, rot_step_deg=15.0, trans_step_mm=5.0,
                sigma0=1.0, popsize=100, maxiter=500, seed=None):
    step = np.r_[[np.deg2rad(rot_step_deg)] * 3, [trans_step_mm] * 3]
    p0 = np.asarray(p0, float)
    opts = {"maxiter": maxiter, "popsize": popsize, "verbose": -9, "verb_disp": 0, "verb_log": 0}
    if seed is not None:
        opts["seed"] = seed
    es = cma.CMAEvolutionStrategy(np.zeros(6), sigma0, opts)
    best_z, best_f = np.zeros(6), cost(p0, P_list, Y_list, xyz)
    while not es.stop():
        Z = es.ask()
        F = [cost(p0 + z * step, P_list, Y_list, xyz) for z in Z]
        es.tell(Z, F)
        i = int(np.argmin(F))
        if F[i] < best_f:
            best_f, best_z = F[i], np.array(Z[i])
    return p0 + best_z * step, best_f, es
