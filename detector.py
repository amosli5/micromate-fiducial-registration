"""
Synthetic X-ray views of the MicroMate fiducials + a classical detector.

Renderer (log / line-integral domain, then Poisson counts):
    A(u) = A_body(u) + A_bone(u) + sum_i mu_glass * chord_i(u)
    I(u) ~ Poisson(I0 * exp(-A(u)))   ->   image = -ln(I / I0)
Detector:
    1. background removal: subtract a wide Gaussian blur (sigma = 3 r)
    2. Laplacian-of-Gaussian at sigma = r / sqrt(2) (r = expected ball radius in px)
    3. local maxima of -LoG above a robust threshold -> candidates
    4. sub-pixel refinement: robust least-squares fit of a projected-sphere profile
       a * sqrt(max(0, rho^2 - |u - c|^2)) + quadratic background, in a 1.3 r window
    5. physics checks: fitted radius within 25% of expected, peak height within 0.5-1.8x mu_glass * 10 mm
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares, linear_sum_assignment

BALL_R_MM = 5.0          # 10 mm glass balls
MU_GLASS = 0.044         # 1/mm, borosilicate glass ~70 keV (assumed)
MU_BONE = 0.04           # 1/mm, bone-like cylinder (assumed)
MU_BODY = 0.02           # 1/mm, water-equivalent body (assumed)


def camera_centre(P):
    M, p4 = P[:, :3], P[:, 3]
    return -np.linalg.solve(M, p4)


def ball_radii_px(T, P, xyz, sdd=1200.0, pix=0.4):
    C = camera_centre(P)
    X = (T[:3, :3] @ xyz.T).T + T[:3, 3]
    return BALL_R_MM * sdd / np.linalg.norm(X - C, axis=1) / pix


def render(uv, r_px, rng, n=976, I0=1e4, n_bones=2, body_mm=200.0, blur_px=1.0):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    # body: smooth thickness field around body_mm
    A = MU_BODY * body_mm * (1 + 0.15 * ndi.gaussian_filter(rng.normal(size=(n, n)), 80) / 0.004)
    # bones: cylinders (chord profile across the axis), passing near the fiducial cluster
    ctr = uv.mean(0)
    for _ in range(n_bones):
        th = rng.uniform(0, np.pi); off = rng.uniform(-120, 120); rad = rng.uniform(40, 70)   # px
        d = (xx - ctr[0]) * np.sin(th) - (yy - ctr[1]) * np.cos(th) - off
        # hollow cortical tube: outer radius rad, inner 0.7 rad; 0.2 mm per px at the object (2x magnification)
        A += MU_BONE * 0.2 * 2 * (np.sqrt(np.clip(rad ** 2 - d ** 2, 0, None)) - np.sqrt(np.clip((0.7 * rad) ** 2 - d ** 2, 0, None)))
    # glass balls: chord length of a sphere, radius in px -> mm at the detector scale
    for (cx, cy), r in zip(uv, r_px):
        rho2 = (xx - cx) ** 2 + (yy - cy) ** 2
        chord_px = 2 * np.sqrt(np.clip(r ** 2 - rho2, 0, None))
        A += MU_GLASS * chord_px * (2 * BALL_R_MM) / (2 * r)          # scale so the max chord = 10 mm
    A = ndi.gaussian_filter(A, blur_px)
    counts = rng.poisson(I0 * np.exp(-A)).astype(float)
    return -np.log(np.clip(counts, 1, None) / I0)


def _sphere_model(q, X, Yg, x0, y0):
    """Projected glass sphere (chord profile) on a quadratic background."""
    cx, cy, rho, a, b0, bx, by, bxx, byy, bxy = q
    dx, dy = X - x0, Yg - y0
    return (a * np.sqrt(np.clip(rho ** 2 - (X - cx) ** 2 - (Yg - cy) ** 2, 0, None))
            + b0 + bx * dx + by * dy + bxx * dx * dx + byy * dy * dy + bxy * dx * dy)


EXPECTED_PEAK = MU_GLASS * 2 * BALL_R_MM      # line integral through the ball centre (assumed known)


def detect(img, r_expected, max_candidates=8, n_raw=25, z_thresh=2.5, radius_tol=0.25, peak_tol=(0.5, 1.8), refine=True):
    """Return up to max_candidates sub-pixel centres (best first) and their scores."""
    hp = img - ndi.gaussian_filter(img, 3 * r_expected)              # 1. background removal
    s = r_expected / np.sqrt(2)
    resp = -ndi.gaussian_laplace(hp, s) * s ** 2                      # 2. scale-normalised LoG
    peaks = (resp == ndi.maximum_filter(resp, size=int(r_expected))) & (resp > 0)
    edge = int(3 * r_expected); peaks[:edge] = peaks[-edge:] = False; peaks[:, :edge] = peaks[:, -edge:] = False
    med = np.median(resp); mad = np.median(np.abs(resp - med)) + 1e-12
    z = (resp - med) / (1.4826 * mad)
    ys, xs = np.nonzero(peaks & (z > z_thresh))
    o = np.argsort(-z[ys, xs])[:n_raw]; ys, xs = ys[o], xs[o]
    if not refine:
        return np.c_[xs, ys].astype(float)[:max_candidates], z[ys, xs][:max_candidates]
    out, sc = [], []
    H, W = img.shape; w = int(np.ceil(1.3 * r_expected))
    for x0, y0 in zip(xs, ys):                                      # 3. sphere + plane fit around each candidate
        sl = (slice(max(0, y0 - w), min(H, y0 + w + 1)), slice(max(0, x0 - w), min(W, x0 + w + 1)))
        patch = img[sl]; Yg, X = np.mgrid[sl[0], sl[1]].astype(float)
        q0 = [x0, y0, r_expected, EXPECTED_PEAK / r_expected, np.median(patch), 0, 0, 0, 0, 0]
        lo = [x0 - r_expected / 2, y0 - r_expected / 2, .5 * r_expected, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
        hi = [x0 + r_expected / 2, y0 + r_expected / 2, 1.5 * r_expected, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        fit = least_squares(lambda q: (_sphere_model(q, X, Yg, x0, y0) - patch).ravel(), q0, bounds=(lo, hi),
                            loss="soft_l1", f_scale=0.05)
        cx, cy, rho, a, *_ = fit.x
        peak = a * rho
        if abs(rho / r_expected - 1) > radius_tol:                    # 4. physics checks: size and peak height
            continue
        if not (peak_tol[0] * EXPECTED_PEAK <= peak <= peak_tol[1] * EXPECTED_PEAK):
            continue
        resid = np.sqrt(np.mean(fit.fun ** 2)) / peak
        if any(np.hypot(cx - u, cy - v) < r_expected / 2 for u, v in out):   # same ball found twice
            continue
        out.append([cx, cy]); sc.append(z[y0, x0] / (1 + 10 * resid))
    o = np.argsort(-np.array(sc))[:max_candidates]
    return np.array(out).reshape(-1, 2)[o], np.array(sc)[o]


def match(det, truth, tol_px=10.0):
    """One-to-one matching of detections to true centres. Returns recall, #false positives, center errors."""
    if len(det) == 0:
        return 0.0, 0, np.array([])
    D = np.linalg.norm(det[:, None] - truth[None], axis=2)
    r, c = linear_sum_assignment(D)
    ok = D[r, c] <= tol_px
    return ok.sum() / len(truth), len(det) - ok.sum(), D[r, c][ok]
