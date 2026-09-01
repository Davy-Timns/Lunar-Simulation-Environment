import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from Sensors.StarTracker import (
    load_triangle_region,
    verify_fourth_star,
    load_gaia_region
)

# =========================================================
# CONFIG
# =========================================================

TOP_K_STARS = 20
MAX_TRIANGLES = 45
TRIANGLE_K_NEIGHBORS = 2

MIN_TRIANGLE_ANGLE_DEG = 4.0
MAX_TRIANGLE_ANGLE_DEG = 80.0

INLIER_THRESHOLD_DEG = 0.02
FAST_VERIFY_STARS = 1000


# =========================================================
# CACHE
# =========================================================

# Helper: convert chord distance (||u-v|| on unit sphere) to angular separation (radians)
def chord_to_angle(d):
    return 2.0 * np.arcsin(np.clip(d / 2.0, -1.0, 1.0))


_catalog_tree_cache = {}




# =========================================================
# NORMALIZATION
# =========================================================

def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (n + 1e-9)


# =========================================================
# ANGLE MATRIX
# =========================================================

def angle_matrix(vecs):
    dot = vecs @ vecs.T
    np.clip(dot, -1.0, 1.0, out=dot)
    return np.arccos(dot)


# =========================================================
# ROBUST TRIANGLE SIGNATURE
# =========================================================

def triangle_signature(a, b, c):
    """
    Strong invariant triangle descriptor.

    OUTPUT (5D):
        [e1, e2, e3, e1/e2, e2/e3]
    """

    def ang(x, y):
        return np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))

    e1 = ang(a, b)
    e2 = ang(b, c)
    e3 = ang(c, a)

    edges = np.array([e1, e2, e3], dtype=np.float32)
    edges.sort()

    edges = np.clip(edges, 1e-6, None)

    s = edges.sum()
    norm = edges / (s + 1e-9)

    r1 = edges[0] / (edges[1] + 1e-9)
    r2 = edges[1] / (edges[2] + 1e-9)

    return np.array([
        norm[0],
        norm[1],
        norm[2],
        r1,
        r2
    ], dtype=np.float32)


# =========================================================
# INLIER SCORING
# =========================================================
def count_inliers(rot_ci, cam_vecs, cat_vecs, ang_thresh_deg=0.1):
    """
    Count inliers by mapping camera vectors into catalog frame and
    comparing angular separation to nearest catalog neighbor.

    Returns integer count of inliers.
    """
    cam_vecs = normalize(np.asarray(cam_vecs, dtype=np.float64))
    cat_vecs = normalize(np.asarray(cat_vecs, dtype=np.float64))

    if len(cam_vecs) == 0 or len(cat_vecs) == 0:
        return 0

    cam_inertial = rot_ci.apply(cam_vecs)

    print("[COUNT_INLIERS] sample cam_vecs (first 5):", cam_vecs[:5])
    print("cam_inertial (first 5):", cam_inertial[:5])
    # sample dot with matched nn
    print("cat_vecs (first 5):", cat_vecs[:5])


    # nearest neighbor search
    cat_tree = cKDTree(cat_vecs.astype(np.float32))
    dists, nn = cat_tree.query(cam_inertial, k=1)

    # convert nearest neighbor vectors to angles using dot product
    nn_vecs = cat_vecs[nn]
    dots = np.clip(np.sum(cam_inertial * nn_vecs, axis=1), -1.0, 1.0)
    ang = np.arccos(dots)

    ang_thresh = np.radians(ang_thresh_deg)
    inliers = ang < ang_thresh

    print("\n[INLIER DEBUG FIXED]")
    print("total matches :", int(np.sum(inliers)))
    if len(ang) > 0:
        print("mean error deg:", float(np.degrees(np.mean(ang))))
        print("max error deg :", float(np.degrees(np.max(ang))))

    return int(np.sum(inliers))



# =========================================================
# TRIANGLE BUILDING
# =========================================================

def build_triangles(vecs, flux):
    vecs = normalize(vecs)
    flux = np.asarray(flux)

    print("\n[BUILD_TRIANGLES]")
    print("Input stars:", len(vecs))
    print("Flux stats: min =", float(np.min(flux)), "max =", float(np.max(flux)))

    if len(vecs) < 3:
        print("Not enough stars for triangles.")
        return [], vecs

    if len(flux) > TOP_K_STARS:
        idx = np.argpartition(flux, -TOP_K_STARS)[-TOP_K_STARS:]
    else:
        idx = np.arange(len(flux))

    vecs = vecs[idx]
    n = len(vecs)

    print("Using TOP_K_STARS =", TOP_K_STARS, "=> selected", n, "stars for triangles")

    angles = angle_matrix(vecs)

    min_a = np.radians(MIN_TRIANGLE_ANGLE_DEG)
    max_a = np.radians(MAX_TRIANGLE_ANGLE_DEG)

    triangles = []
    angle_samples = []

    for i in range(n):
        for j in range(i + 1, n):
            a = angles[i, j]
            if a < min_a or a > max_a:
                continue

            for k in range(j + 1, n):
                b = angles[j, k]
                c = angles[k, i]

                if not (min_a <= b <= max_a):
                    continue
                if not (min_a <= c <= max_a):
                    continue

                angle_samples.extend([a, b, c])

                sig = triangle_signature(vecs[i], vecs[j], vecs[k])
                triangles.append((sig, (i, j, k)))

    if angle_samples:
        angle_samples = np.array(angle_samples)
        print("Triangle edge angle stats (deg):",
              "min =", np.degrees(np.min(angle_samples)),
              "max =", np.degrees(np.max(angle_samples)),
              "mean =", np.degrees(np.mean(angle_samples)))
    else:
        print("No valid triangle edges within angle constraints.")

    print("Total triangles built:", len(triangles), " (before MAX_TRIANGLES truncation)")

    return triangles[:MAX_TRIANGLES], vecs


# =========================================================
# ATTITUDE SOLVE
# =========================================================
def fast_kabsch(A, B):
    """
    Compute the orthogonal rotation that best maps A -> B using the
    classic Kabsch algorithm. A and B must be Nx3 arrays of corresponding
    points (unit vectors recommended). Returns a scipy Rotation object
    with attached quality diagnostics in rot._kabsch_quality.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    print("[FAST_KABSCH INPUT] A.shape, B.shape:", A.shape, B.shape)
    dots = np.sum(A * B, axis=1)
    print("[FAST_KABSCH INPUT] A·B (first 10):", dots[:10])

    print("[FAST_KABSCH INPUTS]")
    print("A (first 3):", A[:3])
    print("B (first 3):", B[:3])
    # check pairwise dot products to detect near-opposite correspondences
    dots = np.sum(A * B, axis=1)
    print("A·B dots (first 10):", dots[:10])
    print("A norms (first 10):", np.linalg.norm(A, axis=1)[:10])
    print("B norms (first 10):", np.linalg.norm(B, axis=1)[:10])



    # For unit direction vectors, do not subtract means
    H = A.T @ B

    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    Rmat = Vt.T @ U.T

    # ensure right-handed rotation
    if np.linalg.det(Rmat) < 0:
        Vt[-1, :] *= -1
        Rmat = Vt.T @ U.T

    # orthogonality diagnostics
    ortho_err = np.linalg.norm(Rmat.T @ Rmat - np.eye(3))
    detR = float(np.linalg.det(Rmat))

    rot = R.from_matrix(Rmat)

    # residuals in Euclidean chord space
    A_rot = rot.apply(A)
    residuals = np.linalg.norm(A_rot - B, axis=1)

    # attach diagnostics
    rot._kabsch_quality = {
        "ortho_err": float(ortho_err),
        "det": detR,
        "residual_mean": float(np.mean(residuals)),
        "residual_median": float(np.median(residuals)),
        "residual_max": float(np.max(residuals)),
        "singular_values": S.tolist() if S is not None else None
    }

    # debug print (kept for parity with previous behavior)
    print("\n[FAST_KABSCH DEBUG]")
    print("A shape:", A.shape, "B shape:", B.shape)
    print("det(R):", detR)
    print("R orthogonality error:", ortho_err)
    print("Residual stats: min =", float(np.min(residuals)),
          "max =", float(np.max(residuals)),
          "mean =", float(np.mean(residuals)))

    return rot




def solve_attitude(cam_vecs, cat_vecs):
    cam_vecs = normalize(cam_vecs)
    cat_vecs = normalize(cat_vecs)

    print("\n[SOLVE_ATTITUDE]")
    print("cam_vecs:", cam_vecs.shape, "cat_vecs:", cat_vecs.shape)

    return fast_kabsch(cam_vecs, cat_vecs)


# =========================================================
# REFINEMENT
# =========================================================
from scipy.optimize import least_squares
def refine_solution(rot_ci, cam_vecs, cat_vecs, cam_flux=None):
    """
    Robust nonlinear angular refinement of rotation rot_ci using matched pairs.
    - Uses least_squares with a soft L1 loss and optional flux weights.
    - Returns a Rotation object. Falls back to rot_ci if refinement fails or
      makes the solution worse.
    """
    cam_vecs = normalize(np.asarray(cam_vecs, dtype=np.float64))
    cat_vecs = normalize(np.asarray(cat_vecs, dtype=np.float64))

    if len(cam_vecs) < 6 or len(cat_vecs) < 6:
        return rot_ci

    # map catalog into camera frame using inverse of rot_ci
    cat_in_cam = rot_ci.inv().apply(cat_vecs)
    tree = cKDTree(cat_in_cam.astype(np.float32))

    # find nearest catalog neighbor for each camera vector
    dists, idx = tree.query(cam_vecs, k=1)
    nn = cat_vecs[idx]

    # compute angular distances via dot product
    dots = np.clip(np.sum(rot_ci.apply(cam_vecs) * nn, axis=1), -1.0, 1.0)
    ang = np.arccos(dots)

    # prefilter: keep pairs within a loose threshold for refinement
    prefilter_thresh = np.radians(1.0)  # 1 degree initial prefilter
    mask = ang < prefilter_thresh

    print("[REFINE PREFILTER] prefilter_thresh_deg:", np.degrees(prefilter_thresh))
    print("[REFINE PREFILTER] total pairs:", len(ang), "kept:", int(np.sum(mask)))
    if np.sum(mask) > 0:
        A = cam_vecs[mask]
        B = nn[mask]
        print("[REFINE PREFILTER] A (first 5):", A[:5])
        print("[REFINE PREFILTER] B (first 5):", B[:5])
        print("[REFINE PREFILTER] ang_kept_deg (first 10):", np.degrees(ang[mask])[:10])

    if np.sum(mask) < 8:
        # not enough good matches to refine
        return rot_ci

    A = cam_vecs[mask]
    B = nn[mask]

    # weights from flux if provided
    if cam_flux is not None:
        cam_flux = np.asarray(cam_flux, dtype=np.float64)
        weights = cam_flux[mask]
        # normalize weights to unit mean to keep scale stable
        if np.mean(weights) > 0:
            weights = weights / np.mean(weights)
        else:
            weights = None
    else:
        weights = None

    # initial rotation vector
    rvec0 = R.from_matrix(rot_ci.as_matrix()).as_rotvec()

    def residuals(rvec):
        Rm = R.from_rotvec(rvec)
        A_rot = Rm.apply(A)
        dots_local = np.clip(np.sum(A_rot * B, axis=1), -1.0, 1.0)
        ang_local = np.arccos(dots_local)
        if weights is not None:
            return ang_local * np.sqrt(weights)
        return ang_local

    # robust least squares with soft L1 loss
    try:
        res = least_squares(residuals, rvec0, method='trf', loss='soft_l1',
                            f_scale=1e-3, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
    except Exception as e:
        print("[REFINE] least_squares failed:", e)
        return rot_ci

    refined_rot = R.from_rotvec(res.x)

    # evaluate improvement: median angular residual before/after
    A_before = rot_ci.apply(A)
    dots_before = np.clip(np.sum(A_before * B, axis=1), -1.0, 1.0)
    med_before = float(np.median(np.degrees(np.arccos(dots_before))))

    A_after = refined_rot.apply(A)
    dots_after = np.clip(np.sum(A_after * B, axis=1), -1.0, 1.0)
    med_after = float(np.median(np.degrees(np.arccos(dots_after))))

    print("\n[REFINE DEBUG]")
    print("Refine inliers:", int(np.sum(mask)))
    print("Median error before (deg):", med_before)
    print("Median error after  (deg):", med_after)
    print("Refine success:", bool(res.success), "cost:", float(res.cost))

    # accept refined rotation only if it improves median angular residual
    if med_after <= med_before and med_after <= 0.25:
        refined_rot._refine_success = bool(res.success)
        refined_rot._refine_cost = float(res.cost)
        return refined_rot

    # otherwise keep original
    return rot_ci




# =========================================================
# MATCHING CORE
# =========================================================

def match_triangles(
    cam_vecs,
    cam_flux,
    region_center,
    radius_deg=10.0
):
    print("\n[MATCH_TRIANGLES]")
    print("Region center (RA,DEC):", region_center, "radius_deg:", radius_deg)
    print("Camera stars (input):", len(cam_vecs))

    # -----------------------------
    # Load catalog triangle region
    # -----------------------------
    cat_hashes, cat_ids, cat_vecs, hash_tree = load_triangle_region(
        region_center,
        radius_deg
    )

    # in main or a small test
    pos_sim, _ = load_gaia_region(region_center=region_center, radius_deg=15.0)
    hashes, ids, vecs_tri, _ = load_triangle_region(region_center=region_center, radius_deg=10.0)

    print("sim pos sample:", pos_sim[:5])
    print("tri vecs sample:", vecs_tri[:5])

    print("[CAT REGION DEBUG]")
    print("Loaded catalog from triangle region:")
    print("cat_hashes:", cat_hashes.shape,
          "cat_ids:", cat_ids.shape,
          "cat_vecs:", cat_vecs.shape)

    if len(cat_vecs) == 0:
        raise RuntimeError("Empty catalog")

    # sanity: cat_ids indices must be within cat_vecs
    if cat_ids.size > 0:
        max_id = np.max(cat_ids)
        min_id = np.min(cat_ids)
        print("cat_ids index range:", int(min_id), "to", int(max_id),
              "vs cat_vecs length:", len(cat_vecs))
        if max_id >= len(cat_vecs) or min_id < 0:
            raise RuntimeError("cat_ids out of range for cat_vecs")

    cat_vecs = normalize(cat_vecs)
    cat_hashes = np.asarray(cat_hashes, dtype=np.float32)

    print("\n[CATALOG HASH DEBUG]")
    print("cat_hashes shape:", cat_hashes.shape)

    if len(cat_hashes.shape) != 2:
        raise RuntimeError(f"Catalog hashes not 2D: {cat_hashes.shape}")
    if cat_hashes.shape[1] != 5:
        raise RuntimeError(
            f"Catalog triangle signature must be 5D, got {cat_hashes.shape[1]}"
        )

    norms = np.linalg.norm(cat_hashes, axis=1, keepdims=True)
    cat_hashes = cat_hashes / (norms + 1e-9)

    hash_tree = cKDTree(cat_hashes.astype(np.float32))
    cat_tree = cKDTree(cat_vecs.astype(np.float32))

    # -----------------------------
    # Build camera triangles
    # -----------------------------
    cam_triangles, cam_sel = build_triangles(cam_vecs, cam_flux)

    print("\n================ TRIANGLE DEBUG ================")
    print("Camera triangles:", len(cam_triangles))
    print("Selected stars   :", len(cam_sel))
    print("Raw camera stars :", len(cam_vecs))

    print("\n===== PYRAMID DEBUG =====")
    print("Catalog stars:", len(cat_vecs))
    print("Catalog triangles:", len(cat_hashes))
    print("Camera stars:", len(cam_vecs))
    print("Camera triangles:", len(cam_triangles))

    if len(cam_triangles) == 0:
        raise RuntimeError("No camera triangles built")

    # -----------------------------
    # Generate candidate rotations
    # -----------------------------
    rot_votes = []
    scores = []
    tri_for_rot = []

    sig_array = np.array([s for s, _ in cam_triangles])
    print("\n[CAMERA TRIANGLE SIG STATS]")
    print("mean:", np.mean(sig_array, axis=0))
    print("std :", np.std(sig_array, axis=0))

    for (sig, tri) in cam_triangles:
        sig_n = sig / (np.linalg.norm(sig) + 1e-9)
        dists, idx = hash_tree.query(sig_n, k=10)
        idx = np.atleast_1d(idx)

        print("\n[TRIANGLE MATCH DEBUG]")
        print("Camera triangle:", tri)
        print("Triangle hash distances:", dists)
        print("Best 3 catalog triangle indices:", idx[:3])

        cam_ids = list(tri)
        cam_tri = cam_sel[cam_ids]

        for i_cat in idx:
            cat_tri_ids = cat_ids[i_cat]
            cat_tri = cat_vecs[cat_tri_ids]

            # -----------------------------
            # 4th STAR VERIFICATION
            # -----------------------------
            # pick a 4th camera star (brightest not in triangle)
            fourth_idx = None
            for idx4 in range(len(cam_sel)):
                if idx4 not in cam_ids:
                    fourth_idx = idx4
                    break

            if fourth_idx is None:
                continue

            obs_fourth = cam_sel[fourth_idx]

            # verify against catalog
            cat_fourth_idx = verify_fourth_star(
                cam_tri,
                obs_fourth,
                cat_tri,
                cat_sel,
                tol_deg=0.15  # slightly loose tolerance
            )

            if cat_fourth_idx is None:
                # reject this triangle match
                continue

            # build full 4‑star correspondence
            cam_quad = np.vstack([cam_tri, obs_fourth])
            cat_quad = np.vstack([cat_tri, cat_sel[cat_fourth_idx]])

            # now solve rotation using 4 stars
            rot = fast_kabsch(cam_quad, cat_quad)

            # quick sanity: avoid flipped solutions with negative mean dot
            cam_mapped = rot.apply(cam_tri)
            dots = np.sum(cam_mapped * cat_tri, axis=1)
            mean_dot = float(np.mean(dots))
            print("[TRI ROT CHECK] mean dot:", mean_dot)
            if mean_dot < 0.0:
                print("[TRI ROT CHECK] REJECT rotation: mean dot < 0 (possible 180 flip)")
                continue

            # score rotation by inlier count over all camera stars
            cam_in_cat = rot.apply(cam_vecs)
            dists_all, _ = cat_tree.query(cam_in_cat, k=1)
            theta = chord_to_angle(dists_all)
            score = np.sum(theta < np.radians(0.1))

            print("[TRIANGLE ROT INLIERS] score:", int(score))

            rot_votes.append(rot)
            scores.append(score)
            tri_for_rot.append(tri)

    if len(scores) == 0:
        raise RuntimeError("No pyramid solution (no valid scores)")

    # -----------------------------
    # Pick best rotation
    # -----------------------------
    best_idx = int(np.argmax(scores))
    best_rot = rot_votes[best_idx]
    best_cam_tri = tri_for_rot[best_idx]

    print("\n[PYRAMID SCORE DEBUG]")
    print("All scores:", scores)
    print("Best score index:", best_idx, "value:", scores[best_idx])
    print("Best camera triangle:", best_cam_tri)

    # -----------------------------
    # Final inlier mask for best_rot
    # -----------------------------
    cam_in_cat = best_rot.apply(cam_vecs)
    tree = cKDTree(cat_vecs)

    dists, nn = tree.query(cam_in_cat, k=1)
    nn_vecs = cat_vecs[nn]
    dots = np.clip(np.sum(cam_in_cat * nn_vecs, axis=1), -1.0, 1.0)
    ang = np.arccos(dots)
    mask = ang < np.radians(0.1)

    theta = chord_to_angle(dists)
    print("[PYRAMID FINAL MASK] inliers:", int(np.sum(mask)),
          "min_err_deg:", float(np.degrees(np.min(theta))),
          "max_err_deg:", float(np.degrees(np.max(theta))))

    if np.sum(mask) < 6:
        raise RuntimeError("No stable solution")

    # refine rotation on inliers
    refined = fast_kabsch(cam_vecs[mask], cat_vecs[nn[mask]])
    score = int(np.sum(mask))

    print("\n===== PYRAMID RESULT =====")
    print("Inliers:", score)

    boresight_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    boresight_cat = refined.apply(boresight_cam)

    ra_rec = np.degrees(np.arctan2(boresight_cat[1], boresight_cat[0])) % 360.0
    dec_rec = np.degrees(np.arcsin(boresight_cat[2]))

    print("\n[PYRAMID ATTITUDE DEBUG]")
    print("Recovered boresight (cat frame):", boresight_cat)
    print(f"Recovered RA  (from match_triangles): {ra_rec:.6f}")
    print(f"Recovered DEC (from match_triangles): {dec_rec:.6f}")

    # refine rotation on inliers
    refined = fast_kabsch(cam_vecs[mask], cat_vecs[nn[mask]])
    score = int(np.sum(mask))

    print("\n===== PYRAMID RESULT =====")
    print("Inliers:", score)

    # -----------------------------
    # DEBUG: is boresight biased toward winning triangle?
    # -----------------------------
    boresight_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    boresight_cat = refined.apply(boresight_cam)

    # winning triangle in catalog frame
    cat_tri_ids_best = cat_ids[best_idx]          # indices into cat_vecs
    cat_tri_best = cat_vecs[cat_tri_ids_best]     # 3×3

    tri_centroid = np.mean(cat_tri_best, axis=0)
    tri_centroid /= np.linalg.norm(tri_centroid) + 1e-9

    dot_bt = np.clip(np.dot(boresight_cat, tri_centroid), -1.0, 1.0)
    ang_bt_deg = np.degrees(np.arccos(dot_bt))

    print("\n[TRIANGLE BIAS DEBUG]")
    print("Boresight (cat frame):", boresight_cat)
    print("Winning tri centroid :", tri_centroid)
    print("Angle boresight ↔ tri centroid (deg):", ang_bt_deg)


    return refined, score, cat_vecs, mask, best_cam_tri, cam_sel, cat_tri_best

def match_triangles_direct(cam_vecs, cam_flux, cat_vecs, cat_flux,
                           region_center=None, radius_deg=None):
    """
    Triangle-based attitude solve using the SAME catalog the simulator used,
    with region gating and a boresight-prior rejection.
    """
    cam_vecs = normalize(cam_vecs)
    cat_vecs = normalize(cat_vecs)

    # ----------------------------------------
    # Optional region gating on catalog
    # ----------------------------------------
    if region_center is not None and radius_deg is not None:
        ra, dec = np.radians(region_center[0]), np.radians(region_center[1])
        center_vec = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)
        ], dtype=np.float64)
        rad = np.radians(radius_deg)

        dots = np.clip(cat_vecs @ center_vec, -1.0, 1.0)
        ang = np.arccos(dots)
        mask = ang <= rad

        cat_vecs = cat_vecs[mask]
        cat_flux = cat_flux[mask]

        print(f"[CAT REGION FILTER] kept {len(cat_vecs)} stars out of {len(mask)}")

    # ----------------------------------------
    # Build triangles
    # ----------------------------------------
    cam_triangles, cam_sel = build_triangles(cam_vecs, cam_flux)
    cat_triangles, cat_sel = build_triangles(cat_vecs, cat_flux)

    if len(cam_triangles) == 0 or len(cat_triangles) == 0:
        raise RuntimeError("No triangles available for matching")

    # KD-tree on catalog triangle signatures
    cat_sigs = np.array([sig for sig, _ in cat_triangles], dtype=np.float32)
    cat_sigs /= (np.linalg.norm(cat_sigs, axis=1, keepdims=True) + 1e-9)
    hash_tree = cKDTree(cat_sigs)

    # KD-tree on catalog stars for scoring
    cat_tree = cKDTree(cat_vecs.astype(np.float32))

    # true boresight from region_center (for prior)
    if region_center is not None:
        ra0, dec0 = np.radians(region_center[0]), np.radians(region_center[1])
        true_boresight = np.array([
            np.cos(dec0) * np.cos(ra0),
            np.cos(dec0) * np.sin(ra0),
            np.sin(dec0)
        ], dtype=np.float64)
    else:
        true_boresight = None

    rot_votes = []
    scores = []
    tri_for_rot = []

    # ----------------------------------------
    # First triangle → candidate rotations
    # ----------------------------------------
    for (sig, tri) in cam_triangles:
        sig_n = sig / (np.linalg.norm(sig) + 1e-9)
        dists, idx = hash_tree.query(sig_n, k=10)
        idx = np.atleast_1d(idx)

        cam_ids = list(tri)
        cam_tri = cam_sel[cam_ids]

        for i_cat in idx:
            cat_ids = list(cat_triangles[i_cat][1])
            cat_tri = cat_sel[cat_ids]

            # ----------------------------------------
            # BRIGHTNESS ORDERING CHECK
            # ----------------------------------------
            # Sort camera triangle vertices by brightness (flux)
            cam_flux_tri = cam_flux[cam_ids]
            cam_order = np.argsort(-cam_flux_tri)  # brightest first
            cam_tri_sorted = cam_tri[cam_order]

            # Sort catalog triangle vertices by brightness
            cat_flux_tri = cat_flux[cat_ids]
            cat_order = np.argsort(-cat_flux_tri)
            cat_tri_sorted = cat_tri[cat_order]

            # If brightness ordering differs, reject this match
            if not np.array_equal(cam_order, cat_order):
                continue

            # Use brightness‑sorted triangles for Kabsch
            cam_tri = cam_tri_sorted
            cat_tri = cat_tri_sorted

            # initial rotation from this triangle
            rot = fast_kabsch(cam_tri, cat_tri)

            #rot = R.from_matrix(rot.as_matrix().T)

            # ----------------------------------------
            # BORESIGHT PRIOR REJECTION
            # ----------------------------------------
            if true_boresight is not None:
                b_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                b_est = rot.apply(b_cam)
                dot_b = np.clip(np.dot(b_est, true_boresight), -1.0, 1.0)
                err_b_deg = np.degrees(np.arccos(dot_b))

                # reject candidates whose boresight is too far from prior
                if err_b_deg > 2.0:   # you can tune this threshold
                    continue

            # ----------------------------------------
            # Score rotation over all camera stars
            # ----------------------------------------
            cam_in_cat = rot.apply(cam_vecs)
            dists_all, nn = cat_tree.query(cam_in_cat, k=1)
            theta = chord_to_angle(dists_all)
            score = np.sum(theta < np.radians(0.1))

            rot_votes.append(rot)
            scores.append(score)
            tri_for_rot.append(tri)

    if not scores:
        raise RuntimeError("No pyramid solution (no valid scores after boresight prior)")

    # ----------------------------------------
    # Pick best rotation
    # ----------------------------------------
    best_idx = int(np.argmax(scores))
    best_rot = rot_votes[best_idx]
    best_cam_tri = tri_for_rot[best_idx]

    # final inlier mask
    cam_in_cat = best_rot.apply(cam_vecs)
    dists, nn = cat_tree.query(cam_in_cat, k=1)
    nn_vecs = cat_vecs[nn]
    dots = np.clip(np.sum(cam_in_cat * nn_vecs, axis=1), -1.0, 1.0)
    ang = np.arccos(dots)
    mask = ang < np.radians(0.1)

    refined = fast_kabsch(cam_vecs[mask], cat_vecs[nn[mask]])
    score = int(np.sum(mask))

    return refined, score, cat_vecs, mask, best_cam_tri, cam_sel

# =========================================================
# MAIN PIPELINE
# =========================================================
def process_star_tracker_output(
    data,
    region_center,
    radius_deg=10.0,
    return_plot=False
):
    print("STAR TRACKER PROCESSING INPUT")
    print("RegionCenter:", region_center)
    print("RadiusDeg:", radius_deg)
    print("Data keys:", list(data.keys()))

    cam_vecs_full = np.asarray(data["camera_vectors"], dtype=np.float32)
    cam_flux_full = np.asarray(data["flux"], dtype=np.float32)

    print("\n[PROCESS_PIPELINE]")
    print("Region center (RA,DEC):", region_center)
    print("Initial camera stars:", len(cam_vecs_full))
    print("Initial flux stats: min =", float(np.min(cam_flux_full)),
          "max =", float(np.max(cam_flux_full)))

    # ----------------------------------------------------
    # Camera filtering
    # ----------------------------------------------------
    cam_vecs = cam_vecs_full.copy()
    cam_flux = cam_flux_full.copy()

    top_k = 500
    if len(cam_flux) > top_k:
        idx = np.argsort(cam_flux)[-top_k:]
        cam_vecs = cam_vecs[idx]
        cam_flux = cam_flux[idx]

    print("\n===== CAMERA FILTERING =====")
    print("Reduced camera stars to:", len(cam_vecs))
    print("Flux range:", float(np.min(cam_flux)), "to", float(np.max(cam_flux)))

    # ----------------------------------------------------
    # Use EXACT same catalog as simulator
    # ----------------------------------------------------
    cat_all = np.asarray(data["catalog_vectors"], dtype=np.float32)
    cat_flux = np.asarray(data["catalog_flux"], dtype=np.float32)

    # Triangle match + refinement using shared catalog
    rot, score, cat_all, inlier_mask, best_cam_tri, cam_sel = match_triangles_direct(
        cam_vecs,
        cam_flux,
        cat_all,
        cat_flux,
        region_center=region_center,
        radius_deg=radius_deg
    )

    rot = refine_solution(rot, cam_vecs, cat_all)



    # ----------------------------------------------------
    # Ground-truth direction from region_center (boresight only)
    # ----------------------------------------------------
    ra_true_deg, dec_true_deg = region_center
    ra_true = np.radians(ra_true_deg)
    dec_true = np.radians(dec_true_deg)

    true_dir = np.array([
        np.cos(dec_true) * np.cos(ra_true),
        np.cos(dec_true) * np.sin(ra_true),
        np.sin(dec_true)
    ])

    print("\n================ ATTITUDE VALIDATION ================")

    boresight = rot.apply([0.0, 0.0, 1.0])
    dot = np.clip(np.dot(true_dir, boresight), -1.0, 1.0)
    ang_err_deg = np.degrees(np.arccos(dot))

    print("Recovered boresight:", boresight)
    print("True direction     :", true_dir)
    print("Dot product        :", dot)
    print("Angular error (deg):", ang_err_deg)

    # ----------------------------------------------------
    # Inlier score using final rotation
    # ----------------------------------------------------
    final_score = count_inliers(rot, cam_vecs, cat_all)

    print("\n================ INLIER ANALYSIS ================")

    cam_inertial = rot.apply(cam_vecs)
    cat_tree = cKDTree(cat_all.astype(np.float32))
    dists, nn = cat_tree.query(cam_inertial, k=1)
    theta = chord_to_angle(dists)

    print("Total camera stars:", len(cam_vecs))
    print("Median error (deg):", np.degrees(np.median(theta)))
    print("90th percentile   :", np.degrees(np.percentile(theta, 90)))
    print("Max error         :", np.degrees(np.max(theta)))
    print("Inliers @0.05°:", int(np.sum(theta < np.radians(0.05))))
    print("Inliers @0.1° :", int(np.sum(theta < np.radians(0.1))))
    print("Inliers @0.5° :", int(np.sum(theta < np.radians(0.5))))

    # ----------------------------------------------------
    # Final boresight RA/DEC from recovered rotation
    # ----------------------------------------------------
    boresight = rot.apply([0.0, 0.0, 1.0])
    ra = np.degrees(np.arctan2(boresight[1], boresight[0])) % 360.0
    dec = np.degrees(np.arcsin(boresight[2]))

    print("\n===== ATTITUDE DEBUG =====")
    print("Boresight vector:", boresight)
    print("True direction:", true_dir)
    print("Boresight dot product:", np.clip(np.dot(true_dir, boresight), -1.0, 1.0))
    print("Boresight angular error (deg):", ang_err_deg)
    print("Recovered RA,DEC:", ra, dec)
    print("Input region_center (RA,DEC):", region_center)

    # ----------------------------------------------------
    # TRUE attitude from simulation (if present)
    # ----------------------------------------------------
    true_rot = None
    if "true_quat" in data:
        q = np.asarray(data["true_quat"], dtype=np.float64)
        true_rot = R.from_quat(q)
        print("[TRUE ATTITUDE] Using true_quat from data.")
    elif "true_rot_matrix" in data:
        Rm_true = np.asarray(data["true_rot_matrix"], dtype=np.float64).reshape(3, 3)
        true_rot = R.from_matrix(Rm_true)
        print("[TRUE ATTITUDE] Using true_rot_matrix from data.")
    else:
        print("[TRUE ATTITUDE] No true attitude in data.")

    if return_plot:
        fig, axes = plt.subplots(1, 2 if true_rot is not None else 1, figsize=(16, 8))
        if true_rot is not None:
            ax_est, ax_true = axes
        else:
            ax_est = axes

        cat_ra = np.degrees(np.arctan2(cat_all[:, 1], cat_all[:, 0])) % 360.0
        cat_dec = np.degrees(np.arcsin(cat_all[:, 2]))

        cam_est = rot.apply(cam_vecs)
        cam_est_ra = np.degrees(np.arctan2(cam_est[:, 1], cam_est[:, 0])) % 360.0
        cam_est_dec = np.degrees(np.arcsin(cam_est[:, 2]))

        ax_est.scatter(cat_ra, cat_dec, s=3, color="gray", alpha=0.4, label="Catalog stars")
        ax_est.scatter(cam_est_ra, cam_est_dec, s=8, color="cyan", alpha=0.8, label="Camera (est)")
        ax_est.scatter(
            cam_est_ra[inlier_mask],
            cam_est_dec[inlier_mask],
            s=20,
            color="yellow",
            label="Matched inliers"
        )

        i, j, k = best_cam_tri
        cam_sel_est = rot.apply(cam_sel)
        tri_ra = np.degrees(np.arctan2(cam_sel_est[[i, j, k, i], 1],
                                       cam_sel_est[[i, j, k, i], 0])) % 360.0
        tri_dec = np.degrees(np.arcsin(cam_sel_est[[i, j, k, i], 2]))
        ax_est.plot(tri_ra, tri_dec, color="red", linewidth=1.5, label="Winning triangle")

        ax_est.set_title("Estimated Sky (Recovered Attitude)")
        ax_est.set_xlabel("RA (deg)")
        ax_est.set_ylabel("DEC (deg)")
        ax_est.legend()

        if true_rot is not None:
            cam_true = true_rot.apply(cam_vecs)
            cam_true_ra = np.degrees(np.arctan2(cam_true[:, 1], cam_true[:, 0])) % 360.0
            cam_true_dec = np.degrees(np.arcsin(cam_true[:, 2]))

            print("\n[ATTITUDE COMPARISON]")
            print("Recovered boresight (rot):", rot.apply([0, 0, 1]))
            print("True boresight (true_rot):", true_rot.apply([0, 0, 1]))

            rel = true_rot * rot.inv()
            rvec_rel = rel.as_rotvec()
            angle_rel = np.linalg.norm(rvec_rel)
            axis_rel = rvec_rel / (np.linalg.norm(axis_rel := np.linalg.norm(rvec_rel)) + 1e-9)

            print("Relative rot angle (deg):", np.degrees(angle_rel))
            print("Relative rot axis:", axis_rel)

            ax_true.scatter(cat_ra, cat_dec, s=3, color="gray", alpha=0.4, label="Catalog stars")
            ax_true.scatter(cam_true_ra, cam_true_dec, s=8, color="lime", alpha=0.8, label="Camera (true)")

            ax_true.set_title("True Sky (Ground Truth Attitude)")
            ax_true.set_xlabel("RA (deg)")
            ax_true.set_ylabel("DEC (deg)")
            ax_true.legend()

            # ---------------------------------------------
            # Draw the winning triangle on the TRUE sky
            # ---------------------------------------------
            i, j, k = best_cam_tri  # same triangle indices

            # Transform the triangle vertices using TRUE rotation
            cam_sel_true = true_rot.apply(cam_sel)

            tri_true_ra = np.degrees(np.arctan2(
                cam_sel_true[[i, j, k, i], 1],
                cam_sel_true[[i, j, k, i], 0]
            )) % 360.0

            tri_true_dec = np.degrees(np.arcsin(
                cam_sel_true[[i, j, k, i], 2]
            ))

            ax_true.plot(
                tri_true_ra,
                tri_true_dec,
                color="red",
                linewidth=1.5,
                label="Winning triangle (true)"
            )

        plt.tight_layout()
        plt.show()



    if true_rot is not None:
        rel = true_rot * rot.inv()
        angle_rel = np.linalg.norm(rel.as_rotvec())
        print("Relative rot angle (deg):", np.degrees(angle_rel))

    return {
        "rotation_camera_to_inertial": rot,
        "quaternion": rot.as_quat(),
        "triangle_score": int(score),
        "inlier_score": int(final_score),
        "matched_stars": len(cam_vecs),
        "boresight_vector": boresight,
        "ra_deg": ra,
        "dec_deg": dec,
    }
