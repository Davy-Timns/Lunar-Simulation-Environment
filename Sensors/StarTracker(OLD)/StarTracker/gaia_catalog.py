import numpy as np
from pathlib import Path
from astroquery.gaia import Gaia
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CACHE_DIR = BASE_DIR / "Starcache"
CACHE_DIR.mkdir(exist_ok=True)

TRI_CACHE_DIR = BASE_DIR / "TriStarCache"
TRI_CACHE_DIR.mkdir(exist_ok=True)

# =========================================================
# CONSTANTS
# =========================================================

TILE_SIZE = 2.0
TILE_PADDING = 0.25
SIGNATURE_VERSION = 2

# =========================================================
# MEMORY CACHE
# =========================================================

_MEMORY_TILE_CACHE = {}

# =========================================================
# HELPERS
# =========================================================

def tile_key(ra_tile, dec_tile):
    return int(ra_tile), int(dec_tile)


def tile_filename(ra_tile, dec_tile):
    return CACHE_DIR / f"tile_RA{ra_tile}_DEC{dec_tile}.npz"


def tri_cache_filename(ra_tile, dec_tile):
    return TRI_CACHE_DIR / f"tri_RA{ra_tile}_DEC{dec_tile}.npz"


def sky_tile(ra_deg, dec_deg):
    ra_deg = float(ra_deg) % 360.0
    dec_deg = float(dec_deg)

    return (
        int(np.floor(ra_deg / TILE_SIZE)),
        int(np.floor((dec_deg + 90.0) / TILE_SIZE))
    )

# =========================================================
# TRIANGLE SIGNATURE (5D robust descriptor)
# =========================================================

def angular_distance(a, b):
    return np.arccos(
        np.clip(np.dot(a, b), -1.0, 1.0)
    )
def verify_fourth_star(
    obs_tri,
    obs_fourth,
    cat_tri,
    cat_vecs,
    tol_deg=0.1
):

    tol = np.radians(tol_deg)

    obs_d = np.array([
        angular_distance(obs_fourth, obs_tri[0]),
        angular_distance(obs_fourth, obs_tri[1]),
        angular_distance(obs_fourth, obs_tri[2]),
    ])

    matches = []

    best_err = np.inf

    for idx, star in enumerate(cat_vecs):

        cat_d = np.array([
            angular_distance(star, cat_tri[0]),
            angular_distance(star, cat_tri[1]),
            angular_distance(star, cat_tri[2]),
        ])

        err = np.max(np.abs(obs_d - cat_d))

        best_err = min(best_err, err)

        if err < tol:
            matches.append(idx)

            if len(matches) > 1:
                return None

    if len(matches) == 1:
        return matches[0]

    return None

def triangle_signature(a, b, c):
    def ang(x, y):
        return np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))

    d_ab = ang(a, b)
    d_bc = ang(b, c)
    d_ca = ang(c, a)

    edges = np.array([d_ab, d_bc, d_ca], dtype=np.float32)
    edges.sort()

    s = edges.sum() + 1e-9
    norm_edges = edges / s

    ratio1 = edges[0] / (edges[1] + 1e-9)
    ratio2 = edges[1] / (edges[2] + 1e-9)

    return np.array([
        norm_edges[0],
        norm_edges[1],
        norm_edges[2],
        ratio1,
        ratio2
    ], dtype=np.float32)

# =========================================================
# TILE COVERAGE
# =========================================================

def required_tiles(region_center, radius_deg):
    ra0, dec0 = region_center

    ra_min = (ra0 - radius_deg) % 360.0
    ra_max = (ra0 + radius_deg) % 360.0

    dec_min = max(dec0 - radius_deg, -90.0)
    dec_max = min(dec0 + radius_deg, 90.0)

    if ra_min < ra_max:
        ra_tiles = range(
            int(np.floor(ra_min / TILE_SIZE)),
            int(np.floor(ra_max / TILE_SIZE)) + 1
        )
    else:
        ra_tiles = list(range(int(np.floor(ra_min / TILE_SIZE)), int(360 / TILE_SIZE)))
        ra_tiles += list(range(0, int(np.floor(ra_max / TILE_SIZE)) + 1))

    dec_tiles = range(
        int(np.floor((dec_min + 90.0) / TILE_SIZE)),
        int(np.floor((dec_max + 90.0) / TILE_SIZE)) + 1
    )

    return [(r, d) for r in ra_tiles for d in dec_tiles]

# =========================================================
# TILE QUERY
# =========================================================

def query_tile(ra_tile, dec_tile):

    key = tile_key(ra_tile, dec_tile)

    if key in _MEMORY_TILE_CACHE:
        return _MEMORY_TILE_CACHE[key]

    cache_file = tile_filename(*key)

    if cache_file.exists():
        data = np.load(cache_file)
        result = (data["star_positions"], data["fluxes"])
        _MEMORY_TILE_CACHE[key] = result
        return result

    ra_min = ra_tile * TILE_SIZE - TILE_PADDING
    ra_max = ra_tile * TILE_SIZE + TILE_SIZE + TILE_PADDING

    dec_min = (dec_tile * TILE_SIZE - 90.0) - TILE_PADDING
    dec_max = (dec_tile * TILE_SIZE - 90.0) + TILE_SIZE + TILE_PADDING

    center_ra = (ra_min + ra_max) * 0.5
    center_dec = (dec_min + dec_max) * 0.5
    radius = max(ra_max - ra_min, dec_max - dec_min) * 0.75

    query = f"""
    SELECT ra, dec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {center_ra}, {center_dec}, {radius})
    ) = 1
    AND phot_g_mean_mag < 21
    """

    job = Gaia.launch_job_async(query)
    stars = job.get_results()

    if len(stars) == 0:
        empty = (np.empty((0, 3)), np.empty((0,)))
        np.savez(cache_file, star_positions=empty[0], fluxes=empty[1])
        _MEMORY_TILE_CACHE[key] = empty
        return empty

    positions = []
    fluxes = []

    for r in stars:
        ra, dec, mag = r["ra"], r["dec"], r["phot_g_mean_mag"]

        if np.isnan(ra) or np.isnan(dec) or np.isnan(mag):
            continue

        # ✅ FIX: PURE DIRECTION ONLY (NO DISTANCE)
        x = np.cos(np.radians(dec)) * np.cos(np.radians(ra))
        y = np.cos(np.radians(dec)) * np.sin(np.radians(ra))
        z = np.sin(np.radians(dec))

        positions.append(np.array([x, y, z], dtype=np.float32))
        fluxes.append(10 ** (-0.4 * mag))

    positions = np.asarray(positions, dtype=np.float32)
    fluxes = np.asarray(fluxes, dtype=np.float32)

    # normalize immediately (CRITICAL FIX)
    positions = positions / (np.linalg.norm(positions, axis=1, keepdims=True) + 1e-9)

    np.savez(cache_file, star_positions=positions, fluxes=fluxes)
    _MEMORY_TILE_CACHE[key] = (positions, fluxes)

    build_triangle_cache(ra_tile, dec_tile)

    return positions, fluxes

# =========================================================
# TRIANGLE CACHE BUILD
# =========================================================

def build_triangle_cache(ra_tile, dec_tile, top_k=8, neighbor_k=12):
    """
    Build triangle cache for a tile but limit triangles by nearest neighbors.

    - top_k: number of brightest stars to consider (kept from original API)
    - neighbor_k: for each star, only consider its nearest neighbor_k neighbors
                  when forming triangles (neighbor_k should be >= 2)
    """
    f = tri_cache_filename(ra_tile, dec_tile)
    if f.exists():
        return

    pos, flux = query_tile(ra_tile, dec_tile)
    if len(pos) < 3:
        return

    vecs = pos.astype(np.float32)

    # normalize (already done in query_tile, but keep safe)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    flux = np.asarray(flux, dtype=np.float32)

    # keep only the brightest stars (reduce N)
    idx_bright = np.argsort(flux)[::-1][:50]
    vecs = vecs[idx_bright]
    flux = flux[idx_bright]

    n = len(vecs)
    if n < 3:
        return

    # Build KD-tree for neighbor queries
    tree = cKDTree(vecs.astype(np.float32))

    hashes = []
    ids = []

    # For each star, get neighbor_k nearest neighbors (including itself)
    # and form triangles only among that small neighborhood.
    # This reduces combinations drastically in dense fields.
    neighbor_k = min(neighbor_k, n - 1)
    dists, neighbors = tree.query(vecs, k=neighbor_k + 1)  # +1 includes self

    for i in range(n):
        neigh = neighbors[i]
        # remove self (first entry)
        neigh = neigh[neigh != i]
        # if fewer than 2 neighbors, skip
        if len(neigh) < 2:
            continue

        # form triangles (i, j, k) with j < k to avoid duplicates
        for a_idx in range(len(neigh)):
            for b_idx in range(a_idx + 1, len(neigh)):
                j = neigh[a_idx]
                k = neigh[b_idx]
                # ensure deterministic ordering of indices
                tri_ids = [i, j, k]
                tri_ids_sorted = sorted(tri_ids)
                # compute signature
                sig = triangle_signature(vecs[tri_ids_sorted[0]],
                                         vecs[tri_ids_sorted[1]],
                                         vecs[tri_ids_sorted[2]])
                hashes.append(sig)
                ids.append(tri_ids_sorted)

    if len(hashes) == 0:
        return

    hashes = np.asarray(hashes, dtype=np.float32)
    ids = np.asarray(ids, dtype=np.int32)

    # optional: deduplicate identical signatures (cheap hash)
    # create a small hash key from rounded signature to remove duplicates
    sig_keys = np.round(hashes, 6)
    _, unique_idx = np.unique(sig_keys, axis=0, return_index=True)
    hashes = hashes[unique_idx]
    ids = ids[unique_idx]

    np.savez(
        f,
        hashes=hashes,
        triangle_ids=ids,
        star_vectors=vecs,
        fluxes=flux,
        sig_version=SIGNATURE_VERSION
    )


# =========================================================
# REGION LOADER
# =========================================================

def load_gaia_region(region_center=(10, 41), radius_deg=3):

    tiles = required_tiles(region_center, radius_deg)

    def load(t):
        return query_tile(*t)

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(load, tiles))

    pos, flux = [], []

    for p, f in results:
        if len(p) > 0:
            pos.append(p)
            flux.append(f)

    if not pos:
        return np.empty((0, 3)), np.empty((0,))

    pos = np.concatenate(pos)
    flux = np.concatenate(flux)

    # 🔴 NEW: discard the lowest 95% flux stars to drastically reduce catalog size
    if len(flux) > 0:
        # keep only the top 5% brightest stars by flux
        thresh = np.percentile(flux, 95.0)
        keep_mask = flux >= thresh
        pos = pos[keep_mask]
        flux = flux[keep_mask]

    # 🔴 FIX: REMOVE DUPLICATES FROM TILE OVERLAP
    # (prevents triangle corruption)
    if len(pos) > 0:
        _, unique_idx = np.unique(pos.round(6), axis=0, return_index=True)
        pos = pos[unique_idx]
        flux = flux[unique_idx]


    return pos, flux


# =========================================================
# TRIANGLE REGION LOADER
# =========================================================

def load_triangle_region(region_center=(10, 41), radius_deg=3):

    ra0, dec0 = region_center
    rad = np.radians(radius_deg)

    # Load tiles as before
    tiles = required_tiles(region_center, radius_deg)

    hashes_all = []
    ids_all = []
    vecs_all = []

    offset = 0

    for r, d in tiles:
        file = tri_cache_filename(r, d)

        if not file.exists():
            build_triangle_cache(r, d)

        if not file.exists():
            continue

        data = np.load(file)

        if "sig_version" in data and int(data["sig_version"]) != SIGNATURE_VERSION:
            continue

        hashes = np.asarray(data["hashes"], dtype=np.float32).reshape(-1, 5)
        tri_ids = np.asarray(data["triangle_ids"], dtype=np.int32).reshape(-1, 3)
        vecs = np.asarray(data["star_vectors"], dtype=np.float32).reshape(-1, 3)


        if len(hashes) == 0:
            continue

        # =========================================================
        # 🔥 FILTER TRIANGLE STARS TO MATCH GAIA REGION EXACTLY
        # =========================================================
        # Convert region center to vector
        center_vec = np.array([
            np.cos(np.radians(dec0)) * np.cos(np.radians(ra0)),
            np.cos(np.radians(dec0)) * np.sin(np.radians(ra0)),
            np.sin(np.radians(dec0))
        ], dtype=np.float32)

        # Angular distance filter
        dots = np.clip(vecs @ center_vec, -1.0, 1.0)
        ang = np.arccos(dots)

        mask = ang <= rad
        if not np.any(mask):
            continue


        # Filter vectors
        vecs = vecs[mask]

        # Filter triangle IDs and hashes accordingly
        # (only keep triangles whose all 3 stars survive)
        keep = []
        for i, tri in enumerate(tri_ids):
            if mask[tri[0]] and mask[tri[1]] and mask[tri[2]]:
                keep.append(i)

        if not keep:
            continue

        hashes = hashes[keep]
        tri_ids = tri_ids[keep]

        # Reindex triangle IDs after filtering
        remap = np.cumsum(mask) - 1
        tri_ids = remap[tri_ids]


        print("[TRI REGION REMAP] original vec count:", mask.shape[0])
        print("[TRI REGION REMAP] kept vec count:", int(np.sum(mask)))
        print("[TRI REGION REMAP] remap sample (first 10):", remap[:10])
        print("[TRI REGION REMAP] tri_ids sample (first 10):", tri_ids[:10])
        # sanity: ensure tri_ids are within [0, kept_count-1]
        if tri_ids.size > 0:
            if np.max(tri_ids) >= np.sum(mask) or np.min(tri_ids) < 0:
                print("[TRI REGION REMAP] ERROR: tri_ids out of range after remap")


        print("[TRI REGION FILTER]")
        print("original vecs count:", len(mask))
        print("kept triangles:", len(keep))
        print("remap sample (first 10):", remap[:10])
        print("tri_ids sample (first 10):", tri_ids[:10])


        # Append
        ids_all.append(tri_ids + offset)
        hashes_all.append(hashes)
        vecs_all.append(vecs)

        offset += vecs.shape[0]

    if not hashes_all:
        return (
            np.empty((0, 5), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            None
        )

    hashes = np.concatenate(hashes_all, axis=0)
    ids = np.concatenate(ids_all, axis=0)
    vecs = np.concatenate(vecs_all, axis=0)

    tree = cKDTree(hashes)

    return hashes, ids, vecs, tree
