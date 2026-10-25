"""Cache a small terrain OBJ for Vizard without loading the source into RAM.

Large terrain meshes are clustered on their two widest axes. Faces are remapped
and deduplicated, rather than sampled (sampling would leave holes). This is a
lossy display mesh only: collision geometry and navigation maps are unchanged.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

class TerrainDisplay:
    def __init__(self,
                 path,
                 minimum,
                 maximum,
                 source_vertices,
                 source_faces,
                 display_vertices,
                 display_triangles,
                 optimized,
                 ):
        self.path = path
        self.minimum = minimum
        self.maximum = maximum
        self.source_vertices = source_vertices
        self.source_faces = source_faces
        self.display_vertices = display_vertices
        self.display_triangles = display_triangles
        self.optimized = optimized


def _vertices(source, batch_size=100000):
    """Read only positions, in bounded batches; ignore normals/UVs/materials."""
    rows = []
    with source.open('rb', buffering=8 * 1024 * 1024) as stream:
        for line in stream:
            parts = line.split()
            if parts and parts[0] == b'v':
                if len(parts) < 4:
                    raise ValueError(f'Malformed OBJ vertex in {source}')
                rows.append(b' '.join(parts[1:4]))
                if len(rows) == batch_size:
                    yield _parse_vertices(rows, source)
                    rows.clear()
    if rows:
        yield _parse_vertices(rows, source)


def _parse_vertices(rows, source):
    xyz = np.fromstring(b' '.join(rows).decode('ascii'), sep=' ')
    if xyz.size != 3 * len(rows) or not np.isfinite(xyz).all():
        raise ValueError(f'Invalid/non-finite OBJ vertices in {source}')
    return xyz.reshape(-1, 3)


def _bounds(source):
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    count = 0
    for xyz in _vertices(source):
        lower = np.minimum(lower, xyz.min(axis=0))
        upper = np.maximum(upper, xyz.max(axis=0))
        count += len(xyz)
    if count < 3:
        raise ValueError(f'OBJ has fewer than three vertices: {source}')
    return lower, upper, count


def _faces(source, vertex_count, batch_size=100000):
    """Triangulate OBJ polygons; negative indices are relative to that line."""
    rows, seen_vertices, polygons = [], 0, 0
    with source.open('rb', buffering=8 * 1024 * 1024) as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == b'v':
                seen_vertices += 1
            elif parts[0] == b'f':
                # OBJ permits a trailing comment and v, v/vt, v//vn, v/vt/vn.
                tokens = list(itertools.takewhile(lambda p: not p.startswith(b'#'), parts[1:]))
                indices = [int(p.split(b'/', 1)[0]) for p in tokens]
                indices = [i - 1 if i > 0 else seen_vertices + i if i < 0 else -1
                           for i in indices]
                if len(indices) < 3 or min(indices) < 0 or max(indices) >= vertex_count:
                    raise ValueError(f'Invalid OBJ face indices in {source}')
                polygons += 1
                rows.extend((indices[0], indices[i], indices[i + 1])
                            for i in range(1, len(indices) - 1))
                if len(rows) >= batch_size:
                    yield np.asarray(rows, dtype=np.int64), polygons
                    rows.clear()
    if rows:
        yield np.asarray(rows, dtype=np.int64), polygons


def _cluster(source, output, lower, upper, count, grid_cells, dimensions=2, surface_sign=1):
    axes = np.argsort(upper - lower)[-dimensions:]
    extent = (upper - lower)[axes]
    if np.count_nonzero(extent > 0) < 2:
        raise ValueError('Mesh needs two nonzero spatial extents')
    extent = np.where(extent > 0, extent, 1)
    width = grid_cells + 1
    cluster_count = width ** dimensions
    sums = np.zeros((cluster_count, 3))
    hits = np.zeros(cluster_count, dtype=np.int64)
    height_axis = int(np.argsort(upper - lower)[0])
    heights = np.full(cluster_count, -np.inf)
    # Disk-backed index map: four bytes per original vertex, no Python object
    # per vertex and no readlines()/full-file strings for multi-GB source OBJs.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='terrain-index-') as folder:
        mapping = np.memmap(Path(folder) / 'indices.bin', dtype=np.int32, mode='w+', shape=(count,))
        try:
            start = 0
            for xyz in _vertices(source):
                cell = np.floor((xyz[:, axes] - lower[axes]) / extent * grid_cells).astype(int)
                cell = np.clip(cell, 0, grid_cells)
                ids = np.ravel_multi_index(cell.T, (width,) * dimensions)
                mapping[start:start + len(xyz)] = ids
                np.add.at(sums, ids, xyz)
                np.add.at(hits, ids, 1)
                if dimensions == 2:
                    # Solid terrain OBJs include a base below the visible DEM.
                    # Averaging both surfaces would halve all elevations.
                    np.maximum.at(heights, ids, surface_sign * xyz[:, height_axis])
                start += len(xyz)
                if start % 5000000 == 0:
                    print(f'Vizard OBJ: mapped {start:,}/{count:,} vertices', flush=True)
            if start != count:
                raise ValueError('Terrain source changed while building display mesh')
            active = np.flatnonzero(hits)
            vertices = sums[active] / hits[active, None]
            if dimensions == 2:
                vertices[:, height_axis] = surface_sign * heights[active]
            compact = np.full(cluster_count, -1, dtype=np.int32)
            compact[active] = np.arange(len(active))
            unique = set()
            source_faces = 0
            report_at = 5000000
            for faces, source_faces in _faces(source, count):
                if source_faces >= report_at:
                    print(f'Vizard OBJ: remapped {source_faces:,} source faces', flush=True)
                    report_at += 5000000
                faces = compact[mapping[faces]]
                faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                              & (faces[:, 2] != faces[:, 0])]
                if not len(faces):
                    continue
                # Canonical cyclic order keeps winding, including opposite faces.
                offsets = faces.argmin(axis=1)
                faces = np.take_along_axis(faces, (offsets[:, None] + np.arange(3)) % 3, axis=1)
                unique.update(map(tuple, np.unique(faces, axis=0).tolist()))
                if len(unique) > 8 * cluster_count:
                    raise ValueError('Mesh simplification exceeds the display face budget')
            if not unique:
                raise ValueError('Terrain simplification produced no triangles; increase grid_cells')
            faces = np.asarray(sorted(unique), dtype=np.int32)
        finally:
            mapping._mmap.close()
            del mapping
    # Remove geometric degenerates and vertices no longer referenced by faces.
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                      vertices[faces[:, 2]] - vertices[faces[:, 0]])
    keep = np.linalg.norm(normal, axis=1) > 1e-12
    if dimensions == 2:
        # Retain the outward terrain surface; underside/opposite faces would
        # overlap after horizontal clustering and cancel the smooth normals.
        keep &= surface_sign * normal[:, height_axis] > 1e-12
    faces, normal = faces[keep], normal[keep]
    if not len(faces):
        raise ValueError('Terrain simplification produced only degenerate triangles')
    used, inverse = np.unique(faces, return_inverse=True)
    vertices, faces = vertices[used], inverse.reshape(-1, 3)
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], normal)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.where(lengths > 0, lengths, 1)[:, None]
    with output.open('w', encoding='ascii', newline='\n') as stream:
        stream.write('# Vizard display terrain; source coordinates; collision mesh unchanged\no lunarTerrain\n')
        for v in vertices:
            stream.write('v %.9g %.9g %.9g\n' % tuple(v))
        for n in normals:
            stream.write('vn %.9g %.9g %.9g\n' % tuple(n))
        stream.write('s 1\n')
        for a, b, c in faces + 1:
            stream.write(f'f {a}//{a} {b}//{b} {c}//{c}\n')
    return source_faces, len(vertices), len(faces)


def prepare_vizard_terrain(source, cache_dir, grid_cells=180, large_threshold_bytes=32 * 1024 * 1024,
                           cluster_dimensions=2, surface_sign=1):
    """Return a cached render model; missing/invalid terrain fails explicitly.

    The cache key includes absolute source path, size, modification time and
    optimizer settings. Changing the selected OBJ or resolution rebuilds it.
    Materials/UVs are omitted from optimized lunar terrain (neutral gray).
    """
    if cluster_dimensions not in (2, 3):
        raise ValueError('cluster_dimensions must be 2 (terrain) or 3 (CAD)')
    if surface_sign not in (-1, 1):
        raise ValueError('surface_sign must be -1 or 1')
    limit = 250 if cluster_dimensions == 2 else 39
    if not isinstance(grid_cells, int) or not 2 <= grid_cells <= limit:
        raise ValueError(f'grid_cells must be an integer from 2 to {limit}')
    if large_threshold_bytes < 0:
        raise ValueError('large_threshold_bytes must be nonnegative')
    source, cache_dir = Path(source).resolve(), Path(cache_dir).resolve()
    stat = source.stat()
    signature = dict(version=2, source=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                     grid_cells=grid_cells, threshold=large_threshold_bytes)
    if cluster_dimensions == 3:
        signature['cluster_dimensions'] = 3
    if surface_sign == -1:
        signature['surface_sign'] = -1
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f'{source.stem}_{key}.obj'
    metadata = output.with_suffix('.json')
    if metadata.is_file():
        try:
            saved = json.loads(metadata.read_text())
            display = TerrainDisplay(**saved['display'])
            if saved['signature'] == signature and Path(display.path).stat().st_size == saved['output_bytes']:
                print(f'Vizard OBJ: cached {display.display_vertices:,} vertices / '
                      f'{display.display_triangles:,} triangles: {display.path}', flush=True)
                return display
        except (OSError, ValueError, KeyError, TypeError):
            pass
    started = time.monotonic()
    print(f'Vizard OBJ: scanning {source.name} ({stat.st_size / 1024**2:.1f} MiB). '
          'The first build can take several minutes; later runs reuse the cache.', flush=True)
    lower, upper, count = _bounds(source)
    optimized = stat.st_size > large_threshold_bytes
    if optimized:
        print(f'Vizard OBJ: clustering {count:,} vertices on a {grid_cells}-cell '
              f'{cluster_dimensions}D grid...', flush=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, suffix='.obj', delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            source_faces, vertices, triangles = _cluster(source, temp_path, lower, upper, count, grid_cells,
                                                        dimensions=cluster_dimensions, surface_sign=surface_sign)
            current = source.stat()
            if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise ValueError('Terrain source changed during optimization; run again')
            temp_path.replace(output)
        finally:
            temp_path.unlink(missing_ok=True)
        render_path = output
    else:
        # Keep small models and their material/texture references intact.
        source_faces = triangles = 0
        for faces, source_faces in _faces(source, count):
            triangles += len(faces)
        vertices, render_path = count, source
    if not triangles:
        raise ValueError(f'OBJ contains no renderable terrain faces: {source}')
    display = TerrainDisplay(str(render_path), tuple(lower), tuple(upper), count, source_faces,
                             vertices, triangles, optimized)
    saved = dict(signature=signature, display=display.__dict__, output_bytes=render_path.stat().st_size)
    metadata.write_text(json.dumps(saved, indent=2))
    print(f'Vizard OBJ ready in {time.monotonic() - started:.1f}s: '
          f'{vertices:,} vertices / {triangles:,} triangles, '
          f'{render_path.stat().st_size / 1024**2:.1f} MiB: {render_path}', flush=True)
    return display


def prepare_vizard_model(source, cache_dir, grid_cells=32):
    """Bound large multi-layer CAD models using 3D rather than terrain clustering."""
    return prepare_vizard_terrain(source, cache_dir, grid_cells=grid_cells, cluster_dimensions=3)


def display_transform(vertices, anchor, rotation, scale, offset=(0., 0., 0.)):
    """Apply Vizard's actual custom-model scale/Euler/offset convention.

    Unity Quaternion.Euler applies Z, then X, then Y; Vizard maps BSK axes to
    Unity as (y,z,-x), including model offsets and scale axes (y,z,x).
    The custom-model rotation fields are [x,y,z], despite the older helper's
    misleading 3-2-1 description; Vizard uses Euler(y,z,-x) in Unity.
    """
    from scipy.spatial.transform import Rotation
    unity_to_bsk = np.array([[0., 0., -1.], [1., 0., 0.], [0., 1., 0.]])
    unity_rotation = Rotation.from_euler('zxy', [-rotation[0], rotation[1], rotation[2]]).as_matrix()
    return ((np.asarray(vertices) * np.asarray(scale)[[1, 2, 0]])
            @ unity_rotation.T @ unity_to_bsk.T + np.asarray(anchor) + np.asarray(offset))


def read_display_mesh(path):
    """Read the bounded display OBJ, never the multi-GB source model."""
    path = Path(path)
    vertices = np.concatenate(list(_vertices(path)))
    faces = np.concatenate([f for f, _ in _faces(path, len(vertices))])
    return vertices, faces


def centered_terrain_offset(display, rotation, scale):
    """Center the rendered horizontal footprint on the south-pole anchor.

    The OBJ origin can be a corner. Center X/Z in the inertial frame without
    changing elevations or the Moon-centered reference radius. Collision and
    visualization must apply this same offset exactly once.
    """
    vertices = np.concatenate(list(_vertices(Path(display.path))))
    local_N = display_transform(vertices, [0., 0., 0.], rotation, scale)
    center = (local_N.min(axis=0) + local_N.max(axis=0)) / 2
    return [-float(center[0]), 0., -float(center[2])]


def surface_y(vertices_N, faces, xz):
    """Interpolate the outward (-Y) rendered surface under a horizontal point."""
    triangles = vertices_N[faces]
    origin = triangles[:, 0, [0, 2]]
    u = triangles[:, 1, [0, 2]] - origin
    v = triangles[:, 2, [0, 2]] - origin
    delta = np.asarray(xz).reshape(2) - origin
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(det) > 1e-12
    a = np.full(len(faces), np.nan)
    b = a.copy()
    a[valid] = (delta[valid, 0] * v[valid, 1] - delta[valid, 1] * v[valid, 0]) / det[valid]
    b[valid] = (u[valid, 0] * delta[valid, 1] - u[valid, 1] * delta[valid, 0]) / det[valid]
    inside = valid & (a >= -1e-9) & (b >= -1e-9) & (a + b <= 1 + 1e-9)
    if not inside.any():
        raise ValueError('Preview point is outside rendered terrain coverage; select a covered point')
    y = (triangles[:, 0, 1] + a * (triangles[:, 1, 1] - triangles[:, 0, 1])
         + b * (triangles[:, 2, 1] - triangles[:, 0, 1]))
    return float(y[inside].min())


def preview_position(display, anchor, rotation, scale, clearance_m=30., offset=(0., 0., 0.)):
    """Place the preview above the actual rendered surface at its center.

    Raw OBJ positions are Unity coordinates; custom rotation fields are BSK x,y,z.
    In this project lunar outward/up is inertial -Y.
    """
    if not math.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError('Preview clearance must be finite and positive')
    vertices, faces = read_display_mesh(display.path)
    vertices_N = display_transform(vertices, anchor, rotation, scale, offset)
    lower, upper = vertices_N.min(axis=0), vertices_N.max(axis=0)
    xz = ((lower + upper) / 2)[[0, 2]]
    y = surface_y(vertices_N, faces, xz)
    return [xz[0], y - clearance_m, xz[1]]
