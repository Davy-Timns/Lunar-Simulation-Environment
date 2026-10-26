import mujoco
import numpy as np
from pathlib import Path
import time
from Sensors.Laser.Altimeter import LaserAltimeter
from Spacecraft.Mujoco.FrameTransforms import (
    TERRAIN_POSITION_MUJOCO,
    TERRAIN_ROTATION_MUJOCO,
    TERRAIN_SCALE,
)
from Environment.Dynamics import r_moon
OBJ_NAME = "SouthPoleUpdates.obj"

OBJ_PATH = f"Environment/Objects/{OBJ_NAME}"
COLLISION_TERRAIN_GRID_CELLS = 180  # explicit simplified DEM resolution; 2..250
MUJOCO_ARENA_MEMORY = '64M'  # terrain contacts use raycasts, not millions of native contact pairs
def clean_obj(obj_path):
    with open(obj_path, "r", errors="replace") as f:
        lines = f.readlines()

    # Find valid vertices
    valid_vertex_indices = set()
    vertex_count = 0

    for line in lines:
        if line.startswith("v "):
            vertex_count += 1
            parts = line.split()

            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])

                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    valid_vertex_indices.add(vertex_count)

            except (ValueError, IndexError):
                pass

    # Build new OBJ
    output = []
    vertex_map = {}

    new_index = 1
    old_index = 0

    # Vertices
    for line in lines:
        if line.startswith("v "):
            old_index += 1

            if old_index in valid_vertex_indices:
                output.append(line)
                vertex_map[old_index] = new_index
                new_index += 1

        elif line.startswith("f "):
            parts = line.split()[1:]

            face_vertices = []

            try:
                for part in parts:
                    vertex_index = int(part.split("/")[0])

                    if vertex_index < 0:
                        vertex_index = vertex_count + vertex_index + 1

                    # If ANY vertex in the face is invalid,
                    # discard the entire face.
                    if vertex_index not in valid_vertex_indices:
                        face_vertices = []
                        break

                    face_vertices.append(vertex_map[vertex_index])

                if face_vertices:
                    output.append(
                        "f " + " ".join(map(str, face_vertices)) + "\n"
                    )

            except (ValueError, IndexError):
                pass

    return "".join(output).encode("utf-8")


def _xml_vec(values):
    return " ".join(str(float(value)) for value in values)


def _spacecraft_collision_geoms_xml(margin):
    return f"""
                        <geom name="spacecraft_core"
                              type="box"
                              pos="0 0 0.05"
                              size="0.42 0.42 0.25"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_leg_1"
                              type="capsule"
                              fromto="0.30 0.30 -0.08 0.78 0.78 -0.58"
                              size="0.035"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_leg_2"
                              type="capsule"
                              fromto="-0.30 0.30 -0.08 -0.78 0.78 -0.58"
                              size="0.035"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_leg_3"
                              type="capsule"
                              fromto="0.30 -0.30 -0.08 0.78 -0.78 -0.58"
                              size="0.035"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_leg_4"
                              type="capsule"
                              fromto="-0.30 -0.30 -0.08 -0.78 -0.78 -0.58"
                              size="0.035"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_foot_1"
                              type="box"
                              pos="0.82 0.82 -0.62"
                              size="0.18 0.18 0.04"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_foot_2"
                              type="box"
                              pos="-0.82 0.82 -0.62"
                              size="0.18 0.18 0.04"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_foot_3"
                              type="box"
                              pos="0.82 -0.82 -0.62"
                              size="0.18 0.18 0.04"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>

                        <geom name="spacecraft_foot_4"
                              type="box"
                              pos="-0.82 -0.82 -0.62"
                              size="0.18 0.18 0.04"
                              contype="1"
                              conaffinity="1"
                              margin="{margin}"/>
"""


def initialize_mujoco(LF: int, scm, sci, samp):
    import hashlib
    import json
    import tempfile
    from types import SimpleNamespace
    from Environment.VizardTerrain import (
        prepare_vizard_terrain,
        read_display_mesh,
        centered_terrain_offset,
        display_transform,
    )
    from Spacecraft.Mujoco import FrameTransforms as frames

    started = time.monotonic()
    terrain = None
    assets = {}

    if LF == 0:
        XML = get_xml(LF)
    else:
        root = Path(__file__).resolve().parents[2]
        print("MuJoCo: preparing collision terrain...", flush=True)

        display = prepare_vizard_terrain(
            root / "Environment" / "Objects" / OBJ_NAME,
            root / "work" / "vizard_terrain",
            grid_cells=COLLISION_TERRAIN_GRID_CELLS,
        )
        vertices, faces = read_display_mesh(display.path)
        offset = centered_terrain_offset(
            display,
            frames.TERRAIN_ROTATION_VIZARD,
            frames.TERRAIN_SCALE,
        )

        # Apply the same transform used by Vizard, then convert to MuJoCo.
        local_N = display_transform(
            vertices,
            [0., 0., 0.],
            frames.TERRAIN_ROTATION_VIZARD,
            frames.TERRAIN_SCALE,
            offset,
        )
        local_M = local_N @ frames.R_BSK_TO_MUJOCO.T
        faces = faces[:, [0, 2, 1]]  # Correct winding after reflection.

        source = Path(display.path).resolve()
        stat = source.stat()
        signature = {
            "version": 3,
            "source": str(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "grid_cells": COLLISION_TERRAIN_GRID_CELLS,
            "rotation": list(frames.TERRAIN_ROTATION_VIZARD),
            "scale": list(frames.TERRAIN_SCALE),
            "offset": offset,
        }
        key = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()[:16]

        cache_dir = root / "work" / "collision_terrain"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{Path(OBJ_NAME).stem}_collision_{key}.obj"

        if not path.is_file():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="ascii",
                    dir=cache_dir,
                    suffix=".obj",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    stream.write("# Baked collision terrain; metres\n")
                    np.savetxt(stream, local_M, fmt="v %.9f %.9f %.9f")
                    np.savetxt(stream, faces + 1, fmt="f %d %d %d")

                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

        cell_m = (
            float(np.max(np.ptp(local_M, axis=0)))
            / COLLISION_TERRAIN_GRID_CELLS
            if display.optimized else 0.
        )
        terrain = SimpleNamespace(
            path=str(path),
            vertices=len(local_M),
            triangles=len(faces),
            approximate_cell_m=cell_m,
        )

        print(
            f"MuJoCo terrain: {terrain.vertices:,} vertices / "
            f"{terrain.triangles:,} triangles; approximate DEM cell "
            f"{cell_m:.2f} m. Compiling collision geometry...",
            flush=True,
        )
        XML = get_xml(
            LF,
            terrain_file=path.name,
            terrain_scale=[1., 1., 1.],
            terrain_rotation=[0., 0., 0.],
        )
        assets[path.name] = path.read_bytes()

    model = mujoco.MjModel.from_xml_string(XML, assets=assets)

    print("MuJoCo: creating simulation data...", flush=True)
    data = mujoco.MjData(model)

    laser_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "laser_altimeter"
    )
    if laser_id < 0:
        raise ValueError("MuJoCo model has no laser_altimeter site.")

    altimeter = LaserAltimeter(model, data, laser_id, scm, sci, samp)
    altimeter.collision_terrain = terrain

    print(
        f"MuJoCo geometry ready in {time.monotonic() - started:.2f}s; "
        f"arena={model.narena / 1024**2:g} MiB",
        flush=True,
    )
    return altimeter


def get_xml(LF, terrain_file=OBJ_NAME, terrain_scale=None, terrain_rotation=None):
    terrain_scale = TERRAIN_SCALE if terrain_scale is None else terrain_scale
    terrain_rotation = TERRAIN_ROTATION_MUJOCO if terrain_rotation is None else terrain_rotation
    if LF == 0:  # Creates a sphere for the altimeter instead of lunar terrain obj
        XML = f"""
            <mujoco model="laser_altimeter">

                <option gravity="0 0 -9.81"/>
                <statistic extent="{r_moon * 2}"/>

                <worldbody>

                    <geom name="moon"
                          type="sphere"
                          size="{r_moon}"
                          pos="0 0 0"
                          margin="0.25"/>

                    <body name="spacecraft" pos="0 0 {r_moon + 1500}">

                        <freejoint name="spacecraft_freejoint"/>

{_spacecraft_collision_geoms_xml(0.25)}

                        <site name="laser_altimeter"
                              pos="0 0 -0.30"
                              euler="180 0 0"
                              size="0.03"/>

                    </body>

                </worldbody>

            </mujoco>
            """
    elif LF == 1:  # Use lunar terrain OBJ
        XML = f"""
         <mujoco model="laser_altimeter">

            <compiler
                meshdir="."
                usethread="true"
            />

            <size
                memory="{MUJOCO_ARENA_MEMORY}"
            />

            <asset>
                <mesh name="lunarTerrain"
                      file="{terrain_file}"
                      scale="{_xml_vec(terrain_scale)}"
                      inertia="shell"
                      maxhullvert="64"/>
            </asset>

            <option gravity="0 0 -9.81"/>

            <statistic extent="50000"/>

            <worldbody>

                <geom name="lunarTerrain_geom"
                      type="mesh"
                      mesh="lunarTerrain"
                      pos="{_xml_vec(TERRAIN_POSITION_MUJOCO)}"
                      euler="{_xml_vec(terrain_rotation)}"
                      contype="0"
                      conaffinity="0"
                      margin="0.0"/>

                <body name="spacecraft">

                    <freejoint name="spacecraft_freejoint"/>

{_spacecraft_collision_geoms_xml(0.0)}

                    <site name="laser_altimeter"
                          pos="0 0 -0.30"
                          euler="180 0 0"
                          size="0.03"/>

                </body>

            </worldbody>

        </mujoco>
        """
    else:  # Use both a smooth spherical surface and lunar terrain obj
        XML = f"""
        <mujoco model="laser_altimeter">

            <asset>
                <mesh name="south_pole"
                      file="{terrain_file}"
                      scale="{_xml_vec(terrain_scale)}"
                      inertia="shell" maxhullvert="64"/>
            </asset>

            <option gravity="0 0 0"/>
            <statistic extent="{r_moon * 2}"/>

            <worldbody>

                <!-- Smooth spherical Moon reference -->
                <geom name="moon_sphere"
                      type="sphere"
                      size="{r_moon}"
                      pos="0 0 0"

                      contype="1"
                        conaffinity="1"
                      margin="0.25"/>

                <!-- Actual lunar terrain -->
                <geom name="south_pole"
                      type="mesh"
                      mesh="south_pole"
                      pos="{_xml_vec(TERRAIN_POSITION_MUJOCO)}"
                      euler="{_xml_vec(terrain_rotation)}"
                    contype="1"
                     conaffinity="1"
                      margin="0.25"/>

                <!-- Spacecraft -->
                <body name="spacecraft">

                    <freejoint name="spacecraft_freejoint"/>

{_spacecraft_collision_geoms_xml(0.25)}

                    <site name="laser_altimeter"
                          pos="0 0 -0.30"
                          euler="180 0 0"
                          size="0.03"/>

                </body>

            </worldbody>

        </mujoco>
        """
    return XML
