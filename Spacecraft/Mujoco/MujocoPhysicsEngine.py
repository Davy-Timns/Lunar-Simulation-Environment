import mujoco
import numpy as np
from Sensors.Laser.Altimeter import LaserAltimeter
from Spacecraft.Mujoco.FrameTransforms import (
    TERRAIN_POSITION_MUJOCO,
    TERRAIN_ROTATION_MUJOCO,
    TERRAIN_SCALE,
)
from Environment.Dynamics import r_moon
OBJ_NAME = "SouthPole_Defragged.obj"

OBJ_PATH = f"Environment/Objects/{OBJ_NAME}"
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


def initialize_mujoco(LF:int,scm,sci,samp):


    obj_data = clean_obj(f"Environment/Objects/{OBJ_NAME}")

    XML= get_xml(LF)

    model = mujoco.MjModel.from_xml_string(XML,assets={OBJ_NAME: obj_data})

    data = mujoco.MjData(model)

    laser_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,"laser_altimeter")

    altimeter = LaserAltimeter(model,data,laser_id,scm,sci,samp)

    return altimeter


def get_xml(LF):
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
                              euler="90 0 0"
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
                memory="1G"
            />

            <asset>
                <mesh name="lunarTerrain"
                      file="{OBJ_NAME}"
                      scale="{_xml_vec(TERRAIN_SCALE)}"
                      maxhullvert="1000000"/>
            </asset>

            <option gravity="0 0 -9.81"/>

            <statistic extent="50000"/>

            <worldbody>

                <geom name="lunarTerrain_geom"
                      type="mesh"
                      mesh="lunarTerrain"
                      pos="{_xml_vec(TERRAIN_POSITION_MUJOCO)}"
                      euler="{_xml_vec(TERRAIN_ROTATION_MUJOCO)}"
                      contype="0"
                      conaffinity="0"
                      margin="0.0"/>

                <body name="spacecraft">

                    <freejoint name="spacecraft_freejoint"/>

{_spacecraft_collision_geoms_xml(0.0)}

                    <site name="laser_altimeter"
                          pos="0 0 -0.30"
                          euler="90 0 0"
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
                      file="{OBJ_NAME}"
                      scale="{_xml_vec(TERRAIN_SCALE)}"/>
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
                      euler="{_xml_vec(TERRAIN_ROTATION_MUJOCO)}"
                    contype="1"
                     conaffinity="1"
                      margin="0.25"/>

                <!-- Spacecraft -->
                <body name="spacecraft">

                    <freejoint name="spacecraft_freejoint"/>

{_spacecraft_collision_geoms_xml(0.25)}

                    <site name="laser_altimeter"
                          pos="0 0 -0.30"
                          euler="90 0 0"
                          size="0.03"/>

                </body>

            </worldbody>

        </mujoco>
        """
    return XML
