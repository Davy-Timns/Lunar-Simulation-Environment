import numpy as np
from scipy.spatial.transform import Rotation
from Environment.Dynamics import r_moon



# Basilisk uses the Moon-centered simulation frame. MuJoCo is used only as a
# geometry/raycast backend, with its Z axis aligned to Basilisk's south-pole Y.
R_BSK_TO_MUJOCO = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])
R_MUJOCO_TO_BSK = R_BSK_TO_MUJOCO.T

TERRAIN_POSITION_BSK = np.array([0.0, -r_moon, 0.0])
TERRAIN_POSITION_MUJOCO = R_BSK_TO_MUJOCO @ TERRAIN_POSITION_BSK

# MuJoCo needs the collision mesh flipped to expose the rough side to raycasts.
TERRAIN_ROTATION_MUJOCO = [180.0, 0.0, 90.0]

# Vizard renders the visible terrain side correctly without the MuJoCo collision
# flip. Keep this separate from the raycast/collision orientation.
TERRAIN_ROTATION_MUJOCO_VISUAL = [0.0, 0.0, 0.0]
TERRAIN_SCALE = [10.0, 10.0, 10.0]


def _matrix_to_vizard_321(matrix):
    return Rotation.from_matrix(matrix).as_euler("zyx", degrees=False).tolist()


TERRAIN_ROTATION_VIZARD = _matrix_to_vizard_321(
    R_MUJOCO_TO_BSK
    @ Rotation.from_euler(
        "xyz",
        TERRAIN_ROTATION_MUJOCO_VISUAL,
        degrees=True,
    ).as_matrix()
)
