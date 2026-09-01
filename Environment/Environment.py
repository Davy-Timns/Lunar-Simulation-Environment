import numpy as np
from Basilisk.simulation import spacecraft
from Spacecraft.Mujoco.FrameTransforms import TERRAIN_POSITION_BSK


def create_terrain_spacecraft():

    terrain = spacecraft.Spacecraft()
    terrain.ModelTag = "lunarTerrain"

    terrain.hub.r_CN_NInit = TERRAIN_POSITION_BSK.copy()

    # Stationary
    terrain.hub.v_CN_NInit = np.array([
        0.0,
        0.0,
        0.0
    ])

    # Orientation
    terrain.hub.sigma_BNInit = np.array([
        0.0,
        0.0,
        0.0
    ])

    # No angular velocity
    terrain.hub.omega_BN_BInit = np.array([
        0.0,
        0.0,
        0.0
    ])

    # Give it a valid mass; this is only a visualization anchor
    terrain.hub.mHub = 0.01

    return terrain
