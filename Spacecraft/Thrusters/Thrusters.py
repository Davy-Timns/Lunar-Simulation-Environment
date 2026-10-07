import numpy as np
from Basilisk.simulation import thrusterDynamicEffector

def make_config(thrust, isp, direction):
    """Build an independent, constant-magnitude thruster config (N, s, body axes)."""
    if not np.isfinite(thrust) or thrust <= 0:
        raise ValueError("thrust must be finite and positive.")
    if not np.isfinite(isp) or isp <= 0:
        raise ValueError("isp must be finite and positive.")
    direction = np.asarray(direction, dtype=float).reshape(3)
    norm = np.linalg.norm(direction)
    if not np.all(np.isfinite(direction)) or not np.isfinite(norm) or norm == 0:
        raise ValueError("direction must be a finite, nonzero vector.")
    config = thrusterDynamicEffector.THRSimConfig()
    config.MaxThrust = thrust
    config.steadyIsp = isp
    config.thrDir_B = (direction / norm).reshape(3, 1).tolist()
    return config

class RCS_thruster:
    """One fixed-direction RCS thruster.

    The supplied configuration is copied so instances can share a template.
    """

    def __init__(self, config, name="RCS_thruster", location=None):
        config = thrusterDynamicEffector.THRSimConfig(config)
        self.name = name
        self.thruster = thrusterDynamicEffector.ThrusterDynamicEffector()
        self.thruster.ModelTag = name

        # Preserve the configured location unless explicitly overridden.
        if location is not None:
            location = np.asarray(location, dtype=float).reshape(3, 1)
            if not np.all(np.isfinite(location)):
                raise ValueError("location must be finite.")
            config.thrLoc_B = location.tolist()

        self.thruster.addThruster(config)

        # addThruster retains this shared config. Keeping it also works in
        # Basilisk 2.10.2, whose thrusterData vector is not Python-indexable.
        self.config = config

    def attach(self, spacecraft, simulation, task_name, priority=10):
        """Schedule above the spacecraft (default priority 0), below its controller."""
        spacecraft.addDynamicEffector(self.thruster)
        simulation.AddModelToTask(task_name, self.thruster, ModelPriority=priority)

    def subscribe_to_commands(self, command_msg):
        """Controller must be linked here!"""
        self.thruster.cmdsInMsg.subscribeTo(command_msg)


class MainEngine(RCS_thruster):
    """Main engine with ideal, instantaneous two-axis thrust vectoring.

    Nominal force direction comes from config.thrDir_B (not Euler angles).
    Mount-frame +X follows that direction; +Y is body +Y projected normal
    to it (body +Z is used when parallel). +Z completes a right-handed frame.
    Rotate about mount +Y for pitch, then mount +Z for yaw. For a body +X
    mount, this is the original body +Y / +Z convention.
    The thrust application point remains fixed.
    """

    def __init__(
        self,
        config,
        name="MainEngine",
        location=None,
        max_gimbal=0.27,
    ):
        super().__init__(config, name=name, location=location)

        # Optional symmetric limit on each axis, in radians.
        if max_gimbal is not None:
            if not np.isfinite(max_gimbal) or max_gimbal < 0:
                raise ValueError("max_gimbal must be finite and nonnegative.")

        self.max_gimbal = max_gimbal

        direction = np.asarray(self.config.thrDir_B, dtype=float).reshape(3)
        norm = np.linalg.norm(direction)
        if not np.all(np.isfinite(direction)) or not np.isfinite(norm) or norm == 0:
            raise ValueError("The main engine direction must be finite and nonzero.")
        mount_x = direction / norm
        reference_y = np.array([0.0, 1.0, 0.0])
        mount_y = reference_y - np.dot(reference_y, mount_x) * mount_x
        if np.linalg.norm(mount_y) < 1e-8:
            reference_y = np.array([0.0, 0.0, 1.0])
            mount_y = reference_y - np.dot(reference_y, mount_x) * mount_x
        mount_y /= np.linalg.norm(mount_y)
        mount_z = np.cross(mount_x, mount_y)
        self._mount_dcm_BM = np.column_stack((mount_x, mount_y, mount_z))
        self.set_gimbal(pitch=0.0, yaw=0.0)

    def set_gimbal(self, pitch, yaw):
        if not np.all(np.isfinite([pitch, yaw])):
            raise ValueError("Gimbal angles must be finite.")

        if self.max_gimbal is not None:
            if max(abs(pitch), abs(yaw)) > self.max_gimbal:
                raise ValueError("Requested angle exceeds the gimbal limit.")

        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # Absolute deflection from the configured mount, not from body +X or
        # the previous gimbal command. Zero angles restore the nominal direction.
        direction_B = self._mount_dcm_BM @ np.array([cp * cy, cp * sy, -sp])
        self.config.thrDir_B = direction_B.reshape(3, 1).tolist()

        self.pitch = float(pitch)
        self.yaw = float(yaw)
