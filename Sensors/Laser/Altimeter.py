import mujoco
import numpy as np
from Basilisk.architecture import messaging, sysModel, bskLogging
from Basilisk.utilities import RigidBodyKinematics
from Spacecraft.Mujoco.FrameTransforms import R_BSK_TO_MUJOCO, R_MUJOCO_TO_BSK

class LaserAltimeter(sysModel.SysModel):
    """
    MuJoCo laser altimeter implemented as a Basilisk Python module.
    Basilisk controls spacecraft dynamics; MuJoCo provides geometry
    and raycasting for the laser measurement.

    For now, the collision dynamics is also home in this class, this MUST be moved to another file, either MujocoPhysicsEngine.py
    or Dynamics.py.
    """

    def __init__(self, model, data, laser_id, spacecraft_mass, spacecraft_inertia, samp):
        super().__init__()

        self.name = "LaserAltimeter1"

        self.ModelTag = "LaserAltimeter"
        self.model = model
        self.data = data
        self.laser_id = laser_id
        self.spacecraft_collision_geom_names = ( #Splits the spacecraft up into different GEOMs for collision detection. Otherwise the spacecraft is treated as a square and the spacecraft is purely visualization.
            "spacecraft_core",
            "spacecraft_leg_1", #Since we only have four legs here, this isnt totally analogous to IMX, but it's close enough
            "spacecraft_leg_2",
            "spacecraft_leg_3",
            "spacecraft_leg_4",
            "spacecraft_foot_1",
            "spacecraft_foot_2",
            "spacecraft_foot_3",
            "spacecraft_foot_4",
            "spacecraft_body",
        )
        self.spacecraft_geom_ids = [
            geom_id
            for geom_id in (
                mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,geom_name)
                for geom_name in self.spacecraft_collision_geom_names
            )
            if geom_id >= 0
        ]
        self.spacecraft_geom_id_set = set(self.spacecraft_geom_ids)
        self.spacecraft_geom_id = (
            self.spacecraft_geom_ids[0]
            if self.spacecraft_geom_ids
            else -1
        )

        #Basilisk input: spacecraft state
        self.scStateInMsg = messaging.SCStatesMsgReader()

        # Basilisk outputs
        self.sensorOutMsg = messaging.NavTransMsg()
        self.forceOutMsg = messaging.CmdForceInertialMsg()
        self.torqueOutMsg = messaging.CmdTorqueBodyMsg()

        # Sensor output initialization
        self.altitude = np.nan
        self.hit_geom = -1

        # logs
        self.fields = ["timeTag", "r_BN_N", "v_BN_N"]





        #Dynamics parameters to tweak.

        self.contact_margin = 0.25
        self.contact_stiffness = 1.0e5
        self.contact_damping = 2.0e4
        self.contact_tangential_damping = 5000.0
        self.resting_tangential_damping = 50000.0
        self.contact_friction_coefficient = 2.0
        self.max_contact_force = 2.0e5
        self.surface_probe_distance = 2.0
        self.surface_normal_sample_offset = 0.5
        self.use_radial_contact_normal = False
        self.apply_contact_torque = True
        self.max_contact_torque = 0.5e3
        self.max_impact_contact_torque = 2.0e4
        self.settle_velocity_threshold = 0.05  # m/s
        self.settle_damping = 50000.0  # N/(m/s)  Damping used to remove small residual tangential motion.
        self.max_settle_force = 2.0e4  # Maximum force available to kill residual motion.
        self.max_contact_penetration = 0.02  # 2 cm
        self.max_impact_contact_penetration = 0.15  # 15 cm
        self.resting_velocity_threshold = 0.5
        self.resting_angular_damping = 500.0
        self.resting_angular_settle_damping = 5000.0
        self.resting_angular_settle_threshold = 0.05
        self.resting_angular_velocity_threshold = 1.0e-6
        self.max_resting_damping_torque = 1000.0
        self.debug_contacts = True
        self.debug_contact_every_n_steps = 20


        #MuJoCo  variable Initialization and Constants
        self.recorder = None
        self.collision = False
        self.contact_force_mjc = np.zeros(3)
        self.contact_point_mjc = np.zeros(3)
        self.contact_normal_mjc = np.zeros(3)
        self.penetration = 0.0
        self.surface_contact_margin = self.contact_margin
        self.resting_penetration_threshold = self.max_contact_penetration
        self.mu_moon = 4.9048695e12
        self._previous_update_time_s = None
        self._time_step_s = samp
        self.spacecraft_mass = spacecraft_mass
        self.spacecraft_inertia_B = spacecraft_inertia
        self._debug_step_counter = 0
        self.surface_contacts = []

        self.terrain_geom_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"lunarTerrain_geom")
        self.use_mesh_surface_contacts = self.terrain_geom_id >= 0

        print("\n========== MUJOCO TERRAIN DEBUG ==========")
        print("Terrain geom ID:", self.terrain_geom_id)

        if self.terrain_geom_id >= 0:

            terrain_mesh_id = model.geom_dataid[self.terrain_geom_id]

            print("Terrain geom type:", model.geom_type[self.terrain_geom_id])
            print("Terrain mesh ID:", terrain_mesh_id)

            if terrain_mesh_id >= 0:
                print("Terrain mesh name:",mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_MESH,terrain_mesh_id))

                print("Terrain mesh vertices:", model.mesh_vertnum[terrain_mesh_id])

                print("Terrain mesh faces:",model.mesh_facenum[terrain_mesh_id])

        print("==========================================\n")

        self.spacecraft_body_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"spacecraft")

        if not self.spacecraft_geom_ids:
            raise RuntimeError("MuJoCo model is missing spacecraft collision geoms.")
        if self.spacecraft_body_id < 0:
            raise RuntimeError("MuJoCo model is missing body 'spacecraft'.")

        self.surface_contact_geom_ids = [
            geom_id
            for geom_id in self.spacecraft_geom_ids
            if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,geom_id)or "").startswith("spacecraft_foot_")
        ]

        self.spacecraft_joint_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,"spacecraft_freejoint")
        if self.spacecraft_joint_id < 0:
            raise RuntimeError("MuJoCo model is missing joint 'spacecraft_freejoint'.")

        self.spacecraft_qpos_adr = model.jnt_qposadr[self.spacecraft_joint_id]
        self.spacecraft_qvel_adr = model.jnt_dofadr[self.spacecraft_joint_id]

    def is_spacecraft_geom(self, geom_id):
        """
        Return spacecraft geometry ID for debugging and other mechanics
        :param geom_id:
        :return:
        """
        return geom_id in self.spacecraft_geom_id_set

    def is_spacecraft_contact(self, geom1, geom2):
        """
        Check if spacecraft is in contact
        :param geom1:
        :param geom2:
        :return:
        """
        return (self.is_spacecraft_geom(geom1) or self.is_spacecraft_geom(geom2))


    def Reset(self, CurrentSimNanos):
        """
        Specifically a MuJoCo dynamics step that deals with resetting variables.Happens in the backend
        :param CurrentSimNanos:
        :return:
        """

        if not self.scStateInMsg.isLinked():
            self.bskLogger.bskLog(bskLogging.BSK_ERROR,"LaserAltimeter.scStateInMsg is not linked.")

        payload = self.sensorOutMsg.zeroMsgPayload
        payload.timeTag = CurrentSimNanos
        payload.r_BN_N = [0.0, 0.0, 0.0]
        payload.v_BN_N = [0.0, 0.0, 0.0]

        self.sensorOutMsg.write(payload, CurrentSimNanos, self.moduleID)

        self.publish_contact_loads(np.zeros(3),np.zeros(3), CurrentSimNanos)

        self.bskLogger.bskLog(bskLogging.BSK_INFORMATION,"LaserAltimeter Reset() complete.")


    def UpdateState(self, CurrentSimNanos):
        """
        Called every simulation step, similar to Reset(), also happens on the backend
        :param CurrentSimNanos:
        :return:
        """

        # BASILISK -> MUJOCO FRAME

        R_bsk_to_mjc = R_BSK_TO_MUJOCO
        R_mjc_to_bsk = R_MUJOCO_TO_BSK
        current_time_s = CurrentSimNanos * 1.0e-9

        if self._previous_update_time_s is not None:
            dt = current_time_s - self._previous_update_time_s

            if dt > 0.0:
                self._time_step_s = dt

        self._previous_update_time_s = current_time_s

        # READ BASILISK STATE

        scState = self.scStateInMsg()

        r_BN_N = np.array(scState.r_BN_N, dtype=float)
        v_BN_N = np.array(scState.v_BN_N, dtype=float)
        sigma_BN = np.array(scState.sigma_BN, dtype=float)
        omega_BN_B = np.array(scState.omega_BN_B, dtype=float)

        r_mjc = R_bsk_to_mjc @ r_BN_N

        C_BN = RigidBodyKinematics.MRP2C(sigma_BN)
        C_NB = C_BN.T
        omega_BN_N = C_NB @ omega_BN_B

        C_mjc = (R_bsk_to_mjc @ C_NB @ R_bsk_to_mjc.T)

        q_mjc = RigidBodyKinematics.C2EP(C_mjc)

        v_mjc = R_bsk_to_mjc @ v_BN_N
        omega_mjc = R_bsk_to_mjc @ omega_BN_N

        qpos_adr = self.spacecraft_qpos_adr
        qvel_adr = self.spacecraft_qvel_adr

        self.data.qpos[qpos_adr:qpos_adr + 3] = r_mjc
        self.data.qpos[qpos_adr + 3:qpos_adr + 7] = q_mjc
        self.data.qvel[qvel_adr:qvel_adr + 3] = v_mjc
        self.data.qvel[qvel_adr + 3:qvel_adr + 6] = omega_mjc

        mujoco.mj_normalizeQuat(self.model,self.data.qpos)

        #UPDATE MUJOCO

        mujoco.mj_forward(self.model,self.data)

        #CHECK CONTACT

        if self.use_mesh_surface_contacts:
            self.surface_contacts = self.find_mesh_surface_contacts()
            self.check_surface_collision()
        else:
            self.check_collision()

        #COMPUTE CONTACT LOADS

        F_N, torque_B = self.get_contact_loads(r_mjc,v_mjc,omega_mjc,C_BN,R_mjc_to_bsk)

        #  MUJOCO -> BASILISK

        print("Contact force: ", F_N, "Torque: ", torque_B)

        self.publish_contact_loads(F_N,torque_B, CurrentSimNanos)

        #LASER ALTIMETER

        self.altitude, self.hit_geom = self.laser_altimeter()

        payload = self.sensorOutMsg.zeroMsgPayload

        payload.timeTag = CurrentSimNanos

        payload.r_BN_N = [0.0, 0.0,self.altitude]

        payload.v_BN_N = [0.0, 0.0, 0.0]

        self.sensorOutMsg.write(payload, CurrentSimNanos,self.moduleID)


    def laser_altimeter(self):
        """
        MuJoCo raycast for the altimeter measurement
        :return:
        """

        origin = self.data.site_xpos[self.laser_id].copy()
        rotation = self.data.site_xmat[self.laser_id].reshape(3, 3)

        direction = rotation[:, 2].copy()
        direction /= np.linalg.norm(direction)

        local_down = -origin
        local_down /= np.linalg.norm(local_down)

        if np.dot(direction, local_down) <= 0.0:
            return -1.0, -1

        geomgroup = np.ones(6, dtype=np.uint8)
        geom_id = np.array([-1], dtype=np.int32)

        spacecraft_body_id = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"spacecraft")

        distance = mujoco.mj_ray(self.model, self.data,origin,direction,geomgroup,1,spacecraft_body_id,geom_id)
        return distance, geom_id[0]

    def raycast_terrain(self, origin, direction):
        """
        A specific raycast that takes the current orientation of the lander to get the downward direction, "Raycast" down in order to collide
        with the terrain mesh rather than the terrain mesh bounding box.
        :param origin:
        :param direction:
        :return:
        """

        geomgroup = np.ones(6, dtype=np.uint8)
        geom_id = np.array([-1], dtype=np.int32)

        distance = mujoco.mj_ray(self.model,self.data,origin,direction,geomgroup,1,self.spacecraft_body_id,geom_id)

        if distance < 0.0 or geom_id[0] != self.terrain_geom_id:
            return None

        return distance, origin + distance * direction

    def terrain_normal_at(self, surface_point, down_direction):
        """
        Gets the normal direction to terrain as a point, used in the raycast_terrain process to find the surface contacts
        :param surface_point:
        :param down_direction:
        :return:
        """

        up_direction = -down_direction
        reference = np.array([1.0, 0.0, 0.0])

        if abs(np.dot(reference, down_direction)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])

        tangent_1 = np.cross(down_direction, reference)
        tangent_1 /= max(np.linalg.norm(tangent_1), 1e-12)

        tangent_2 = np.cross(down_direction, tangent_1)
        tangent_2 /= max(np.linalg.norm(tangent_2), 1e-12)

        sample_offset = self.surface_normal_sample_offset
        ray_start_offset = self.surface_probe_distance

        sample_points = []

        for tangent in (tangent_1, tangent_2):
            ray_origin = (surface_point +sample_offset * tangent - ray_start_offset * down_direction)
            hit = self.raycast_terrain(ray_origin, down_direction)

            if hit is None:
                return up_direction

            sample_points.append(hit[1])

        normal = np.cross(sample_points[0] - surface_point,sample_points[1] - surface_point)
        normal_norm = np.linalg.norm(normal)

        if normal_norm < 1e-12:
            return up_direction

        normal /= normal_norm

        if np.dot(normal, up_direction) < 0.0:
            normal *= -1.0

        return normal

    def spacecraft_foot_sample_points(self, geom_id):
        """
        Get the spacecraft foot points for contact finding
        :param geom_id:
        :return:
        """

        geom_type = self.model.geom_type[geom_id]

        if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
            return []

        center = self.data.geom_xpos[geom_id].copy()
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
        half_size = self.model.geom_size[geom_id].copy()

        sample_points = []

        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    local_corner = half_size * np.array([sx, sy, sz])
                    sample_points.append(center + rotation @ local_corner)

        return sample_points

    def find_mesh_surface_contacts(self):
        """
        Actually finds the surface contacts of the mesh
        :return:
        """

        contacts = []

        for geom_id in self.surface_contact_geom_ids:
            geom_name = mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom_id)

            for sample_point in self.spacecraft_foot_sample_points(geom_id):
                radial_norm = np.linalg.norm(sample_point)

                if radial_norm < 1e-12:
                    continue

                down_direction = -sample_point / radial_norm
                ray_origin = (sample_point-self.surface_probe_distance * down_direction)
                hit = self.raycast_terrain(ray_origin, down_direction)

                if hit is None:
                    continue

                distance, surface_point = hit
                clearance = distance - self.surface_probe_distance

                if clearance > self.surface_contact_margin:
                    continue

                normal = self.terrain_normal_at(surface_point,down_direction)

                contacts.append({
                    "geom_id": geom_id,
                    "geom_name": geom_name,
                    "pos": surface_point,
                    "normal": normal,
                    "clearance": clearance,
                    "penetration": max(0.0,self.surface_contact_margin - clearance)
                })

        return contacts

    def check_surface_collision(self):
        """
        Check if meshes collide with the surface
        :return:
        """

        self.collision = False
        self.contact_force_mjc[:] = 0.0
        self.contact_point_mjc[:] = 0.0
        self.contact_normal_mjc[:] = 0.0
        self.penetration = 0.0

        self._debug_step_counter += 1

        if not self.surface_contacts:
            return

        deepest_contact = max(self.surface_contacts,key=lambda contact: contact["penetration"])

        self.collision = True
        self.contact_point_mjc = deepest_contact["pos"].copy()
        self.contact_normal_mjc = deepest_contact["normal"].copy()
        self.penetration = deepest_contact["penetration"]

        if (
                self.debug_contacts
                and self._debug_step_counter % self.debug_contact_every_n_steps == 0
        ):
            print("\n" + "=" * 70)
            print("RAYCAST MESH CONTACT")
            print("=" * 70)
            print("Contact samples:", len(self.surface_contacts))
            print("Geom:", deepest_contact["geom_name"])
            print("Clearance:", deepest_contact["clearance"])
            print("Penetration:", deepest_contact["penetration"])
            print("\nContact position [MuJoCo]:")
            print(deepest_contact["pos"])
            print("\nContact normal [MuJoCo]:")
            print(deepest_contact["normal"])
            print("\n" + "=" * 70)

    def check_collision(self):
        """
        Fallback function that checks bounding box collision instead of mesh collision if there is an issue with the mesh collision
        :return:
        """

        self.collision = False
        self.contact_force_mjc[:] = 0.0
        self.contact_point_mjc[:] = 0.0
        self.contact_normal_mjc[:] = 0.0
        self.penetration = 0.0

        self._debug_step_counter += 1

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,geom1)

            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,geom2)

            spacecraft_contact = self.is_spacecraft_contact(geom1, geom2)

            if not spacecraft_contact:
                continue

            self.collision = True

            normal = contact.frame[:3].copy()
            normal /= max(np.linalg.norm(normal), 1e-12)

            penetration = max(0.0, -contact.dist)

            self.contact_point_mjc = contact.pos.copy()
            self.contact_normal_mjc = normal
            self.penetration = penetration

            if (
                    self.debug_contacts
                    and self._debug_step_counter % self.debug_contact_every_n_steps == 0
            ):

                print("\n" + "=" * 70)
                print("MUJOCO CONTACT")
                print("=" * 70)

                print("Contact index:", i)

                print(
                    "Geom 1:",
                    geom1,
                    name1
                )

                print(
                    "Geom 2:",
                    geom2,
                    name2
                )

                print("Contact distance:", contact.dist)
                print("Penetration:", penetration)

                print("\nContact position [MuJoCo]:")
                print(contact.pos)

                print("\nContact normal [MuJoCo]:")
                print(normal)

                print("\nFull contact frame:")
                print(contact.frame.reshape(3, 3))

                qpos_adr = self.spacecraft_qpos_adr

                spacecraft_pos = self.data.qpos[qpos_adr:qpos_adr + 3]

                print("\nSpacecraft COM:")
                print(spacecraft_pos)

                r_contact = contact.pos - spacecraft_pos

                print("\nCOM -> contact:")
                print(r_contact)

                print("COM -> contact distance:", np.linalg.norm(r_contact))

                radial = spacecraft_pos.copy()
                radial_norm = np.linalg.norm(radial)

                if radial_norm > 0.0:
                    radial /= radial_norm

                    print("\nRadial normal:")
                    print(radial)

                    print(
                        "Angle between radial and MuJoCo normal [deg]:",
                        np.degrees(np.arccos(np.clip(np.dot(radial, normal),-1.0,1.0))))


                contact_force = np.zeros(6)

                mujoco.mj_contactForce( self.model,self.data, i,contact_force)

                print("\nMuJoCo contact force:")
                print(contact_force)

                print("\n" + "=" * 70)

    def get_mesh_surface_contact_loads(self,r_spacecraft_mjc,v_spacecraft_mjc,omega_spacecraft_mjc,C_BN,R_mjc_to_bsk):
        """
        Get specific loads on the mesh surface for applying dynamics
        """

        F_mjc = np.zeros(3)
        torque_B = np.zeros(3)

        if not self.surface_contacts:
            return R_mjc_to_bsk @ F_mjc, torque_B

        active_contact_count = len(self.surface_contacts)
        omega_spacecraft_N = R_mjc_to_bsk @ omega_spacecraft_mjc
        omega_spacecraft_B = C_BN @ omega_spacecraft_N
        impact_contact_count = 0
        resting_contact_count = 0

        r_com_norm = np.linalg.norm(r_spacecraft_mjc)

        if r_com_norm > 1e-12:
            gravity_mjc = (-self.mu_moon * r_spacecraft_mjc/r_com_norm ** 3)
        else:
            gravity_mjc = np.zeros(3)

        for contact in self.surface_contacts:
            normal_mjc = contact["normal"].copy()
            normal_mjc /= max(np.linalg.norm(normal_mjc), 1e-12)
            r_contact_from_com_mjc = contact["pos"] - r_spacecraft_mjc
            contact_velocity_mjc = (v_spacecraft_mjc + np.cross(omega_spacecraft_mjc, r_contact_from_com_mjc))
            normal_velocity = np.dot(contact_velocity_mjc, normal_mjc)
            tangential_velocity_mjc = (contact_velocity_mjc-normal_velocity * normal_mjc)
            tangential_speed = np.linalg.norm(tangential_velocity_mjc)

            if (
                    contact["penetration"] <= self.resting_penetration_threshold
                    and
                    abs(normal_velocity) <= self.resting_velocity_threshold
                    and
                    tangential_speed <= self.resting_velocity_threshold
            ):
                resting_contact_count += 1

        for contact in self.surface_contacts:
            normal_mjc = contact["normal"].copy()
            normal_mjc /= max(np.linalg.norm(normal_mjc), 1e-12)
            penetration_raw = contact["penetration"]

            if penetration_raw <= 0.0:
                continue

            penetration = min(penetration_raw,self.max_contact_penetration)

            r_contact_from_com_mjc = contact["pos"] - r_spacecraft_mjc
            contact_velocity_mjc = (v_spacecraft_mjc + np.cross(omega_spacecraft_mjc,r_contact_from_com_mjc))
            normal_velocity = np.dot(contact_velocity_mjc, normal_mjc)
            tangential_velocity_mjc = (contact_velocity_mjc - normal_velocity * normal_mjc)
            tangential_speed = np.linalg.norm(tangential_velocity_mjc)

            gravity_normal = np.dot(gravity_mjc, normal_mjc)
            required_normal_force = max(0.0,-self.spacecraft_mass * gravity_normal)

            resting_contact = (
                    penetration <= self.resting_penetration_threshold
                    and
                    abs(normal_velocity) <= self.resting_velocity_threshold
                    and
                    tangential_speed <= self.resting_velocity_threshold
            )

            if resting_contact:
                force_magnitude = (required_normal_force -self.contact_damping * normal_velocity)
                force_magnitude = max(0.0, force_magnitude)
                force_magnitude = min(force_magnitude,self.max_contact_force)
                force_magnitude /= max(resting_contact_count, 1)
            else:
                impact_contact_count += 1
                impact_penetration = min(penetration_raw,self.max_impact_contact_penetration)
                force_magnitude = (self.contact_stiffness * impact_penetration - self.contact_damping * normal_velocity)
                force_magnitude = max(0.0, force_magnitude)
                force_magnitude = min(force_magnitude,self.max_contact_force)
                force_magnitude /= max(active_contact_count, 1)

            force_mjc = force_magnitude * normal_mjc
            tangential_force_mjc = np.zeros(3)
            friction_limit = (self.contact_friction_coefficient * force_magnitude)

            if tangential_speed > 1e-8 and friction_limit > 0.0:
                tangential_direction = (tangential_velocity_mjc/tangential_speed)
                tangential_force_mjc = (-friction_limit * tangential_direction)

                if self.contact_tangential_damping > 0.0:
                    tangential_force_mjc += (-self.contact_tangential_damping*tangential_velocity_mjc)

                tangential_force_norm = np.linalg.norm(tangential_force_mjc)

                if tangential_force_norm > friction_limit:
                    tangential_force_mjc *= (friction_limit/max(tangential_force_norm, 1e-12))

            if resting_contact:
                com_normal_velocity = np.dot(v_spacecraft_mjc, normal_mjc)
                com_tangential_velocity_mjc = (v_spacecraft_mjc - com_normal_velocity * normal_mjc)
                gravity_tangential_mjc = (gravity_mjc - gravity_normal * normal_mjc)
                static_settle_force_mjc = (
                            -self.spacecraft_mass * gravity_tangential_mjc - self.resting_tangential_damping * com_tangential_velocity_mjc)
                static_settle_force_mjc /= max(resting_contact_count, 1)
                tangential_force_mjc += static_settle_force_mjc

                tangential_force_norm = np.linalg.norm(tangential_force_mjc)

                if tangential_force_norm > friction_limit:
                    tangential_force_mjc *= (friction_limit / max(tangential_force_norm, 1e-12))

            force_mjc += tangential_force_mjc
            F_mjc += force_mjc

            if self.apply_contact_torque:
                r_contact_from_com_N = (R_mjc_to_bsk @ r_contact_from_com_mjc)
                force_N = R_mjc_to_bsk @ force_mjc
                torque_N = np.cross(r_contact_from_com_N, force_N)
                torque_B += C_BN @ torque_N

            self.contact_force_mjc = force_mjc.copy()
            self.contact_point_mjc = contact["pos"].copy()
            self.contact_normal_mjc = normal_mjc.copy()
            self.penetration = penetration

        if self.apply_contact_torque and resting_contact_count > 0:
            omega_mag = np.linalg.norm(omega_spacecraft_B)

            if omega_mag > self.resting_angular_velocity_threshold:
                damping_torque_B = (-self.resting_angular_damping * omega_spacecraft_B)
                damping_torque_norm = np.linalg.norm(damping_torque_B)

                if damping_torque_norm > self.max_resting_damping_torque:
                    damping_torque_B *= (self.max_resting_damping_torque / max(damping_torque_norm, 1e-12))

                torque_B += damping_torque_B

        max_contact_torque = (self.max_impact_contact_torque if impact_contact_count > 0 else self.max_contact_torque)
        torque_norm = np.linalg.norm(torque_B)

        if torque_norm > max_contact_torque:
            torque_B *= max_contact_torque / torque_norm

        F_N = R_mjc_to_bsk @ F_mjc

        return F_N, torque_B

    def get_contact_loads(self, r_spacecraft_mjc, v_spacecraft_mjc, omega_spacecraft_mjc, C_BN, R_mjc_to_bsk):
        """
        Fallback get contact loads, meant if mesh contact loads doesnt work then it does default calculation, otherwise goes to get_mesh_surface_contact_loads
        :param r_spacecraft_mjc:
        :param v_spacecraft_mjc:
        :param omega_spacecraft_mjc:
        :param C_BN:
        :param R_mjc_to_bsk:
        :return:
        """
        if self.use_mesh_surface_contacts:
            return self.get_mesh_surface_contact_loads(r_spacecraft_mjc, v_spacecraft_mjc, omega_spacecraft_mjc, C_BN,R_mjc_to_bsk)

        F_mjc = np.zeros(3)
        torque_B = np.zeros(3)
        omega_spacecraft_N = R_mjc_to_bsk @ omega_spacecraft_mjc
        omega_spacecraft_B = C_BN @ omega_spacecraft_N

        resting_contact_count = 0
        impact_contact_count = 0

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            if not self.is_spacecraft_contact(geom1, geom2):
                continue

            normal_mjc = contact.frame[:3].copy()
            normal_norm = np.linalg.norm(normal_mjc)

            if normal_norm < 1e-12:
                continue

            normal_mjc /= normal_norm

            if self.is_spacecraft_geom(geom1):
                normal_mjc *= -1.0

            penetration_raw = max(0.0, -contact.dist)

            if penetration_raw <= 0.0:
                continue

            penetration = min(penetration_raw, self.max_contact_penetration)

            spacecraft_com_mjc = self.data.qpos[self.spacecraft_qpos_adr:self.spacecraft_qpos_adr + 3]

            r_contact_from_com_mjc = (contact.pos - spacecraft_com_mjc)

            normal_lever_arm = np.dot(r_contact_from_com_mjc, normal_mjc)

            if normal_lever_arm > 0.0:
                r_contact_from_com_mjc -= (2.0 * normal_lever_arm * normal_mjc)

            contact_velocity_mjc = (v_spacecraft_mjc + np.cross(omega_spacecraft_mjc, r_contact_from_com_mjc))

            normal_velocity = np.dot(contact_velocity_mjc, normal_mjc)
            tangential_velocity_mjc = (contact_velocity_mjc - normal_velocity * normal_mjc)
            tangential_speed = np.linalg.norm(tangential_velocity_mjc)

            if (penetration <= self.resting_penetration_threshold and abs(normal_velocity) <= self.resting_velocity_threshold and tangential_speed <= self.resting_velocity_threshold):
                resting_contact_count += 1

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            if not self.is_spacecraft_contact(geom1, geom2):
                continue



            normal_mjc = contact.frame[:3].copy()

            normal_norm = np.linalg.norm(normal_mjc)

            if normal_norm < 1e-12:
                continue

            normal_mjc /= normal_norm

            if self.is_spacecraft_geom(geom1):
                # MuJoCo normal is spacecraft -> terrain
                # Reverse it so force points terrain -> spacecraft.
                normal_mjc *= -1.0

            elif self.is_spacecraft_geom(geom2):
                # MuJoCo normal is terrain -> spacecraft.
                # Already correct.
                pass

            # ==============================================================
            # PENETRATION
            # ==============================================================

            penetration_raw = max(0.0, -contact.dist)

            if penetration_raw <= 0.0:
                continue



            penetration = min(penetration_raw, self.max_contact_penetration)

            # ==============================================================
            # CONTACT POINT RELATIVE TO SPACECRAFT COM
            # ==============================================================

            spacecraft_com_mjc = self.data.qpos[
                self.spacecraft_qpos_adr:
                self.spacecraft_qpos_adr + 3
            ]

            r_contact_from_com_mjc = (contact.pos - spacecraft_com_mjc)

            normal_lever_arm = np.dot(r_contact_from_com_mjc, normal_mjc)

            if normal_lever_arm > 0.0:
                r_contact_from_com_mjc -= (2.0 * normal_lever_arm * normal_mjc)

            # ==============================================================
            # CONTACT POINT VELOCITY
            # ==============================================================

            contact_velocity_mjc = (v_spacecraft_mjc + np.cross(omega_spacecraft_mjc, r_contact_from_com_mjc))


            normal_velocity = np.dot(contact_velocity_mjc, normal_mjc)

            # ==============================================================
            # LUNAR GRAVITY AT SPACECRAFT COM
            # ==============================================================

            r_com_norm = np.linalg.norm(spacecraft_com_mjc)

            if r_com_norm > 1e-12:

                gravity_mjc = (-self.mu_moon * spacecraft_com_mjc / r_com_norm ** 3)

            else:

                gravity_mjc = np.zeros(3)

            # ==============================================================
            # NORMAL CONTACT FORCE
            # ==============================================================



            gravity_normal = np.dot(gravity_mjc, normal_mjc)

            # Positive value means gravity pulls INTO the terrain.
            required_normal_force = max(0.0, -self.spacecraft_mass * gravity_normal)

            # ==============================================================
            # RESTING CONTACT
            # ==============================================================

            tangential_velocity_mjc = (contact_velocity_mjc - normal_velocity * normal_mjc)

            tangential_speed = np.linalg.norm(tangential_velocity_mjc)

            resting_contact = (penetration <= self.resting_penetration_threshold and abs(
                normal_velocity) <= self.resting_velocity_threshold and tangential_speed <= self.resting_velocity_threshold)

            if not resting_contact:
                impact_contact_count += 1

            if resting_contact:


                force_magnitude = (required_normal_force - self.contact_damping * normal_velocity)

                force_magnitude = max(0.0, force_magnitude)

                force_magnitude = min(force_magnitude, self.max_contact_force)

                force_magnitude = (force_magnitude / max(resting_contact_count, 1))

            else:

                # ----------------------------------------------------------
                # IMPACT / PENETRATION RESPONSE
                # ----------------------------------------------------------

                impact_penetration = min(penetration_raw, self.max_impact_contact_penetration)

                force_magnitude = (self.contact_stiffness * impact_penetration - self.contact_damping * normal_velocity)

                force_magnitude = max(0.0, force_magnitude)

                force_magnitude = min(force_magnitude, self.max_contact_force)

            force_mjc = (force_magnitude * normal_mjc)

            print("\nCONTACT FORCE CONTRIBUTION")
            print("contact index:", i)
            print("geom1:", geom1)
            print("geom2:", geom2)
            print("dist:", contact.dist)
            print("penetration:", penetration)
            print("normal:", normal_mjc)
            print("normal velocity:", normal_velocity)
            print("force magnitude:", force_magnitude)
            print("force MJC:", force_mjc)

            # ==============================================================
            # TANGENTIAL FRICTION
            # ==============================================================

            com_normal_velocity = np.dot(v_spacecraft_mjc, normal_mjc)
            com_tangential_velocity_mjc = (v_spacecraft_mjc - com_normal_velocity * normal_mjc)

            tangential_force_mjc = np.zeros(3)
            friction_limit = (self.contact_friction_coefficient * force_magnitude)

            # ==============================================================
            # STATIC / SLIDING FRICTION
            # ==============================================================

            if tangential_speed > 1e-8:

                tangential_direction = (tangential_velocity_mjc / tangential_speed)

                # ----------------------------------------------------------
                # SLIDING FRICTION
                # ----------------------------------------------------------

                tangential_force_mjc = (-friction_limit * tangential_direction)

                # ----------------------------------------------------------
                # Optional viscous damping
                # ----------------------------------------------------------

                if self.contact_tangential_damping > 0.0:
                    viscous_force = (-self.contact_tangential_damping * tangential_velocity_mjc)

                    tangential_force_mjc += viscous_force

                # ----------------------------------------------------------
                # Coulomb friction limit
                # ----------------------------------------------------------

                tangential_force_norm = np.linalg.norm(tangential_force_mjc)

                if tangential_force_norm > friction_limit:
                    tangential_force_mjc *= (friction_limit / max(tangential_force_norm, 1e-12))

            # ==============================================================
            # RESTING VELOCITY SNAP
            # ==============================================================

            if resting_contact:
                gravity_tangential_mjc = (gravity_mjc - gravity_normal * normal_mjc)
                static_settle_force_mjc = (
                            -self.spacecraft_mass * gravity_tangential_mjc - self.resting_tangential_damping * com_tangential_velocity_mjc)

                static_settle_force_mjc /= max(resting_contact_count, 1)

                if 1e-10 < tangential_speed < self.settle_velocity_threshold:
                    settle_force_mjc = (-self.settle_damping * tangential_velocity_mjc)

                    settle_force_norm = np.linalg.norm(settle_force_mjc)

                    settle_limit = min(friction_limit, self.max_settle_force / max(resting_contact_count, 1))

                    if settle_force_norm > settle_limit:
                        settle_force_mjc *= (settle_limit / max(settle_force_norm, 1e-12))

                    tangential_force_mjc = settle_force_mjc

                tangential_force_mjc += static_settle_force_mjc

                tangential_force_norm = np.linalg.norm(tangential_force_mjc)

                if tangential_force_norm > friction_limit:
                    tangential_force_mjc *= (friction_limit / max(tangential_force_norm, 1e-12))

            # --------------------------------------------------------------
            # TOTAL CONTACT FORCE
            # --------------------------------------------------------------

            force_mjc += tangential_force_mjc

            # --------------------------------------------------------------
            # LINEAR FORCE
            # --------------------------------------------------------------

            F_mjc += force_mjc



            if self.apply_contact_torque and resting_contact:
                r_friction_from_com_mjc = (np.dot(r_contact_from_com_mjc, normal_mjc) * normal_mjc)

                r_friction_from_com_N = (R_mjc_to_bsk @ r_friction_from_com_mjc)

                friction_force_N = (R_mjc_to_bsk @ tangential_force_mjc)

                torque_N = np.cross(r_friction_from_com_N, friction_force_N)

                torque_B += (C_BN @ torque_N)

            elif self.apply_contact_torque:
                r_contact_from_com_N = (R_mjc_to_bsk @ r_contact_from_com_mjc)

                force_N = (R_mjc_to_bsk @ force_mjc)

                torque_N = np.cross(r_contact_from_com_N, force_N)

                torque_B += (C_BN @ torque_N)

            # ==============================================================
            # STORE DEBUG STATE
            # ==============================================================

            self.collision = True
            self.contact_force_mjc = force_mjc.copy()
            self.contact_point_mjc = contact.pos.copy()
            self.contact_normal_mjc = normal_mjc.copy()
            self.penetration = penetration

        # ==============================================================
        # RESTING ROTATIONAL DAMPING
        # ==============================================================

        if self.apply_contact_torque and resting_contact_count > 0:

            omega_mag = np.linalg.norm(omega_spacecraft_B)

            if omega_mag > self.resting_angular_velocity_threshold:
                inertia_B = np.array(self.spacecraft_inertia_B, dtype=float)
                angular_momentum_B = inertia_B @ omega_spacecraft_B
                angular_momentum_norm = np.linalg.norm(angular_momentum_B)
                angular_damping = self.resting_angular_damping

                if omega_mag < self.resting_angular_settle_threshold:
                    angular_damping = self.resting_angular_settle_damping

                damping_torque_B = (-angular_damping * omega_spacecraft_B)

                damping_torque_norm = np.linalg.norm(damping_torque_B)
                dt = max(self._time_step_s, 1.0e-9)
                stopping_torque_limit = angular_momentum_norm / dt
                damping_torque_limit = min(self.max_resting_damping_torque, stopping_torque_limit)

                if damping_torque_norm > damping_torque_limit:
                    damping_torque_B *= (damping_torque_limit / max(damping_torque_norm, 1e-12))

                torque_B += damping_torque_B

        if (self.apply_contact_torque and resting_contact_count > 0):
            omega_mag = np.linalg.norm(omega_spacecraft_B)

            if (self.resting_angular_velocity_threshold < omega_mag < self.resting_angular_settle_threshold):
                omega_direction_B = (omega_spacecraft_B / omega_mag)
                torque_along_omega = np.dot(torque_B, omega_direction_B)

                if torque_along_omega < 0.0:
                    torque_B = (torque_along_omega * omega_direction_B)
                else:
                    torque_B[:] = 0.0

                inertia_B = np.array(self.spacecraft_inertia_B, dtype=float)
                angular_momentum_B = inertia_B @ omega_spacecraft_B
                stopping_torque_limit = (np.linalg.norm(angular_momentum_B) / max(self._time_step_s, 1.0e-9))
                torque_norm = np.linalg.norm(torque_B)

                if torque_norm > stopping_torque_limit:
                    torque_B *= (stopping_torque_limit / max(torque_norm, 1e-12))

        # ==============================================================
        # TORQUE LIMIT
        # ==============================================================

        # print(" TOTAL CONTACT FORCE")
        # print("F_mjc:", F_mjc)
        # print("F_N:", R_mjc_to_bsk @ F_mjc)
        # print("magnitude:", np.linalg.norm(F_mjc))

        max_contact_torque = (
            self.max_impact_contact_torque if impact_contact_count > 0 else self.max_contact_torque)
        torque_before_clamp = torque_B.copy()
        torque_norm = np.linalg.norm(torque_B)

        if torque_norm > max_contact_torque:
            torque_B *= (max_contact_torque / torque_norm)

        print(" TORQUE DEBUG")

        print("torque before clamp:")
        print(torque_before_clamp)

        print("torque magnitude:")
        print(np.linalg.norm(torque_B))

        print("omega_B:")
        print(omega_spacecraft_B)

        if np.linalg.norm(omega_spacecraft_B) > 1e-12:
            print("torque dot omega:", np.dot(torque_B, omega_spacecraft_B))

        # ==============================================================
        # MUJOCO -> BASILISK
        # ==============================================================

        F_N = R_mjc_to_bsk @ F_mjc

        return F_N, torque_B

    def publish_contact_loads(self, force_N, torque_B, CurrentSimNanos):

        force_payload = self.forceOutMsg.zeroMsgPayload
        force_payload.forceRequestInertial = [
            force_N[0],
            force_N[1],
            force_N[2]
        ]
        self.forceOutMsg.write(force_payload, CurrentSimNanos, self.moduleID)

        torque_payload = self.torqueOutMsg.zeroMsgPayload
        torque_payload.torqueRequestBody = [
            torque_B[0],
            torque_B[1],
            torque_B[2]
        ]
        self.torqueOutMsg.write(torque_payload, CurrentSimNanos, self.moduleID)