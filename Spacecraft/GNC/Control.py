"""Actuator allocation, closed/open-loop commands, and contact diagnostics."""
import math
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import RigidBodyKinematics
from Spacecraft.GNC.Guidance import DescentState, Guidance, gravity, coast_state, a2g_acceleration, vector3, G0
from Spacecraft.Thrusters.Thrusters import MainEngine, RCS_thruster, make_config


class ActuatorCommand:
    def __init__(self, main_thrust,pitch,yaw,rcs_thrusts,force_B,torque_B,mass_flow,requested_force_N,saturated):
        self.main_thrust = main_thrust
        self.pitch = pitch
        self.yaw = yaw
        self.rcs_thrusts = rcs_thrusts
        self.force_B = force_B
        self.torque_B = torque_B
        self.mass_flow = mass_flow
        self.requested_force_N = requested_force_N
        self.saturated = saturated


class DescentAllocator:
    """Attitude PD plus constrained gimbal/RCS allocation.

    Main gimbal lateral-force columns are linearized for allocation, then
    converted to actual limited angles. Propagation uses the resulting REAL
    force and moment. RCS levels represent averaged duty-cycle thrust.
    """
    def __init__(self, main_engine, rcs_engines, inertia_B, settings):
        self.settings = settings
        self.inertia = np.asarray(inertia_B, dtype=float).reshape(3, 3)
        if (not np.all(np.isfinite(self.inertia)) or
                not np.allclose(self.inertia, self.inertia.T) or
                np.min(np.linalg.eigvalsh(self.inertia)) <= 0):
            raise ValueError("Inertia must be finite, symmetric and positive definite.")
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.mount = main_engine._mount_dcm_BM.copy()
        self.location = vector3(main_engine.config.thrLoc_B)
        self.limit = main_engine.max_gimbal
        if self.limit is None or not 0 < self.limit < np.pi / 2:
            raise ValueError("Descent requires a finite gimbal limit between 0 and pi/2.")
        if settings.max_thrust > main_engine.config.MaxThrust:
            raise ValueError("Descent Tmax exceeds the configured main-engine rating.")
        self.main_isp = float(main_engine.config.steadyIsp)
        self.rcs_names = list(rcs_engines)
        self.rcs_max = np.array([e.config.MaxThrust for e in rcs_engines.values()])
        self.rcs_isp = np.array([e.config.steadyIsp for e in rcs_engines.values()])
        self.directions = np.array([vector3(e.config.thrDir_B) for e in rcs_engines.values()]).reshape(-1, 3).T
        self.moments = np.array([np.cross(vector3(e.config.thrLoc_B), vector3(e.config.thrDir_B)) for e in rcs_engines.values()]).reshape(-1, 3).T
        if self.main_isp <= 0 or np.any(self.rcs_isp <= 0) or np.any(self.rcs_max <= 0):
            raise ValueError("Engine ratings and Isp must be positive.")
        self.force_columns = np.column_stack((self.mount[:, 1:], self.directions))
        self.torque_columns = np.column_stack((
            np.cross(self.location, self.mount[:, 1]),
            np.cross(self.location, self.mount[:, 2]), self.moments))
        self.allocation_matrix = np.vstack((self.torque_columns,settings.force_tracking_weight * self.force_columns))
        wrenches = np.vstack((self.directions, self.moments))
        self.opposed_rcs_pairs = [(i, j) for i in range(len(self.rcs_max))
                                  for j in range(i + 1, len(self.rcs_max))
                                  if np.allclose(wrenches[:, i], -wrenches[:, j], rtol=0, atol=1e-12)]

    def cancel_opposed_rcs(self, levels):
        """Remove equal/opposite firings without changing force or torque."""
        levels = np.asarray(levels, dtype=float).copy()
        for i, j in self.opposed_rcs_pairs:
            cancelled = min(levels[i], levels[j])
            levels[i] -= cancelled
            levels[j] -= cancelled
        return levels

    def command(self, state, target_state, time_to_go, *, main_enabled=True):
        cfg = self.settings
        acceleration = a2g_acceleration(state.position, state.velocity, *target_state, max(time_to_go, cfg.terminal_guard_s), cfg.mu)
        requested = state.mass * acceleration
        requested_B = state.dcm_NB.T @ requested
        magnitude = np.linalg.norm(requested_B)
        # The engine can push only along its current axis. Using ||T_requested||
        # turns a downward/sideways correction into upward thrust when terminal
        # attitude is upright. Project onto the realizable axis instead.
        nominal_axis = self.mount[:, 0]
        axial_request = float(np.dot(requested_B, nominal_axis))
        thrust = float(np.clip(axial_request, cfg.min_thrust, cfg.max_thrust)) if main_enabled else 0.0
        desired_axis = requested_B / magnitude if magnitude > 1e-12 else self.mount[:, 0]
        # Attitude must track the translational force reference through approach.
        # A clock-only blend to vertical discards needed braking/position control;
        # the body becomes upright naturally when the required force is vertical.
        # Phase times label telemetry; a separate staged trajectory is required
        # to prescribe a particular pitchover maneuver or terminal descent rate.
        axis = np.cross(nominal_axis, desired_axis)
        sine = np.linalg.norm(axis)
        cosine = np.clip(np.dot(nominal_axis, desired_axis), -1, 1)
        error = (axis * np.arctan2(sine, cosine) / sine if sine > 1e-10 else
                 (self.mount[:, 1] * np.pi if cosine < 0 else np.zeros(3)))
        # The roll degree of freedom is damped, without prescribing an arbitrary roll attitude.
        torque = self.inertia @ (cfg.attitude_frequency**2 * error - 2 * cfg.attitude_damping * cfg.attitude_frequency * state.omega_B)
        torque += np.cross(state.omega_B, self.inertia @ state.omega_B)
        nominal_force = thrust * nominal_axis
        rhs = np.concatenate((torque - np.cross(self.location, nominal_force),
                              cfg.force_tracking_weight * (requested_B - nominal_force)))
        # Tiny nonzero bounds keep scipy's strictly ordered bounds valid while off.
        lateral_limit = max(thrust * np.tan(self.limit), 1e-12)
        lower = np.concatenate(([-lateral_limit] * 2, np.zeros(len(self.rcs_max))))
        upper = np.concatenate(([lateral_limit] * 2, self.rcs_max))
        solution = lsq_linear(self.allocation_matrix, rhs, bounds=(lower, upper),
                              method="bvls", tol=1e-8, max_iter=30)
        if not solution.success:
            raise RuntimeError("Gimbal/RCS allocation did not converge.")
        local = np.array([thrust, solution.x[0], solution.x[1]])
        norm = np.linalg.norm(local)
        local = local / norm if norm > 1e-12 else np.array([1., 0., 0.])
        pitch = float(np.clip(-np.arcsin(np.clip(local[2], -1, 1)), -self.limit, self.limit))
        yaw = float(np.clip(np.arctan2(local[1], local[0]), -self.limit, self.limit))
        actual_direction = self.mount @ np.array([np.cos(pitch) * np.cos(yaw),
                                                  np.cos(pitch) * np.sin(yaw), -np.sin(pitch)])
        main_force = thrust * actual_direction
        rcs = self.cancel_opposed_rcs(np.clip(solution.x[2:], 0, self.rcs_max))
        force = main_force + self.directions @ rcs
        moment = np.cross(self.location, main_force) + self.moments @ rcs
        flow = thrust / (self.main_isp * G0) + np.sum(rcs / (self.rcs_isp * G0))
        return ActuatorCommand(thrust, pitch, yaw, rcs, force, moment, float(flow), requested,
                               bool(main_enabled and (axial_request > cfg.max_thrust or axial_request < cfg.min_thrust)))

class Control(sysModel.SysModel):
    def __init__(self, main_engine, rcs_engines, settings, target, initial_state,
                 inertia_B, period_s, state_msg, fuel_msg, terrain_map=None, landing_foot_points_B=None):
        super().__init__()
        self.ModelTag = "A2GControl"
        if not np.isfinite(period_s) or period_s <= 0:
            raise ValueError("Controller period must be positive.")
        self.period_s = float(period_s)
        self.settings = settings
        self.target = target
        self.initial_state = initial_state.copy()
        self.main_engine = main_engine
        self.rcs_engines = dict(rcs_engines)
        self.engines = {main_engine.name: main_engine, **self.rcs_engines}
        if len(self.engines) != 1 + len(self.rcs_engines):
            raise ValueError("Main and RCS names must be distinct.")
        for engine in self.engines.values():
            if (len(engine.config.ThrusterOnRamp) or len(engine.config.ThrusterOffRamp)
                    or len(engine.config.thrBlowDownCoeff) or len(engine.config.ispBlowDownCoeff)):
                raise ValueError("Descent currently requires engines without ramps or blow-down.")
        self.allocator = DescentAllocator(main_engine, self.rcs_engines, inertia_B, settings)
        self.planner = Guidance(self.allocator, target, terrain_map, landing_foot_points_B)
        self.pdi_state = coast_state(initial_state, settings.coast_duration_s, settings.mu,
                                     settings.surface_radius)
        self.plan = self.planner.optimize(self.pdi_state, settings.coast_duration_s)
        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.scStateInMsg.subscribeTo(state_msg)
        self.fuelInMsg = messaging.FuelTankMsgReader()
        self.fuelInMsg.subscribeTo(fuel_msg)
        self.command_msgs = {}
        for name, engine in self.engines.items():
            self.command_msgs[name] = messaging.THRArrayOnTimeCmdMsg()
            engine.subscribe_to_commands(self.command_msgs[name])
        # Inertial requested/applied forces are also available as Basilisk messages.
        self.requestedForceOutMsg = messaging.CmdForceInertialMsg()
        self.appliedForceOutMsg = messaging.CmdForceInertialMsg()
        self.history = []
        self.status = "planned"
        self.last_command = None
        self.start_s = 0.0
        self.mission_start_s = None  # original mission epoch when skipping the unpowered coast
        self.reset_s = 0.0
        self.navigation = None
        self.contact_sensor = None
        self.main_engine_cutoff = False
        self.cutoff_event = None
        self.terminal_start_s = None

    def Reset(self, CurrentSimNanos):
        self.reset_s = CurrentSimNanos * 1e-9
        self.start_s = self.reset_s if self.mission_start_s is None else self.mission_start_s
        if self.start_s != 0 and self.mission_start_s is None:
            self.plan = self.planner.optimize(self.pdi_state, self.start_s + self.settings.coast_duration_s)
        ignition = self.start_s + self.settings.coast_duration_s
        self.status = "coast" if self.reset_s < ignition else "descending"
        self.history.clear()
        self.last_command = None
        self.main_engine_cutoff = False
        self.cutoff_event = None
        self.terminal_start_s = None
        self._off(CurrentSimNanos)

    def _off(self, time_ns):
        for name, engine in self.engines.items():
            engine.config.MaxThrust = 0.0
            payload = messaging.THRArrayOnTimeCmdMsgPayload()
            payload.OnTimeRequest = [0.0]
            self.command_msgs[name].write(payload, time_ns, self.moduleID)
        zero = messaging.CmdForceInertialMsgPayload()
        self.requestedForceOutMsg.write(zero, time_ns, self.moduleID)
        self.appliedForceOutMsg.write(zero, time_ns, self.moduleID)

    def _state(self, time_s):
        state = self.initial_state.copy()
        if self.navigation is not None:
            message = self.navigation.attitudeOutMsg.read()
            state.dcm_NB = RigidBodyKinematics.MRP2C(message.sigma_BN).T
            state.omega_B = np.asarray(message.omega_BN_B, dtype=float)
            attitude_age = max(0., time_s - message.timeTag)
            age = 0.
        elif self.scStateInMsg.isWritten():
            message = self.scStateInMsg()
            state.position = np.asarray(message.r_BN_N, dtype=float)
            state.velocity = np.asarray(message.v_BN_N, dtype=float)
            state.dcm_NB = RigidBodyKinematics.MRP2C(message.sigma_BN).T
            state.omega_B = np.asarray(message.omega_BN_B, dtype=float)
            age = max(0., time_s - self.scStateInMsg.timeWritten() * 1e-9)
            attitude_age = age
        else:
            age = 0.
            attitude_age = 0.
        if self.fuelInMsg.isWritten():
            fuel = float(self.fuelInMsg().fuelMass)
            fuel_age = max(0., time_s - self.fuelInMsg.timeWritten() * 1e-9)
            if self.last_command is not None:
                fuel -= self.last_command.mass_flow * fuel_age
            state.mass = self.settings.dry_mass + max(0., fuel)
        if self.navigation is not None:
            message = self.navigation.navOutMsg.read()
            state.position = np.asarray(message.r_BN_N,dtype=float)
            state.velocity = np.asarray(message.v_BN_N,dtype=float)
            age = max(0.,time_s-message.timeTag)
        # State messages from the hub precede this controller tick. Propagate
        # their timestamp to the command epoch using the previous applied load.
        if age:
            accel = gravity(state.position, self.settings.mu)
            if self.last_command is not None:
                accel += state.dcm_NB @ self.last_command.force_B / state.mass
            state.position += age * state.velocity + 0.5 * age**2 * accel
            state.velocity += age * accel
        if attitude_age:
            state.dcm_NB = state.dcm_NB @ Rotation.from_rotvec(attitude_age * state.omega_B).as_matrix()
        return state

    def _lowest_foot_clearance(self, state):
        """Estimated vertical foot-to-terrain clearance, using calibrated geometry."""
        terrain = self.planner.terrain_map
        points_B = getattr(self.contact_sensor, 'landing_foot_points_B', None)
        if terrain is None or points_B is None or len(points_B) == 0:
            return None
        return self.planner.lowest_foot_clearance(state, points_B)

    def UpdateState(self, CurrentSimNanos):
        if self.status not in ("coast", "descending", "free_fall"):
            self._off(CurrentSimNanos)
            return
        now = CurrentSimNanos * 1e-9
        elapsed = now - self.start_s - self.settings.coast_duration_s
        if elapsed < -1e-9:
            self._off(CurrentSimNanos)
            return
        self.status = "free_fall" if self.main_engine_cutoff else "descending"
        if self.navigation is not None and not self.navigation.healthy:
            # Control precedes sensors on the first task tick. Allow a short
            # startup interval for the first absolute attitude fix.
            if now - self.reset_s <= 2 * self.period_s + 1e-9:
                self._off(CurrentSimNanos)
                return
            self.status = 'navigation_invalid'
            self._off(CurrentSimNanos)
            return
        state = self._state(now)
        target_state = self.target.state(self.start_s + self.settings.coast_duration_s + self.plan.flight_time)
        er = float(np.linalg.norm(state.position - target_state[0]))
        ev = float(np.linalg.norm(state.velocity - target_state[1]))
        if elapsed >= self.plan.flight_time and not self.settings.engine_cutoff_m:
            self.status = ("terminal_constraints_met" if er <= self.settings.position_tolerance
                           and ev <= self.settings.velocity_tolerance else "terminal_miss")
            self._off(CurrentSimNanos)
            self.history.append({"time_s": now, "mass": state.mass, "status": self.status,
                                 "position_error": er, "velocity_error": ev})
            return
        try:
            guidance_target,tgo,phase = self.planner.guidance_target(
                self.start_s+self.settings.coast_duration_s,self.plan.flight_time,elapsed,state)
        except ValueError:
            self.status = 'terrain_map_invalid'
            self._off(CurrentSimNanos)
            self.history.append(dict(time_s=now, mass=state.mass, status=self.status,
                main_thrust=0., main_engine_cutoff=self.main_engine_cutoff, lowest_foot_clearance_m=None))
            return
        if phase == 'terminal' and self.terminal_start_s is None:
            self.terminal_start_s = now
        if self.contact_sensor is not None and self.contact_sensor.collision:
            # The planned zero-speed endpoint is above terrain. Touchdown after
            # the commanded drop is assessed independently by diagnostics.
            self.status = ('landed' if self.main_engine_cutoff and phase == 'terminal' and er < 3.
                           else 'landed' if not self.settings.engine_cutoff_m and phase == 'terminal' and ev < .5 and er < 3.
                           else 'terrain_contact_before_landing')
            self._off(CurrentSimNanos)
            self.history.append(dict(time_s=now,mass=state.mass,status=self.status,
                                     position_error=er,velocity_error=ev))
            return
        foot_clearance = None
        if phase == 'terminal' and self.settings.engine_cutoff_m > 0:
            foot_clearance = self._lowest_foot_clearance(state)
            if foot_clearance is not None and foot_clearance <= self.settings.engine_cutoff_m + 1e-9:
                self.main_engine_cutoff = True
            elif elapsed >= self.plan.flight_time:
                endpoint_error = float(np.linalg.norm(state.position - guidance_target[0]))
                # Integration/estimator roundoff can leave a nearly stationary
                # vehicle millimetres above threshold at the planned endpoint.
                if (foot_clearance is not None
                        and foot_clearance <= self.settings.engine_cutoff_m + self.settings.surface_tolerance_m
                        and endpoint_error <= self.settings.position_tolerance
                        and np.linalg.norm(state.velocity - guidance_target[1]) <= self.settings.velocity_tolerance):
                    self.main_engine_cutoff = True
            if self.main_engine_cutoff and self.cutoff_event is None:
                up = (self.planner.terrain_map.up_N if hasattr(self.planner.terrain_map, 'up_N')
                      else state.position / np.linalg.norm(state.position))
                self.cutoff_event = dict(time_s=now, lowest_foot_clearance_m=foot_clearance,
                    vertical_velocity_m_s=float(np.dot(state.velocity, up)),
                    velocity_N_m_s=state.velocity.tolist())
                self.status = 'free_fall'
        if elapsed > self.plan.flight_time + 10. and not self.main_engine_cutoff:
            self.status = 'cutoff_target_miss'
            self._off(CurrentSimNanos)
            return
        # Allocate with the main engine already disabled so applied force,
        # torque and fuel flow describe the actual RCS-only command.
        command = self.allocator.command(state, guidance_target, tgo,
                                         main_enabled=not self.main_engine_cutoff)
        # Reserve two controller ticks of fuel so finite-step integration cannot
        # consume past dry mass. No clipped thrust is presented as an optimal plan.
        if state.mass - self.settings.dry_mass <= 2 * self.period_s * command.mass_flow:
            self.status = "fuel_depleted"
            self._off(CurrentSimNanos)
            self.history.append({"time_s": now, "mass": state.mass, "status": self.status,
                                 "position_error": er, "velocity_error": ev})
            return
        self.main_engine.set_gimbal(command.pitch, command.yaw)
        levels = {self.main_engine.name: command.main_thrust}
        levels.update(zip(self.allocator.rcs_names, command.rcs_thrusts))
        for name, thrust in levels.items():
            engine = self.engines[name]
            engine.config.MaxThrust = float(thrust)
            payload = messaging.THRArrayOnTimeCmdMsgPayload()
            payload.OnTimeRequest = [max(2 * self.period_s, engine.config.MinOnTime) if thrust > 1e-12 else 0.0]
            self.command_msgs[name].write(payload, CurrentSimNanos, self.moduleID)
        requested = messaging.CmdForceInertialMsgPayload()
        requested.forceRequestInertial = command.requested_force_N.tolist()
        applied = messaging.CmdForceInertialMsgPayload()
        applied.forceRequestInertial = (state.dcm_NB @ command.force_B).tolist()
        self.requestedForceOutMsg.write(requested, CurrentSimNanos, self.moduleID)
        self.appliedForceOutMsg.write(applied, CurrentSimNanos, self.moduleID)
        self.last_command = command
        up = state.position / np.linalg.norm(state.position)
        radial_speed = float(np.dot(state.velocity, up))
        applied_N = state.dcm_NB @ command.force_B
        self.history.append({"time_s": now, "mass": state.mass, "status": self.status,
                             "phase": phase,
                             "lowest_foot_clearance_m": foot_clearance,
                             "main_engine_cutoff": self.main_engine_cutoff,
                             "position_error": er, "velocity_error": ev,
                             "main_thrust": command.main_thrust, "pitch": command.pitch,
                             "yaw": command.yaw, "rcs_thrusts": command.rcs_thrusts.copy(),
                             "reference_altitude_m": float(np.linalg.norm(state.position) - self.settings.surface_radius),
                             "radial_speed_m_s": radial_speed,
                             "horizontal_speed_m_s": float(np.linalg.norm(state.velocity - radial_speed * up)),
                             "requested_radial_thrust_N": float(np.dot(command.requested_force_N, up)),
                             "applied_radial_thrust_N": float(np.dot(applied_N, up)),
                             "force_error_N": float(np.linalg.norm(command.requested_force_N - applied_N)),
                             "saturated": command.saturated})


class ConstantControl(sysModel.SysModel):
    """Publish one on-time command per engine on every controller tick.

    Run on the same task as the engines, with controller priority 20,
    engine priority 10, and spacecraft priority 0. Selected engines deliver
    their configured MaxThrust in newtons; this module does not implement PWM,
    throttle, feedback, or actuator dynamics. Use configs without ramps.
    """

    def __init__(self, engines, period_s, enabled_engines=(), name="ConstantControl"):
        super().__init__()
        self.ModelTag = name
        self.period_s = float(period_s)
        if not math.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("period_s must be finite and positive.")

        self.engines = dict(engines)
        if len({id(engine.thruster) for engine in self.engines.values()}) != len(self.engines):
            raise ValueError("Each engine name must refer to a distinct effector.")
        for engine in self.engines.values():
            config = engine.config
            if len(config.ThrusterOnRamp) or len(config.ThrusterOffRamp):
                raise ValueError("ConstantControl requires thrusters without on/off ramps.")
            if len(config.thrBlowDownCoeff) or len(config.ispBlowDownCoeff):
                raise ValueError("ConstantControl requires configs without blow-down curves.")
            if not math.isfinite(config.MinOnTime) or config.MinOnTime < 0:
                raise ValueError("MinOnTime must be finite and nonnegative.")

        self.set_enabled_engines(enabled_engines)
        self.command_msgs = {}
        for name, engine in self.engines.items():
            msg = messaging.THRArrayOnTimeCmdMsg()
            self.command_msgs[name] = msg
            engine.subscribe_to_commands(msg)

    @property
    def enabled_engines(self):
        return self._enabled_engines

    def set_enabled_engines(self, names):
        """Replace the selected engines; changes take effect on the next tick."""
        if isinstance(names, str):
            names = [names]
        selected = frozenset(names)
        unknown = selected.difference(self.engines)
        if unknown:
            raise ValueError(f"Unknown engines: {sorted(unknown)}")
        self._enabled_engines = selected

    def set_engine_enabled(self, name, enabled=True):
        """Enable/disable one engine without changing the other selections."""
        if name not in self.engines:
            raise ValueError(f"Unknown engine: {name}")
        selected = set(self._enabled_engines)
        if enabled:
            selected.add(name)
        else:
            selected.discard(name)
        self.set_enabled_engines(selected)

    def Reset(self, CurrentSimNanos):
        self.UpdateState(CurrentSimNanos)

    def UpdateState(self, CurrentSimNanos):
        for name, engine in self.engines.items():
            command = messaging.THRArrayOnTimeCmdMsgPayload()
            # Overlap successive requests so every integration substep sees
            # continuous thrust. Each effector contains one engine: index 0.
            on_time = max(2.0 * self.period_s, engine.config.MinOnTime)
            command.OnTimeRequest = [on_time if name in self._enabled_engines else 0.0]
            self.command_msgs[name].write(command, CurrentSimNanos, self.moduleID)


def touchdown_metrics(position, velocity, target_position, terrain, fuel_remaining, initial_fuel, time_s, ignition_s, braking_s, terminal_s):
    """First-contact geometry/velocity, with local vertical positive upward."""
    position, velocity = np.asarray(position), np.asarray(velocity)
    map_position = terrain.to_map(position)
    map_target = terrain.to_map(target_position)
    # Coordinate axes are fixed by this terrain, not by vehicle body attitude.
    velocity_map = terrain.to_map(position + velocity) - map_position
    speed = float(np.linalg.norm(velocity_map))
    direction = velocity_map / speed if speed > 1e-12 else np.zeros(3)
    site = terrain.assess_site(map_position[:2], radius_m=3.)
    return dict(time_s=float(time_s), intended_site_map_xy_m=map_target[:2].tolist(),
        touchdown_site_map_xy_m=map_position[:2].tolist(),
        distance_to_intended_site_m=float(np.linalg.norm(map_position[:2] - map_target[:2])),
        fuel_remaining_kg=float(fuel_remaining), fuel_used_kg=float(initial_fuel - fuel_remaining),
        landed_area_slope_deg=site.get('slope_deg'),
        ignition_to_touchdown_s=float(time_s - ignition_s),
        braking_to_touchdown_s=float(time_s - braking_s),
        terminal_to_touchdown_s=None if terminal_s is None else float(time_s - terminal_s),
        velocity_N_m_s=velocity.tolist(), velocity_map_m_s=velocity_map.tolist(),
        speed_m_s=speed, direction_map_unit=direction.tolist(),
        horizontal_velocity_map_m_s=velocity_map[:2].tolist(),
        horizontal_speed_m_s=float(np.linalg.norm(velocity_map[:2])),
        vertical_velocity_m_s=float(velocity_map[2]),
        descent_angle_deg=float(np.rad2deg(np.arctan2(-velocity_map[2], np.linalg.norm(velocity_map[:2])))),
        horizontal_heading_deg=(float(np.rad2deg(np.arctan2(velocity_map[1], velocity_map[0])))
                                if np.linalg.norm(velocity_map[:2]) > 1e-12 else None))


def print_touchdown_metrics(event):
    def number(value):
        return 'unavailable' if value is None else f'{value:.3f}'
    print('\nTouchdown diagnostics (t0 = first detected foot contact)', flush=True)
    print(f"  Distance to intended landing site: {event['distance_to_intended_site_m']:.3f} m", flush=True)
    print(f"  Fuel remaining: {event['fuel_remaining_kg']:.3f} kg; used: {event['fuel_used_kg']:.3f} kg", flush=True)
    print(f"  Maximum acceleration including contact so far: {event['maximum_acceleration_m_s2']:.3f} m/s^2", flush=True)
    print(f"  Maximum engine acceleration: {event['maximum_engine_acceleration_m_s2']:.3f} m/s^2 "
          f"(main alone {event['maximum_main_engine_acceleration_m_s2']:.3f} m/s^2)", flush=True)
    print(f"  Landed area slope: {number(event['landed_area_slope_deg'])} deg", flush=True)
    print(f"  Ignition / braking / terminal to t0: {event['ignition_to_touchdown_s']:.3f} / "
          f"{event['braking_to_touchdown_s']:.3f} / {number(event['terminal_to_touchdown_s'])} s", flush=True)
    print(f"  Touchdown velocity: {event['speed_m_s']:.3f} m/s; local XYZ {event['velocity_map_m_s']} m/s", flush=True)
    print(f"  Horizontal: {event['horizontal_speed_m_s']:.3f} m/s, XY {event['horizontal_velocity_map_m_s']} m/s; "
          f"vertical (+up): {event['vertical_velocity_m_s']:.3f} m/s", flush=True)
    print(f"  Direction (local unit vector): {event['direction_map_unit']}; "
          f"descent angle {event['descent_angle_deg']:.3f} deg; "
          f"horizontal heading {number(event['horizontal_heading_deg'])} deg", flush=True)


class TouchdownDiagnostics(sysModel.SysModel):
    """Observe actual first contact and loads; never supplies controller feedback."""
    def __init__(self, vehicle, print_on_contact=True):
        super().__init__()
        self.ModelTag = 'TouchdownDiagnostics'
        self.vehicle = vehicle
        self.print_on_contact = print_on_contact
        self.state = vehicle.lander.scStateOutMsg.addSubscriber()
        self.mass = vehicle.lander.scMassOutMsg.addSubscriber()
        contact = vehicle.thruster_controller.contact_sensor
        self.contact_force = contact.forceOutMsg.addSubscriber() if contact is not None else None
        imu = vehicle.imus.get('IMU')
        self.imu = imu.model.sensorOutMsg.addSubscriber() if imu is not None else None
        self.accel_bias = np.asarray(imu.model.senTransBias).reshape(3) if imu is not None else np.zeros(3)
        self.Reset(0)

    def Reset(self, now):
        self.touchdown = None
        self.history = []
        self.peaks = dict(maximum_acceleration_m_s2=0., maximum_imu_acceleration_m_s2=0.,
                          maximum_engine_acceleration_m_s2=0., maximum_main_engine_acceleration_m_s2=0.,
                          maximum_contact_acceleration_m_s2=0.)

    def UpdateState(self, now):
        vehicle = self.vehicle
        controller = vehicle.thruster_controller
        t = now * 1e-9
        ignition = controller.start_s + controller.settings.coast_duration_s
        if t < ignition or not self.state.isWritten():
            return
        state = self.state()
        mass = float(self.mass().massSC) if self.mass.isWritten() else controller.initial_state.mass
        c_NB = RigidBodyKinematics.MRP2C(state.sigma_BN).T
        engine_force = sum((np.asarray(e.thruster.forceExternal_B).reshape(3)
                            for e in vehicle.engines.values()), np.zeros(3))
        main_force = np.asarray(controller.main_engine.thruster.forceExternal_B).reshape(3)
        contact = controller.contact_sensor
        contact_force = (np.asarray(self.contact_force().forceRequestInertial)
                         if self.contact_force is not None and self.contact_force.isWritten() else np.zeros(3))
        specific = float(np.linalg.norm(c_NB @ engine_force + contact_force) / mass)
        imu_accel = 0.
        if self.imu is not None and self.imu.isWritten():
            imu_accel = float(np.linalg.norm(np.asarray(self.imu().AccelPlatform) - self.accel_bias))
        values = dict(maximum_acceleration_m_s2=specific,
                      maximum_imu_acceleration_m_s2=imu_accel,
                      maximum_engine_acceleration_m_s2=float(np.linalg.norm(engine_force) / mass),
                      maximum_main_engine_acceleration_m_s2=float(np.linalg.norm(main_force) / mass),
                      maximum_contact_acceleration_m_s2=float(np.linalg.norm(contact_force) / mass))
        for key, value in values.items():
            self.peaks[key] = max(self.peaks[key], value)
        self.history.append(dict(time_s=t, specific_acceleration_m_s2=specific,
            imu_acceleration_m_s2=imu_accel, engine_acceleration_m_s2=values['maximum_engine_acceleration_m_s2'],
            contact_acceleration_m_s2=values['maximum_contact_acceleration_m_s2']))
        if self.touchdown is None and contact is not None and contact.collision:
            fuel = float(vehicle.fuel_tank.fuelTankOutMsg.read().fuelMass)
            target = controller.target.state(t)[0]
            self.touchdown = touchdown_metrics(state.r_BN_N, state.v_BN_N, target,
                vehicle.terrain_map, fuel, controller.initial_state.mass - controller.settings.dry_mass,
                t, ignition, ignition, controller.terminal_start_s)
            self.touchdown.update(self.peaks)
            self.touchdown['cutoff'] = controller.cutoff_event
            self.touchdown['acceleration_definition'] = 'specific load |engine + contact force| / mass; gravity excluded'
            if self.print_on_contact:
                print_touchdown_metrics(self.touchdown)

    def report(self):
        if self.touchdown is None:
            return None
        event = dict(self.touchdown)
        event['peaks_at_first_contact'] = {key: self.touchdown[key] for key in self.peaks}
        # Contact-load peak may occur after the first contact detection. Keep
        # first-contact velocity/fuel/time fixed while updating load maxima.
        event.update(self.peaks)
        event['acceleration_peak_interval'] = 'powered ignition through end of settling'
        return event


def terminal_metrics(position, velocity, target_position, target_velocity, radius):
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    er = float(np.linalg.norm(r - target_position))
    ev = float(np.linalg.norm(v - target_velocity))
    return dict(position_error_m=er, velocity_error_m_s=ev,
                reference_altitude_m=float(np.linalg.norm(r)-radius),
                speed_m_s=float(np.linalg.norm(v)),
                radial_speed_m_s=float(np.dot(r,v)/np.linalg.norm(r)),
                precision_target_met=bool(er <= .5 and ev <= .1),
                contact_landing_verified=False)


def save_descent_report(vehicle, directory):
    import matplotlib.pyplot as plt
    from Basilisk.utilities import RigidBodyKinematics
    controller = vehicle.thruster_controller
    cfg = controller.settings
    time = np.asarray(vehicle.sc_recorder.times())*1e-9
    r = np.asarray(vehicle.sc_recorder.r_BN_N)
    v = np.asarray(vehicle.sc_recorder.v_BN_N)
    target = controller.target.state(time[-1])
    report = terminal_metrics(r[-1],v[-1],*target[:2],cfg.surface_radius)
    altitude = np.linalg.norm(r,axis=1)-cfg.surface_radius
    report.update(controller_status=controller.status,
                  simulation_start_s=getattr(vehicle, 'simulation_start_s', 0.),
                  initialization=('powered-descent ignition; sensors initialized at ignition'
                      if getattr(vehicle, 'simulation_start_s', 0.) else 'full mission'),
                  configured_position_tolerance_m=cfg.position_tolerance,
                  configured_velocity_tolerance_m_s=cfg.velocity_tolerance,
                  minimum_reference_altitude_m=float(np.min(altitude)),
                  powered_flight_time_s=controller.plan.flight_time,
                  dry_mass_kg=cfg.dry_mass,
                  engine_cutoff_m=getattr(cfg, 'engine_cutoff_m', 0.),
                  main_engine_cutoff_triggered=getattr(controller, 'main_engine_cutoff', False),
                  remaining_propellant_kg=float(vehicle.fuel_recorder.fuelMass[-1]))
    report['precision_target_met'] = bool(report['position_error_m'] <= cfg.position_tolerance
                                          and report['velocity_error_m_s'] <= cfg.velocity_tolerance)
    contact = getattr(controller, 'contact_sensor', None)
    report['contact_detected_at_end'] = bool(contact is not None and contact.collision)
    report['contact_landing_verified'] = bool(controller.status == 'landed'
        and report['contact_detected_at_end'] and report['precision_target_met'])
    monitor = getattr(vehicle, 'touchdown_diagnostics', None)
    report['touchdown'] = monitor.report() if monitor is not None else None
    report['cutoff'] = getattr(controller, 'cutoff_event', None)
    c_NB=RigidBodyKinematics.MRP2C(vehicle.sc_recorder.sigma_BN[-1]).T
    report['final_body_tilt_deg']=float(np.rad2deg(np.arccos(np.clip(
        np.dot(c_NB[:,2],r[-1]/np.linalg.norm(r[-1])),-1,1))))
    report['final_angular_speed_rad_s']=float(np.linalg.norm(vehicle.sc_recorder.omega_BN_B[-1]))
    if hasattr(vehicle, 'terrain_map'):
        report['final_terrain_clearance_m'] = vehicle.terrain_map.clearance(r[-1])
        report['landing_site'] = getattr(vehicle, 'landing_site', None)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    navigation = getattr(vehicle,'navigation',None)
    if navigation is not None and navigation.history:
        nh = navigation.history
        nt = np.array([h['time_s'] for h in nh])
        nr = np.array([h['position_N'] for h in nh])
        nv = np.array([h['velocity_N'] for h in nh])
        nr_true = np.column_stack([np.interp(nt,time,r[:,i]) for i in range(3)])
        nv_true = np.column_stack([np.interp(nt,time,v[:,i]) for i in range(3)])
        errors = np.linalg.norm(nr-nr_true,axis=1)
        valid = np.array([h['range_valid'] for h in nh])
        report['navigation'] = dict(all_healthy=all(h['healthy'] for h in nh),
            filter_enabled=navigation.cfg.filter_enabled,
            final_position_error_m=float(errors[-1]), maximum_position_error_m=float(errors.max()),
            laser_updates=int(valid.sum()),
            first_laser_update_s=float(nt[valid][0]) if valid.any() else None,
            inputs='Gauss-Markov map XY, map-referenced laser range, star tracker, IMU, commanded thrust model')
        if not navigation.cfg.filter_enabled:
            report['navigation']['inputs']='Raw Gauss-Markov map XY and map-referenced laser fixes; calibrated IMU dead reckoning during outages'
        status=np.array([h['range_status'] for h in nh])
        reasons,counts=np.unique(status,return_counts=True)
        report['navigation']['laser_status_counts']=dict(zip(reasons.tolist(),counts.tolist()))
        laser_fields=('raw_slant_range_m','used_slant_range_m','laser_observed_map_z_m',
                      'predicted_map_z_m','estimated_map_z_m','beam_down_cosine','measured_map_xy_m')
        laser_data={key:np.array([h[key] for h in nh]) for key in laser_fields}
        np.savez_compressed(directory/'navigation.npz',time_s=nt,position_N=nr,velocity_N=nv,
            truth_position_N=nr_true,truth_velocity_N=nv_true,range_valid=valid,
            position_sigma_m=[h['position_sigma_m'] for h in nh],range_status=status,
            filter_enabled=navigation.cfg.filter_enabled,**laser_data)
        laserfig,laseraxes=plt.subplots(3,1,figsize=(11,9),layout='constrained',sharex=True)
        lt=(nt-cfg.coast_duration_s)/60
        raw=laser_data['raw_slant_range_m'].copy()
        raw[(raw<0)|~np.isfinite(raw)]=np.nan
        laseraxes[0].plot(lt,raw,label='Raw return',alpha=.65)
        laseraxes[0].plot(lt,laser_data['used_slant_range_m'],label='Accepted noisy range')
        laseraxes[0].set(ylabel='Beam slant range (m)')
        laseraxes[1].plot(lt,laser_data['estimated_map_z_m'],label='Navigation map Z')
        laseraxes[1].plot(lt,laser_data['laser_observed_map_z_m'],'.',ms=2,label='Laser + DEM observed Z')
        if hasattr(vehicle,'terrain_map'):
            laseraxes[1].plot(lt,vehicle.terrain_map.to_map(nr_true)[:,2],':',label='Truth Z (comparison only)')
        laseraxes[1].set(ylabel='Map Z (m)')
        codes={name:i for i,name in enumerate(('no_return','stale','over_range','beam_not_downward',
                                              'footprint_unmapped','negative_noisy_range','accepted'))}
        laseraxes[2].step(lt,[codes[s] for s in status],where='post',label='Navigation range decision')
        laseraxes[2].set(yticks=list(codes.values()),yticklabels=list(codes),xlabel='Time since ignition (min)')
        for ax in laseraxes:
            ax.grid(alpha=.25);ax.legend()
        laserfig.suptitle('Laser measurement use — navigation predicts Z whenever a range is rejected')
        laserfig.savefig(directory/'laser_navigation_diagnostics.png',dpi=150)
        # A second view prevents the long coast from hiding the useful returns.
        if valid.any():
            laseraxes[2].set_xlim(lt[valid][0]-.1,lt[-1]+.1)
            z_values=np.r_[laser_data['estimated_map_z_m'][valid],laser_data['laser_observed_map_z_m'][valid]]
            margin=max(1.,float(np.ptp(z_values))*.05)
            laseraxes[1].set_ylim(float(np.min(z_values))-margin,float(np.max(z_values))+margin)
            laseraxes[0].set_ylim(0,max(1.,float(np.max(laser_data['used_slant_range_m'][valid]))*1.1))
            laserfig.savefig(directory/'laser_navigation_closeup.png',dpi=150)
        plt.close(laserfig)
        navfig,navaxes=plt.subplots(2,1,figsize=(9,6),layout='constrained')
        navaxes[0].plot((nt-cfg.coast_duration_s)/60,errors)
        navaxes[0].set(ylabel='Position estimate error (m)',yscale='symlog')
        navaxes[1].plot((nt-cfg.coast_duration_s)/60,valid.astype(int))
        navaxes[1].set(ylabel='Laser update valid',yticks=[0,1],xlabel='Time since ignition (min)')
        for ax in navaxes: ax.grid(alpha=.25)
        navfig.savefig(directory/'navigation_diagnostics.png',dpi=150)
        plt.close(navfig)
    attitude = getattr(vehicle, 'navigation', None)
    if attitude is not None and attitude.attitude_history:
        ah = attitude.attitude_history
        report['attitude_navigation'] = dict(inputs='joint position/attitude UKF: calibrated IMU gyros + star tracker' if attitude.cfg.filter_enabled else 'raw calibrated IMU gyros + star tracker',
            all_healthy=all(h['healthy'] for h in ah), samples=len(ah))
        np.savez_compressed(directory/'attitude_navigation.npz',
            time_s=[h['time_s'] for h in ah], dcm_NB=[h['dcm_NB'] for h in ah],
            omega_B_rad_s=[h['omega_B_rad_s'] for h in ah])
    if report['touchdown'] is not None:
        print_touchdown_metrics(report['touchdown'])
    if monitor is not None and monitor.history:
        np.savez_compressed(directory/'touchdown_acceleration.npz',
            **{key: np.array([h[key] for h in monitor.history]) for key in monitor.history[0]})
    (directory/'descent_report.json').write_text(json.dumps(report,indent=2))
    fig,axes = plt.subplots(3,2,figsize=(11,10),layout='constrained')
    powered = time >= cfg.coast_duration_s
    t = (time[powered]-cfg.coast_duration_s)/60
    axes[0,0].plot(t,altitude[powered]); axes[0,0].axhline(0,color='black',lw=.8)
    axes[0,0].set(ylabel='Reference altitude (m)')
    up = r / np.linalg.norm(r,axis=1)[:,None]
    radial = np.sum(v*up,axis=1)
    horizontal = np.linalg.norm(v-radial[:,None]*up,axis=1)
    axes[0,1].plot(t,radial[powered],label='Radial (+ upward)')
    axes[0,1].plot(t,horizontal[powered],label='Horizontal')
    axes[0,1].set(ylabel='Velocity (m/s)'); axes[0,1].legend()
    history = [h for h in controller.history if 'main_thrust' in h]
    axes[1,0].plot([(h['time_s']-cfg.coast_duration_s)/60 for h in history],
                   [h['main_thrust'] for h in history])
    axes[1,0].set(ylabel='Main thrust (N)')
    ht = np.array([(h['time_s']-cfg.coast_duration_s)/60 for h in history])
    for field,label in [('requested_radial_thrust_N','Requested'),('applied_radial_thrust_N','Applied')]:
        axes[1,1].plot(ht,[h.get(field, np.nan) for h in history],label=label)
    axes[1,1].axhline(0,color='black',lw=.8)
    axes[1,1].set(ylabel='Radial thrust (+ upward, N)',yscale='symlog')
    axes[1,1].legend()
    ft = np.asarray(vehicle.fuel_recorder.times())*1e-9
    fm = np.asarray(vehicle.fuel_recorder.fuelMass)
    axes[2,0].plot((ft[ft>=cfg.coast_duration_s]-cfg.coast_duration_s)/60,fm[ft>=cfg.coast_duration_s])
    axes[2,0].set(ylabel='Propellant remaining (kg)')
    axes[2,1].plot(ht,[h.get('position_error', np.nan) for h in history])
    axes[2,1].set(ylabel='Target position error (m)')
    keys = ('time_s','reference_altitude_m','radial_speed_m_s','horizontal_speed_m_s',
            'requested_radial_thrust_N','applied_radial_thrust_N','force_error_N','main_thrust','mass')
    np.savez_compressed(directory/'descent_guidance.npz',
                        **{key: np.array([h.get(key, np.nan) for h in history]) for key in keys})
    for ax in axes.flat:
        ax.set_xlabel('Time since ignition (min)'); ax.grid(alpha=.25)
    verdict = ('Contact landing verified in simulation' if report['contact_landing_verified'] else
               'Precision target met; contact unverified' if report['precision_target_met'] else 'Landing target NOT achieved')
    fig.suptitle(f"{verdict}: error {report['position_error_m']:.2f} m, {report['velocity_error_m_s']:.2f} m/s")
    fig.savefig(directory/'descent_diagnostics.png',dpi=160)
    plt.close(fig)
    print(f"Descent assessment: {verdict}. Position error {report['position_error_m']:.3f} m; "
          f"velocity error {report['velocity_error_m_s']:.3f} m/s.",flush=True)
    print('Descent report:',(directory/'descent_report.json').resolve(),flush=True)
    return report


def example_engines(main_thrust=4003.4, main_isp=325., rcs_thrust=4.45):
    engines = {"MainEngine": MainEngine(make_config(main_thrust, main_isp, [0, 0, 1]),
                                        location=[0, 0, -1], max_gimbal=0.27)}
    for pod, (location, directions) in enumerate((
        ([.5, 0, .8], ([0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])),
        ([-.5, 0, .8], ([0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])),
        ([0, .5, .8], ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1])),
        ([0, -.5, .8], ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1])),
    ), 1):
        for index, direction in enumerate(directions, 1):
            name = f"RCS_{pod}_{index}"
            engines[name] = RCS_thruster(make_config(rcs_thrust, 220., direction), name, location)
    return engines
