"""Basilisk Python SysModel adapter for variable-gravity A2G descent."""
import numpy as np
from scipy.spatial.transform import Rotation
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import RigidBodyKinematics

from Spacecraft.Thrusters.DescentGuidance import (
    DescentAllocator, DescentState, FlightTimePlanner, gravity, coast_state,
)


class DescentController(sysModel.SysModel):
    def __init__(self, main_engine, rcs_engines, settings, target, initial_state,
                 inertia_B, period_s, state_msg, fuel_msg, terrain_map=None, landing_foot_points_B=None):
        super().__init__()
        self.ModelTag = "A2GDescentController"
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
        self.planner = FlightTimePlanner(self.allocator, target, terrain_map, landing_foot_points_B)
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
        self.attitude_navigation = None
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
        if self.attitude_navigation is not None:
            message = self.attitude_navigation.navOutMsg.read()
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
            self.status = 'navigation_invalid'
            self._off(CurrentSimNanos)
            return
        if self.attitude_navigation is not None and not self.attitude_navigation.healthy:
            if now - self.reset_s <= 2 * self.period_s + 1e-9:
                # Control is scheduled before sensors on the initial tick.
                self._off(CurrentSimNanos)
                return
            self.status = 'attitude_invalid'
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
