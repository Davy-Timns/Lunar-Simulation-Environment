"""
Classical Apollo guidance, bounded allocation, and flight-time search.

The historical identifier 'a2g' denotes the supplied fixed 12/6-gain law.
This is not the complete Augmented Apollo Powered Descent Guidance (A2PDG)
architecture: tunable guidance gain, UPG-based ignition selection and terminal
gravity-turn descent-rate regulation are not implemented.

All translational states use a Moon-centered INERTIAL frame, metres/seconds/kg.
The planner minimizes fuel within this feedback-law family, not over arbitrary
thrust histories. Feasibility is numerical and checked again on a finer grid.
"""
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import lsq_linear, minimize
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

G0 = 9.80665


def vector3(value):
    result = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(result)):
        raise ValueError("Vectors must contain three finite components.")
    return result


def gravity(position, mu):
    position = vector3(position)
    radius = np.linalg.norm(position)
    if radius <= 0 or not np.isfinite(mu) or mu <= 0:
        raise ValueError("Gravity requires positive mu and nonzero radius.")
    return -mu * position / radius**3


def a2g_acceleration(position, velocity, target_position, target_velocity,
                     target_acceleration, time_to_go, mu):
    """Quadratic NET-acceleration boundary law minus instantaneous gravity.

    target_acceleration is inertial kinematic acceleration, not thrust/mass.
    This polynomial law does not itself minimize propellant or enforce actuator
    and path constraints. Clipping its output changes its predicted path.
    """
    if not np.isfinite(time_to_go) or time_to_go <= 0:
        raise ValueError("A2G time-to-go must be positive.")
    return (12 * (vector3(target_position) - vector3(position)) / time_to_go**2
            - 6 * (vector3(target_velocity) + vector3(velocity)) / time_to_go
            + vector3(target_acceleration) - gravity(position, mu))


@dataclass
class LandingTarget:
    """Constant inertial target, or a site rotating uniformly about the Moon.

    position is the site at epoch_s; omega_N is the Moon rotation vector.
    With omega_N=0, velocity/acceleration describe a polynomial inertial target.
    With nonzero omega_N, do not also provide velocity/acceleration: these are
    computed from rotation. This does not introduce a rotating dynamics frame.
    """
    position: object
    velocity: object = (0, 0, 0)
    acceleration: object = (0, 0, 0)
    omega_N: object = (0, 0, 0)
    epoch_s: float = 0.0

    def __post_init__(self):
        for name in ("position", "velocity", "acceleration", "omega_N"):
            setattr(self, name, vector3(getattr(self, name)))
        if not np.isfinite(self.epoch_s):
            raise ValueError("Target epoch must be finite.")
        if np.linalg.norm(self.omega_N) and (np.any(self.velocity) or np.any(self.acceleration)):
            raise ValueError("A rotating site derives its velocity and acceleration from omega_N.")

    def state(self, time_s):
        dt = time_s - self.epoch_s
        if np.linalg.norm(self.omega_N):
            r = Rotation.from_rotvec(self.omega_N * dt).apply(self.position)
            v = np.cross(self.omega_N, r)
            return r, v, np.cross(self.omega_N, v)
        return (self.position + dt * self.velocity + 0.5 * dt**2 * self.acceleration,
                self.velocity + dt * self.acceleration, self.acceleration.copy())


@dataclass
class DescentSettings:
    mu: float
    dry_mass: float
    min_thrust: float
    max_thrust: float
    tf_min: float
    tf_max: float
    position_tolerance: float = 0.5
    velocity_tolerance: float = 0.1
    surface_radius: float = 1737400.0
    surface_tolerance_m: float = 0.01
    terminal_guard_s: float = 0.5
    attitude_frequency: float = 0.6
    attitude_damping: float = 1.0
    force_tracking_weight: float = 0.001
    search_samples: int = 17
    max_search_evaluations: int = 60
    max_search_wall_s: float = 180.0
    print_progress: bool = True
    rollout_step_s: float = 0.2
    coast_duration_s: float = 0.0
    terminal_gate_height_m: float = 0.0  # 0 preserves single-endpoint examples
    terminal_duration_s: float = 120.0
    terrain_clearance_m: float = 20.0
    landing_com_clearance_m: float = 0.66  # physical foot depth; contact margin stops thrust earlier
    engine_cutoff_m: float = 0.0  # powered endpoint above lowest foot; 0 preserves surface endpoint

    def __post_init__(self):
        numeric = [v for v in vars(self).values()]
        if not np.all(np.isfinite(numeric)):
            raise ValueError("Descent settings must be finite.")
        if not (self.mu > 0 and self.dry_mass > 0 and 0 <= self.min_thrust < self.max_thrust):
            raise ValueError("Require mu > 0, dry_mass > 0 and 0 <= Tmin < Tmax.")
        if not (0 < self.tf_min < self.tf_max):
            raise ValueError("Require 0 < tf_min < tf_max.")
        if min(self.position_tolerance, self.velocity_tolerance, self.terminal_guard_s,
               self.attitude_frequency, self.attitude_damping, self.force_tracking_weight,
               self.rollout_step_s) <= 0 or self.surface_radius < 0 or self.surface_tolerance_m < 0:
            raise ValueError("Tolerances, gains and time steps must be positive.")
        if int(self.search_samples) != self.search_samples or self.search_samples < 5:
            raise ValueError("search_samples must be an integer of at least five.")
        if (int(self.max_search_evaluations) != self.max_search_evaluations or
                self.max_search_evaluations < self.search_samples or self.max_search_wall_s <= 0):
            raise ValueError("Search budget must cover the grid and have a positive wall-time limit.")
        if self.coast_duration_s < 0:
            raise ValueError("Require nonnegative coast duration.")
        if (self.terminal_gate_height_m < 0 or self.terrain_clearance_m < 0 or self.engine_cutoff_m < 0 or
                self.landing_com_clearance_m <= 0 or self.terminal_duration_s <= 0):
            raise ValueError('Terrain/gate clearances and terminal duration must be positive')
        if self.terminal_gate_height_m and self.terminal_duration_s >= self.tf_min:
            raise ValueError('Terminal duration must be shorter than total powered flight time')


@dataclass
class DescentState:
    position: np.ndarray
    velocity: np.ndarray
    mass: float
    dcm_NB: np.ndarray
    omega_B: np.ndarray

    def __post_init__(self):
        self.position = vector3(self.position).copy()
        self.velocity = vector3(self.velocity).copy()
        self.omega_B = vector3(self.omega_B).copy()
        self.dcm_NB = np.asarray(self.dcm_NB, dtype=float).reshape(3, 3).copy()
        if not np.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Mass must be positive and finite.")
        if (not np.all(np.isfinite(self.dcm_NB)) or
                not np.allclose(self.dcm_NB.T @ self.dcm_NB, np.eye(3), atol=1e-8) or
                not np.isclose(np.linalg.det(self.dcm_NB), 1.)):
            raise ValueError("dcm_NB must be a proper orthogonal rotation.")

    def copy(self):
        return DescentState(self.position.copy(), self.velocity.copy(), float(self.mass),
                            self.dcm_NB.copy(), self.omega_B.copy())


@dataclass
class ActuatorCommand:
    main_thrust: float
    pitch: float
    yaw: float
    rcs_thrusts: np.ndarray
    force_B: np.ndarray
    torque_B: np.ndarray
    mass_flow: float
    requested_force_N: np.ndarray
    saturated: bool


def coast_state(initial, duration_s, mu, surface_radius=0.):
    """Propagate the post-DOI unpowered coast in the same inertial gravity model.

    The demo starts with zero angular rate; nonzero rate is rejected rather than
    silently approximating torque-free rigid-body dynamics over a long coast.
    """
    state = initial.copy()
    if not np.isfinite(duration_s) or not np.isfinite(surface_radius) or surface_radius < 0:
        raise ValueError("Coast duration and reference surface must be finite.")
    if np.linalg.norm(state.position) <= surface_radius:
        raise ValueError("Coast starts at or below the reference surface.")
    if duration_s == 0:
        return state
    if np.linalg.norm(initial.omega_B) > 1e-10:
        raise ValueError("Set zero initial body rate for the unpowered DOI coast.")
    def surface_event(t, y):
        return np.linalg.norm(y[:3]) - surface_radius
    
    surface_event.terminal = True
    result = solve_ivp(lambda t, y: np.concatenate((y[3:], gravity(y[:3], mu))),
                       (0., duration_s), np.concatenate((state.position, state.velocity)),
                       rtol=1e-11, atol=1e-7, max_step=30., events=surface_event)
    if not result.success:
        raise RuntimeError("Coast propagation failed: " + result.message)
    if result.t_events[0].size:
        raise ValueError("Unpowered coast intersects the reference lunar surface.")
    state.position = result.y[:3, -1]
    state.velocity = result.y[3:, -1]
    return state


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
        self.moments = np.array([np.cross(vector3(e.config.thrLoc_B), vector3(e.config.thrDir_B))
                                 for e in rcs_engines.values()]).reshape(-1, 3).T
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


@dataclass
class TrajectoryResult:
    flight_time: float
    fuel_used: float
    position_error: float
    velocity_error: float
    min_radius: float
    final_mass: float
    feasible: bool
    state: DescentState
    stop_reason: str = "touchdown_time"
    elapsed_s: float = 0.0
    peak_requested_thrust: float = 0.0
    saturated_s: float = 0.0
    path_violation_m: float = 0.0


class InfeasibleDescent(ValueError):
    pass


class DescentSearchLimit(RuntimeError):
    """
    Search stopped without completing verification; not proof of infeasibility.
    Thus no error thrown (ie pass)
    """


class _EvaluationLimit(RuntimeError):
    pass


class _RefinementLimit(RuntimeError):
    pass


class FlightTimePlanner:
    def __init__(self, allocator, target, terrain_map=None, landing_foot_points_B=None):
        self.allocator = allocator
        self.settings = allocator.settings
        self.target = target
        self.terrain_map = terrain_map
        self.landing_foot_points_B = landing_foot_points_B

    def lowest_foot_clearance(self, state, points_B=None):
        points_B = self.landing_foot_points_B if points_B is None else points_B
        if points_B is None or len(points_B) == 0:
            clearance = (self.terrain_map.clearance(state.position) if self.terrain_map is not None
                         else np.linalg.norm(state.position) - self.settings.surface_radius)
            return None if clearance is None else float(clearance - self.settings.landing_com_clearance_m)
        points_N = state.position + np.asarray(points_B) @ state.dcm_NB.T
        clearances = [(self.terrain_map.clearance(point) if self.terrain_map is not None
                       else np.linalg.norm(point) - self.settings.surface_radius) for point in points_N]
        if any(value is None or not np.isfinite(value) for value in clearances):
            return None
        return float(min(clearances))

    def powered_target_state(self, time_s, state=None):
        """Zero-speed endpoint at the cutoff height, accounting for foot pose and terrain."""
        final = self.target.state(time_s)
        if not self.settings.engine_cutoff_m:
            return final
        up = (np.asarray(self.terrain_map.up_N) if self.terrain_map is not None
              and hasattr(self.terrain_map, 'up_N') else final[0] / np.linalg.norm(final[0]))
        if state is None:
            # Upright nominal orientation; live/rollout calls use the estimated/propagated pose.
            z = up
            y = np.array([1., 0., 0.])
            if abs(np.dot(y, z)) > .9:
                y = np.array([0., 1., 0.])
            y -= np.dot(y, z) * z
            y /= np.linalg.norm(y)
            pose = DescentState(final[0], final[1], 1., np.column_stack((np.cross(y, z), y, z)), np.zeros(3))
        else:
            pose = state.copy()
            pose.position = final[0].copy()
        clearance = self.lowest_foot_clearance(pose)
        if clearance is None:
            raise ValueError('Powered cutoff target requires mapped landing-foot terrain')
        position = final[0] + (self.settings.engine_cutoff_m - clearance) * up
        return position, final[1], final[2]

    def guidance_target(self, start_s, flight_time, elapsed, state=None):
        """Select the braking gate or the previously chosen landing target.

        This does not search for safe terrain or redesignate a site. The static
        map search occurs before planning. Switch at tf-terminal_duration_s;
        reaching the gate altitude/velocity is an objective, not a switch guard.
        """
        final = self.target.state(start_s + flight_time)
        gate_end = flight_time - self.settings.terminal_duration_s
        if self.settings.terminal_gate_height_m and elapsed < gate_end:
            up = final[0] / np.linalg.norm(final[0])
            gate = (final[0] + self.settings.terminal_gate_height_m * up, final[1], final[2])
            return gate, gate_end-elapsed, 'braking'
        return (self.powered_target_state(start_s + flight_time, state), flight_time-elapsed,
                'terminal' if self.settings.terminal_gate_height_m or self.settings.engine_cutoff_m else 'powered')

    def rollout(self, initial, flight_time, start_s=0., step_s=None, heartbeat=None):
        cfg = self.settings
        step_s = cfg.rollout_step_s if step_s is None else step_s
        if not np.all(np.isfinite([flight_time, start_s, step_s])) or min(flight_time, step_s) <= 0:
            raise ValueError("Flight time and integration step must be positive and finite.")
        state = initial.copy()
        target_state = self.powered_target_state(start_s + flight_time, state)
        steps = max(1, int(np.ceil(flight_time / step_s)))
        min_radius = np.linalg.norm(state.position)
        stop_reason = "cutoff_time" if cfg.engine_cutoff_m else "touchdown_time"
        peak_requested = 0.0
        saturated_s = 0.0
        path_violation = 0.0

        for i in range(steps):
            elapsed = i * step_s
            dt = min(step_s, flight_time - elapsed)
            if heartbeat is not None and i % 25 == 0:
                heartbeat(elapsed, flight_time)
            guidance_target, tgo, phase = self.guidance_target(start_s,flight_time,elapsed,state)
            if cfg.engine_cutoff_m and phase == 'terminal':
                foot_clearance = self.lowest_foot_clearance(state)
                if foot_clearance is not None and foot_clearance <= cfg.engine_cutoff_m + 1e-9:
                    stop_reason = 'engine_cutoff'
                    dt = 0.
                    break
            command = self.allocator.command(state, guidance_target, tgo)
            peak_requested = max(peak_requested, float(np.linalg.norm(command.requested_force_N)))
            if state.mass - command.mass_flow * dt <= cfg.dry_mass:
                return TrajectoryResult(flight_time, initial.mass - state.mass,
                                        float(np.linalg.norm(state.position - target_state[0])),
                                        float(np.linalg.norm(state.velocity - target_state[1])),
                                        min_radius, state.mass - command.mass_flow * dt, False, state,
                                        "fuel_floor", elapsed, peak_requested, saturated_s)
            if command.saturated:
                saturated_s += dt

            # Midpoint rigid-body/translation propagation of the applied, bounded
            # actuator forces. Inertia is held fixed, matching a point-mass tank.
            wdot = self.allocator.inv_inertia @ (command.torque_B - np.cross(
                state.omega_B, self.allocator.inertia @ state.omega_B))
            w_mid = state.omega_B + 0.5 * dt * wdot
            c_mid = state.dcm_NB @ Rotation.from_rotvec(0.5 * dt * w_mid).as_matrix()
            mass_mid = state.mass - 0.5 * dt * command.mass_flow
            a0 = gravity(state.position, cfg.mu) + state.dcm_NB @ command.force_B / state.mass
            r_mid = state.position + 0.5 * dt * state.velocity
            a_mid = gravity(r_mid, cfg.mu) + c_mid @ command.force_B / mass_mid
            state.position += dt * (state.velocity + 0.5 * dt * a0)
            state.velocity += dt * a_mid
            state.dcm_NB = state.dcm_NB @ Rotation.from_rotvec(dt * w_mid).as_matrix()
            state.omega_B += dt * (self.allocator.inv_inertia @ (command.torque_B - np.cross(
                w_mid, self.allocator.inertia @ w_mid)))
            state.mass -= dt * command.mass_flow
            min_radius = min(min_radius, np.linalg.norm(state.position))
            clearance = self.terrain_map.clearance(state.position) if self.terrain_map is not None else None

            if clearance is not None:
                required = cfg.terrain_clearance_m if phase == 'braking' else cfg.landing_com_clearance_m
                violation = required-clearance-cfg.surface_tolerance_m

            else:

                violation = cfg.surface_radius-cfg.surface_tolerance_m-np.linalg.norm(state.position)
            path_violation = max(path_violation,violation)

            if violation > 0:
                stop_reason = "terrain_clearance" if clearance is not None else "surface_intersection"
                break

        target_state = self.powered_target_state(start_s + flight_time, state)
        er = float(np.linalg.norm(state.position - target_state[0]))
        ev = float(np.linalg.norm(state.velocity - target_state[1]))
        feasible = (stop_reason in ("touchdown_time", "cutoff_time", "engine_cutoff") and er <= cfg.position_tolerance and ev <= cfg.velocity_tolerance
                    and path_violation <= 0
                    and state.mass >= cfg.dry_mass)

        return TrajectoryResult(float(flight_time), initial.mass - state.mass, er, ev, min_radius, state.mass, feasible, state, stop_reason, elapsed + dt, peak_requested, saturated_s, path_violation)

    def optimize(self, initial, start_s=0.):
        """Search one scalar burn duration with the initial/PDI state held fixed.

        Grid rollouts -> least-violation Powell search when needed -> constrained
        SLSQP fuel refinement. Verify the first feasible candidate immediately,
        retain it, and reserve time to check improvements with a half-size step.
        Fuel includes main and RCS use. This is a budgeted local numerical search
        within a fixed feedback-law/target family, not a globally optimal control
        solution or a noisy-navigation/terrain-contact simulation.
        """
        cfg = self.settings
        if not np.isfinite(initial.mass) or initial.mass <= cfg.dry_mass:
            raise ValueError("Initial mass must exceed dry mass.")

        cache = {}
        started = time.monotonic()
        last_progress = started
        phase = "grid"
        verified_plan = None
        verification_cache = {}
        verification_cost_s = 0.

        def log(message):
            """
            Just generates a log of the time and output message for the a2pdg solver
            """
            if cfg.print_progress:
                print(f"[A2G {time.monotonic() - started:.1f}s] {message}", flush=True)

        def heartbeat(elapsed, tf):
            """
            Regular diagnostic output to ensure program is still running

            """
            nonlocal last_progress #Deal with the variable of last_progress not stored in heartbeat
            now = time.monotonic()
            if now - started >= cfg.max_search_wall_s:
                raise DescentSearchLimit(
                    f"A2G search reached its {cfg.max_search_wall_s:g} s wall-time limit "
                    f"after {len(cache)} completed search trajectories, during {phase} at tf={tf:.2f} s. "
                    "No verified plan was returned; this does not establish infeasibility. "
                    "Review the trajectory diagnostics before increasing the search budget."
                )
            # Refinement is optional once a checked trajectory exists. Keep
            # time for fine verification of the improved candidates it finds.
            reserve = max(.1 * cfg.max_search_wall_s, 2 * verification_cost_s)
            if (verified_plan is not None and phase in ('feasibility search', 'fuel search')
                    and cfg.max_search_wall_s - (now - started) <= reserve):
                raise _RefinementLimit()

            if now - last_progress >= 5.:
                log(f"{phase}: tf={tf:.2f} s, propagated {elapsed:.1f}/{tf:.1f} s; "
                    f"{len(cache)} search trajectories completed")
                last_progress = now

        def verify(candidate):
            nonlocal phase, verified_plan, verification_cost_s
            if candidate.flight_time in verification_cache:
                return verification_cache[candidate.flight_time]
            previous_phase = phase
            phase = 'fine verification'
            before = time.monotonic()
            try:
                log(f"Verifying tf={candidate.flight_time:.3f} s at half the integration step")
                checked = self.rollout(initial, candidate.flight_time, start_s, cfg.rollout_step_s / 2,
                                       heartbeat=heartbeat)
                heartbeat(checked.elapsed_s, checked.flight_time)
                verification_cache[candidate.flight_time] = checked
                verification_cost_s = max(verification_cost_s, time.monotonic() - before)
                if checked.feasible and (verified_plan is None or checked.fuel_used < verified_plan.fuel_used):
                    verified_plan = checked
                    log(f"Retained verified plan: tf={checked.flight_time:.3f} s, fuel={checked.fuel_used:.3f} kg")
                return checked
            finally:
                phase = previous_phase

        def evaluate(tf):
            """
            Evaluates current spacecraft state at time
            """
            tf = float(np.asarray(tf).ravel()[0])
            heartbeat(0., tf)
            if tf not in cache:
                if len(cache) >= cfg.max_search_evaluations:
                    raise _EvaluationLimit()
                log(f"{phase}: trajectory {len(cache)+1}/{cfg.max_search_evaluations}, tf={tf:.2f} s")
                result = self.rollout(initial, tf, start_s, heartbeat=heartbeat)
                cache[tf] = result
                log(f"{result.stop_reason} at {result.elapsed_s:.1f} s: "
                    f"target errors {result.position_error:.2f} m, {result.velocity_error:.3f} m/s; "
                    f"fuel {result.fuel_used:.3f} kg; feasible={result.feasible}; "
                    f"peak requested {result.peak_requested_thrust:.1f} N, "
                    f"throttle saturated {result.saturated_s:.1f} s")
                if result.feasible and verified_plan is None:
                    # Retain a checked plan before optional refinement can
                    # consume the entire wall-time budget.
                    verify(result)
            return cache[tf]


        def violation(result):
            return max(result.position_error / cfg.position_tolerance,
                       result.velocity_error / cfg.velocity_tolerance,
                       1 + result.path_violation_m / max(.01, cfg.surface_tolerance_m),
                       1 + (cfg.dry_mass - result.final_mass) / initial.mass) - 1

        times = np.linspace(cfg.tf_min, cfg.tf_max, cfg.search_samples)
        log(f"Searching [{cfg.tf_min:g}, {cfg.tf_max:g}] s: {cfg.search_samples} grid candidates, "
            f"up to {cfg.max_search_evaluations} search evaluations; "
            f"wall-time limit {cfg.max_search_wall_s:g} s including verification.")

        results = []
        try:
            for t in times:
                result = evaluate(t)
                results.append(result)
        except DescentSearchLimit:
            if verified_plan is None:
                raise
            log('Wall-time limit reached during grid search; returning the retained verified plan.')
            return verified_plan
        # Local constrained searches from the best feasible/least-violating grid seeds.
        seeds = sorted(results, key=lambda r: (max(0., violation(r)), r.fuel_used))[:3]
        phase = "feasibility search"
        evaluation_limited = False
        refinement_limited = False

        try:
            # Fuel consumed before an early impact is not a landing objective.
            if not any(r.feasible for r in results):
                for seed in seeds:
                    minimize(lambda x: max(0., violation(evaluate(x))), [seed.flight_time],
                             method="Powell", bounds=[(cfg.tf_min, cfg.tf_max)],
                             options={"maxiter": 5, "maxfev": 12, "xtol": 0.05, "ftol": 1e-4})
                    if any(r.feasible for r in cache.values()):
                        break

            phase = "fuel search"
            seeds = sorted((r for r in cache.values() if r.feasible), key=lambda r: r.fuel_used)[:3]

            for seed in seeds:
                minimize(lambda x: evaluate(x).fuel_used, [seed.flight_time], method="SLSQP",
                         bounds=[(cfg.tf_min, cfg.tf_max)],
                         constraints=[{"type": "ineq", "fun": lambda x: -violation(evaluate(x))}],
                         options={"maxiter": 20, "ftol": 1e-5, "eps": 0.05})

        except _EvaluationLimit:
            evaluation_limited = True
            log("Search evaluation limit reached; verifying any feasible candidates already found.")
        except _RefinementLimit:
            refinement_limited = True
            log('Stopping fuel refinement to reserve time for fine verification.')
        except DescentSearchLimit:
            if verified_plan is None:
                raise
            log('Wall-time limit reached during refinement; returning the retained verified plan.')
            return verified_plan
        candidates = sorted((r for r in cache.values() if r.feasible), key=lambda r: r.fuel_used)
        phase = "fine verification"

        for candidate in candidates[:8]:
            try:
                checked = verify(candidate)
            except DescentSearchLimit:
                if verified_plan is None:
                    raise
                log('Wall-time limit reached during verification; returning the retained verified plan.')
                return verified_plan
            if checked.feasible:
                log(f"Verified feasible plan: tf={verified_plan.flight_time:.3f} s, fuel={verified_plan.fuel_used:.3f} kg"
                    + (" (search evaluation limit reached; best verified candidate found)" if evaluation_limited else
                       " (fuel refinement stopped to preserve verification time)" if refinement_limited else ""))
                return verified_plan

        if verified_plan is not None:
            log('Refined candidates failed verification; returning the retained verified plan.')
            return verified_plan

        best = min(cache.values(), key=lambda r: violation(r))

        if evaluation_limited:
            raise DescentSearchLimit(
                f"A2G reached {cfg.max_search_evaluations} search evaluations without a verified plan. "
                f"Best sampled trajectory stopped due to {best.stop_reason} at {best.elapsed_s:.1f} s "
                f"of requested {best.flight_time:.1f} s; errors {best.position_error:.2f} m, "
                f"{best.velocity_error:.3f} m/s. Search limit is not proof of infeasibility."
            )

        raise InfeasibleDescent(
            f"No verified feasible A2G descent in [{cfg.tf_min:g}, {cfg.tf_max:g}] s. "
            f"Best sampled target errors at rollout stop: {best.position_error:.3f} m, "
            f"{best.velocity_error:.3f} m/s; remaining mass {best.final_mass:.3f} kg. "
            f"Stop reason: {best.stop_reason} at {best.elapsed_s:.1f} s of {best.flight_time:.1f} s. "
            "Check initial conditions, target, fuel, thrust/attitude authority and time bounds."
        )
