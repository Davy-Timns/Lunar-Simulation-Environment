"""Numeric ML telemetry and physically assessed landing labels.

All numerical exports load with np.load(..., allow_pickle=False). Logging never
feeds truth or labels back into navigation/control.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from SimulationBatch import write_json

CHANNELS = ("IMU_x", "IMU_y", "IMU_z", "IMU_xrot", "IMU_yrot", "IMU_zrot",
            "ST_Q1", "ST_Q2", "ST_Q3", "ST_Q4", "Altimeter", "Trans_x", "Trans_y")
INNOVATIONS = {"gyro": (0, 3), "star": (3, 6), "command_acceleration": (6, 9),
               "accelerometer": (9, 12), "xy": (12, 14), "range": (14, 15)}
STATE_FIELDS = ("position_N_m[3]", "velocity_N_m_s[3]", "specific_acceleration_N_m_s2[3]",
                "quaternion_N_to_B_wxyz[4]", "angular_rate_B_rad_s[3]")


def state_vector(nav):
    q = Rotation.from_matrix(nav.ukf.rotation.T).as_quat()[[3, 0, 1, 2]]
    return np.r_[nav.ukf.translation, q, nav.ukf.omega]


class TickRecorder:
    """Store fixed-shape numeric columns in blocks, avoiding per-tick dict history."""
    def __init__(self, full_covariance=False, block_size=4096):
        self.full_covariance = full_covariance
        self.block_size = block_size
        self.blocks = {}
        self.count = 0
        self.row = None
        self.first_ns = None

    def begin(self, now, nav, ideal, noisy, faulted):
        from Basilisk.utilities import RigidBodyKinematics
        tick = nav.fi_tick - 1
        if tick != self.count:
            raise ValueError(f"Telemetry expected FI tick {self.count}, got {tick}")
        if self.first_ns is None:
            self.first_ns = int(now)
        expected_ns = self.first_ns + tick * round(nav.sensor_period_s * 1e9)
        if int(now) != expected_ns:
            raise ValueError(f"Fault-injection clock mismatch: expected {expected_ns}, got {now}")
        values = lambda data: np.array([data.get(k, np.nan) for k in CHANNELS], dtype=float)
        truth = np.full(16, np.nan)
        if nav.truth.isWritten():
            msg = nav.truth()
            c_NB = RigidBodyKinematics.MRP2C(msg.sigma_BN).T
            truth = np.r_[msg.r_BN_N, msg.v_BN_N, c_NB @ np.asarray(msg.nonConservativeAccelpntB_B),
                          RigidBodyKinematics.MRP2EP(msg.sigma_BN), msg.omega_BN_B]
        active, realized = nav.fault_injector.telemetry(tick)
        source_ns = np.full(13, -1, dtype=np.int64)
        for key, indices in (("imu", slice(0, 6)), ("star", slice(6, 10)), ("range", slice(10, 11))):
            if key in nav.shared_sensor_faults._samples:
                source_ns[indices] = nav.shared_sensor_faults._samples[key][0]
        if nav.truth.isWritten():
            source_ns[11:] = nav.truth.timeWritten()
        noisy_values, faulted_values = values(noisy), values(faulted)
        channel_active = np.zeros(len(CHANNELS), dtype=bool)
        for enabled, fault in zip(active, nav.fault_injector.fault_schedule):
            if enabled:
                channel_active[CHANNELS.index(fault[0])] = True
        valid = np.isfinite(faulted_values)
        valid[10] &= bool(nav.altimeter.range_valid) and faulted_values[10] >= 0
        valid[6:10] &= np.linalg.norm(faulted_values[6:10]) > 1e-12
        self.row = dict(tick=np.int64(tick), mission_time_ns=np.int64(now),
                        elapsed_time_s=(int(now) - self.first_ns) * 1e-9,
                        state_true=truth, state_true_valid=np.isfinite(truth), state_predicted=np.full(16, np.nan),
                        prediction_available=False, covariance_predicted_diag=np.full(15, np.nan),
                        measurements_ideal=values(ideal), measurements_noisy=noisy_values,
                        measurement_noise=noisy_values-values(ideal),
                        measurements_faulted=faulted_values, fault_effect=faulted_values-noisy_values,
                        fault_active=active, fault_active_by_channel=channel_active, fault_realized_value=realized,
                        sensor_source_time_ns=source_ns, sensor_fresh=source_ns == int(now),
                        sensor_valid=valid, measurements_ideal_valid=np.isfinite(values(ideal)),
                        measurements_noisy_valid=np.isfinite(noisy_values), innovation=np.full(15, np.nan),
                        innovation_variance=np.full(15, np.nan), innovation_available=np.zeros(15, dtype=bool),
                        range_update_performed=False, range_accepted=False, range_status="not_updated")
        if self.full_covariance:
            self.row["covariance_predicted"] = np.full((15, 15), np.nan)

    def innovation(self, name, residual, covariance):
        if self.row is None or name not in INNOVATIONS:
            return
        start, end = INNOVATIONS[name]
        self.row["innovation"][start:end] = residual
        self.row["innovation_variance"][start:end] = np.diag(covariance)
        self.row["innovation_available"][start:end] = True

    def predicted(self, nav):
        if self.row is None:
            return
        self.row["state_predicted"] = state_vector(nav)
        self.row["prediction_available"] = True
        if nav.cfg.filter_enabled:
            self.row["covariance_predicted_diag"] = np.diag(nav.ukf.P).copy()
            if self.full_covariance:
                self.row["covariance_predicted"] = nav.ukf.P.copy()

    def finish(self, nav):
        if self.row is None:
            return
        self.row.update(state_estimated=state_vector(nav), navigation_healthy=bool(nav.healthy),
                        navigation_failure_reason=nav.fault_reason or "", filter_enabled=nav.cfg.filter_enabled,
                        covariance_estimated_diag=np.diag(nav.ukf.P).copy() if nav.cfg.filter_enabled else np.full(15, np.nan))
        self.row["state_estimated_valid"] = np.isfinite(self.row["state_estimated"])
        if self.full_covariance:
            self.row["covariance_estimated"] = nav.ukf.P.copy() if nav.cfg.filter_enabled else np.full((15,15), np.nan)
        # First-tick initialization/invalid returns remain visible as rows.
        block_index, offset = divmod(self.count, self.block_size)
        for key, value in self.row.items():
            array = np.asarray(value)
            dtype = "U128" if array.dtype.kind in "US" else array.dtype
            blocks = self.blocks.setdefault(key, [])
            if len(blocks) <= block_index:
                blocks.append(np.empty((self.block_size,) + array.shape, dtype=dtype))
            blocks[block_index][offset] = array
        self.count += 1
        self.row = None

    def arrays(self):
        return {key: np.concatenate(parts, axis=0)[:self.count] for key, parts in self.blocks.items()}


def terrain_tilt_deg(c_NB, terrain, position):
    """Body +Z tilt from a central-difference terrain normal at contact."""
    xy = terrain.to_map(position)[:2]
    heights = [terrain.height(xy + delta) for delta in ((.5, 0), (-.5, 0), (0, .5), (0, -.5))]
    if any(h is None or not np.isfinite(h) for h in heights):
        return None
    normal_M = np.array([heights[1]-heights[0], heights[3]-heights[2], 1.])
    normal_N = terrain.to_inertial(normal_M) - terrain.to_inertial(np.zeros(3))
    normal_N /= np.linalg.norm(normal_N)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(c_NB[:, 2], normal_N), -1, 1))))


def landing_outcome(vehicle, criteria=None, execution_status="completed", error=None):
    limits = dict(max_touchdown_speed_m_s=3., max_touchdown_tilt_deg=10.,
                  max_final_angular_rate_rad_s=.2)
    limits.update(criteria or {})
    criteria = limits
    if not all(np.isfinite(value) and value >= 0 for value in criteria.values()):
        raise ValueError("Landing limits must be finite and nonnegative")
    controller = getattr(vehicle, "thruster_controller", None)
    monitor = getattr(vehicle, "touchdown_diagnostics", None)
    event = monitor.report() if monitor is not None else None
    outcome = dict(execution_status=execution_status, landing_outcome="failure", landing_success=False,
                   landed=event is not None, failure_reason=error, landing_diagnostics=event,
                   controller_status=getattr(controller, "status", None), criteria=criteria)
    if error or execution_status != "completed":
        return outcome
    if controller is None or not hasattr(controller, "settings"):
        outcome["failure_reason"] = "landing_assessment_unavailable_for_this_mode"
        return outcome
    cfg = controller.settings
    criteria.update(max_final_position_error_m=cfg.position_tolerance,
                    max_final_velocity_error_m_s=cfg.velocity_tolerance)
    criteria.setdefault("max_touchdown_distance_m", cfg.position_tolerance)
    criteria.setdefault("max_final_tilt_deg", criteria["max_touchdown_tilt_deg"])
    recorder = vehicle.sc_recorder
    if not len(recorder.times()):
        outcome["failure_reason"] = "no_state_samples"
        return outcome
    from Basilisk.utilities import RigidBodyKinematics
    t = float(recorder.times()[-1] * 1e-9)
    target_position, target_velocity, _ = controller.target.state(t)
    position, velocity = np.asarray(recorder.r_BN_N[-1]), np.asarray(recorder.v_BN_N[-1])
    c_NB = RigidBodyKinematics.MRP2C(recorder.sigma_BN[-1]).T
    contact = getattr(controller, "contact_sensor", None)
    settled = dict(time_s=t, position_N_m=position, velocity_N_m_s=velocity,
                   position_error_m=float(np.linalg.norm(position-target_position)),
                   velocity_error_m_s=float(np.linalg.norm(velocity-target_velocity)),
                   angular_rate_rad_s=float(np.linalg.norm(recorder.omega_BN_B[-1])),
                   tilt_from_local_vertical_deg=float(np.rad2deg(np.arccos(np.clip(np.dot(c_NB[:,2], vehicle.terrain_map.up_N), -1, 1)))) if hasattr(vehicle, "terrain_map") else None,
                   fuel_remaining_kg=float(vehicle.fuel_recorder.fuelMass[-1]) if hasattr(vehicle, "fuel_recorder") else None,
                   contact_at_end=bool(contact is not None and contact.collision))
    outcome["final_state_diagnostics"] = settled
    checks = dict(touchdown_detected=event is not None,
                  controller_landed=controller.status == "landed", contact_at_end=settled["contact_at_end"],
                  final_position_within_tolerance=settled["position_error_m"] <= cfg.position_tolerance,
                  final_velocity_within_tolerance=settled["velocity_error_m_s"] <= cfg.velocity_tolerance,
                  final_angular_rate_within_limit=settled["angular_rate_rad_s"] <= criteria["max_final_angular_rate_rad_s"])
    checks["final_tilt_within_limit"] = bool(settled["tilt_from_local_vertical_deg"] is not None
        and settled["tilt_from_local_vertical_deg"] <= criteria["max_final_tilt_deg"])
    checks["nonnegative_remaining_fuel"] = bool(settled["fuel_remaining_kg"] is not None
        and np.isfinite(settled["fuel_remaining_kg"]) and settled["fuel_remaining_kg"] >= 0)
    for key, field, limit in (("touchdown_speed_within_limit", "speed_m_s", "max_touchdown_speed_m_s"),
                              ("touchdown_tilt_within_limit", "tilt_from_local_vertical_deg", "max_touchdown_tilt_deg"),
                              ("touchdown_position_within_tolerance", "distance_to_intended_site_m", "max_touchdown_distance_m")):
        value = event.get(field) if event is not None else None
        checks[key] = bool(value is not None and np.isfinite(value) and value <= criteria[limit])
    outcome["success_checks"] = checks
    outcome["landing_success"] = all(checks.values())
    outcome["landing_outcome"] = "success" if outcome["landing_success"] else "failure"
    outcome["failure_reason"] = None if outcome["landing_success"] else ", ".join(k for k,v in checks.items() if not v)
    return outcome


def export_run(vehicle, config, settings, execution_status="completed", error=None):
    directory = Path(config["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    nav = getattr(vehicle, "navigation", None)
    data = nav.data_recorder.arrays() if nav is not None and nav.data_recorder is not None else {}
    schedule = nav.fault_injector.schedule_metadata() if nav is not None else []
    if hasattr(vehicle, "sc_recorder"):
        recorder = vehicle.sc_recorder
        data["truth_mission_time_ns"] = recorder.times()
        for field in ("r_BN_N", "v_BN_N", "r_CN_N", "v_CN_N", "sigma_BN", "omega_BN_B"):
            data["truth_" + field] = np.asarray(getattr(recorder, field))
    for name, recorder in getattr(vehicle, "thruster_recorders", {}).items():
        prefix = f"thruster_{name}_"
        data[prefix + "mission_time_ns"] = recorder.times()
        for field in ("thrustForce_B", "thrustFactor", "thrusterDirection"):
            data[prefix + field] = np.asarray(getattr(recorder, field))
    if hasattr(vehicle, "fuel_recorder"):
        data["fuel_mass_kg"] = np.asarray(vehicle.fuel_recorder.fuelMass)
        data["fuel_mission_time_ns"] = vehicle.fuel_recorder.times()
    monitor = getattr(vehicle, "touchdown_diagnostics", None)
    if monitor is not None and monitor.history:
        for key in monitor.history[0]:
            data["touchdown_" + key] = np.asarray([row[key] for row in monitor.history])
    # Retain native IMU integrated readings and other sensor payload fields too.
    for group, sensors in (("imu", vehicle.imus), ("star_tracker", vehicle.star_trackers)):
        for name, sensor in sensors.items():
            for stage, recorder in (("noisy", sensor.clean_recorder), ("faulted", sensor.recorder)):
                prefix = f"{group}_{name}_{stage}_"
                data[prefix + "mission_time_ns"] = recorder.times()
                for field in sensor.fields:
                    data[prefix + field] = np.asarray(getattr(recorder, field))
    try:
        outcome = landing_outcome(vehicle, config.get("landing_criteria"), execution_status, error)
    except Exception as exc:
        outcome = dict(execution_status="error", landing_success=False, landing_outcome="failure",
                       landed=False, failure_reason=f"landing_assessment_failed: {exc}", landing_diagnostics=None)
    # Outcome JSON is the authoritative label; the archive contains a scalar copy.
    data["landing_success"] = np.asarray(outcome["landing_success"])
    data["landing_outcome"] = np.asarray(outcome["landing_outcome"])
    temporary = directory / "data.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    temporary.replace(directory / "data.npz")
    write_json(directory / "outcome.json", outcome)
    write_json(directory / "fault_schedule.json", schedule)
    controller = vehicle.thruster_controller
    write_json(directory / "control_history.json", getattr(controller, "history", []))
    if controller is not None and getattr(controller, "plan", None) is not None:
        plan = controller.plan
        write_json(directory / "descent_plan.json", {key: value for key, value in vars(plan).items()
                   if isinstance(value, (str, int, float, bool, tuple, list, dict, np.ndarray))})
    import Basilisk
    import scipy
    import mujoco
    write_json(directory / "metadata.json", dict(schema_version=1, run_id=config["run_id"],
        library_versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                              Basilisk=Basilisk.__version__, mujoco=mujoco.__version__),
        source_sha256=config.get("source_sha256"), seeds=config["seeds"], settings=settings,
        navigation_settings=vars(nav.cfg) if nav is not None else None,
        initial_state=vars(nav.initial) if nav is not None else None,
        landing_target=vars(controller.target) if controller is not None and hasattr(controller, "target") else None,
        selected_landing_site=getattr(vehicle, "landing_site", None),
        imu_config={name: {field: getattr(sensor.model, field) for field in
                    ("sensorPos_B", "dcm_PB", "senRotBias", "senTransBias", "accelScale", "gyroScale", "PMatrixAccel", "PMatrixGyro", "navErrorsAccel", "navErrorsGyro")}
                    for name, sensor in vehicle.imus.items()},
        star_tracker_config={name: {field: getattr(sensor.model, field) for field in
                             ("dcm_CB", "PMatrix", "mrpErrors", "navErrors")}
                             for name, sensor in vehicle.star_trackers.items()},
        fault_options=nav.batch_config.get("fault_options") if nav is not None and nav.batch_config else None,
        descent_settings=vars(controller.settings) if hasattr(controller, "settings") else None,
        measurement_channels=CHANNELS,
        measurement_units=["m/s^2"]*3 + ["rad/s"]*3 + ["dimensionless"]*4 + ["m"]*3,
        state_fields=STATE_FIELDS, state_point="spacecraft body reference point B",
        covariance_order="r_N(3),v_N(3),a_N(3),delta_theta_B(3),omega_B(3)",
        prediction_definition="after gyro correction and propagation; before star/acceleration/terrain corrections",
        ideal_measurements="Basilisk native trueValues; ideal raycast range; truth-derived map XY",
        invalid_values="NaN retained in numerical arrays; JSON nonfinite values become null",
        innovation_slots=INNOVATIONS, fi_calls=nav.fi_tick if nav is not None else 0,
        recorded_ticks=nav.data_recorder.count if nav is not None and nav.data_recorder else 0,
        array_shapes={key: list(np.shape(value)) for key,value in data.items()},
        vizard_file="vizard_UnityViz.bin", vizard_enabled=settings.get("vizard_enabled", True)))
    return outcome
