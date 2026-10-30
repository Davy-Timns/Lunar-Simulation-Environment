# FaultInjection.py
# SERVAL Sensor Fault Injection Module
#
# Supported faults:
# 1. Transient Spike
# 2. Step Fault
# 3. Permanent Stuck Fault
# 4. Gradual Drift
#
# Sensors:
# - IMU (Gyroscope and Accelerometer)
# - Star Tracker
#
# All times are in seconds.
# Axis: 0 = X, 1 = Y, 2 = Z

import numpy as np
import time as t
import random

def random_bool(probability):
    """Return True with the given probability (0–1)."""
    return random.random() < probability

class FaultInjector:

    def __init__(self, dt, fault_count: int = 1, seed=None, runtime_seed=None, start_time_bounds_s=None, step_interval_bounds_s=None, dead_duration_max_s=1000., drift_per_second=False, total_run_fault_rate= 0.5):

        self.fault_types = ["Drift", "Transient", "Step", "Permanent", "Dead"]
        self.fault_locations = ["IMU_x", "IMU_y", "IMU_z", "IMU_xrot", "IMU_yrot", "IMU_zrot", "ST_Q1", "ST_Q2", "ST_Q3", "ST_Q4", "Altimeter","Trans_x","Trans_y"]

        self.is_fault_injected = random_bool(total_run_fault_rate)

        # Random number generator
        self.rng = np.random.default_rng(seed)
        self.start_time_bounds_s = start_time_bounds_s
        self.step_interval_bounds_s = step_interval_bounds_s
        self.dead_duration_max_s = float(dead_duration_max_s)
        self.drift_per_second = drift_per_second
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Fault-injection period must be positive seconds")
        for bounds in (start_time_bounds_s, step_interval_bounds_s):
            if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all()
                                       or bounds[0] < 0 or bounds[1] < bounds[0]):
                raise ValueError("Fault timing bounds must be ordered nonnegative seconds")
        if not np.isfinite(self.dead_duration_max_s) or self.dead_duration_max_s < 0:
            raise ValueError("Dead fault duration must be nonnegative seconds")

        # Probability of each fault occurring
        self.fault_count = fault_count

        self.sim_timestep = dt

        # Schedule times and intervals are in seconds relative to simulation initialization.
        self.fault_schedule = []

        # Store previous sensor measurements
        self.previous_values = {}

        # Store frozen measurements for permanent faults
        self.stuck_values = {}

        # Generate all fault times and magnitudes ONCE
        if self.is_fault_injected: #Only if run is going to have faults
            print("FAULTS INCLUDED ON THIS RUN")
            self.generate_fault_schedule()
        else:
            print("NO FAULTS ON THIS RUN")
            self.fault_schedule = []

        if runtime_seed is not None:
            self.rng = np.random.default_rng(runtime_seed)

    def generate_fault_schedule(self):
        t1 = self.rng.uniform(1.001, 1.5)
        t2 = self.rng.uniform(t1, 13)
        transient_fault_seed_range = [t1,t2]

        s1 = self.rng.uniform(1.001, 1.5)
        s2 = self.rng.uniform(s1, 8)
        step_fault_seed_range = [s1,s2]

        p1 = self.rng.uniform(0, 1.5)
        p2 = self.rng.uniform(t1, 5)
        permanent_fault_seed_range = [p1,p2]

        d1 = self.rng.uniform(0.00001, 0.001)
        d2 = self.rng.uniform(d1, 0.01)
        drift_fault_seed_range = [d1,d2]

        for i in range(self.fault_count):
            fault_type = self.rng.choice(self.fault_types)
            fault_time = self.rng.uniform(*(self.start_time_bounds_s or (self.sim_timestep * 100, self.sim_timestep * 50000)))
            fault_location = self.rng.choice(self.fault_locations)
            if fault_type == "Transient":
                fault_magnitude = self.rng.uniform(transient_fault_seed_range[0], transient_fault_seed_range[1])
                self.fault_schedule.append((fault_location,fault_type, fault_time, fault_magnitude))

            elif fault_type == "Step":
                fault_magnitude = self.rng.uniform(step_fault_seed_range[0], step_fault_seed_range[1])
                step_interval = self.rng.uniform(*(self.step_interval_bounds_s or (self.sim_timestep * 1000, self.sim_timestep * 5000)))
                step_variation = self.rng.uniform(0.001, 0.1)
                fault_magnitude_variations = self.rng.uniform(0.001, 1)
                self.fault_schedule.append((fault_location,fault_type, fault_time, fault_magnitude, step_interval, step_variation, fault_magnitude_variations))

            elif fault_type == "Permanent":
                fault_magnitude = self.rng.uniform(permanent_fault_seed_range[0], permanent_fault_seed_range[1])
                self.fault_schedule.append((fault_location,fault_type, fault_time, fault_magnitude))

            elif fault_type == "Drift":
                fault_magnitude = self.rng.uniform(drift_fault_seed_range[0], drift_fault_seed_range[1])
                self.fault_schedule.append((fault_location,fault_type, fault_time, fault_magnitude))

            elif fault_type == "Dead":
                fault_end_time = self.rng.uniform(fault_time, fault_time + self.dead_duration_max_s)
                self.fault_schedule.append((fault_location,fault_type, fault_time, fault_end_time))
        print("--Fault Schedule--\n",self.fault_schedule)

    def transient_spike(self, measurement, time, start_time, magnitude):
        """Multiply one sample; time and start_time are integer tick indices."""
        return measurement * magnitude if time == start_time else measurement

    def step_fault(self, measurement, time, start_time, magnitude,
                   step_interval, step_variation, magnitude_variation, fault_id):
        """Add a piecewise-constant offset. step_interval is in seconds."""
        if time < start_time:
            return measurement

        state = self._step_states.get(fault_id)
        if state is None:
            state = {"offset": magnitude, "next_tick": start_time}
            self._step_states[fault_id] = state

        # Generate each plateau once, rather than drawing new noise every tick.
        while time >= state["next_tick"]:
            if state["next_tick"] != start_time:
                state["offset"] += self.rng.uniform(-magnitude_variation, magnitude_variation)
            interval_s = step_interval * (1.0 + self.rng.uniform(-step_variation, step_variation))
            state["next_tick"] += max(1, self._seconds_to_tick(interval_s))

        return measurement + state["offset"]

    def permanent_fault(self, measurement, time, start_time, magnitude):
        """Replace the measurement with a fixed constant from activation onward."""
        return magnitude if time >= start_time else measurement

    def gradual_fault(self, measurement, time, start_time, drift_rate):
        """Add per-tick drift, or a per-second rate when explicitly configured."""
        if time < start_time:
            return measurement
        increment = drift_rate * self.sim_timestep if self.drift_per_second else drift_rate
        return measurement + increment * (time - start_time + 1)

    def dead_fault(self, measurement, time, start_time, end_time, fault_id):
        """Hold the raw sample captured at activation over [start_time, end_time)."""
        if not start_time <= time < end_time:
            return measurement
        if fault_id not in self.stuck_values:
            self.stuck_values[fault_id] = measurement
        return self.stuck_values[fault_id]

    def _seconds_to_tick(self, seconds):
        """Select the first sampling tick at or after a scheduled time."""
        # nextafter avoids rounding an exact boundary such as 0.3 / 0.1
        # upward because of floating-point representation.
        return int(np.ceil(np.nextafter(float(seconds) / self.sim_timestep, -np.inf)))

    def reset_fault_state(self):
        """Reset runtime state for a new run; retain the existing schedule/RNG."""
        self.previous_values.clear()
        self.stuck_values.clear()
        self._step_states = {}
        self._fi_next_tick = 0

    def FI(self, current_timestep, measurements):
        """Return a new flat measurement dict, once per tick starting at zero.

        Quaternion fault magnitudes pass through a sigmoid into [-1, 1].
        Final quaternion components are clipped to [-1, 1].
        Concurrent faults compose in schedule-list order.
        """
        quaternion_keys = {"ST_Q1", "ST_Q2", "ST_Q3", "ST_Q4"}

        def sigmoid(value):
            """Map a fault magnitude into [-1, 1], preserving its sign."""
            return float(np.tanh(float(value) / 2.0))

        if not np.isfinite(self.sim_timestep) or self.sim_timestep <= 0:
            raise ValueError("sim_timestep must be positive seconds")

        if (isinstance(current_timestep, (bool, np.bool_)) or
                not isinstance(current_timestep, (int, np.integer))):
            raise TypeError("current_timestep must be an integer tick index")

        # Reject missing channels before advancing any fault state.
        missing = {fault[0] for fault in self.fault_schedule} - measurements.keys()
        if missing:
            raise KeyError(f"Missing fault-injection channels: {sorted(missing)}")

        if not hasattr(self, "_fi_next_tick"):
            self.reset_fault_state()

        if current_timestep != self._fi_next_tick:
            raise ValueError(
                f"FI expected tick {self._fi_next_tick}, got {current_timestep}; "
                "call FI exactly once for every tick, starting at zero"
            )

        raw = dict(measurements)
        faulty = dict(raw)

        for fault_id, fault in enumerate(self.fault_schedule):
            sensor_key, fault_type, start_s = fault[:3]
            start_tick = self._seconds_to_tick(start_s)
            value = float(faulty[sensor_key])

            # Dead faults contain an end time rather than a magnitude.
            if fault_type in {"Transient", "Step", "Permanent", "Drift"}:
                magnitude = fault[3]
                if sensor_key in quaternion_keys:
                    magnitude = sigmoid(magnitude)

            if fault_type == "Transient":
                value = self.transient_spike(
                    value, current_timestep, start_tick, magnitude
                )

            elif fault_type == "Step":
                value = self.step_fault(
                    value, current_timestep, start_tick,
                    magnitude, fault[4], fault[5], fault[6], fault_id
                )

            elif fault_type == "Permanent":
                value = self.permanent_fault(
                    value, current_timestep, start_tick, magnitude
                )

            elif fault_type == "Drift":
                value = self.gradual_fault(
                    value, current_timestep, start_tick, magnitude
                )

            elif fault_type == "Dead":
                end_tick = self._seconds_to_tick(fault[3])
                if start_tick <= current_timestep < end_tick:
                    value = self.dead_fault(
                        float(raw[sensor_key]), current_timestep, start_tick,
                        end_tick, fault_id
                    )

            else:
                raise ValueError(f"Unknown fault type: {fault_type!r}")

            faulty[sensor_key] = value

        # Bound the final outputs after all overlapping faults are composed.
        for sensor_key in quaternion_keys.intersection(faulty):
            faulty[sensor_key] = float(
                np.clip(faulty[sensor_key], -1.0, 1.0)
            )

        self.previous_values.update(raw)
        self._fi_next_tick += 1
        return faulty

    def schedule_metadata(self):
        records = []
        for fault_id, fault in enumerate(self.fault_schedule):
            channel, kind, start_s = fault[:3]
            tick = self._seconds_to_tick(start_s)
            record = dict(fault_id=fault_id, channel=channel, type=kind,
                          requested_start_s=float(start_s), start_tick=tick,
                          actual_start_s=tick*self.sim_timestep)
            if kind == "Dead":
                record.update(requested_end_s=float(fault[3]), end_tick=self._seconds_to_tick(fault[3]))
            else:
                magnitude = float(fault[3])
                effective = float(np.tanh(magnitude/2)) if channel in {"ST_Q1","ST_Q2","ST_Q3","ST_Q4"} else magnitude
                record.update(magnitude=magnitude, effective_magnitude=effective, operation={"Transient":"multiply", "Step":"add", "Permanent":"replace", "Drift":"add_drift"}[kind])
                if kind == "Drift":
                    record["rate_units"] = "channel_units_per_second" if self.drift_per_second else "channel_units_per_tick"
                if kind == "Step":
                    record.update(step_interval_s=float(fault[4]), step_variation=float(fault[5]), magnitude_variation=float(fault[6]))
            records.append(record)
        return records

    def telemetry(self, tick):
        """Schedule activity and realized multiplier/offset/replacement per fault."""
        active = np.zeros(len(self.fault_schedule), dtype=bool)
        realized = np.full(len(self.fault_schedule), np.nan)
        for fault_id, fault in enumerate(self.fault_schedule):
            channel, kind, start_s = fault[:3]
            start = self._seconds_to_tick(start_s)
            enabled = (tick == start if kind == "Transient" else
                       start <= tick < self._seconds_to_tick(fault[3]) if kind == "Dead" else tick >= start)
            active[fault_id] = enabled
            if not enabled:
                continue
            if kind == "Dead":
                realized[fault_id] = self.stuck_values.get(fault_id, np.nan)
                continue
            magnitude = float(fault[3])
            if channel in {"ST_Q1","ST_Q2","ST_Q3","ST_Q4"}:
                magnitude = float(np.tanh(magnitude/2))
            if kind == "Step":
                magnitude = self._step_states[fault_id]["offset"]
            elif kind == "Drift":
                magnitude *= (tick-start+1) * (self.sim_timestep if self.drift_per_second else 1.)
            realized[fault_id] = magnitude
        return active, realized
