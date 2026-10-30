from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft

import numpy as np
from Basilisk.architecture import messaging
from .IMU import IMU
from .StarTracker import StarTracker


class SensorsManager:
    def __init__(self, Vehicle, sampling_ns=10_000_000, seeds=None):
        self.sampling_ns = sampling_ns   # Sensor sampling period in integer nanoseconds
        self.seeds = seeds or {}

        self.vehicle = Vehicle                              #Holds the vehicle class for assigning tasks and models
        # FI is run once per tick by the navigation acquisition stage.

    def add_imu(self, name, sc_state_msg, config=None):
        """
        Mainly just makes an instance of the IMU in the simulation
        :param name: ID Of the IMU
        :param sc_state_msg: Basilisk State Message (? What is a state message)
        :param config: Any additional configuration that is required (?)
        :return:
        """
        imu = IMU(name)
        if "imu" in self.seeds:
            imu.model.RNGSeed = self.seeds["imu"]
        imu.connect_state(sc_state_msg)                     #connects IMU input message to the spacecraft state output message (??)
        if config:
            imu.configure(config)                           #add additional config parameters such as noise, bias, and FOV (for some sensors).

        rec = imu.attach_recorder(self.sampling_ns)         #Initialize the recorder/logger for the imu
        self.vehicle.imus[name] = imu                               #store IMU in the array of IMUs on board
        self.vehicle.loggers[f"imu:{name}"] = rec                   #store logger in the array of loggers for the craft
        return imu                                          #returns the IMU instance

    def add_star_tracker(self, name, sc_state_msg, config=None):
        """
        Mainly just makes an instance of the Star Tracker in the simulation
        :param name: ID Of the ST
        :param sc_state_msg: Basilisk State Message (? What is a state message)
        :param config: Any additional configuration that is required (?)
        :return:
        """
        st = StarTracker(name)
        if "star_tracker" in self.seeds:
            st.model.RNGSeed = self.seeds["star_tracker"]
        st.connect_state(sc_state_msg)                      #connects IMU input message to the spacecraft state output message (??)
        if config:
            st.configure(config)                            #add additional config parameters such as noise, bias, and FOV (for some sensors).

        rec = st.attach_recorder(self.sampling_ns)          #Initialize the recorder/logger for the ST
        self.vehicle.star_trackers[name] = st                       #store ST in the array of STs on board
        self.vehicle.loggers[f"star:{name}"] = rec                  #store logger in the array of loggers for the craft
        return st                                           #returns the ST instance

    def add_altimeter(self,altimeterclass,sc_state_msg, config=None):
        """
        Mainly just makes an instance of the Mujoco Altimeter in the simulation
        :param altimeterclass: Reference to the instance of the altimeter class, assigned in initialize_sensors function
        :param name: Defined in the Altimeter.py class, NOT HERE
        :param sc_state_msg:
        :param config:
        :return:
        """
        alt = altimeterclass
        name = alt.name
        alt.scStateInMsg.subscribeTo(sc_state_msg)
        if config:
            alt.configure(config)
        rec = alt.sensorOutMsg.recorder(self.sampling_ns)  # Initialize the recorder/logger for the ST
        alt.recorder = rec
        alt.cleanSensorOutMsg = messaging.NavTransMsg()
        alt.clean_recorder = alt.cleanSensorOutMsg.recorder(self.sampling_ns)
        self.vehicle.altimeters[name] = alt  # store ST in the array of STs on board
        self.vehicle.loggers[f"star:{name}"] = rec  # store logger in the array of loggers for the craft
        return alt

    def register_to_task(self, sim, task_name):
        """Producers (-1), navigation/FI (-3), sensor recorders (-4)."""
        for imu in self.vehicle.imus.values():
            sim.AddModelToTask(task_name, imu.model, ModelPriority=-1)
            sim.AddModelToTask(task_name, imu.recorder, ModelPriority=-4)
            sim.AddModelToTask(task_name, imu.clean_recorder, ModelPriority=-4)
        for st in self.vehicle.star_trackers.values():
            sim.AddModelToTask(task_name, st.model, ModelPriority=-1)
            sim.AddModelToTask(task_name, st.recorder, ModelPriority=-4)
            sim.AddModelToTask(task_name, st.clean_recorder, ModelPriority=-4)
        for alt in self.vehicle.altimeters.values():
            sim.AddModelToTask(task_name, alt, ModelPriority=-1)
            sim.AddModelToTask(task_name, alt.recorder, ModelPriority=-4)
            sim.AddModelToTask(task_name, alt.clean_recorder, ModelPriority=-4)

    def output(self, index):
        """Read the existing public (post-FI) recorders; never inject twice."""
        return {
            "imu": {name: imu.output(index) for name, imu in self.vehicle.imus.items()},
            "star_tracker": {name: st.output(index) for name, st in self.vehicle.star_trackers.items()},
        }


def _copy_payload(payload):
    # Basilisk payload setters copy their fixed-size arrays.
    # Reader-returned SWIG payloads do not expose the convenience __fields__
    # method; use the exported messaging payload factory for the same type.
    factory = getattr(messaging, type(payload).__name__)
    result = factory()
    for field in factory.__fields__():
        setattr(result, field, getattr(payload, field))
    return result


class SharedSensorFaults:
    def __init__(self, imu, star_tracker, altimeter):
        self.imu = imu
        self.star = star_tracker
        self.altimeter = altimeter
        self._sources = {
            "imu": imu.model.sensorOutMsg,
            "star": star_tracker.model.sensorOutMsg,
            "range": altimeter.sensorOutMsg,
        }
        self._readers = {key: msg.addSubscriber()
                         for key, msg in self._sources.items()}
        self.reset()

    def reset(self):
        self._samples = {}
        self.ideal_values = {}

    def acquire(self, range_rng, range_sigma_m):
        for key, reader in self._readers.items():
            if not reader.isWritten():
                continue
            stamp = reader.timeWritten()
            if key in self._samples and stamp == self._samples[key][0]:
                # The public message may contain our previous faulty output.
                # Reuse the cached CLEAN sample until its producer writes anew.
                continue
            payload = _copy_payload(reader())
            if key == "imu":
                ideal = self.imu.model.trueValues
                self.ideal_values.update(zip(("IMU_x", "IMU_y", "IMU_z"), ideal.AccelPlatform))
                self.ideal_values.update(zip(("IMU_xrot", "IMU_yrot", "IMU_zrot"), ideal.AngVelPlatform))
            elif key == "star":
                self.ideal_values.update(zip(("ST_Q1", "ST_Q2", "ST_Q3", "ST_Q4"), self.star.model.trueValues.qInrtl2Case))
            else:
                self.ideal_values["Altimeter"] = float(payload.r_BN_N[2])
            if key == "range":
                distance = float(payload.r_BN_N[2])
                if (getattr(self.altimeter, "range_valid", False)
                        and np.isfinite(distance) and distance >= 0):
                    distance += range_rng.normal(0, range_sigma_m)
                vector = list(payload.r_BN_N)
                vector[2] = distance
                payload.r_BN_N = vector
            self._samples[key] = (stamp, payload)

        values = dict.fromkeys((
            "IMU_x", "IMU_y", "IMU_z", "IMU_xrot", "IMU_yrot", "IMU_zrot",
            "ST_Q1", "ST_Q2", "ST_Q3", "ST_Q4", "Altimeter",
        ), np.nan)
        if "imu" in self._samples:
            sample = self._samples["imu"][1]
            values.update(zip(("IMU_x", "IMU_y", "IMU_z"), sample.AccelPlatform))
            values.update(zip(("IMU_xrot", "IMU_yrot", "IMU_zrot"), sample.AngVelPlatform))
        if "star" in self._samples:
            values.update(zip(("ST_Q1", "ST_Q2", "ST_Q3", "ST_Q4"),
                              self._samples["star"][1].qInrtl2Case))
        if "range" in self._samples:
            values["Altimeter"] = float(self._samples["range"][1].r_BN_N[2])
        return values

    def publish(self, checked, module_id):
        clean_messages = {
            "imu": self.imu.cleanSensorOutMsg,
            "star": self.star.cleanSensorOutMsg,
            "range": self.altimeter.cleanSensorOutMsg,
        }
        for key, (stamp, clean) in self._samples.items():
            clean_messages[key].write(clean, stamp, module_id)
            public = _copy_payload(clean)
            if key == "imu":
                public.AccelPlatform = [checked[k] for k in ("IMU_x", "IMU_y", "IMU_z")]
                public.AngVelPlatform = [checked[k] for k in ("IMU_xrot", "IMU_yrot", "IMU_zrot")]
            elif key == "star":
                public.qInrtl2Case = [checked[k] for k in ("ST_Q1", "ST_Q2", "ST_Q3", "ST_Q4")]
            else:
                vector = list(public.r_BN_N)
                vector[2] = float(checked["Altimeter"])
                public.r_BN_N = vector
                self.altimeter.altitude = float(checked["Altimeter"])
            # Preserve the acquisition timestamp: overwriting a value must not
            # make a stale or frozen source appear to have acquired a new sample.
            self._sources[key].write(public, stamp, module_id)
