from Basilisk.utilities import SimulationBaseClass, macros
from Basilisk.simulation import spacecraft

from .IMU import IMU
from .StarTracker import StarTracker


class SensorsManager:
    def __init__(self, Vehicle, sampling_ns=0.01):
        self.sampling_ns = sampling_ns   #how often to sample sensors in ns

        self.vehicle = Vehicle                              #Holds the vehicle class for assigning tasks and models

    def add_imu(self, name, sc_state_msg, config=None):
        """
        Mainly just makes an instance of the IMU in the simulation
        :param name: ID Of the IMU
        :param sc_state_msg: Basilisk State Message (? What is a state message)
        :param config: Any additional configuration that is required (?)
        :return:
        """
        imu = IMU(name)                                     #creates an instance of the IMU class
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
        st = StarTracker(name)                              #creates an instance of the StarTracker class
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
        self.vehicle.altimeters[name] = alt  # store ST in the array of STs on board
        self.vehicle.loggers[f"star:{name}"] = rec  # store logger in the array of loggers for the craft
        return alt

    def register_to_task(self, sim, task_name):
        """
        Registers the basilisk modules to the sensors
        :param sim: Simulation environment variable
        :param task_name: Name/ID of the task
        :return:
        """
        for imu in self.vehicle.imus.values():
            sim.AddModelToTask(task_name, imu.model)    #Add the model of the sensor to the simulation task variable
            sim.AddModelToTask(task_name, imu.recorder) #Add recorder to the simulation task variable

        for st in self.vehicle.star_trackers.values():
            sim.AddModelToTask(task_name, st.model)     #Add the model of the sensor to the simulation task variable
            sim.AddModelToTask(task_name, st.recorder)  #Add recorder to the simulation task variable

        for alt in self.vehicle.altimeters.values():
            sim.AddModelToTask(task_name, alt)
            sim.AddModelToTask(task_name, alt.recorder)

    def output(self, index):
        """
        Formats and returns the sensor output
        :param index: The index to categorize sensor logs
        :return:
        """
        return {
            "imu": {name: imu.output(index) for name, imu in self.vehicle.imus.items()},                #IMU Logging and Output
            "star_tracker": {name: st.output(index) for name, st in self.vehicle.star_trackers.items()} #ST Logging and Output
            }

