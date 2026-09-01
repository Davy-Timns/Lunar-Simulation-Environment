from Basilisk.utilities import SimulationBaseClass, macros, vizSupport, unitTestSupport
from Basilisk.simulation import spacecraft, extForceTorque
from Spacecraft.Mujoco import MujocoPhysicsEngine as MPE

class Vehicle:
    def __init__(self, sim, name, sampling_ns, spacecraft_attitude_MRP, spacecraft_attitude_rate, spacecraft_mass, spacecraft_inertia):
        self.imus = {}              # sensor IMU initialization, if you have multiple IMUs they get stored here
        self.star_trackers = {}     # sensor Star Tracker initialization, if you have multiple IMUs they get stored here
        self.altimeters = {}
        self.lander = spacecraft.Spacecraft()        # Create a spacecraft instance from the spacecraft basilisk module
        self.lander.ModelTag = name             # Tag the spacecraft with a name
        self.sampling_ns = sampling_ns
        self.terrain = None

        # External force/torque module
        self.contact_force = extForceTorque.ExtForceTorque()
        self.contact_force.ModelTag = "TerrainContactForce"
        self.lander.addDynamicEffector(self.contact_force)
        self.lander.hub.sigma_BNInit = spacecraft_attitude_MRP
        self.lander.hub.omega_BN_BInit = spacecraft_attitude_rate
        self.lander.hub.mHub = spacecraft_mass  # CoM currently undefined to my knowledge
        self.lander.hub.IHubPntBc_B = spacecraft_inertia



        self.sim = sim
        self.SM = None                          #make none because it will be applied later


        self.loggers = {}                                   #Holds the logger instances


        #self.sim.AddModelToTask("record", self.lander)  # Add spacecraft to task




    def initialize_sensors(self, SM, samp, LF:int):
        self.SM = SM

        # Add sensors to spacecraft
        self.SM.add_imu("IMU", self.lander.scStateOutMsg)
        self.SM.add_star_tracker("ST", self.lander.scStateOutMsg)
        self.SM.altimeter = MPE.initialize_mujoco(LF,self.lander.hub.mHub, self.lander.hub.IHubPntBc_B, samp)

        self.SM.add_altimeter(self.SM.altimeter,self.lander.scStateOutMsg)

        # Connect MuJoCo contact loads to Basilisk.
        self.contact_force.cmdForceInertialInMsg.subscribeTo(
            self.SM.altimeter.forceOutMsg
        )
        self.contact_force.cmdTorqueInMsg.subscribeTo(
            self.SM.altimeter.torqueOutMsg
        )


        self.SM.register_to_task(self.sim, "record")  # Register task to the simulation environment
        sc_recorder = self.lander.scStateOutMsg.recorder()
        self.sim.AddModelToTask("record", sc_recorder)
        self.sim.AddModelToTask("record", self.terrain)
        self.sim.AddModelToTask("record", self.contact_force)

        self.sc_recorder = sc_recorder



    def initialize_recorder(self,sim):
        """
        has its own function because if the recorder setup is not timed properly then Vizard wont run
        :param sim:
        :return:
        """

        scRec = self.lander.scStateOutMsg.recorder(self.sampling_ns)
        sim.AddModelToTask("record", scRec)


    def output(self):
        """
        Minimal Basilisk bootstrap, this will need to be majorly rewritten when the simulation environment is more defined,
        IE when we actually have a Spacecraft.py and an Environment.py.
        :param num_steps: Number of timesteps to run
        """



        # Return all logged data
        imu_output = {}
        for name, imu in self.imus.items():
            imu_output[name] = {
                field: getattr(imu.recorder, field)
                for field in imu.fields
            }

        st_output = {}
        for name, st in self.star_trackers.items():
            st_output[name] = {
                field: getattr(st.recorder, field)
                for field in st.fields
            }

        alt_output = {}
        for name, alt in self.altimeters.items():
            alt_output[name] = {
                "timeTag": alt.recorder.timeTag,
                "altitude": alt.recorder.r_BN_N,
            }





        return {
            "imu": imu_output,
            "star_tracker": st_output,
            "altimeter": alt_output,
            "true_data": self.lander_true_status()
        }

    def lander_true_status(self):
        lander_pos = self.sc_recorder.r_CN_N
        lander_vel = self.sc_recorder.v_CN_N
        lander_attitude = self.sc_recorder.sigma_BN
        lander_angular_rate = self.sc_recorder.omega_BN_B
        #lander_mass = self.sc_recorder.mHub
        #lander_CoM = self.sc_recorder.r_BcB_B
        return [lander_pos, lander_vel, lander_attitude, lander_angular_rate]
