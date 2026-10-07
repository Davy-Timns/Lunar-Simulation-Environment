from Basilisk.utilities import SimulationBaseClass, macros, vizSupport, unitTestSupport
from Basilisk.simulation import spacecraft, extForceTorque
from Spacecraft.Mujoco import MujocoPhysicsEngine as MPE
from Spacecraft.Thrusters.DummyController import DummyController

class Vehicle:
    def __init__(self, sim, name, sampling_ns, spacecraft_attitude_MRP, spacecraft_attitude_rate, spacecraft_mass, spacecraft_inertia):
        self.imus = {}              # sensor IMU initialization, if you have multiple IMUs they get stored here
        self.star_trackers = {}     # sensor Star Tracker initialization, if you have multiple IMUs they get stored here
        self.altimeters = {}
        self.engines = {}
        self.thruster_recorders = {}
        self.thruster_controller = None
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




    def initialize_thrusters(self, engines, enabled_engines=(), task_name="record", controller=None):
        """Attach named engines and a supplied or constant-on controller."""
        if self.thruster_controller is not None:
            raise RuntimeError("Thrusters have already been initialized.")
        engines = dict(engines)
        if controller is None:
            controller = DummyController(
                engines, self.sampling_ns * macros.NANO2SEC, enabled_engines
            )
        self.engines = engines
        self.thruster_controller = controller
        self.sim.AddModelToTask(task_name, controller, ModelPriority=20)
        for name, engine in engines.items():
            engine.attach(self.lander, self.sim, task_name, priority=10)
            recorder = engine.thruster.thrusterOutMsgs[0].recorder(self.sampling_ns)
            self.thruster_recorders[name] = recorder
            self.sim.AddModelToTask(task_name, recorder, ModelPriority=-10)

    def initialize_descent(self, engines, settings, target, main_name="MainEngine", task_name="record"):
        """Plan and attach A2G, gimbal/RCS attitude control, and a point-mass fuel tank.

        Call after setting the hub's initial state and before InitializeSimulation.
        The existing hub mass is the initial WET mass; its inertia is held fixed.
        """
        import numpy as np
        from Basilisk.simulation import fuelTank
        from Basilisk.utilities import RigidBodyKinematics
        from Spacecraft.Thrusters.DescentGuidance import DescentState
        from Spacecraft.Thrusters.DescentController import DescentController
        if self.thruster_controller is not None:
            raise RuntimeError("Choose either dummy or descent propulsion before initialization.")
        wet_mass = float(self.lander.hub.mHub)
        if wet_mass <= settings.dry_mass:
            raise ValueError("Wet mass must exceed descent dry mass.")
        if np.linalg.norm(self.lander.hub.r_BcB_B) > 1e-12:
            raise ValueError("The descent point-mass tank currently requires the hub COM at B.")
        main = engines[main_name]
        rcs = {name: e for name, e in engines.items() if name != main_name}
        initial = DescentState(
            np.asarray(self.lander.hub.r_CN_NInit, dtype=float).reshape(3),
            np.asarray(self.lander.hub.v_CN_NInit, dtype=float).reshape(3), wet_mass,
            RigidBodyKinematics.MRP2C(np.asarray(self.lander.hub.sigma_BNInit).reshape(3)).T,
            np.asarray(self.lander.hub.omega_BN_BInit, dtype=float).reshape(3),
        )
        tank_model = fuelTank.FuelTankModelConstantVolume()
        tank_model.propMassInit = wet_mass - settings.dry_mass
        tank_model.maxFuelMass = tank_model.propMassInit
        tank_model.radiusTankInit = 0.0  # centered point mass: fixed supplied spacecraft inertia
        tank_model.r_TcT_TInit = [[0.0], [0.0], [0.0]]
        tank = fuelTank.FuelTank()
        tank.ModelTag = "DescentFuelTank"
        tank.setTankModel(tank_model)
        # Plan before modifying the spacecraft; infeasible setup raises here.
        controller = DescentController(main, rcs, settings, target, initial,
                                       self.lander.hub.IHubPntBc_B,
                                       self.sampling_ns * macros.NANO2SEC,
                                       self.lander.scStateOutMsg, tank.fuelTankOutMsg,
                                       terrain_map=getattr(self,'terrain_map',None),
                                       landing_foot_points_B=getattr(getattr(self.SM, 'altimeter', None),
                                                                    'landing_foot_points_B', None))
        self.fuel_tank_model = tank_model  # preserve native object lifetimes
        self.fuel_tank = tank
        self.lander.hub.mHub = settings.dry_mass
        self.lander.addStateEffector(tank)
        for engine in engines.values():
            tank.addThrusterSet(engine.thruster)
        self.sim.AddModelToTask(task_name, tank, ModelPriority=-5)
        self.fuel_recorder = tank.fuelTankOutMsg.recorder(self.sampling_ns)
        self.sim.AddModelToTask(task_name, self.fuel_recorder, ModelPriority=-10)
        self.initialize_thrusters(engines, task_name=task_name, controller=controller)
        return controller

    def initialize_sensors(self, SM, samp, LF:int):
        self.SM = SM

        # Add sensors to spacecraft
        self.SM.add_imu("IMU", self.lander.scStateOutMsg)
        self.SM.add_star_tracker("ST", self.lander.scStateOutMsg)
        self.SM.altimeter = MPE.initialize_mujoco(LF,self.lander.hub.mHub, self.lander.hub.IHubPntBc_B, samp)

        self.SM.add_altimeter(self.SM.altimeter,self.lander.scStateOutMsg)
        self.SM.altimeter.scMassInMsg.subscribeTo(self.lander.scMassOutMsg)

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
                "time_s": alt.recorder.times() * 1e-9,
                "timeTag": alt.recorder.timeTag,
                "altitude": alt.recorder.r_BN_N,
            }





        return {
            "imu": imu_output,
            "star_tracker": st_output,
            "altimeter": alt_output,
            "thrusters": {
                name: {
                    "time_ns": recorder.times(),
                    "thrustForce_B": recorder.thrustForce_B,
                    "thrustFactor": recorder.thrustFactor,
                    "thrusterDirection": recorder.thrusterDirection,
                }
                for name, recorder in self.thruster_recorders.items()
            },
            "true_data": self.lander_true_status(),
            "descent": {
                "status": getattr(self.thruster_controller, "status", None),
                "history": getattr(self.thruster_controller, "history", []),
                "plan": getattr(self.thruster_controller, "plan", None),
                "fuel_mass": self.fuel_recorder.fuelMass if hasattr(self, "fuel_recorder") else None,
                "fuel_time_ns": self.fuel_recorder.times() if hasattr(self, "fuel_recorder") else None,
            }
        }

    def lander_true_status(self):
        lander_pos = self.sc_recorder.r_CN_N
        lander_vel = self.sc_recorder.v_CN_N
        lander_attitude = self.sc_recorder.sigma_BN
        lander_angular_rate = self.sc_recorder.omega_BN_B
        #lander_mass = self.sc_recorder.mHub
        #lander_CoM = self.sc_recorder.r_BcB_B
        return [lander_pos, lander_vel, lander_attitude, lander_angular_rate]
