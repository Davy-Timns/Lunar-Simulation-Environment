from Spacecraft.Vehicle import Vehicle
from Spacecraft.Thrusters.Thrusters import RCS_thruster, MainEngine, make_config
from Spacecraft.Thrusters.DescentGuidance import DescentSettings, LandingTarget, InfeasibleDescent, DescentSearchLimit
from Spacecraft.Thrusters.DescentScenario import (
    example_post_doi_state, apply_initial_state, prepare_powered_descent_start, initialize_simulation_at,
)
import matplotlib
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
import math as mat
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from Basilisk.utilities import SimulationBaseClass, macros, RigidBodyKinematics
from Sensors.Sensors import SensorsManager
import Environment.Dynamics as Dynamics
from Visualization import initialize_vizard
from Environment import Environment as Env
from Environment.TerrainMap import TerrainMap
from Sensors.TerrainNavigation import TerrainNavigation, NavigationSettings
from Sensors.AttitudeNavigation import AttitudeNavigation
from Spacecraft.Thrusters.DescentDiagnostics import TouchdownDiagnostics

matplotlib.use("TkAgg")

CLOSE_TO_LUNAR_SURFACE = True #Turn to False if you want orbital, turn to true if you want suborbital. This only controls the visualization
                                # If True-> Moon visualization shrunk to avoid visual clipping with the lunar terrain OBJ
                                # If False-> Moon visualization not shrunk, leads to clipping if close to surface of both lunar terrain obj and spacecraft.
LOW_FIDELITY_SURFACE = 1 #If CLOSE_TO_LUNAR_SURFACE is false, you may want this setting to be 0 or 2, as you will quickly pass over the lunar surface texture, and the altimeter will stop working
                        # Setting 0-> Creates a smooth sphere mesh for the altimeter raycast collision
                        # Setting 1-> Creates a lunar terrain obj only, currently at the lunar south pole only
                        # Setting 2-> Creates both a smooth sphere mesh and lunar terrain obj, but this is UNTESTED and may result in unintended clipping when points on the OBJ are below lunar sea level
DISABLE_GRAVITY = False #True -> Turns off all gravity with the exception of the sun.

# "a2g": post-DOI coast, then fuel-optimized powered descent with real gimbal/RCS limits.
# "constant": the previous dummy controller, using CONSTANT_ON_ENGINES below.
PROPULSION_MODE = "a2g"
MODEL_CHECK_ONLY = True  # True: quick spacecraft/plume preview in open space; bypasses descent
MODEL_CHECK_DURATION_S = 5.0
START_AT_POWERED_DESCENT = True  # False: full post-DOI coast; True: begin at powered-descent ignition
DESCENT_DRY_MASS_KG = 700.0
DESCENT_MIN_THROTTLE = 0.1
DESCENT_COAST_S = 55.0 * 60.0  # editable example: 70 min total minus 15 min powered
DESCENT_TIME_BOUNDS_S = (14.0 * 60.0, 25.0 * 60.0)
DESCENT_SEARCH_MAX_WALL_S = 180.0  # real computer time, not simulated flight time
DESCENT_SEARCH_MAX_EVALUATIONS = 60
VIZ_SAMPLE_S = 0.1  # 10 Hz playback; physics and controller remain at samp below
DESCENT_TARGET_N = [0.0, -1737401.0, -50.0]  # illustrative COM target; set to actual terrain landing height
DESCENT_TARGET_OMEGA_N = [0.0, 0.0, 0.0]  # static terrain in this simulation; set a rotation vector for a rotating site
DESCENT_POSITION_TOLERANCE_M = 0.5
DESCENT_VELOCITY_TOLERANCE_MPS = 0.1
DESCENT_EXAMPLE_PERILUNE_M = 15000.0
DESCENT_EXAMPLE_APOLUNE_M = 100000.0
DESCENT_EXAMPLE_APPROACH_DEG = 30.0
DESCENT_TERMINAL_GATE_M = 500.0  # brake to near rest this far above landing COM target
DESCENT_TERMINAL_DURATION_S = 120.0  # final near-vertical descent; tf remains optimized
ENGINE_CUTOFF_M = 2.0  # lowest landing-foot clearance above local terrain; terminal main-engine cutoff
DESCENT_TERRAIN_CLEARANCE_M = 20.0
LANDING_SITE_SEARCH_RADIUS_M = 150.0
LANDING_SITE_FOOTPRINT_RADIUS_M = 3.0  # includes horizontal navigation-error allowance
LANDING_SITE_MAX_SLOPE_DEG = 5.0
LANDING_SITE_MAX_ROUGHNESS_M = 0.1
ENABLE_NAVIGATION_FILTER = True  # False: raw sensor fixes + inertial propagation during laser outages
NAVIGATION_SETTINGS = NavigationSettings(filter_enabled=ENABLE_NAVIGATION_FILTER,
    xy_sigma_m=.25, xy_correlation_s=10., seed=42)
CONTACT_LOG_EVERY_N_STEPS = 100  # set to 1 for every physics step; 100 = once/s at samp=.01
DESCENT_SETTLING_S = 5.0

# Dummy propulsion settings. These are illustrative values, not calibrated IMX hardware.
# Selected engines stay on at their configured thrust for the entire simulation.
# Use () to coast; for example ("MainEngine", "RCS_1_1") fires those two engines.
CONSTANT_ON_ENGINES = ("MainEngine")
MAIN_ENGINE_THRUST_N = 4003.4 # CONFIRMED - From IM SEC filing "The workhorse of our engine fleet is the 900lbf thrust class VR900." Convering into newtons gives 4003.4N
MAIN_ENGINE_ISP_S = 325.0 #CONFIRMED - Source IM2 press kit
MAIN_ENGINE_LOCATION_B = [0.0, 0.0, -1.0]  # metres; bottom mount on body -Z
MAIN_ENGINE_DIRECTION_B = [0.0, 0.0, 1.0]  # force vector, NOT angles; exhaust is -Z
MAIN_ENGINE_GIMBAL_DEG = (0.0, 0.0)  # pitch/yaw relative to the engine mount
MAIN_ENGINE_MAX_GIMBAL_RAD = 0.0001
RCS_THRUST_N = 8.45
RCS_ISP_S = 220.0
RCS_LAYOUT = {  # name: (location in body metres, thrust direction in body axes)
    "RCS_1_1": ([0.5, 0, 0.8], [0, 1, 0]),
    "RCS_1_2": ([0.5, 0, 0.8], [0, -1, 0]),
    "RCS_1_3": ([0.5, 0, 0.8], [0, 0, 1]),
    "RCS_1_4": ([0.5, 0, 0.8], [0, 0, -1]),

    "RCS_2_1": ([-0.5, 0, 0.8], [0, 1, 0]),
    "RCS_2_2": ([-0.5, 0, 0.8], [0, -1, 0]),
    "RCS_2_3": ([-0.5, 0, 0.8], [0, 0, 1]),
    "RCS_2_4": ([-0.5, 0, 0.8], [0, 0, -1]),

    "RCS_3_1": ([0.0, 0.5, 0.8], [1, 0, 0]),
    "RCS_3_2": ([0.0, 0.5, 0.8], [-1, 0, 0]),
    "RCS_3_3": ([0.0, 0.5, 0.8], [0, 0, 1]),
    "RCS_3_4": ([0.0, 0.5, 0.8], [0, 0, -1]),

    "RCS_4_1": ([0.0, -0.5, 0.8], [1, 0, 0]),
    "RCS_4_2": ([0.0, -0.5, 0.8], [-1, 0, 0]),
    "RCS_4_3": ([0.0, -0.5, 0.8], [0, 0, 1]),
    "RCS_4_4": ([0.0, -0.5, 0.8], [0, 0, -1]),


}


runtime = MODEL_CHECK_DURATION_S if MODEL_CHECK_ONLY else 20.0 # simulation duration in seconds
samp = 0.05 if MODEL_CHECK_ONLY else 0.01 # coarse preview step; normal physics/sensors at 100 Hz
sampling_ns = macros.sec2nano(samp)  # how often to sample sensors in ns
simulation_start_s = (DESCENT_COAST_S
    if not MODEL_CHECK_ONLY and START_AT_POWERED_DESCENT and PROPULSION_MODE == 'a2g' else 0.0)
if MODEL_CHECK_ONLY and (not np.isfinite(runtime) or runtime <= 0):
    raise ValueError("MODEL_CHECK_DURATION_S must be finite and positive.")
if not np.isfinite(simulation_start_s) or simulation_start_s < 0:
    raise ValueError("Descent coast duration must be finite and nonnegative.")
start_ticks = round(simulation_start_s / samp)
if not np.isclose(start_ticks * samp, simulation_start_s, atol=1e-8, rtol=0):
    raise ValueError("Descent coast duration must be a multiple of samp when skipping the coast.")
start_ns = start_ticks * sampling_ns
sim = SimulationBaseClass.SimBaseClass()                        # Initialize/instantiate a simulation environment

orbital_parameters = {
            "altitude": 10000.0, # Will define orbit Semi-major axis (m). Takes the moon radius + altitude to define it
            "eccentricity": 0.0, # Eccentricity (0 = circular orbit, 0 < e < 1 = elliptical)
            "inclination deg": 0.0, # Inclination (rad)
            "right ascension of ascending node deg": 0.0, # Right Ascension of the Ascending Node (RAAN) (rad)
            "argument of periapsis deg": 0.0, # Argument of Periapsis (rad)
            "true anomaly": -90.0, # True Anomaly (rad)
        }


spacecraft_velocity_override = [0, 0, 0] #At the south pole, +Y is downward toward the Moon and Z is horizontal/tangent to the surface.
spacecraft_position_override = [0,-1737500.0,-50] #If you wanted to change the position relative to the moon you could do it here. I haven't found a real important use for this yet.

mrp1, mrp2, mrp3 = RigidBodyKinematics.euler3212MRP(np.deg2rad([0.0, 0.0, 90.0])) #input as degrees here for spacecraft rotation!

spacecraft_attitude_MRP = [[mrp1], [mrp2], [mrp3]]  #Starting attitude of spacecraft with respect to the body and inertial frame as a modified rodrigues parameter  (N->P)
spacecraft_attitude_rate = [[0.0], [0.0], [0.0]]    # Current angular velocity with body frame relative to inertial frame (rad/s)
spacecraft_mass = 2120.0                            #kg, not sure yet how to deal with CoM or where that is defined
spacecraft_inertia = [
    [1014.49, 0.0, 0.0],
    [0.0, 1014.49, 0.0],
    [0.0, 0.0, 1413.33],
]


SPACECRAFT_BODY_NAME = "IMX"

# Create simulation tasks and processes
process = sim.CreateNewProcess("proc")                          # Create a new simulation process
task = sim.CreateNewTask("record", sampling_ns, FirstStart=start_ns)      # Create a new task in the simulation
process.addTask(task)                                               # Add created task to the process

#Continued simulation modules setup
sc=Vehicle(sim, SPACECRAFT_BODY_NAME, sampling_ns, spacecraft_attitude_MRP, spacecraft_attitude_rate, spacecraft_mass, spacecraft_inertia)                                #initialize the vehicle

# The model preview needs only the spacecraft, engines and playback writer.
if not MODEL_CHECK_ONLY:
    # The planner and navigation adapter use the same compiled terrain as contacts.
    sc.terrain = Env.create_terrain_spacecraft()
    SM = SensorsManager(sc, sampling_ns)
    sc.initialize_sensors(SM, samp, LOW_FIDELITY_SURFACE)
    SM.altimeter.debug_contact_every_n_steps = CONTACT_LOG_EVERY_N_STEPS
    SM.altimeter.spacecraft_mass = spacecraft_mass
    SM.altimeter.spacecraft_inertia_B = np.array(spacecraft_inertia, dtype=float)
    if LOW_FIDELITY_SURFACE == 1:
        sc.terrain_map = TerrainMap(SM.altimeter)


main_engine = MainEngine(
    make_config(MAIN_ENGINE_THRUST_N, MAIN_ENGINE_ISP_S, MAIN_ENGINE_DIRECTION_B),
    location=MAIN_ENGINE_LOCATION_B,
    max_gimbal=MAIN_ENGINE_MAX_GIMBAL_RAD,
)
main_engine.set_gimbal(*np.deg2rad(MAIN_ENGINE_GIMBAL_DEG))
engines = {main_engine.name: main_engine}
for name, (location, direction) in RCS_LAYOUT.items():
    if name in engines:
        raise ValueError(f"Duplicate engine name: {name}")
    engines[name] = RCS_thruster(
        make_config(RCS_THRUST_N, RCS_ISP_S, direction), name=name, location=location
    )
if MODEL_CHECK_ONLY:
    sc.initialize_thrusters(engines, tuple(engines))  # all main/RCS engines at rated thrust
    sc.lander.hub.r_CN_NInit = [0., -(Dynamics.r_moon + 100000.), 0.]
    sc.lander.hub.v_CN_NInit = [0., 0., 0.]
    print(f"Model check: {runtime:g} seconds, all {len(engines)} thrusters ON, "
          "100 km above the Moon. Descent planning and sensors skipped.", flush=True)
elif PROPULSION_MODE == "constant":
    sc.initialize_thrusters(engines, CONSTANT_ON_ENGINES)
elif PROPULSION_MODE == "a2g":
    if DISABLE_GRAVITY:
        raise ValueError("A2G requires lunar gravity; set DISABLE_GRAVITY=False.")
    descent_settings = DescentSettings(
        mu=Dynamics.mu_moon, dry_mass=DESCENT_DRY_MASS_KG,
        min_thrust=DESCENT_MIN_THROTTLE * MAIN_ENGINE_THRUST_N,
        max_thrust=MAIN_ENGINE_THRUST_N,
        tf_min=DESCENT_TIME_BOUNDS_S[0], tf_max=DESCENT_TIME_BOUNDS_S[1],
        max_search_wall_s=DESCENT_SEARCH_MAX_WALL_S,
        max_search_evaluations=DESCENT_SEARCH_MAX_EVALUATIONS,
        position_tolerance=DESCENT_POSITION_TOLERANCE_M,
        velocity_tolerance=DESCENT_VELOCITY_TOLERANCE_MPS,
        surface_radius=Dynamics.r_moon, coast_duration_s=DESCENT_COAST_S,
        terminal_gate_height_m=DESCENT_TERMINAL_GATE_M,
        terminal_duration_s=DESCENT_TERMINAL_DURATION_S,
        engine_cutoff_m=ENGINE_CUTOFF_M,
        terrain_clearance_m=DESCENT_TERRAIN_CLEARANCE_M,
    )
    if not hasattr(sc, 'terrain_map') or np.any(DESCENT_TARGET_OMEGA_N):
        raise ValueError('Terrain descent currently requires LOW_FIDELITY_SURFACE=1 and a static terrain/target.')
    preferred_xy = sc.terrain_map.to_map(DESCENT_TARGET_N)[:2]
    site_xy, sc.landing_site = sc.terrain_map.find_site(
        preferred_xy, search_radius_m=LANDING_SITE_SEARCH_RADIUS_M,
        radius_m=LANDING_SITE_FOOTPRINT_RADIUS_M, max_slope_deg=LANDING_SITE_MAX_SLOPE_DEG,
        max_roughness_m=LANDING_SITE_MAX_ROUGHNESS_M)
    site_position = sc.terrain_map.to_inertial([
        *site_xy, sc.landing_site['maximum_height_m'] + descent_settings.landing_com_clearance_m])
    print(f"Terrain landing site: map XY={site_xy}, slope={sc.landing_site['slope_deg']:.2f} deg, "
          f"roughness={sc.landing_site['roughness_m']:.3f} m; braking gate={DESCENT_TERMINAL_GATE_M:g} m", flush=True)
    descent_target = LandingTarget(site_position)
    doi_state = example_post_doi_state(
        descent_settings, descent_target, spacecraft_mass,
        DESCENT_EXAMPLE_PERILUNE_M, DESCENT_EXAMPLE_APOLUNE_M, DESCENT_EXAMPLE_APPROACH_DEG,
    )
    apply_initial_state(sc, doi_state)
    print("Searching powered-descent flight time with gimbal/RCS and fuel constraints...", flush=True)
    try:
        controller = sc.initialize_descent(engines, descent_settings, descent_target)
    except (InfeasibleDescent, DescentSearchLimit) as exc:
        raise SystemExit(f"A2G planning stopped: {exc}\n"
                         "See Spacecraft/Thrusters/DESCENT.md for example-orbit settings and validation limits.") from None
    simulation_initial_state = prepare_powered_descent_start(sc) if START_AT_POWERED_DESCENT else doi_state
    if START_AT_POWERED_DESCENT:
        print(f"Starting native simulation at powered-descent ignition, t={simulation_start_s:g} s. "
              "Unpowered coast skipped; sensors initialize at the ignition state.", flush=True)
    navigation = TerrainNavigation(simulation_initial_state, sc.terrain_map, sc.lander.scStateOutMsg,
        sc.imus['IMU'], sc.star_trackers['ST'], SM.altimeter, descent_settings.mu, NAVIGATION_SETTINGS)
    print(f"Navigation filter: {'UKF enabled' if ENABLE_NAVIGATION_FILTER else 'OFF — raw fixes / inertial propagation'}; "
          f"laser acceptance limit {NAVIGATION_SETTINGS.max_range_m:g} m. "
          "Z is predicted whenever no laser measurement is accepted.", flush=True)
    navigation.command_source = controller
    attitude_navigation = AttitudeNavigation(simulation_initial_state.dcm_NB, sc.imus['IMU'], sc.star_trackers['ST'], samp)
    navigation.attitude_navigation = attitude_navigation
    sc.attitude_navigation = controller.attitude_navigation = attitude_navigation
    sim.AddModelToTask('record', attitude_navigation, ModelPriority=-1)
    sc.navigation = controller.navigation = navigation
    controller.contact_sensor = SM.altimeter
    sim.AddModelToTask('record', navigation, ModelPriority=-2)
    sc.touchdown_diagnostics = TouchdownDiagnostics(sc)
    sim.AddModelToTask('record', sc.touchdown_diagnostics, ModelPriority=-15)
    runtime = DESCENT_COAST_S + np.ceil(controller.plan.flight_time / samp) * samp + DESCENT_SETTLING_S
    print(f"A2G plan: {controller.plan.flight_time:.3f} s powered, "
          f"{controller.plan.fuel_used:.3f} kg predicted propellant", flush=True)
else:
    raise ValueError("PROPULSION_MODE must be 'constant' or 'a2g'.")

#Gravity Model

Dynamics.initialize_dynamics(sim, sc, DISABLE_GRAVITY,
                             CLOSE=False if MODEL_CHECK_ONLY else CLOSE_TO_LUNAR_SURFACE,
                             LUNAR_ONLY=(MODEL_CHECK_ONLY or PROPULSION_MODE == "a2g"))
if not MODEL_CHECK_ONLY and PROPULSION_MODE == "constant":
    Dynamics.initialize_vehicle_dynamics_parameters(sc, orbital_parameters, spacecraft_velocity_override, spacecraft_position_override)
if MODEL_CHECK_ONLY:
    sc.sc_recorder = sc.lander.scStateOutMsg.recorder(sampling_ns)
    sim.AddModelToTask('record', sc.sc_recorder)
else:
    sc.initialize_recorder(sim)


process.addTask(sim.CreateNewTask("viz", macros.sec2nano(VIZ_SAMPLE_S), FirstStart=start_ns))
model_check_file = Path(__file__).resolve().parent / '_VizFiles' / 'ModelCheck_UnityViz.bin'
if MODEL_CHECK_ONLY:
    model_check_file.parent.mkdir(parents=True, exist_ok=True)
initialize_vizard(sim, sc, task_name="viz",
                  save_file=str(model_check_file) if MODEL_CHECK_ONLY else None)


# Run simulation
initialize_simulation_at(sim, start_ns)  # Initialize physics/sensors at the selected mission time
stop_ticks = round(runtime / samp) if PROPULSION_MODE == "a2g" else mat.floor(runtime / samp)
sim.ConfigureStopTime(stop_ticks * sampling_ns)
sim.ExecuteSimulation()

if MODEL_CHECK_ONLY:
    print(f"Model check complete. Open {model_check_file} in Vizard.", flush=True)
    raise SystemExit(0)  # omit sensor extraction, landing reports, plots and animations

out = sc.output()
if PROPULSION_MODE == "a2g":
    print("Descent result:", controller.status)
    if controller.history:
        print("Terminal guidance telemetry:", controller.history[-1])
    from Spacecraft.Thrusters.DescentDiagnostics import save_descent_report
    save_descent_report(sc, Path(__file__).resolve().parent / 'results')

#vvvvvv everything else is just diagnostics and plotting below vvvvvv
##region plotting
if PROPULSION_MODE == 'a2g':
    # Use controller commands, normalized by rated thrust (not mutable MaxThrust).
    command_history = sc.thruster_controller.history
    command_time = np.array([h['time_s'] for h in command_history])
    throttle_percent = np.array([100. * h.get('main_thrust', 0.) / MAIN_ENGINE_THRUST_N
                                 for h in command_history])
    if len(command_time):
        # Include ignition and the final hold interval after contact/stop.
        command_time = np.r_[DESCENT_COAST_S, command_time, sc.sc_recorder.times()[-1] * 1e-9]
        throttle_percent = np.r_[0., throttle_percent,
                                  100. * sc.thruster_controller.main_engine.config.MaxThrust / MAIN_ENGINE_THRUST_N]
    fig_throttle, ax_throttle = plt.subplots(figsize=(10, 4))
    ax_throttle.step(command_time - DESCENT_COAST_S, throttle_percent, where='post', label='Main engine command')
    ax_throttle.set(xlabel='Time since powered descent initiation (s)', ylabel='Main engine command (%)',
                    ylim=(0, 100), title='Main engine thrust command')
    if sc.thruster_controller.cutoff_event is not None:
        ax_throttle.axvline(sc.thruster_controller.cutoff_event['time_s'] - DESCENT_COAST_S,
                           linestyle='--', label=f'{ENGINE_CUTOFF_M:g} m foot-clearance cutoff')
    if sc.touchdown_diagnostics.touchdown is not None:
        ax_throttle.axvline(sc.touchdown_diagnostics.touchdown['time_s'] - DESCENT_COAST_S,
                           linestyle=':', label='Touchdown')
    ax_throttle.grid(alpha=.3)
    ax_throttle.legend()
    fig_throttle.tight_layout()
    fig_throttle.savefig(Path(__file__).resolve().parent / 'results' / 'main_engine_throttle.png', dpi=160)

for line in out:
    print(line)
    print()

true = out["true_data"]

position      = true[0]
velocity      = true[1]
attitude      = true[2]
angular_rate  = true[3]


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot trajectory
ax.plot(position[:, 0], position[:, 1], position[:, 2], lw=2)

# Mark start and end points
ax.scatter(position[0, 0], position[0, 1], position[0, 2],
           color='green', s=50, label='Start')
ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2],
           color='red', s=50, label='End')

ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.set_zlabel('Z Position [m]')
ax.set_title('3D Spacecraft Trajectory')

# Make all axes use the same scale
x = position[:, 0]
y = position[:, 1]
z = position[:, 2]

max_range = np.array([
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min()
]).max() / 2

mid_x = (x.max() + x.min()) / 2
mid_y = (y.max() + y.min()) / 2
mid_z = (z.max() + z.min()) / 2

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Floor (bottom of plot)
z_floor = z.min()

# Start point
x0, y0, z0 = position[0]

# End point
x1, y1, z1 = position[-1]
# Start point
ax.plot([x0, x0], [y0, y0], [z_floor, z0], 'g--', lw=1)     # vertical
ax.plot([x0, x0], [y.min(), y0], [z_floor, z_floor], 'g:', lw=1)
ax.plot([x.min(), x0], [y0, y0], [z_floor, z_floor], 'g:', lw=1)

# End point
ax.plot([x1, x1], [y1, y1], [z_floor, z1], 'r--', lw=1)
ax.plot([x1, x1], [y.min(), y1], [z_floor, z_floor], 'r:', lw=1)
ax.plot([x.min(), x1], [y1, y1], [z_floor, z_floor], 'r:', lw=1)

ax.legend()
plt.tight_layout()
plt.show()
# Extract IMU and ST dictionaries
imu = out["imu"]["IMU"]
st  = out["star_tracker"]["ST"]

# Extract IMU, star tracker, and altimeter dictionaries
imu = out["imu"]["IMU"]
st  = out["star_tracker"]["ST"]
altimeter = out["altimeter"]["LaserAltimeter1"]

# --------------------------------------------------
# Altimeter altitude vs time
# --------------------------------------------------

altimeter_time = np.array(altimeter["time_s"])  # recorder clock, already in seconds
altimeter_data = np.array(altimeter["altitude"])

# Legacy message slot 2 holds beam slant range, NOT spacecraft map Z.
altitude = altimeter_data[:, 2].astype(float)

# Treat -1 as no measurement
altitude[(altitude < 0) | ~np.isfinite(altitude)] = np.nan

plt.figure(figsize=(10, 5))

plt.plot(
    altimeter_time,
    altitude,
    lw=2,
    label="Raw laser slant range (before navigation validity checks)"
)

plt.xlabel("Time [s]")
plt.ylabel("Slant range [m]")
plt.title("Laser Altimeter Measurement — gaps mean no return")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



# Number of samples (3 samples for 2 timesteps: t=0, 0.01, 0.02)
num_samples = len(next(iter(imu.values())))
t = sc.imus['IMU'].recorder.times() * 1e-9

# -----------------------------
# Plot IMU fields
# -----------------------------
imu_fields = [
    "AccelPlatform",
    "AngVelPlatform"
]
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

axs = axs.flatten()

for ax, field in zip(axs, imu_fields):
    arr = imu[field]   # shape (N, 3)

    ax.plot(t, arr[:, 0], label="X")
    ax.plot(t, arr[:, 1], label="Y")
    ax.plot(t, arr[:, 2], label="Z")

    ax.set_title(field)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(field)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Plot StarTracker quaternion
# -----------------------------
q = st["qInrtl2Case"]    # shape (N,4)

#FILTERING BELOW! May not be akin to space-based filtering, was just first attempt
#alpha = 0.05
#q_filt = np.zeros_like(q)
#q_filt[0] = q[0]

#for i in range(1, len(q)):
#    q_filt[i] = alpha*q[i] + (1-alpha)*q_filt[i-1]
#    q_filt[i] /= np.linalg.norm(q_filt[i])
#q=q_filt

plt.figure(figsize=(12, 8))
for i in range(4):
    plt.plot(t, q[:, i+0], label=f"qInrtl2Case[{i+0}]")

plt.title("StarTracker Quaternion Over Time")
plt.xlabel("Time [s]")
plt.ylabel("Quaternion Component")
plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()



# Basilisk uses scalar-first Euler parameters and a passive inertial-to-case DCM.
# Reorder for SciPy's scalar-last convention and transpose its active rotation.
R_mats = R.from_quat(q[:, [1, 2, 3, 0]]).as_matrix().transpose(0, 2, 1)

# Canonical body axes
xB = np.array([1,0,0])
yB = np.array([0,1,0])
zB = np.array([0,0,1])

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')




def update(i):
    ax.cla()
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_title(f"Attitude at t={t[i]:.2f}s")

    R_i = R_mats[i]

    # Rotate body axes
    xR = R_i.T @ xB #transponse required because R_i maps inertial -> body
    yR = R_i.T @ yB
    zR = R_i.T @ zB

    # Plot rotated axes
    scale = 0.6
    ax.quiver(0,0,0, *(scale * xR), color='r', label='X axis')
    ax.quiver(0,0,0, *(scale * yR), color='g', label='Y axis')
    ax.quiver(0,0,0, *(scale * zR), color='b', label='Z axis')

    ax.legend()

ani = FuncAnimation(fig, update, frames=num_samples, interval=10)


# Save as GIF
writer = PillowWriter(fps=30)
#ani.save("attitude_animation.gif", writer=writer) #uncomment this in order to save gif of star tracker output

plt.show(block=True)

##endregion

