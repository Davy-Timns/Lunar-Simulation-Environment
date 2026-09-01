from Spacecraft.Vehicle import Vehicle
import matplotlib
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
import math as mat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from Basilisk.utilities import SimulationBaseClass, macros, RigidBodyKinematics
from Sensors.Sensors import SensorsManager
import Environment.Dynamics as Dynamics
from Visualization import initialize_vizard
from Environment import Environment as Env

matplotlib.use("TkAgg")

CLOSE_TO_LUNAR_SURFACE = True #Turn to False if you want orbital, turn to true if you want suborbital. This only controls the visualization
                                # If True-> Moon visualization shrunk to avoid visual clipping with the lunar terrain OBJ
                                # If False-> Moon visualization not shrunk, leads to clipping if close to surface of both lunar terrain obj and spacecraft.
LOW_FIDELITY_SURFACE = 1 #If CLOSE_TO_LUNAR_SURFACE is false, you may want this setting to be 0 or 2, as you will quickly pass over the lunar surface texture, and the altimeter will stop working
                        # Setting 0-> Creates a smooth sphere mesh for the altimeter raycast collision
                        # Setting 1-> Creates a lunar terrain obj only, currently at the lunar south pole only
                        # Setting 2-> Creates both a smooth sphere mesh and lunar terrain obj, but this is UNTESTED and may result in unintended clipping when points on the OBJ are below lunar sea level
DISABLE_GRAVITY = True #True -> Turns off all gravity with the exception of the sun.


runtime = 25.0 #how long to run the simulation for, in seconds
samp = 0.01 #sampling rate, ie how often to take measurements, in seconds
sampling_ns = macros.sec2nano(samp)  # how often to sample sensors in ns
sim = SimulationBaseClass.SimBaseClass()                        # Initialize/instantiate a simulation environment

orbital_parameters = {
            "altitude": 10000.0, # Will define orbit Semi-major axis (m). Takes the moon radius + altitude to define it
            "eccentricity": 0.0, # Eccentricity (0 = circular orbit, 0 < e < 1 = elliptical)
            "inclination deg": 0.0, # Inclination (rad)
            "right ascension of ascending node deg": 0.0, # Right Ascension of the Ascending Node (RAAN) (rad)
            "argument of periapsis deg": 0.0, # Argument of Periapsis (rad)
            "true anomaly": -90.0, # True Anomaly (rad)
        }


spacecraft_velocity_override = [10, 0, -10] #At the south pole, +Y is downward toward the Moon and Z is horizontal/tangent to the surface.
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
task = sim.CreateNewTask("record", sampling_ns)      # Create a new task in the simulation
process.addTask(task)                                               # Add created task to the process

#Continued simulation modules setup
sc=Vehicle(sim, SPACECRAFT_BODY_NAME, sampling_ns, spacecraft_attitude_MRP, spacecraft_attitude_rate, spacecraft_mass, spacecraft_inertia)                                #initialize the vehicle


sc.terrain = Env.create_terrain_spacecraft()  #Must happen before sc.initialize_sensors()

SM = SensorsManager(sc, sampling_ns)                                #initialize the sensors manager

sc.initialize_sensors(SM,samp, LOW_FIDELITY_SURFACE)                                   #attach sensors to vehicle and set up mujoco physics for the altimeter sensor
SM.altimeter.spacecraft_mass = spacecraft_mass
SM.altimeter.spacecraft_inertia_B = np.array(spacecraft_inertia, dtype=float)



#Gravity Model

Dynamics.initialize_dynamics(sim, sc,DISABLE_GRAVITY,CLOSE_TO_LUNAR_SURFACE)
Dynamics.initialize_vehicle_dynamics_parameters(sc, orbital_parameters, spacecraft_velocity_override, spacecraft_position_override)
sc.initialize_recorder(sim)


initialize_vizard(sim, sc)


# Run simulation
sim.InitializeSimulation()  # Start the simulation
sim.ConfigureStopTime((mat.floor(runtime / samp) * sampling_ns))  # When the simulation should stop
sim.ExecuteSimulation()

out = sc.output()

#vvvvvv everything else is just diagnostics and plotting below vvvvvv
##region plotting
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

altimeter_time = np.array(altimeter["timeTag"]) * 1e-9  # ns -> seconds
altimeter_data = np.array(altimeter["altitude"])

# Z component contains the altitude measurement
altitude = altimeter_data[:, 2].astype(float)

# Treat -1 as no measurement
altitude[altitude == -1] = np.nan

plt.figure(figsize=(10, 5))

plt.plot(
    altimeter_time,
    altitude,
    lw=2,
    label="Laser Altimeter"
)

plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("Laser Altimeter Measurement")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



# Number of samples (3 samples for 2 timesteps: t=0, 0.01, 0.02)
num_samples = len(next(iter(imu.values())))
t = np.arange(num_samples) * samp

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



# Convert quaternions (Basilisk gives [q1,q2,q3,q0] = [x,y,z,w])
# SciPy expects [x,y,z,w], so this is already correct ordering.
R_mats = R.from_quat(q).as_matrix()   # shape (N,3,3)

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

