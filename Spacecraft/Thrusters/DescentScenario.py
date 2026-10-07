"""Editable example post-DOI orbit and actuator geometry for A2G demonstrations."""
import numpy as np
from Basilisk.utilities import RigidBodyKinematics
from Spacecraft.Thrusters.DescentGuidance import DescentState, a2g_acceleration, coast_state
from Spacecraft.Thrusters.Thrusters import MainEngine, RCS_thruster, make_config


def example_post_doi_state(settings, target, wet_mass, pdi_altitude=15000.,
                           apolune_altitude=100000., approach_angle_deg=21.0):
    """Illustrative polar ellipse, NOT reconstructed Apollo/IM navigation data.

    Construct the PDI state at perilune, upstream of the requested south-pole
    target, then propagate backwards through the specified post-DOI coast.
    """
    touchdown = settings.coast_duration_s + 0.5 * (settings.tf_min + settings.tf_max)
    target_state = target.state(touchdown)
    normal = target_state[0] / np.linalg.norm(target_state[0])
    tangent = np.array([0., 0., 1.])
    tangent -= np.dot(tangent, normal) * normal
    if np.linalg.norm(tangent) < 1e-8:
        tangent = np.array([1., 0., 0.])
        tangent -= np.dot(tangent, normal) * normal
    tangent /= np.linalg.norm(tangent)
    angle = np.deg2rad(approach_angle_deg)
    pdi_normal = np.cos(angle) * normal - np.sin(angle) * tangent
    pdi_tangent = np.sin(angle) * normal + np.cos(angle) * tangent
    rp = settings.surface_radius + pdi_altitude
    ra = settings.surface_radius + apolune_altitude
    if rp <= settings.surface_radius or ra < rp:
        raise ValueError("Example orbit requires 0 < perilune altitude <= apolune altitude.")
    speed = np.sqrt(settings.mu * (2 / rp - 2 / (rp + ra)))
    pdi = DescentState(rp * pdi_normal, speed * pdi_tangent, wet_mass, np.eye(3), np.zeros(3))
    thrust_axis = a2g_acceleration(pdi.position, pdi.velocity, *target_state,
                                   touchdown - settings.coast_duration_s, settings.mu)
    z_axis = thrust_axis / np.linalg.norm(thrust_axis)
    y_axis = np.array([1., 0., 0.])
    y_axis -= np.dot(y_axis, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis)
    pdi.dcm_NB = np.column_stack((np.cross(y_axis, z_axis), y_axis, z_axis))
    return coast_state(pdi, -settings.coast_duration_s, settings.mu, settings.surface_radius)


def apply_initial_state(vehicle, state):
    vehicle.lander.hub.r_CN_NInit = state.position.tolist()
    vehicle.lander.hub.v_CN_NInit = state.velocity.tolist()
    vehicle.lander.hub.sigma_BNInit = RigidBodyKinematics.C2MRP(state.dcm_NB.T).tolist()
    vehicle.lander.hub.omega_BN_BInit = state.omega_B.tolist()


def prepare_powered_descent_start(vehicle):
    """Initialize the hub at PDI, skipping native propagation of the coast.

    Fuel is unchanged during coast. Preserve the original mission/ignition
    epochs, and initialize sensor navigation separately from the returned state.
    """
    controller = vehicle.thruster_controller
    cfg = controller.settings
    vehicle.simulation_start_s = float(cfg.coast_duration_s)
    state = controller.pdi_state.copy()
    apply_initial_state(vehicle, state)
    controller.mission_start_s = 0.0
    return state


def initialize_simulation_at(sim, start_ns=0):
    """Initialize each Basilisk model once at the requested mission clock.

    Tasks must be created with FirstStart=start_ns. Resetting the hub at that
    same epoch prevents integrating the skipped interval on its first update.
    """
    if not start_ns:
        sim.InitializeSimulation()
        return
    if sim.simulationInitialized:
        raise ValueError("A late start requires a new simulation.")
    sim.TotalSim.assignRemainingProcs()
    sim.TotalSim.ResetSimulation()
    sim.TotalSim.selfInitSimulation()
    for task in sim.TaskList:
        task.resetTask(start_ns)
    sim.TotalSim.CurrentNanos = start_ns
    sim.simulationInitialized = True


def example_engines(main_thrust=4003.4, main_isp=325., rcs_thrust=4.45):
    engines = {"MainEngine": MainEngine(make_config(main_thrust, main_isp, [0, 0, 1]),
                                        location=[0, 0, -1], max_gimbal=0.27)}
    for pod, (location, directions) in enumerate((
        ([.5, 0, .8], ([0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])),
        ([-.5, 0, .8], ([0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])),
        ([0, .5, .8], ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1])),
        ([0, -.5, .8], ([1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1])),
    ), 1):
        for index, direction in enumerate(directions, 1):
            name = f"RCS_{pod}_{index}"
            engines[name] = RCS_thruster(make_config(rcs_thrust, 220., direction), name, location)
    return engines
