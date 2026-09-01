from Basilisk.utilities import macros, simIncludeGravBody
from Basilisk.utilities import orbitalMotion
from typing import Dict

r_moon = 1737400  # radius of the moon in meters
mu_moon = 4.9048695e12  # meters^3/s^2
mu_earth = 0.3986004415E+15  # meters^3/s^2
r_earth = 6378136.6  # meters^3/s^2





def initialize_dynamics(sim, sc,DEBUG_DISABLE_GRAVITY = False, CLOSE=False):
    gravFactory = simIncludeGravBody.gravBodyFactory()
    gravFactory.createSun()

    mu_moonL = mu_moon
    mu_earthL = mu_earth

    if DEBUG_DISABLE_GRAVITY == True:
        print("GRAVITY IS CURRENTLY DISABLED!")
        mu_moonL = 0.01
        mu_earthL = 0.01

    if CLOSE:
        moon = gravFactory.createCustomGravObject("moon", mu_moonL, radEquator=10000) # Make lunar texture small
    else:
        moon = gravFactory.createCustomGravObject("moon", mu_moonL, radEquator=r_moon) #Normal lunar texture
    moon.isCentralBody= True

    earth = gravFactory.createCustomGravObject("earth",mu_earthL, radEquator=r_earth)
    earth.isCentralBody = False

    timeInitString = "2026 JUL 28 00:00:00.0"

    gravFactory.createSpiceInterface(
        time=timeInitString,
        epochInMsg=True
    )

    gravFactory.spiceObject.zeroBase = "moon"


    sim.AddModelToTask("record", gravFactory.spiceObject)

    gravFactory.addBodiesTo(sc.lander)
    sim.AddModelToTask("record", sc.lander)


def initialize_vehicle_dynamics_parameters(sc, orbital_params:Dict[str,float] | None = None,spacecraft_velocity_override = None, spacecraft_position_override = None):
    """
    :param sc: Spacecraft object
    :param orbital_params: Dictionary containing all the orbital parameters: altitude, eccentricity, inclination deg, right ascension of ascending node deg, argument of periapsis deg, true anomaly
    :param spacecraft_velocity_override: If you want to override the spacecraft velocity that is otherwise calculated to maintain a circular non-elliptical orbit, do so here. Is a matrix of format [1,2,3] representing velocity in m/s
    :param spacecraft_position_override: If you want to override the spacecraft position that is otherwise calculated, do so here with format [1,2,3] representing position in meters.
    :return:
    """

    if orbital_params is None:
        orbital_params = {
            "altitude": 10000.0,
            "eccentricity": 0.0,
            "inclination deg": 0.0,
            "right ascension of ascending node deg": 0.0,
            "argument of periapsis deg": 0.0,
            "true anomaly": 90.0,
        }
    # Setup initial orbit about the Moon
    oe = orbitalMotion.ClassicElements()

    altitude = orbital_params["altitude"]

    oe.a = r_moon + altitude  # Semi-major axis (m). For a circular orbit, this is Moon radius + altitude. Here: 173km + 10km = 183km altitude.
    oe.e = orbital_params["eccentricity"]  # Eccentricity (0 = circular orbit, 0 < e < 1 = elliptical).
    oe.i = orbital_params["inclination deg"] * macros.D2R  # Inclination (rad)
    oe.Omega = orbital_params["right ascension of ascending node deg"] * macros.D2R # Right Ascension of the Ascending Node (RAAN) (rad).
    oe.omega = orbital_params["argument of periapsis deg"] * macros.D2R # Argument of Periapsis (rad).
    oe.f = orbital_params["true anomaly"] * macros.D2R # True Anomaly (rad).
    rN, vN = orbitalMotion.elem2rv(mu_moon,
                                   oe)  # Convert the Keplerian orbital elements into inertial position and velocity vectors

    # To set the spacecraft initial conditions, the following initial position and velocity variables are set:
    if spacecraft_velocity_override is None:
        sc.lander.hub.v_CN_NInit = vN  # m/s - v_BN_N
    else:
        sc.lander.hub.v_CN_NInit = spacecraft_velocity_override
    if spacecraft_position_override is None:
        sc.lander.hub.r_CN_NInit = rN  # m   - r_BN_N
    else:
        sc.lander.hub.r_CN_NInit = spacecraft_position_override

