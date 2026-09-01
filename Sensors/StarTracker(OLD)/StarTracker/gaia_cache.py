import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from Sensors.StarTracker import load_gaia_database


# -----------------------------
# CAMERA / FIELD SETTINGS
# -----------------------------
image_size = 1024
fov_deg = .4
region_center_coords = (165.0, 56.0)   # (RA, Dec) in degrees
radius_deg_gaia = 30.0

image = np.zeros((image_size, image_size), dtype=np.float32)


# -----------------------------
# PSF (POINT SPREAD FUNCTION)
# -----------------------------
def add_star_psf(img, x0, y0, flux, sigma=1.6):
    size = img.shape[0]
    r = 6

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            x = x0 + dx
            y = y0 + dy

            if 0 <= x < size and 0 <= y < size:
                w = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                img[y, x] += flux * w


# -----------------------------
# RA/Dec → UNIT VECTOR (ICRS)
# -----------------------------
def radec_to_vec(ra_deg, dec_deg):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.array([x, y, z], dtype=np.float64)


# -----------------------------
# LOAD STAR FIELD (GAIA)
# -----------------------------
stars_inertial, fluxes = load_gaia_database(
    region_center=region_center_coords,
    radius_deg=radius_deg_gaia
)

print("Stars loaded:", len(stars_inertial))
if len(stars_inertial) == 0:
    raise RuntimeError("No stars loaded from Gaia")

print("Flux range:", np.min(fluxes), np.max(fluxes))


# -----------------------------
# CAMERA ORIENTATION
# -----------------------------
# 1) Point camera boresight at the Gaia region center
boresight = radec_to_vec(*region_center_coords)  # inertial unit vector

# Choose an "up" reference that is not parallel to boresight
up_ref = np.array([0.0, 0.0, 1.0])
if np.abs(np.dot(up_ref, boresight)) > 0.9:
    up_ref = np.array([0.0, 1.0, 0.0])

# Build camera basis: +Z = boresight, +X = right, +Y = up
x_cam = np.cross(up_ref, boresight)
x_cam /= np.linalg.norm(x_cam)
y_cam = np.cross(boresight, x_cam)
y_cam /= np.linalg.norm(y_cam)
z_cam = boresight

R_inertial_to_cam = R.from_matrix(np.vstack([x_cam, y_cam, z_cam]))  # rows = cam axes in inertial frame

# 2) Optional body attitude offsets (roll, pitch, yaw) in camera frame
roll = 0
pitch = 0
yaw = 0
R_body = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)

# Total inertial → camera rotation
R_total = R_body * R_inertial_to_cam

# -----------------------------
# TRANSFORM STARS INTO CAMERA FRAME
# -----------------------------
stars_camera = R_total.apply(stars_inertial)

print("Camera-frame stars shape:", stars_camera.shape)
z_vals = stars_camera[:, 2]
print("Z range (camera frame):", np.min(z_vals), np.max(z_vals))


# -----------------------------
# PROJECTION MODEL (PINHOLE CAMERA)
# -----------------------------
fov = np.radians(fov_deg)
scale = (image_size / 2.0) / np.tan(fov / 2.0)
print("Focal scale:", scale)


# -----------------------------
# PROJECT TO IMAGE PLANE
# -----------------------------
hit = 0
total = 0

for s, flux in zip(stars_camera, fluxes):
    total += 1
    x, y, z = s

    # star behind camera
    if z <= 0.0:
        continue

    u = x / z
    v = y / z

    px = int(image_size / 2.0 + scale * u)
    py = int(image_size / 2.0 + scale * v)

    if 0 <= px < image_size and 0 <= py < image_size:
        add_star_psf(image, px, py, flux)
        hit += 1

print("Projected stars:", hit, "/", total)


# -----------------------------
# VISUALIZATION
# -----------------------------
if np.max(image) > 0:
    img = image **0.3
    img /= np.max(img)
else:
    img = image

plt.imshow(img, origin="lower", cmap="gray")
plt.title("Star Tracker Viewport")
plt.colorbar()
plt.show()
