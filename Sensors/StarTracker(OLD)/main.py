from Sensors.StarTracker import generate_startracker_view
from Sensors.StarTracker import process_star_tracker_output

import numpy as np
from scipy.spatial.transform import Rotation as R
import time

# ==========================================
# TEST TARGET
# ==========================================

input_area = (180.0, 55.0)

data = generate_startracker_view(
    return_plot=False,
    region_center_coords=input_area
)

print("StarTracker View Generated!")

t0 = time.time()

result = process_star_tracker_output(
    data,
    region_center=input_area,
    return_plot = True
)

elapsed = time.time() - t0

print("Processing Complete!")
print(f"Processing Took: {elapsed:.3f}s")

# ==========================================
# ATTITUDE RESULTS
# ==========================================

q = result["quaternion"]

print("\n=== ATTITUDE SOLUTION ===")
print(
    f"Quaternion (x,y,z,w): "
    f"[{q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}]"
)

print(f"Quaternion Norm: {np.linalg.norm(q):.8f}")

print(f"Triangle Score: {result['triangle_score']}")
print(f"Inlier Score:   {result['inlier_score']}")
print(f"Matched Stars:  {result['matched_stars']}")

print()
print(f"Recovered RA  : {result['ra_deg']:.6f}")
print(f"Recovered DEC : {result['dec_deg']:.6f}")

# ==========================================
# VERIFY BORESIGHT
# ==========================================

rot = R.from_quat(q)

forward = np.array([0.0, 0.0, 1.0])

sky_direction = rot.apply(forward)
sky_direction /= np.linalg.norm(sky_direction)

print()
print("Recovered Boresight:")
print(sky_direction)

sky_dir_try = R.from_quat(q).inv().apply(np.array([0.,0.,1.]))
print("sky_dir_try", sky_dir_try)

print("returned q:", q)
print("scipy as_quat:", R.from_quat(q).as_quat())

for f in [np.array([0,0,1.]), np.array([0,0,-1.])]:
    print(f, R.from_quat(q).apply(f))

print("norm:", np.linalg.norm(q))



# ==========================================
# TRUE DIRECTION
# ==========================================

ra_true_deg, dec_true_deg = input_area

print("[SIM TRUE BORESIGHT]", ra_true_deg, dec_true_deg, input_area)


ra_true = np.radians(ra_true_deg)
dec_true = np.radians(dec_true_deg)

true_dir = np.array([
    np.cos(dec_true) * np.cos(ra_true),
    np.cos(dec_true) * np.sin(ra_true),
    np.sin(dec_true)
])

# ==========================================
# ANGULAR ERROR
# ==========================================

dot = np.clip(
    np.dot(true_dir, sky_direction),
    -1.0,
    1.0
)

angle_error_deg = np.degrees(
    np.arccos(dot)
)

print()
print("=== ACCURACY ===")
print(f"True RA  : {ra_true_deg:.6f}")
print(f"True DEC : {dec_true_deg:.6f}")

print(f"Angular Error : {angle_error_deg:.6f} deg")

accuracy = (
    max(0.0, 1.0 - angle_error_deg / 180.0)
    * 100.0
)

print(f"Accuracy % : {accuracy:.6f}")