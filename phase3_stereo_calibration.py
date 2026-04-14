import cv2
import numpy as np
import glob
import os


CHESSBOARD  = (9, 7)
square_size = 0.019  # 19mm in meters


K0    = np.load("calibration_results/cam0/K.npy")
dist0 = np.load("calibration_results/cam0/dist.npy")
K1    = np.load("calibration_results/cam1/K.npy")
dist1 = np.load("calibration_results/cam1/dist.npy")

print(f"K0:\n{K0}\n")
print(f"K1:\n{K1}\n")


print("Loading captured stereo image pairs...")

corners0_files = sorted(glob.glob("stereo_images/cam0/corners_*.npy"))
corners1_files = sorted(glob.glob("stereo_images/cam1/corners_*.npy"))

if len(corners0_files) == 0:
    print("ERROR: No corner files found in stereo_images/")
    print("Run phase2a_capture.py first.")
    exit()

print(f"Found {len(corners0_files)} pairs.")

# Build object points and image points
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * square_size

objpoints  = []
imgpoints0 = []
imgpoints1 = []

for f0, f1 in zip(corners0_files, corners1_files):
    c0 = np.load(f0)
    c1 = np.load(f1)
    objpoints.append(objp)
    imgpoints0.append(c0)
    imgpoints1.append(c1)

# Get image size from first captured image
sample_img = cv2.imread("stereo_images/cam0/capture_00.jpg")
image_size  = (sample_img.shape[1], sample_img.shape[0])
print(f"Image size: {image_size}")

#OUTLIER REMOVAL — remove pairs with large Y difference
print("\nRemoving outlier pairs...")

cleaned_obj  = []
cleaned_img0 = []
cleaned_img1 = []
threshold    = 80  # px

for i, (pts0, pts1) in enumerate(zip(imgpoints0, imgpoints1)):
    y0   = pts0[:, 0, 1]
    y1   = pts1[:, 0, 1]
    diff = np.mean(np.abs(y0 - y1))

    if diff < threshold:
        cleaned_obj.append(objpoints[i])
        cleaned_img0.append(pts0)
        cleaned_img1.append(pts1)
        print(f"  pair {i:02d}: kept    (Y diff={diff:.1f}px)")
    else:
        print(f"  pair {i:02d}: REMOVED (Y diff={diff:.1f}px)")

print(f"\nKept {len(cleaned_obj)} of {len(objpoints)} pairs.")

if len(cleaned_obj) < 4:
    print("ERROR: Not enough valid pairs. Recapture with phase2a_capture.py")
    exit()

objpoints  = cleaned_obj
imgpoints0 = cleaned_img0
imgpoints1 = cleaned_img1

# STEREO CALIBRATION
print("\nRunning stereo calibration...")

flags = cv2.CALIB_FIX_INTRINSIC

error, K0, dist0, K1, dist1, R, T, E, F = cv2.stereoCalibrate(
    objpoints,
    imgpoints0,
    imgpoints1,
    K0, dist0,
    K1, dist1,
    image_size,
    flags=flags
)

print(f"\nReprojection error: {error:.4f} px")
print(f"\nR (rotation between cameras):\n{R}")
print(f"\nT (translation between cameras):\n{T}")
print(f"\nBaseline distance: {abs(T[0][0])*100:.2f} cm")

print("Computing rectification maps...")

R0, R1, P0, P1, Q, roi0, roi1 = cv2.stereoRectify(
    K0, dist0,
    K1, dist1,
    image_size,
    R, T,
    alpha=0
)

map0x, map0y = cv2.initUndistortRectifyMap(
    K0, dist0, R0, P0, image_size, cv2.CV_32FC1)
map1x, map1y = cv2.initUndistortRectifyMap(
    K1, dist1, R1, P1, image_size, cv2.CV_32FC1)

print("\nSaving results...")

os.makedirs("calibration_results/stereo", exist_ok=True)

np.save("calibration_results/stereo/R.npy",     R)
np.save("calibration_results/stereo/T.npy",     T)
np.save("calibration_results/stereo/E.npy",     E)
np.save("calibration_results/stereo/F.npy",     F)
np.save("calibration_results/stereo/R0.npy",    R0)
np.save("calibration_results/stereo/R1.npy",    R1)
np.save("calibration_results/stereo/P0.npy",    P0)
np.save("calibration_results/stereo/P1.npy",    P1)
np.save("calibration_results/stereo/Q.npy",     Q)
np.save("calibration_results/stereo/map0x.npy", map0x)
np.save("calibration_results/stereo/map0y.npy", map0y)
np.save("calibration_results/stereo/map1x.npy", map1x)
np.save("calibration_results/stereo/map1y.npy", map1y)

print("\nAll stereo calibration results saved.")
print(f"""Reprojection error: {error:.4f} px
Baseline: {abs(T[0][0])*100:.2f} cm
""")
