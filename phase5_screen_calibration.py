import os
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np
import mediapipe as mp

# === LOAD CALIBRATION ===
P0    = np.load("calibration_results/stereo/P0.npy")
P1    = np.load("calibration_results/stereo/P1.npy")
map0x = np.load("calibration_results/stereo/map0x.npy")
map0y = np.load("calibration_results/stereo/map0y.npy")
map1x = np.load("calibration_results/stereo/map1x.npy")
map1y = np.load("calibration_results/stereo/map1y.npy")

print("Calibration loaded.")

# === SCREEN SETTINGS ===
SCREEN_W = 2560
SCREEN_H = 1664

# === MEDIAPIPE ===
mp_face_mesh = mp.solutions.face_mesh

face_mesh_0 = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh_1 = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_IRIS      = 468
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33

# === FUNCTIONS ===

def get_iris_and_gaze(frame, face_mesh_instance):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb)

    if not results.multi_face_landmarks:
        return None, None, None

    landmarks  = results.multi_face_landmarks[0].landmark
    h, w       = frame.shape[:2]

    iris = (
        int(landmarks[LEFT_IRIS].x * w),
        int(landmarks[LEFT_IRIS].y * h)
    )
    inner = np.array([landmarks[LEFT_EYE_INNER].x * w,
                      landmarks[LEFT_EYE_INNER].y * h])
    outer = np.array([landmarks[LEFT_EYE_OUTER].x * w,
                      landmarks[LEFT_EYE_OUTER].y * h])

    eye_center  = (inner + outer) / 2
    eye_width   = np.linalg.norm(outer - inner)
    iris_arr    = np.array([iris[0], iris[1]], dtype=float)

    gaze_offset = np.array([
        (iris_arr[0] - eye_center[0]) / eye_width,
        (iris_arr[1] - eye_center[1]) / eye_width
    ])

    return iris, eye_center, gaze_offset

def triangulate_iris(pt0, pt1):
    pt0_arr  = np.array([[pt0[0]], [pt0[1]]], dtype=np.float64)
    pt1_arr  = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    point_4d = cv2.triangulatePoints(P0, P1, pt0_arr, pt1_arr)
    point_3d = point_4d[:3] / point_4d[3]
    return point_3d.flatten()

def get_gaze_features(rect0, rect1):
    iris0, _, gaze0 = get_iris_and_gaze(rect0, face_mesh_0)
    iris1, _, gaze1 = get_iris_and_gaze(rect1, face_mesh_1)

    if iris0 is None or iris1 is None:
        return None, None

    eye_3d   = triangulate_iris(iris0, iris1)
    avg_gaze = (gaze0 + gaze1) / 2

    features = np.array([
        avg_gaze[0],  # horizontal gaze offset
        avg_gaze[1],  # vertical gaze offset
        eye_3d[2]     # depth in meters
    ])

    return features, eye_3d

# === 9-POINT CALIBRATION GRID ===
MARGIN_X = int(SCREEN_W * 0.1)
MARGIN_Y = int(SCREEN_H * 0.1)

CALIB_POINTS = []
for row in range(3):
    for col in range(3):
        x = MARGIN_X + col * (SCREEN_W - 2*MARGIN_X) // 2
        y = MARGIN_Y + row * (SCREEN_H - 2*MARGIN_Y) // 2
        CALIB_POINTS.append((x, y))

print(f"9 calibration points ready.")

# === OPEN CAMERAS ===
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

print("Warming up cameras...")
for _ in range(30):
    cap0.read()
    cap1.read()
print("Ready!")

# Fullscreen calibration window
cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration",
                       cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)

collected_features = []
collected_points   = []

# === CALIBRATION LOOP ===
for point_idx, (target_x, target_y) in enumerate(CALIB_POINTS):

    screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    # Draw completed points in gray
    for px, py in collected_points:
        cv2.circle(screen, (px, py), 10, (50, 50, 50), -1)

    # Draw current target in red
    cv2.circle(screen, (target_x, target_y), 30, (0, 0, 255), -1)
    cv2.circle(screen, (target_x, target_y), 35, (0, 0, 200),  2)

    cv2.putText(screen,
        f"Look at the RED dot and press SPACE  ({point_idx+1}/9)",
        (SCREEN_W//2 - 400, SCREEN_H - 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    cv2.imshow("Calibration", screen)
    cv2.waitKey(1)

    print(f"\nPoint {point_idx+1}/9 — look at ({target_x}, {target_y})")
    print("Press SPACE when looking at the dot. Hold very still.")

    samples = []

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            break

        rect0 = cv2.remap(frame0, map0x, map0y, cv2.INTER_LINEAR)
        rect1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)

        features, eye_3d = get_gaze_features(rect0, rect1)

        status_screen = screen.copy()

        if features is not None:
            cv2.putText(status_screen,
                f"Eye detected  Z={eye_3d[2]:.2f}m  "
                f"gaze=({features[0]:.3f}, {features[1]:.3f})",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(status_screen,
                f"Samples: {len(samples)}",
                (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(status_screen,
                "No face detected — move into frame",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Calibration", status_screen)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' ') and features is not None:
            print(f"Collecting 30 samples — hold still...")
            for _ in range(30):
                ret0, frame0 = cap0.read()
                ret1, frame1 = cap1.read()
                rect0 = cv2.remap(frame0, map0x, map0y, cv2.INTER_LINEAR)
                rect1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)
                f, _  = get_gaze_features(rect0, rect1)
                if f is not None:
                    samples.append(f)
            break

        elif key == ord('q'):
            print("Calibration cancelled.")
            cap0.release()
            cap1.release()
            cv2.destroyAllWindows()
            exit()

    if len(samples) > 0:
        avg_features = np.mean(samples, axis=0)
        collected_features.append(avg_features)
        collected_points.append((target_x, target_y))
        print(f"Point {point_idx+1} collected: {avg_features}")
    else:
        print(f"Point {point_idx+1} skipped — no samples.")

cap0.release()
cap1.release()
cv2.destroyAllWindows()

print(f"\nTotal points collected: {len(collected_features)}")

# === FIT HOMOGRAPHY ===
if len(collected_features) >= 4:

    # Flip both X and Y to match screen coordinate directions
    src_points = np.array([
        [-f[0], -f[1]] for f in collected_features
    ], dtype=np.float32)

    dst_points = np.array(
        collected_points,
        dtype=np.float32
    )

    print(f"\nSrc points (gaze):\n{src_points}")
    print(f"Dst points (screen):\n{dst_points}")

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    print(f"\nHomography matrix H:\n{H}")
    print(f"Inliers: {mask.sum()}/{len(collected_features)}")

    os.makedirs("calibration_results", exist_ok=True)
    np.save("calibration_results/H_gaze.npy", H)
    print("\nSaved calibration_results/H_gaze.npy")
    print("Now run phase5_gaze_demo.py")

else:
    print("Not enough points. Need at least 4. Run again.")