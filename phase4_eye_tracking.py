import os
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np
import mediapipe as mp

K0    = np.load("calibration_results/cam0/K.npy")
dist0 = np.load("calibration_results/cam0/dist.npy")
K1    = np.load("calibration_results/cam1/K.npy")
dist1 = np.load("calibration_results/cam1/dist.npy")
P0    = np.load("calibration_results/stereo/P0.npy")
P1    = np.load("calibration_results/stereo/P1.npy")
map0x = np.load("calibration_results/stereo/map0x.npy")
map0y = np.load("calibration_results/stereo/map0y.npy")
map1x = np.load("calibration_results/stereo/map1x.npy")
map1y = np.load("calibration_results/stereo/map1y.npy")

print("Calibration loaded.")

# MEDIAPIPE — one instance per camera
mp_face_mesh = mp.solutions.face_mesh

face_mesh_0 = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh_1 = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_IRIS  = 468
RIGHT_IRIS = 473

#FUNCTIONS

def get_iris_centers(frame, face_mesh_instance):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w      = frame.shape[:2]

    left_iris = (
        int(landmarks[LEFT_IRIS].x * w),
        int(landmarks[LEFT_IRIS].y * h)
    )
    right_iris = (
        int(landmarks[RIGHT_IRIS].x * w),
        int(landmarks[RIGHT_IRIS].y * h)
    )

    return left_iris, right_iris

def triangulate_iris(pt0, pt1):
    pt0_arr  = np.array([[pt0[0]], [pt0[1]]], dtype=np.float64)
    pt1_arr  = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    point_4d = cv2.triangulatePoints(P0, P1, pt0_arr, pt1_arr)
    point_3d = point_4d[:3] / point_4d[3]
    return point_3d.flatten()

# Main loop

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

# Warmup
for _ in range(30):
    cap0.read()
    cap1.read()

print("Phase 3 - 3D Eye Tracking")
print("Press Q to quit.")
print("When eye is detected and Z looks correct — run phase4_screen_calibrate.py next.")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        break

    rect0 = cv2.remap(frame0, map0x, map0y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)

    left0, right0 = get_iris_centers(rect0, face_mesh_0)
    left1, right1 = get_iris_centers(rect1, face_mesh_1)

    if left0 is not None and left1 is not None:
        pos_3d  = triangulate_iris(left0, left1)
        X, Y, Z = pos_3d

        cv2.circle(rect0, left0, 8, (0, 255, 0), 2)
        cv2.circle(rect1, left1, 8, (0, 255, 0), 2)

        cv2.putText(rect0,
            f"X:{X:.3f}m  Y:{Y:.3f}m  Z:{Z:.3f}m",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(rect0,
            f"Eye is {Z*100:.1f}cm from cameras",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(rect0,
            "No face detected",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    combined = np.hstack([rect0, rect1])
    scale    = 0.6
    display  = cv2.resize(combined,
                   (int(combined.shape[1]*scale),
                    int(combined.shape[0]*scale)))
    cv2.imshow("Phase 3 - 3D Eye Tracking", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
print("Done.")
