import os
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np
import mediapipe as mp


P0    = np.load("calibration_results/stereo/P0.npy")
P1    = np.load("calibration_results/stereo/P1.npy")
map0x = np.load("calibration_results/stereo/map0x.npy")
map0y = np.load("calibration_results/stereo/map0y.npy")
map1x = np.load("calibration_results/stereo/map1x.npy")
map1y = np.load("calibration_results/stereo/map1y.npy")
H     = np.load("calibration_results/H_gaze.npy")

print("All calibration loaded.")

SCREEN_W = 2560
SCREEN_H = 1664

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

# Functions

def get_iris_and_gaze(frame, face_mesh_instance):
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh_instance.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w      = frame.shape[:2]

    iris  = (int(landmarks[LEFT_IRIS].x * w),
             int(landmarks[LEFT_IRIS].y * h))
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

    return iris, gaze_offset

def triangulate(pt0, pt1):
    p0 = np.array([[pt0[0]], [pt0[1]]], dtype=np.float64)
    p1 = np.array([[pt1[0]], [pt1[1]]], dtype=np.float64)
    p4 = cv2.triangulatePoints(P0, P1, p0, p1)
    return (p4[:3] / p4[3]).flatten()

def gaze_to_screen(gaze_x, gaze_y):
    pt     = np.array([[[-gaze_x, -gaze_y]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    px     = int(np.clip(result[0][0][0], 0, SCREEN_W - 1))
    py     = int(np.clip(result[0][0][1], 0, SCREEN_H - 1))
    return px, py

# VIDEO RECORDING SETUP
RECORD  = True #you can change it to False
CAM_W   = 640
CAM_H   = 480
MINI_W  = 640
MINI_H  = 480
OUT_W   = CAM_W + MINI_W
OUT_H   = max(CAM_H, MINI_H)

if RECORD:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter('gaze_demo.mp4', fourcc, 20.0, (OUT_W, OUT_H))
    print(f"Recording to gaze_demo.mp4 ({OUT_W}x{OUT_H})")

# SMOOTHING
history = []
SMOOTH  = 8


cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

print("Warming up cameras...")
for _ in range(30):
    cap0.read()
    cap1.read()

# Fullscreen gaze display
cv2.namedWindow("Gaze", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze", cv2.WND_PROP_FULLSCREEN,
                       cv2.WINDOW_FULLSCREEN)

print("Phase 5 - Gaze Demo running.")
print("Press Q to quit and save recording.")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        break

    rect0 = cv2.remap(frame0, map0x, map0y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(frame1, map1x, map1y, cv2.INTER_LINEAR)

    iris0, gaze0 = get_iris_and_gaze(rect0, face_mesh_0)
    iris1, gaze1 = get_iris_and_gaze(rect1, face_mesh_1)

    # Black screen for gaze dot
    screen = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

    # Light grid for reference
    for x in range(0, SCREEN_W, SCREEN_W // 4):
        cv2.line(screen, (x, 0), (x, SCREEN_H), (25, 25, 25), 1)
    for y in range(0, SCREEN_H, SCREEN_H // 4):
        cv2.line(screen, (0, y), (SCREEN_W, y), (25, 25, 25), 1)

    if iris0 is not None and iris1 is not None:
        avg_gaze = (gaze0 + gaze1) / 2
        px, py   = gaze_to_screen(avg_gaze[0], avg_gaze[1])

        # Smooth gaze point
        history.append((px, py))
        if len(history) > SMOOTH:
            history.pop(0)
        spx = int(np.mean([g[0] for g in history]))
        spy = int(np.mean([g[1] for g in history]))

        # Draw gaze dot
        cv2.circle(screen, (spx, spy), 20, (0, 255, 0), -1)
        cv2.circle(screen, (spx, spy), 28, (0, 180, 0),  2)

        cv2.putText(screen,
            f"Gaze: ({spx}, {spy})",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
        cv2.putText(screen,
            f"Offset: ({avg_gaze[0]:.3f}, {avg_gaze[1]:.3f})",
            (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)

        # Draw iris circle on camera feed
        cv2.circle(rect0, iris0, 8, (0, 255, 0), 2)
        cv2.circle(rect1, iris1, 8, (0, 255, 0), 2)

    else:
        cv2.putText(screen,
            "No face detected",
            (SCREEN_W//2 - 200, SCREEN_H//2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow("Gaze", screen)

    if RECORD:
        # Camera feed resized
        cam_display = cv2.resize(rect0, (CAM_W, CAM_H))

        # Re-draw iris on resized frame
        if iris0 is not None:
            sx = CAM_W / rect0.shape[1]
            sy = CAM_H / rect0.shape[0]
            iris_s = (int(iris0[0]*sx), int(iris0[1]*sy))
            cv2.circle(cam_display, iris_s, 8, (0, 255, 0), 2)

        cv2.putText(cam_display, "cam0 - iris tracking",
            (10, CAM_H - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (150, 150, 150), 1)

        # Mini gaze screen
        mini_screen = cv2.resize(screen, (MINI_W, MINI_H))
        cv2.putText(mini_screen, "gaze on screen",
            (10, MINI_H - 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (150, 150, 150), 1)

        combined_frame = np.hstack([cam_display, mini_screen])
        out.write(combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()

if RECORD:
    out.release()
    print("Saved gaze_demo.mp4")

cv2.destroyAllWindows()
print("Done.")
