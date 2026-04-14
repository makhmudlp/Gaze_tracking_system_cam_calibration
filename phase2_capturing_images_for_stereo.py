import cv2
import numpy as np
import os

# Settings of CHESSBOARD
CHESSBOARD = (9, 7)
square_size = 0.019  # 19mm in meters

#prepare real world coordinates
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2) * square_size

os.makedirs("stereo_images/cam0", exist_ok=True)
os.makedirs("stereo_images/cam1", exist_ok=True)

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

img_count = 0

print("Press SPACE to capture — only when BOTH DETECTED is green.")
print("Press Q to quit.")
print("Aim for 25-30 pairs with varied angles and positions.")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Can't read one or both cameras.")
        break

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    ret_cb0, corners0 = cv2.findChessboardCorners(gray0, CHESSBOARD, None)
    ret_cb1, corners1 = cv2.findChessboardCorners(gray1, CHESSBOARD, None)

    both_detected = ret_cb0 and ret_cb1

    # Draw corners if detected
    if ret_cb0:
        cv2.drawChessboardCorners(frame0, CHESSBOARD, corners0, ret_cb0)
    if ret_cb1:
        cv2.drawChessboardCorners(frame1, CHESSBOARD, corners1, ret_cb1)

    # Status overlay
    status_color = (0, 255, 0) if both_detected else (0, 0, 255)
    status_text  = "BOTH DETECTED - press SPACE" if both_detected else "WAITING..."

    cv2.putText(frame0, f"CAM0 | {status_text} | pairs: {img_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame1, f"CAM1 | {status_text} | pairs: {img_count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    combined = np.hstack([frame0, frame1])
    cv2.imshow("Stereo Capture", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' ') and both_detected:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Refine corners to subpixel accuracy
        corners0_refined = cv2.cornerSubPix(gray0, corners0, (11,11), (-1,-1), criteria)
        corners1_refined = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)

        # Save RAW frames (before drawing) for re-detection later
        fname0 = f"stereo_images/cam0/capture_{img_count:02d}.jpg"
        fname1 = f"stereo_images/cam1/capture_{img_count:02d}.jpg"
        cv2.imwrite(fname0, frame0)
        cv2.imwrite(fname1, frame1)

        # Save refined corners alongside images
        np.save(f"stereo_images/cam0/corners_{img_count:02d}.npy", corners0_refined)
        np.save(f"stereo_images/cam1/corners_{img_count:02d}.npy", corners1_refined)

        img_count += 1
        print(f"Captured pair {img_count:02d} — saved to stereo_images/")

cap0.release()
cap1.release()
cv2.destroyAllWindows()

print(f"\nDone. Total pairs captured: {img_count}")
print("Now run phase3_stereo_calibration.py to compute stereo calibration.")
