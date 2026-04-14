import cv2
import numpy as np
import glob
import os

CHESSBOARD=(9,7)
cam_id=0

#creating real world coordinates of corners = each square = 1 unit
#so corners are (0,0,0), (1,0,0)...
objp=np.zeros((CHESSBOARD[0]*CHESSBOARD[1], 3), dtype=np.float32)
objp[:,:2]=np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1,2)

objpoints =[]
imgpoints =[]

os.makedirs(f"calibration_images_{cam_id}", exist_ok=True)
cap=cv2.VideoCapture(cam_id)
img_count=0


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error with reading camera")
        break

    #Convert to grayscale, because opencv finds corners on grayscale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_cb, corners=cv2.findChessboardCorners(gray, CHESSBOARD, None)

    if ret_cb:
        cv2.drawChessboardCorners(frame, CHESSBOARD, corners, ret_cb)
        cv2.putText(frame, "Detected - press SPACE to Capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0), 2)
    else:
        cv2.putText(frame, "No chessboard detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    cv2.putText(frame, f"Captures: {img_count}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.imshow(f"Calibration of camera {cam_id}", frame)
    key=cv2.waitKey(1) & 0xFF

    if key==ord('q'):
        break
    elif key == ord(' ') and ret_cb and corners is not None:
        corners_refined=cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners_refined)

        filename=f"calibration_images_{cam_id}/capture_{img_count:02d}.jpg"
        cv2.imwrite(filename, frame)
        img_count+=1

cap.release()
cv2.destroyAllWindows()
print(f"\nDone collecting. Total captures: {img_count}")

print("CALCULATING CAMERA MATRIX")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print(f"\nCamera Matrix K:\n{K}")
print(f"\nDistortion Coefficients:\n{dist}")
print(f"\nReprojection Error: {ret:.4f} px")

# === STAGE 4: SAVE ===

os.makedirs(f"calibration_results/{cam_id}", exist_ok=True)

np.save(f"calibration_results/{cam_id}/K.npy", K)
np.save(f"calibration_results/{cam_id}/dist.npy", dist)

print(f"\nSaved K.npy and dist.npy to calibration_results/{cam_id}/")





