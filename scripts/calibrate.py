import cv2
import numpy as np
import json


# Checkerboard settings
checkerboard_size = (10, 7)  # Number of inner corners (columns, rows)
square_size = 0.021  # Size of each square in meters


# Capture video from /dev/video0
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open video capture")
    exit()


# Arrays to store object points and image points from all the calibration images
obj_points = []  # 3D points in the world coordinate space
img_points = []  # 2D points in the image plane


# Prepare object points, like (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (8, 5, 0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size


frame_count = 0
# Calibration loop
while True:
    # Capture frame from video stream
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # If corners are found, add object points and image points
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret)
        cv2.imshow('Calibration', frame)
        frame_count = frame_count +1

    # Exit calibration loop if 'q' is pressed
    if cv2.waitKey(500) == ord('q'):
        print("Captured " + str(frame_count) + " frames with chessboard corners")
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()


# Perform camera calibration
ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

if ret:
    print("Camera calibration successful")
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(distortion_coeffs)

    # Generate extrinsics.json
    extrinsics = {
        "R": rvecs[0].tolist(),
        "t": tvecs[0].tolist()
    }

    with open("extrinsics.json", "w") as f:
        json.dump(extrinsics, f, indent=4)
        print("extrinsics.json file generated")
else:
    print("Camera calibration failed")