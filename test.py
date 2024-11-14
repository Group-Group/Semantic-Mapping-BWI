import cv2
import numpy as np
import pykinect_azure as pykinect

pykinect.initialize_libraries()

device_config = pykinect.default_configuration
device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

kinect = pykinect.start_device(config=device_config)

def main():

    capture = kinect.update()
    # Open a connection to the camera
    ret_color, cap = capture.get_color_image()

    if not (ret_color):
        return

    # Initialize ORB detector
    orb = cv2.ORB_create()
    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    prev_pts = None
    camera_matrix = np.array([[800, 0, 320],
                               [0, 800, 240],
                               [0, 0, 1]])  # Example camera intrinsic matrix

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and descriptors
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

        if prev_frame is not None:
            # Match features between frames
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched keypoints
            prev_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches])
            curr_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])

            # Estimate motion using Essential matrix (if we have intrinsic parameters)
            if len(prev_pts) >= 5:  # Need at least 5 points
                E, mask = cv2.findEssentialMat(prev_pts, curr_pts, camera_matrix)
                _, R, t, _ = cv2.recoverPose(E, prev_pts, curr_pts, camera_matrix)

                # Print rotation and translation
                print("Rotation:\n", R)
                print("Translation:\n", t)

        # Display current frame and keypoints
        cv2.imshow('Current Frame', frame)
        cv2.drawKeypoints(frame, keypoints, frame, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Keypoints', frame)

        # Update previous frame and keypoints
        prev_frame = gray_frame
        prev_keypoints = keypoints
        prev_descriptors = descriptors

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while(True):
        main()
