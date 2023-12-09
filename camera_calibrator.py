import cv2
import numpy as np
import pandas as pd


try:
    CAMERA_PARAMETERS = pd.read_csv("cm.csv", header=0, index_col=0).to_numpy()
    DISTORTION_PARAMETERS = pd.read_csv("cd.csv", header=0, index_col=0).to_numpy()
except:
    print("Parameters not found...")

# Define the dimensions of checkerboard
CHECKERBOARD = (7, 7)
MIN_POINTS = 80
# Stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Vector for the 3D points:
threedpoints = []
# Vector for 2D points:
twodpoints = []
# 3D points real world coordinates:
objectp3d = np.zeros((1, CHECKERBOARD[0]
                      * CHECKERBOARD[1],
                      3), np.float32)

objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                      0:CHECKERBOARD[1]].T.reshape(-1, 2)

prev_img_shape = None
matrix, distortion = None, None


arucoDict_4_4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
arucoParams = cv2.aruco.DetectorParameters()

aruco_dicts = [arucoDict_4_4]
bbox_color = (0, 255, 0)
t_vecs = dict()
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280//2)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720//2)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:
    ret, frame = vid.read()
    if ret:
        output_image = frame
        ##############################################################################
        ##### CALIBRATE CAMERA HERE ########
        grayColor = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        progress_message = "Points: " + str(len(twodpoints)) + " of " + str(MIN_POINTS)
        cv2.putText(frame, progress_message,
                    (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 255, 0), 2)

        # Find the chess board corners
        # if desired number of corners are
        # found in the image then cors = true:
        cors = False
        cors, corners = cv2.findChessboardCorners(
            grayColor, CHECKERBOARD,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK +
            cv2.CALIB_CB_NORMALIZE_IMAGE)

        if cors:
            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            threedpoints.append(objectp3d)
            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, CHECKERBOARD, (-1, -1), criteria)

            twodpoints.append(corners2)

            output_image = cv2.drawChessboardCorners(frame,
                                                     CHECKERBOARD,
                                                     corners2, ret)

            # When we have minimum number of data points, stop:
            if len(twodpoints) > MIN_POINTS:
                # Perform camera calibration by
                # passing the value of above found out 3D points (threedpoints)
                # and its corresponding pixel coordinates of the
                # detected corners (twodpoints):
                vid.release()
                cv2.destroyAllWindows()
                print("Calibrating....")
                ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
                    threedpoints, twodpoints, grayColor.shape[::-1], None, None)

                cm = pd.DataFrame(matrix)
                cm.to_csv(r'cm.csv')
                cd = pd.DataFrame(distortion)
                cd.to_csv(r'cd.csv')

                print("Finished")
                break

        cv2.imshow("frame", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            break
