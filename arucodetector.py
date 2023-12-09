import cv2
import numpy as np
import pandas as pd
from collections import deque

class ArucoDetector():
    __slots__ = ['camera_parameters', 'distortion_parameters', 'marker_size',
                 'arucoParams', 'aruco_dicts', 'marker_points', 'bbox_color',
                 'distances', 'logo', 'lim', 'resize_factor',
                 ]

    def __init__(self):
        try:
            self.camera_parameters = pd.read_csv("cm.csv", header=0, index_col=0).to_numpy()
            self.distortion_parameters = pd.read_csv("cd.csv", header=0, index_col=0).to_numpy()
        except FileNotFoundError:
            print("Parameters not found...")

        dict_4_4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        dict_5_5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        dict_6_6 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        self.marker_size = 0.10  # 0.016*6
        self.aruco_dicts = [dict_5_5, dict_6_6, dict_4_4]
        self.arucoParams = cv2.aruco.DetectorParameters()
        self.marker_points = np.array([[-self.marker_size / 2, self.marker_size / 2, 0],
                                  [self.marker_size / 2, self.marker_size / 2, 0],
                                  [self.marker_size / 2, -self.marker_size / 2, 0],
                                  [-self.marker_size / 2, -self.marker_size / 2, 0]], dtype=np.float32)
        self.bbox_color = (0, 255, 0)
        self.distances = deque([0.0], maxlen=5)
        self.resize_factor = 1

        try:
            logo_file = 'logo.png'
            logo = cv2.imread(logo_file)
            new_size = 100
            self.logo = cv2.resize(logo, (new_size, new_size), interpolation=cv2.INTER_AREA)
            self.lim = -new_size - 1
        except cv2.error:
            pass

    def add_logo(self):
        # Add our logo if present:
        try:
            l = self.lim
            self.frame[l:-1, l:-1, 0:3] = self.logo
            self.frame[l:-1, l:-1, 2] = 1
        except Exception as error:
            print(error)
            pass

    def detect_markers(self, frame):
        resize_factor = self.resize_factor
        tvecs = dict()
        rvecs = dict()
        # Loop over aruco dictionaries:
        for aruco_dict in self.aruco_dicts:
            detector = cv2.aruco.ArucoDetector(aruco_dict, self.arucoParams)
            # resize image for speed:
            resized_image = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            (corners, ids, rejected) = detector.detectMarkers(resized_image)

            # verify that there are detections in frame first:
            if len(corners) > 0:
                ids = ids.flatten()
                # loop over the detected ArUCo markers:
                for (marker_corner, marker_id) in zip(corners, ids):
                    corner_points = marker_corner.reshape((4, 2))
                    (top_left, top_right, bottom_right, bottom_left) = corner_points / resize_factor
                    top_right = (int(top_right[0]), int(top_right[1]))
                    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                    top_left = (int(top_left[0]), int(top_left[1]))

                    # draw the bounding box of the ArUCo detection
                    for loc in (
                            (top_left, top_right),
                            (top_right, bottom_right),
                            (bottom_right, bottom_left),
                            (bottom_left, top_left)
                                ):
                        cv2.line(frame, loc[0], loc[1], self.bbox_color, 2)

                    # compute and draw the center (x, y) of the marker
                    cX = int((top_left[0] + bottom_right[0]) / 2.0)
                    cY = int((top_left[1] + bottom_right[1]) / 2.0)
                    cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

                    # Find the translation vector of the marker:
                    _, rvec, tvec = cv2.solvePnP(self.marker_points,
                                                 marker_corner,
                                                 self.camera_parameters,
                                                 self.distortion_parameters,
                                                 False, cv2.SOLVEPNP_IPPE_SQUARE)
                    # Draw Axis
                    cv2.drawFrameAxes(frame, self.camera_parameters,
                                      self.distortion_parameters, rvec, tvec, 0.20)

                    tvecs[marker_id] = tvec
                    rvecs[marker_id] = rvec

                    # Write the ArUco marker ID text:
                    cv2.putText(frame, str(marker_id),
                                (top_left[0] - 15, top_left[1] - 15),
                                cv2.FONT_HERSHEY_DUPLEX,
                                1,
                                (0, 255, 0),
                                2)

        return frame, tvecs, rvecs
