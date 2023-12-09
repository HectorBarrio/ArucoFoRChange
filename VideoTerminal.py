import cv2
import numpy as np
from arucodetector import ArucoDetector
import requests
import time
from screeninfo import get_monitors
from collections import deque


class VideoTerminal:
    __slots__ = ["frame", "ret", "w", "h", "smooth_tvecs", "smooth_rvecs",
                 "vid", "fps_list", "average_fps", "bbox_color",
                 'aruco_detector']

    def __init__(self, usb_cam=True, ip_cam=False):
        self.aruco_detector = ArucoDetector()
        self.frame = False
        self.ret = False
        self.fps_list = deque([0, 0], maxlen=50)
        self.average_fps = 0
        self.bbox_color = (0, 255, 0)
        self.smooth_tvecs = dict()
        self.smooth_rvecs = dict()

        if usb_cam:
            ip_cam = not usb_cam

        for m in get_monitors():
            self.w = m.width
            self.h = m.height

        if usb_cam:
            self.vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280//2)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720//2)
            self.vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if ip_cam:
            command = 'sudo gst-launch-1.0 rtspsrc'
            video_stream = 'rtsp://192.168.1.221/video0.sdp'
            gstreamer_str = f"{command} location={video_stream} latency=1 buffer-mode=auto ! rtph264depay ! avdec_h264 ! videorate ! videoconvert ! appsink max-buffers=1 drop=True sync=False"
            self.vid = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)

    def start_video_feed(self):
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            start = time.time()
            self.ret, self.frame = self.vid.read()
            if not self.ret:
                print("Missing frame")
                continue

            self.frame, tvecs, rvecs = self.aruco_detector.detect_markers(self.frame)
            self.frame = cv2.resize(self.frame, (self.w, self.h))

            try:
                through_time = time.time() - start
                through_fps = 1/through_time
                self.average_fps = np.mean(np.array(self.fps_list))
                self.fps_list.append(through_fps)
            except ZeroDivisionError:
                print("Missing data for through time.")

            self.stamp_data(tvecs, rvecs)
            cv2.imshow('frame', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stamp_data(self, tvecs, rvecs):
        self.frame[0:100, :] = 0
        text_line = 1

        if not tvecs:
            text = f"{self.average_fps:.0f} FPS - P: No Marker"
            cv2.putText(self.frame, text, (20, text_line * 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, self.bbox_color, 4, cv2.LINE_AA)
        else:
            rmat = 0
            for marker_id in [*tvecs]:
                # Smooth points out if not present:
                if marker_id not in self.smooth_tvecs:
                    self.smooth_tvecs[marker_id] = deque(maxlen=30)
                    self.smooth_rvecs[marker_id] = deque(maxlen=30)

                self.smooth_tvecs[marker_id].append(tvecs[marker_id])
                self.smooth_rvecs[marker_id].append(rvecs[marker_id])

                # Set the origin point: 8 in this case
                if marker_id == 8:
                    print(self.smooth_rvecs[marker_id])
                    R = np.array(self.smooth_rvecs[marker_id])
                    R = np.median(R, axis=0)
                    print(R)
                    T = np.array(self.smooth_tvecs[marker_id])
                    T = np.median(T, axis=0)
                    rvec_matrix = cv2.Rodrigues(R)[0]
                    rmat = np.matrix(rvec_matrix)
                    tmat = np.matrix(T)
                    continue

                cam_point = np.array(self.smooth_tvecs[marker_id]).mean(axis=0)
                if type(rmat) != int:
                    world_point = np.array(rmat ** -1 * (cam_point - tmat))
                    px = world_point[0][0]
                    py = world_point[1][0]
                    pz = world_point[2][0]

                    text = f'{self.average_fps:.0f} FPS - Dist: {px:.2f}, {py:.2f}, {pz:.2f}'
                else:
                    text = "No reference"


                cv2.putText(self.frame, text, (20, text_line * 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, self.bbox_color, 4, cv2.LINE_AA)
                text_line += 1
