import cv2
import os
import time
import argparse
import imutils
import numpy as np
from cameravideostream import CameraVideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from tpu_model import *
from track_distance import *
import threading

class AImodel_tpu(threading.Thread):
    def __init__(self, input_model, input_cam, input_lim, input_pts):
        self._stopevent = threading.Event()
        self._sleepperiod = 1.0

        threading.Thread.__init__(self)

        cam_78 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        cam_81 = 'rtsp://192.168.200.81:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        model_People = 'detection_toco_edgetpu.tflite'
        model_SSD = 'mobilenet_ssd_v2_edgetpu.tflite'

        if input_model == "SSD Mobilenet v2 detection":
            model = model_SSD
        elif input_model == "SSD Custom People detection":
            model = model_People
        if input_cam == "192.168.200.78":
            camip = cam_78
        elif input_cam == "192.168.200.81":
            camip = cam_81
        
        self.limit = int(input_lim)
        self.two_points = input_pts

        # Define a DNN model
        self.DNN = model_tpu(model)
        # Get video handle
        self.fvs = CameraVideoStream(camip).start()

        self.run_flag = True

    def terminate(self, timeout=0):
        self.run_flag = False
        self._stopevent.set()
        threading.Thread.join(self, timeout)

    def run(self):
        # Initialize necessary variables
        SOLID_BACK_COLOR = (41, 41, 41)
        ct = CentroidTracker(maxDisappeared=2, maxDistance=45)
        trackableObjects = dict()
        totalCount = 0
        countedID = 0
        ROI = 350
        frame_num = 0
        total_six_feet_violations = 0

        # Process each frame, until end of video
        while not self._stopevent.isSet():
            if not self.run_flag:
                break

            direction_str = "..."
            distance = "..."
            frame_num += 1
            frame = self.fvs.read()

            if frame is None:
                continue

            H = 480
            W = 640

            if frame_num == 1:
                # Get threshold distance and bird image
                d_thresh = np.sqrt(
                    (self.two_points[0][0] - self.two_points[1][0]) ** 2
                    + (self.two_points[0][1] - self.two_points[1][1]) ** 2
                )
                bird_image = np.zeros(
                    (H, W, 3), np.uint8
                )
                bird_image[:] = SOLID_BACK_COLOR

            #print("Processing frame: ", frame_num)

            t_dtc = time.time()
            # Detect person and bounding boxes using DNN
            frame, pedestrian_boxes = self.DNN.detect_distance(frame)

            frame_ctr = cv2.line(frame, (ROI, 0), (ROI, H), (0, 255, 255), 2)
            frame_ctr, countedID, totalCount, direction_str, ct, trackableObjects = append_objs_counter(
                    frame_ctr, countedID, pedestrian_boxes, ROI, ct, trackableObjects, totalCount)

            frame_dist, dist_violation = append_objs_distance(frame, pedestrian_boxes, d_thresh)
            te_dtc = time.time()
            dtc_rate = te_dtc - t_dtc
            print('Detection: {}'.format(dtc_rate))

            info_ctr = [
                ("Direction", direction_str),
                ("Count", totalCount),
            ]
            for (i, (k, v)) in enumerate(info_ctr):
                text = "{}: {}".format(k, v)
                cv2.putText(frame_ctr, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            dist_count = []
            if len(dist_violation) > 0:
                for i in range(len(dist_violation)):
                    dist_m = "{:.2f}".format(dist_violation[i] / d_thresh * self.limit)
                    total_six_feet_violations += 1
                    dist_count.append(dist_m)
            if len(dist_count) > 0:
                distance = ""
                for i in range(len(dist_count)):
                    distance = distance + str(dist_count[i]) + " "
            
            info_dist = [
                ("Distance(m)", distance),
                ("Violation", total_six_feet_violations),
            ]
            for (i, (k, v)) in enumerate(info_dist):
                text = "{}: {}".format(k, v)
                cv2.putText(frame_dist, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Frame counter", frame_ctr)
            cv2.imshow("Frame distance", frame_dist)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
        self.fvs.stop()
        return