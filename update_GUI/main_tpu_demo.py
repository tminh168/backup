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
            labels = load_labels('coco_labels.txt')
            self.DNN = model_tpu(model_SSD, labels)

        elif input_model == "SSD Custom People detection":
            labels = load_labels('people_label.txt')
            self.DNN = model_tpu(model_People, labels)


        if input_cam == "192.168.200.78":
            self.fvs = CameraVideoStream(cam_78).start()
        elif input_cam == "192.168.200.81":
            self.fvs = CameraVideoStream(cam_81).start()
        
        self.limit = int(input_lim)
        self.two_points = input_pts

        # Initialize necessary variables
        self.ct = CentroidTracker(maxDisappeared=2, maxDistance=45)
        self.trackableObjects = dict()
        self.totalCount = 0
        self.countedID = 0
        self.ROI = 350
        self.frame_num = 0
        self.total_six_feet_violations = 0

        self.run_flag = True

    def terminate(self, timeout=0):
        self.run_flag = False
        self._stopevent.set()
        threading.Thread.join(self, timeout)

    def run(self):
        
        # Process each frame, until end of video
        while not self._stopevent.isSet():
            if not self.run_flag:
                break

            direction_str = "..."
            distance = "..."
            self.frame_num += 1
            frame = self.fvs.read()

            if frame is None:
                continue

            H = 480
            W = 640

            if self.frame_num == 1:
                # Get threshold distance and bird image
                d_thresh = np.sqrt(
                    (self.two_points[0][0] - self.two_points[1][0]) ** 2
                    + (self.two_points[0][1] - self.two_points[1][1]) ** 2
                )
                
            #print("Processing frame: ", frame_num)

            t_dtc = time.time()
            # Detect person and bounding boxes using DNN
            frame, pedestrian_boxes = self.DNN.detect_distance(frame)

            frame_ctr = cv2.line(frame, (self.ROI, 0), (self.ROI, H), (0, 255, 255), 2)
            frame_ctr, self.countedID, self.totalCount, direction_str, self.ct, self.trackableObjects = append_objs_counter(
                    frame_ctr, self.countedID, pedestrian_boxes, self.ROI, self.ct, self.trackableObjects, self.totalCount)

            frame_dist, dist_violation = append_objs_distance(frame, pedestrian_boxes, d_thresh)
            te_dtc = time.time()
            dtc_rate = te_dtc - t_dtc
            print('Detection: {}'.format(dtc_rate))

            info_ctr = [
                ("Direction", direction_str),
                ("Count", self.totalCount),
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
                ("Violation", self.total_six_feet_violations),
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