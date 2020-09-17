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
import multiprocessing


class AImodel_tpu(multiprocessing.Process):
    def __init__(self, input_model, input_cam, input_lim, input_pts):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()

        cam_81 = 'rtsp://192.168.200.81:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        cam_82 = 'rtsp://192.168.200.82:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        model_People = 'detection_toco_edgetpu.tflite'
        model_SSD = 'mobilenet_ssd_v2_edgetpu.tflite'

        if input_model == "SSD Mobilenet v2 detection":
            labels = load_labels('coco_labels.txt')
            self.DNN = model_tpu(model_SSD, labels)
        elif input_model == "SSD Custom People detection":
            labels = load_labels('people_label.txt')
            self.DNN = model_tpu(model_People, labels)

        # if input_cam == "192.168.200.78":
        #     self.fvs = CameraVideoStream(cam_78).start()
        # elif input_cam == "192.168.200.81":
        #     self.fvs = CameraVideoStream(cam_81).start()
        self.fvs = cv2.VideoCapture(input_cam)
        self.limit = int(input_lim)

        # Initialize necessary variables
        self.ct = CentroidTracker(maxDisappeared=2, maxDistance=45)
        self.trackableObjects = dict()
        self.totalCount = 0
        self.countedID = 0
        self.ROI = 350
        self.total_six_feet_violations = 0
        self.d_thresh = np.sqrt(
            (input_pts[0][0] - input_pts[1][0]) ** 2
            + (input_pts[0][1] - input_pts[1][1]) ** 2
        )

        #self.run_flag = True

    def shutdown(self):
        self.exit.set()
        print("Process shutdown!")

    def run(self):
        while not self.exit.is_set():

            direction_str = "..."
            distance = "..."
            ret, frame = self.fvs.read()

            if frame is None:
                continue

            H = 480
            W = 640

            t_dtc = time.time()
            # Detect person and bounding boxes using DNN
            frame, pedestrian_boxes = self.DNN.detect_distance(frame)

            frame_ctr = cv2.line(frame, (self.ROI, 0),
                                 (self.ROI, H), (0, 255, 255), 2)
            frame_ctr, self.countedID, self.totalCount, direction_str, self.ct, self.trackableObjects = append_objs_counter(
                frame_ctr, self.countedID, pedestrian_boxes, self.ROI, self.ct, self.trackableObjects, self.totalCount)

            frame_dist, dist_violation = append_objs_distance(
                frame, pedestrian_boxes, self.d_thresh)
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
                    dist_m = "{:.2f}".format(
                        dist_violation[i] / self.d_thresh * self.limit)
                    self.total_six_feet_violations += 1
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
