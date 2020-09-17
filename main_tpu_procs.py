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


def AImodel_tpu(input_model, input_cam, input_lim, input_pts):

    cam_81 = 'rtsp://192.168.200.81:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    cam_82 = 'rtsp://192.168.200.82:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    model_People = 'detection_toco_edgetpu.tflite'
    model_SSD = 'mobilenet_ssd_v2_edgetpu.tflite'

    if input_model == "SSD Mobilenet v2 detection":
        labels = load_labels('coco_labels.txt')
        DNN = model_tpu(model_SSD, labels)
    elif input_model == "SSD Custom People detection":
        labels = load_labels('people_label.txt')
        DNN = model_tpu(model_People, labels)

    # if input_cam == "192.168.200.81":
    #     fvs = CameraVideoStream(cam_81).start()
    # elif input_cam == "192.168.200.82":
    #     fvs = CameraVideoStream(cam_82).start()

    fvs = cv2.VideoCapture(input_cam)
    limit = int(input_lim)

    # Initialize necessary variables
    ct = CentroidTracker(maxDisappeared=2, maxDistance=45)
    trackableObjects = dict()
    totalCount = 0
    countedID = 0
    ROI = 350
    total_six_feet_violations = 0
    d_thresh = np.sqrt(
        (input_pts[0][0] - input_pts[1][0]) ** 2
        + (input_pts[0][1] - input_pts[1][1]) ** 2
    )

    # Process each frame, until end of video
    while True:

        direction_str = "..."
        distance = "..."
        ret, frame = fvs.read()

        if frame is None:
            continue

        H = 480
        W = 640

        t_dtc = time.time()
        # Detect person and bounding boxes using DNN
        frame, pedestrian_boxes = DNN.detect_distance(frame)

        frame_ctr = cv2.line(frame, (ROI, 0),
                             (ROI, H), (0, 255, 255), 2)
        frame_ctr, countedID, totalCount, direction_str, ct, trackableObjects = append_objs_counter(
            frame_ctr, countedID, pedestrian_boxes, ROI, ct, trackableObjects, totalCount)

        frame_dist, dist_violation = append_objs_distance(
            frame, pedestrian_boxes, d_thresh)
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
                dist_m = "{:.2f}".format(
                    dist_violation[i] / d_thresh * limit)
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
    fvs.stop()
    return
