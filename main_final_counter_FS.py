import cv2
import sys
import os
import time
import argparse
import imutils
import numpy as np
import csv
import calendar
import requests
import base64
from frame_submit import FrameSubmit
from datetime import datetime
from queue import Queue
from cameravideostream import CameraVideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from tpu_model import *
from track_distance import *

cam_78 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
cam_80 = 'rtsp://192.168.200.80:555/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
model_1 = 'detection_1_edgetpu.tflite'
model_2 = 'detection_2_edgetpu.tflite'

# Define a DNN model
labels = load_labels('people_label.txt')
DNN_1 = model_tpu(model_1, labels)
DNN_2 = model_tpu(model_2, labels)
# Get video handle
fvs = CameraVideoStream(cam_78, cam_80).start()

# initialize API parameters
FS = FrameSubmit().start()

ct1 = CentroidTracker(maxDisappeared=3, maxDistance=65)
ct2 = CentroidTracker(maxDisappeared=3, maxDistance=65)
trackableObjects1 = dict()
trackableObjects2 = dict()
totalCount1 = 0
totalCount2 = 0
countedID1 = 0
countedID2 = 0
ROI = 350


# Process each frame, until end of video
while True:

    direction_str1 = "..."
    direction_str2 = "..."
    frame1, frame2 = fvs.read()

    if frame1 is None or frame2 is None:
        continue

    H = 480
    W = 640

    t_dtc = time.time()
    # Detect person and bounding boxes using DNN
    frame1, pedestrian_boxes1 = DNN_1.detect_distance(frame1)
    frame2, pedestrian_boxes2 = DNN_2.detect_distance(frame2)

    frame_ctr1 = cv2.line(frame1, (ROI, 0), (ROI, H), (0, 255, 255), 2)
    frame_ctr2 = cv2.line(frame2, (ROI, 0), (ROI, H), (0, 255, 255), 2)
    frame_ctr1, countedID1, totalCount1, direction_str1, ct1, trackableObjects1 = append_objs_counter(
        frame_ctr1, countedID1, pedestrian_boxes1, ROI, ct1, trackableObjects1, totalCount1)
    frame_ctr2, countedID2, totalCount2, direction_str2, ct2, trackableObjects2 = append_objs_counter(
        frame_ctr2, countedID2, pedestrian_boxes2, ROI, ct2, trackableObjects2, totalCount2)

    info_ctr1 = [
        ("Direction", direction_str1),
        ("Count", totalCount1),
    ]
    for (i, (k, v)) in enumerate(info_ctr1):
        text1 = "{}: {}".format(k, v)
        cv2.putText(frame_ctr1, text1, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    info_ctr2 = [
        ("Direction", direction_str2),
        ("Count", totalCount2),
    ]
    for (i, (k, v)) in enumerate(info_ctr2):
        text2 = "{}: {}".format(k, v)
        cv2.putText(frame_ctr2, text2, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if direction_str1 == "In" or direction_str1 == "Out":
        current_time = calendar.timegm(time.gmtime())

        # Convert captured image to JPG
        ret, buffer = cv2.imencode('.jpg', frame_ctr1)
        # Convert to base64 encoding and show start of data
        jpg_as_text = base64.b64encode(buffer)

        FS.q_push(jpg_as_text, current_time, totalCount1, direction_str1, 1)

    if direction_str2 == "In" or direction_str2 == "Out":
        current_time = calendar.timegm(time.gmtime())

        # Convert captured image to JPG
        ret, buffer = cv2.imencode('.jpg', frame_ctr2)
        # Convert to base64 encoding and show start of data
        jpg_as_text = base64.b64encode(buffer)

        FS.q_push(jpg_as_text, current_time, totalCount2, direction_str2, 2)

    cv2.imshow("Front counter", frame_ctr1)
    cv2.imshow("Rear counter", frame_ctr2)
    cv2.waitKey(1)

    te_dtc = time.time()
    dtc_rate = te_dtc - t_dtc
    print('Detection: {}'.format(dtc_rate))
