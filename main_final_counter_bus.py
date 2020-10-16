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
fvs1 = CameraVideoStream(cam_78).start()
fvs2 = CameraVideoStream(cam_80).start()

# initialize API parameters
url = 'http://ai-camera.dfm-europe.com/api/v1/admin/public/uplink'
q1 = Queue(maxsize=50)
q2 = Queue(maxsize=50)

with open('counter_1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Count", "Time", "Direction", "Status"])

with open('counter_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Count", "Time", "Direction", "Status"])

ct1 = CentroidTracker(maxDisappeared=3, maxDistance=55)
ct2 = CentroidTracker(maxDisappeared=3, maxDistance=55)
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
    dequeue = True
    frame1 = fvs1.read()
    frame2 = fvs2.read()

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

    te_dtc = time.time()
    dtc_rate = te_dtc - t_dtc
    print('Detection: {}'.format(dtc_rate))

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
        status = "Success"
        dequeue = False

        with open('counter_1.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [totalCount1, current_time, direction_str1, status])

        try:
            # Convert captured image to JPG
            ret, buffer = cv2.imencode('.jpg', frame_ctr1)
            # Convert to base64 encoding and show start of data
            jpg_as_text = base64.b64encode(buffer)
            print(jpg_as_text)

            data = {'image': jpg_as_text,
                    'timestamp': current_time,
                    'bus': 1,
                    'shift': 1,
                    'cam_no': 1,
                    'direction': direction_str1,
                    'count': totalCount1}
            print(sys.getsizeof(data))
            r = requests.post(url, data=data, verify=False)

            # check API response
            if r.status_code == 200:
                print("Success")
            else:
                q1.put(data)

        except ConnectionError as e:
            print(e)
            r = "No response. "
            print(r + "Check internet connection. Detection frame on standby!")
            q1.put(data)

    if direction_str2 == "In" or direction_str2 == "Out":
        current_time = calendar.timegm(time.gmtime())
        status = "Success"
        dequeue = False

        with open('counter_2.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                [totalCount2, current_time, direction_str2, status])

        try:
            # Convert captured image to JPG
            ret, buffer = cv2.imencode('.jpg', frame_ctr2)
            # Convert to base64 encoding and show start of data
            jpg_as_text = base64.b64encode(buffer)
            print(jpg_as_text)

            data = {'image': jpg_as_text,
                    'timestamp': current_time,
                    'bus': 1,
                    'shift': 1,
                    'cam_no': 2,
                    'direction': direction_str2,
                    'count': totalCount2}
            print(sys.getsizeof(data))
            r = requests.post(url, data=data, verify=False)

            # check API response
            if r.status_code == 200:
                print("Success")
            else:
                q2.put(data)

        except ConnectionError as e:
            print(e)
            r = "No response. "
            print(r + "Check internet connection. Detection frame on standby!")
            q2.put(data)

    if dequeue:
        if not q1.empty():
            data = q1.get()
            try:
                r = requests.post(url, data=data, verify=False)

                # check API response
                if r.status_code == 200:
                    print("Success")
                else:
                    q1.put(data)

            except ConnectionError as e:
                print(e)
                r = "No response. "
                print(
                    r + "Check internet connection. Detection frame on standby!")
                q1.put(data)

        if not q2.empty():
            data = q2.get()
            try:
                r = requests.post(url, data=data, verify=False)

                # check API response
                if r.status_code == 200:
                    print("Success")
                else:
                    q2.put(data)

            except ConnectionError as e:
                print(e)
                r = "No response. "
                print(
                    r + "Check internet connection. Detection frame on standby!")
                q2.put(data)

    cv2.imshow("Front counter", frame_ctr1)
    cv2.imshow("Rear counter", frame_ctr2)
    cv2.waitKey(1)
