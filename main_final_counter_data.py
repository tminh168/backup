import cv2
import os
import time
import calendar
import argparse
import base64
import json
import random
import string
import ctypes
from frame_submit_fcn import frameSubmit
from datetime import datetime
from multiprocessing import Process, Queue, Value
from cameravideostream import CameraVideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from tpu_model import *
from track_distance import *


def counter_run(q, check_temp):

    cam_78 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    cam_80 = 'rtsp://192.168.200.80:555/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    model_1 = 'detection_1_edgetpu.tflite'
    model_2 = 'detection_2_edgetpu.tflite'

    def get_random(length):
        letters_and_digits = string.ascii_letters + string.digits
        result_str = ''.join((random.choice(letters_and_digits)
                              for i in range(length)))
        return result_str

    # Define a DNN model
    labels = load_labels('people_label.txt')
    DNN_1 = model_tpu(model_1, labels)
    DNN_2 = model_tpu(model_2, labels)
    # Get video handle
    fvs1 = CameraVideoStream(cam_78).start()
    fvs2 = CameraVideoStream(cam_80).start()

    ct1 = CentroidTracker(maxDisappeared=3, maxDistance=55)
    ct2 = CentroidTracker(maxDisappeared=3, maxDistance=55)
    trackableObjects1 = dict()
    trackableObjects2 = dict()
    totalCount1 = 0
    totalCount2 = 0
    countedID1 = 0
    countedID2 = 0
    ROI = 320

    # Process each frame, until end of video
    while True:

        if check_temp.value > 70:
            check_temp.value = 0
            print('Overheat detected..!')
            time.sleep(5.0)

        direction_str1 = "..."
        direction_str2 = "..."
        n_1 = 0
        n_2 = 0
        frame1 = fvs1.read()
        frame2 = fvs2.read()

        if frame1 is None or frame2 is None:
            continue

        H = 480
        W = 640

        #t_dtc = time.time()
        # Detect person and bounding boxes using DNN
        frame1, pedestrian_boxes1 = DNN_1.detect_distance(frame1)
        frame2, pedestrian_boxes2 = DNN_2.detect_distance(frame2)

        if len(pedestrian_boxes1) > 0:
            if n_1 % 10 == 0:
                current_time = calendar.timegm(time.gmtime())

                # Convert captured image to JPG
                ret, buffer = cv2.imencode('.jpg', frame1)
                # Convert to base64 encoding and show start of data
                jpg_as_text = base64.b64encode(buffer)

                data = {'image': jpg_as_text.decode('utf-8'),
                        'timestamp': current_time,
                        'bus': 1,
                        'shift': 1,
                        'cam_no': 1,
                        'direction': "Data",
                        'count': 0}

                if q.qsize() > 200:
                    rand_str = get_random(15)
                    with open('temp/' + rand_str + '.json', 'w') as f:
                        json.dump(data, f)

                else:
                    q.put(data)
            n_1 += 1

        if len(pedestrian_boxes2) > 0:
            if n_2 % 10 == 0:
                current_time = calendar.timegm(time.gmtime())

                # Convert captured image to JPG
                ret, buffer = cv2.imencode('.jpg', frame2)
                # Convert to base64 encoding and show start of data
                jpg_as_text = base64.b64encode(buffer)

                data = {'image': jpg_as_text.decode('utf-8'),
                        'timestamp': current_time,
                        'bus': 1,
                        'shift': 1,
                        'cam_no': 2,
                        'direction': "Data",
                        'count': 0}

                if q.qsize() > 200:
                    rand_str = get_random(15)
                    with open('temp/' + rand_str + '.json', 'w') as f:
                        json.dump(data, f)

                else:
                    q.put(data)
            n_2 += 1

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

            data = {'image': jpg_as_text.decode('utf-8'),
                    'timestamp': current_time,
                    'bus': 1,
                    'shift': 1,
                    'cam_no': 1,
                    'direction': direction_str1,
                    'count': totalCount1}

            if q.qsize() > 200:
                rand_str = get_random(15)
                with open('temp/' + rand_str + '.json', 'w') as f:
                    json.dump(data, f)

            else:
                q.put(data)

        if direction_str2 == "In" or direction_str2 == "Out":
            current_time = calendar.timegm(time.gmtime())

            # Convert captured image to JPG
            ret, buffer = cv2.imencode('.jpg', frame_ctr2)
            # Convert to base64 encoding and show start of data
            jpg_as_text = base64.b64encode(buffer)

            data = {'image': jpg_as_text.decode('utf-8'),
                    'timestamp': current_time,
                    'bus': 1,
                    'shift': 1,
                    'cam_no': 2,
                    'direction': direction_str2,
                    'count': totalCount2}

            if q.qsize() > 200:
                rand_str = get_random(15)
                with open('temp/' + rand_str + '.json', 'w') as f:
                    json.dump(data, f)

            else:
                q.put(data)

        if q.qsize() < 150:
            if os.listdir('temp/'):
                for root, dirs, files in os.walk("temp/", topdown=False):
                    for name in files:
                        with open(str(os.path.join(root, name)), 'r') as f:
                            data = json.load(f)
                            q.put(data)
                            os.remove(os.path.join(root, name))
                            break

        print(q.qsize())
        #cv2.imshow("Front counter", frame_ctr1)
        #cv2.imshow("Rear counter", frame_ctr2)
        # cv2.waitKey(1)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-stt", "--status", type=int, default=1,
                    help="pass update status code")
    args = vars(ap.parse_args())

    img_queue = Queue(maxsize=256)
    check_temp = Value(ctypes.c_int)
    check_temp.value = 0

    p_submit = Process(target=frameSubmit, args=(img_queue, check_temp, args["status"]))
    p_counter = Process(target=counter_run, args=(img_queue, check_temp,))
    p_submit.start()
    p_counter.start()
    p_counter.join()
    p_submit.join()
