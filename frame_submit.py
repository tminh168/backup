# import the necessary packages
from threading import Thread
import sys
import cv2
import time
import csv
import calendar
import requests
import base64
from datetime import datetime
from queue import Queue
from requests.exceptions import ConnectionError


class FrameSubmit:
    def __init__(self, file, queue_size=128):
        # initialize parameters
        self.url = 'http://ai-camera.dfm-europe.com/api/v1/admin/public/uplink'
        self.q = Queue(maxsize=queue_size)
        self.file = file
        self._run = True

        with open(self.file, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Count", "Time", "Direction", "Status"])

        self.thread = Thread(target=self.dequeue, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def dequeue(self):
        # keep looping infinitely
        while self._run:
            if not self.q.empty():
                data = self.q.get()
                try:
                    r = requests.post(self.url, data=data, verify=False)

                    # check API response
                    if r.status_code == 200:
                        print("Success")
                    else:
                        self.q.put(data)

                except ConnectionError as e:
                    print(e)
                    r = "No response. "
                    print(
                        r + "Check internet connection. Detection frame on standby!")
                    self.q.put(data)

        return

    def stop(self):
        self._run = False
        self.thread.join()

    def q_push(self, frame, totalCount, direction_str, cam_no):
        current_time = calendar.timegm(time.gmtime())
        status = "Success"

        with open(self.file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [totalCount, current_time, direction_str, status])

        try:
            # Convert captured image to JPG
            ret, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64 encoding and show start of data
            jpg_as_text = base64.b64encode(buffer)
            # print(jpg_as_text[:80])

            data = {'image': jpg_as_text,
                    'timestamp': current_time,
                    'bus': 123,
                    'shift': 456,
                    'cam_no': cam_no,
                    'direction': direction_str,
                    'count': totalCount}

            self.q.put(data)
        except:
            print("Error. Frame missed!")
