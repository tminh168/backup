# import the necessary packages
from threading import Thread
import cv2
import time
import subprocess
import requests
import base64
from queue import Queue
from requests.exceptions import ConnectionError


class FrameSubmit:
    def __init__(self, queue_size=128, name="FrameSubmit"):
        # initialize parameters
        self.url = 'http://ai-camera.dfm-europe.com/api/v1/admin/public/uplink'
        self.q = Queue(maxsize=queue_size)
        self.t_init = time.time()
        self._run = True

        self.thread = Thread(target=self.dequeue, name=name, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def dequeue(self):
        # keep looping infinitely
        while self._run:
            t_check = time.time()
            if t_check >= self.t_init + 60:
                out = subprocess.Popen(['cat', '', '/sys/class/thermal/thermal_zone0/temp'],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT)
                
                stdout, stderr = out.communicate()
                print('temp: {}'.format(stdout.split('000')[0]))
                print(stderr)

                self.t_init = time.time()

            if not self.q.empty():
                t_send = time.time()
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

                print('Send: {}'.format(time.time() - t_send))

        return

    def stop(self):
        self._run = False
        self.thread.join()

    def q_push(self, jpg_as_text, current_time, totalCount, direction_str, cam_no):

        data = {'image': jpg_as_text,
                'timestamp': current_time,
                'bus': 123,
                'shift': 456,
                'cam_no': cam_no,
                'direction': direction_str,
                'count': totalCount}

        self.q.put(data)
