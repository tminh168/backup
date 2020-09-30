import cv2
import os
import time
import argparse
import numpy as np
from cameravideostream import CameraVideoStream
from tpu_model import *

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

cam_ip = 'rtsp://192.168.200.80:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
model = 'ssd_face_edgetpu.tflite'

# Define a DNN model
labels = None
DNN = model_tpu(model, labels)
# Get video handle
fvs = CameraVideoStream(cam_ip).start()

# Process each frame, until end of video
while True:
    
    frame = fvs.read()

    if frame is None:
        continue

    height = 480
    width = 640

    t_dtc = time.time()
    # Detect person and bounding boxes using DNN
    frame, pedestrian_boxes = DNN.detect_distance(frame)

    for i in range(len(pedestrian_boxes)):
        (ymin, xmin, ymax, xmax) = pedestrian_boxes[i]
        (x0, y0, x1, y1) = (int(round(xmin * width)), int(round(ymin * height)),
                            int(round(xmax * width)), int(round(ymax * height)))

        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
    
    
    te_dtc = time.time()
    dtc_rate = te_dtc - t_dtc
    print('Detection: {}'.format(dtc_rate))

    cv2.imshow("Face detection", frame)
    cv2.waitKey(1)

