import cv2
import os
import time
import argparse
import numpy as np
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from tpu_model import *
from track_distance import *

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mouse_pts = []


def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)


# Command-line input setup
# parser = argparse.ArgumentParser(description="SocialDistancing")
# parser.add_argument(
#     "--videopath", type=str, default="15fps.mp4", help="Path to the video file"
# )
# args = parser.parse_args()

# input_video = args.videopath

stream_1 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
stream_2 = 'rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
model_1 = 'detection_1_edgetpu.tflite'
model_2 = 'detection_2_edgetpu.tflite'
base_dir = '/home/mendel/coral/DFM_counter'

# Define a DNN model
DNN = model_tpu(model_1)
# Get video handle
fvs = WebcamVideoStream(stream_1).start()

SOLID_BACK_COLOR = (41, 41, 41)

# Initialize necessary variables
frame_num = 0
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0

ct = CentroidTracker(maxDisappeared=2, maxDistance=55)
trackableObjects = dict()
totalCount = 0
countedID = 0
ROI = 350
log_img = False

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
first_frame_display = True
fps_count = FPS().start()

# Process each frame, until end of video
while True:
    direction_str = "..."
    frame_num += 1
    frame = fvs.read()

    if frame is None:
        continue

    # frame = imutils.resize(frame, width=600)
    # frame_h = frame.shape[0]
    # frame_w = frame.shape[1]
    H = 400
    W = 600

    if frame_num == 1:
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        while True:
            image = frame
            image = imutils.resize(image, width=W, height=H)
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 3:
                cv2.destroyWindow("image")
                break
            first_frame_display = False
        two_points = mouse_pts

        # Get threshold distance and bird image
        d_thresh = np.sqrt(
            (two_points[0][0] - two_points[1][0]) ** 2
            + (two_points[0][1] - two_points[1][1]) ** 2
        )
        bird_image = np.zeros(
            (int(frame_h), int(frame_w), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR


    print("Processing frame: ", frame_num)

    t_dtc = time.time()
    # Detect person and bounding boxes using DNN
    frame, pedestrian_boxes = DNN.detect_distance(frame)

    frame_ctr = cv2.line(frame, (ROI, 0), (ROI, H), (0, 255, 255), 2)
    frame_ctr, countedID, totalCount, direction_str, ct, trackableObjects = append_objs_counter(
            frame_ctr, countedID, pedestrian_boxes, ROI, ct, trackableObjects, totalCount)

    frame_dist = append_objs_distance(frame, pedestrian_boxes, d_thresh)
    te_dtc = time.time()
    print('Detection: {}'.format(te_dtc - t_dtc))

    info = [
        ("Direction", direction_str),
        ("Count", totalCount),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame_ctr, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame counter", frame_ctr)
    cv2.imshow("Frame distance", frame_dist)
    cv2.waitKey(1)
    # output_movie.write(pedestrian_detect)
    # bird_movie.write(bird_image)
    #te_write=time.time()
    #print('Write: {}'.format(te_write - te_text))
    fps_count.update()

fps_count.stop()
print("[INFO] elapsed time: {:.2f}".format(fps_count.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps_count.fps()))
