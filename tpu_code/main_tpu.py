import cv2
import os
import time
import argparse
import imutils
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from tpu_model import *
from aux_functions import *

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

def append_objs_to_img(cv2_im, objs, labels, ROI, ct, trackableObjects, totalCount):
    height, width, channels = cv2_im.shape
    rects = []

    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        if x0 < 100 or x1 > 550:
              continue
        rects.append((x0, y0, x1, y1))
        percent = int(100 * obj.score)
        label = '{}-{}%'.format(labels.get(obj.id, obj.id), percent)

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    objects = ct.update(rects)
    direction_str = "..."
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
              to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
             # check to see if the object has been counted or not
             if not to.counted:
                    # if the previous centroids from one side
                    # count as soon as the updated centroid reach other side
                    for c in to.centroids:
                          if c[0] < ROI and centroid[0] < ROI:
                             direction_str = "..."
                          elif c[0] < ROI and centroid[0] > ROI:
                             totalCount += 1
                             to.counted = True
                             direction_str = "In"
                             break
                          elif c[0] > ROI and centroid[0] > ROI:
                             direction_str = "..."
                          elif c[0] > ROI and centroid[0] < ROI:
                             totalCount += 1
                             to.counted = True
                             direction_str = "Out"
                             break

             # update new centroid to trackable object
             to.centroids.append(centroid)

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(cv2_im, text, (centroid[0] - 10, centroid[1] - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(cv2_im, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return cv2_im, totalCount, direction_str, trackableObjects

# Command-line input setup
#labels = load_labels('people_label.txt')
stream_1 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
stream_2 = 'rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
model_1 = 'detection_1_edgetpu.tflite'
model_2 = 'detection_2_edgetpu.tflite'
model = 'detection_toco_edgetpu.tflite'


# Define a DNN model
DNN_distance = model_tpu(model_1)
DNN_count = model_tpu(model_2)
# Get video handle
fvs1 = WebcamVideoStream(src=stream_1).start()
fvs2 = WebcamVideoStream(src=stream_1).start()
height = 400
width = 600
ct = CentroidTracker(maxDisappeared=2, maxDistance=45)
trackableObjects = {}
totalCount = 0
ROI = 350
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#fps = int(cap.get(cv2.CAP_PROP_FPS))

scale_w = 1.2 / 2
scale_h = 4 / 2

SOLID_BACK_COLOR = (41, 41, 41)
# Setup video writer
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# output_movie = cv2.VideoWriter(
#     "Pedestrian_detect.avi", fourcc, fps, (width, height))
# bird_movie = cv2.VideoWriter(
#     "Pedestrian_bird.avi", fourcc, fps, (int(
#         width * scale_w), int(height * scale_h))
# )

# Initialize necessary variables
frame_num = 0
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1
sc_index = 1

cv2.namedWindow("define Region of Interest")
cv2.setMouseCallback("define Region of Interest", get_mouse_points)
num_mouse_points = 0
first_frame_display = True
#fps_count = FPS().start()

# Process each frame, until end of video
while True:
    t_dtc = time.time()
    direction_str = "..."
    frame1 = fvs1.read()
    frame2 = fvs2.read()

    if frame1 is None or frame2 is None:
        continue

    frame_num += 1
    H = 400
    W = 600

    if frame_num == 1:
        frame1 = imutils.resize(frame1, width=W, height=H)
        print('Please specify ROI for distance measurement.')
        # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        while True:
            image = frame1
            cv2.imshow("define Region of Interest", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 7:
                cv2.destroyWindow("define Region of Interest")
                break
            first_frame_display = False
        four_points = mouse_pts

        # Get perspective
        M, Minv = get_camera_perspective(frame1, four_points[0:4])
        pts = src = np.float32(np.array([four_points[4:]]))
        warped_pt = cv2.perspectiveTransform(pts, M)[0]
        d_thresh = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        )
        bird_image = np.zeros(
            (int(H * scale_h), int(W * scale_w), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR
        pedestrian_detect = frame1

    print("Processing frame: ", frame_num)

    # draw polygon of ROI
    pts = np.array(
        [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
    )

    # Detect person and bounding boxes using DNN
    pedestrian_boxes, num_pedestrians, frame1 = DNN_distance.detect_distance(
        frame1)
    cv2.polylines(frame1, [pts], True, (0, 255, 255), thickness=4)
    
    if len(pedestrian_boxes) > 0:
        pedestrian_detect = plot_pedestrian_boxes_on_image(
            frame1, pedestrian_boxes)
        warped_pts, bird_image = plot_points_on_bird_eye_view(
            frame1, pedestrian_boxes, M, scale_w, scale_h
        )
        six_feet_violations, ten_feet_violations, pairs = plot_lines_between_nodes(
            warped_pts, bird_image, d_thresh
        )
        # plot_violation_rectangles(pedestrian_boxes, )
        total_pedestrians_detected += num_pedestrians
        total_pairs += pairs

        # total_six_feet_violations += six_feet_violations / fps
        # abs_six_feet_violations += six_feet_violations
        # pedestrian_per_sec, sh_index=calculate_stay_at_home_index(
        #     total_pedestrians_detected, frame_num, fps
        # )

    # last_h=75
    # text="# distance violations: " + str(int(total_six_feet_violations))
    # pedestrian_detect, last_h=put_text(
    #     pedestrian_detect, text, text_offset_y=last_h)

    # text="Stay-at-home Index: " + str(np.round(100 * sh_index, 1)) + "%"
    # pedestrian_detect, last_h=put_text(
    #     pedestrian_detect, text, text_offset_y=last_h)

    # if total_pairs != 0:
    #     sc_index=1 - abs_six_feet_violations / total_pairs

    # text="Social-distancing Index: " + str(np.round(100 * sc_index, 1)) + "%"
    # pedestrian_detect, last_h=put_text(
    #     pedestrian_detect, text, text_offset_y=last_h)
    # te_text=time.time()
    # print('Text: {}'.format(te_text - te_view))

    cv2_im, objs = DNN_count.detect_count(frame2)
    cv2_im = cv2.line(cv2_im, (ROI, 0), (ROI, H), (0, 255, 255), 2)
    cv2_im, totalCount, direction_str, trackableObjects = append_objs_to_img(cv2_im, objs, labels, ROI, ct, trackableObjects, totalCount)

    info = [
        ("Direction", direction_str),
        ("Count", totalCount),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(cv2_im, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Camera_distance", pedestrian_detect)
    cv2.imshow("Bird-eye_view", bird_image)
    cv2.imshow('Camera_count', cv2_im)
    cv2.waitKey(1)
    # output_movie.write(pedestrian_detect)
    # bird_movie.write(bird_image)
    # fps_count.update()
    te_dtc = time.time()
    print('Detection in: {}'.format(t_dtc - te_dtc))

fvs1.stop()
fvs2.stop()
cv2.destroyAllWindows()

# fps_count.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps_count.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps_count.fps()))

