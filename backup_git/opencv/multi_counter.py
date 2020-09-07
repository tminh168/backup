import argparse
import collections
import common
import cv2
import numpy as np
import os
import imutils
import time
from PIL import Image
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
#from pyimagesearch.camerastream import CameraStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import re
import tflite_runtime.interpreter as tflite

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

def counter(stream, out_cam):
    default_model_dir = '../all_models'
    default_model = 'detection_toco_edgetpu.tflite'
    default_labels = 'people_label.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=default_model)
    parser.add_argument('--labels', help='label file path',
                        default=default_labels)
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='classifier score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = common.make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    H = None
    W = None
    ct = CentroidTracker(maxDisappeared=2, maxDistance=80)
    trackableObjects = {}
    totalCount = 0
    ROI = 500

    fvs = WebcamVideoStream(stream).start()
    #cap = cv2.VideoCapture(args.camera_idx)
    time.sleep(1.0)
    fps = FPS().start()

    while True: #fvs.more():
        direction_str = "..."
        frame = fvs.read()
        if frame is None:
              break

        #start_t = time.time()
        frame = imutils.resize(frame, width=800)
        if W is None or H is None:
              (H, W) = frame.shape[:2]
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)

        common.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = get_output(interpreter, score_threshold=args.threshold, top_k=args.top_k)
        cv2_im = cv2.line(cv2_im, (ROI, 0), (ROI, H), (0, 255, 255), 2)
        #cv2_im = cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
         #                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2_im, totalCount, direction_str, trackableObjects = append_objs_to_img(cv2_im, objs, labels, ROI, ct, trackableObjects, totalCount)

        info = [
           ("Direction", direction_str),
           ("Count", totalCount),
        ]
        for (i, (k, v)) in enumerate(info):
           text = "{}: {}".format(k, v)
           cv2.putText(cv2_im, text, (10, H - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(out_cam, cv2_im)

        #if direction_str == '...':
        #   print('scanning...')
        #else: print('detected..!!!')

        #end_t = time.time()
        #interval = end_t - start_t
        #print('frame: {}'.format(interval))
        #print('..')

        fps.update()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    fvs.stop()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, objs, labels, ROI, ct, trackableObjects, totalCount):
    height, width, channels = cv2_im.shape
    rects = []

    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        if x0 < 200 or x1 > 750:
              continue
        rects.append((x0, y0, x1, y1))
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

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
