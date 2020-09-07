import collections
import common
import cv2
import numpy as np
import time
import re
import imutils
from PIL import Image
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import WebcamVideoStream
from imutils.video import FPS
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

labels = load_labels('people_label.txt')
stream_1 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
stream_2 = 'rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
model_1 = 'detection_1_edgetpu.tflite'
model_2 = 'detection_2_edgetpu.tflite'
model = 'detection_toco_edgetpu.tflite'
base_dir = '/home/mendel/coral/DFM_counter'

def tf_inference():
    #print('Loading model..')
    interpreter_1 = tflite.Interpreter(model_1, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter_2 = tflite.Interpreter(model_2, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter_1.allocate_tensors()
    interpreter_2.allocate_tensors()
    H = 400
    W = 600
    log_img = False
    top_k = 3
    threshold = 0.8
    ct1 = CentroidTracker(maxDisappeared=2, maxDistance=45)
    ct2 = CentroidTracker(maxDisappeared=2, maxDistance=45)
    trackableObjects1 = {}
    trackableObjects2 = {}
    totalCount1 = 0
    totalCount2 = 0
    ROI = 350

    fvs1 = WebcamVideoStream(src=stream_1).start()
    fvs2 = WebcamVideoStream(src=stream_2).start()
    time.sleep(1.0)
    #fps = FPS().start()

    while True:
        #start_t = time.time()
        direction_str1 = "..."
        direction_str2 = "..."
        frame1 = fvs1.read()
        frame2 = fvs2.read()

        #frame1 = imutils.resize(frame1, width=600)
        #frame2 = imutils.resize(frame2, width=600)
        cv2_im1 = frame1
        cv2_im2 = frame2

        cv2_im_rgb1 = cv2.cvtColor(cv2_im1, cv2.COLOR_BGR2RGB)
        cv2_im_rgb2 = cv2.cvtColor(cv2_im2, cv2.COLOR_BGR2RGB)
        pil_im1 = Image.fromarray(cv2_im_rgb1)
        pil_im2 = Image.fromarray(cv2_im_rgb2)
        pil_im1 = pil_im1.resize((W, H), Image.NEAREST)
        pil_im2 = pil_im2.resize((W, H), Image.NEAREST)

        common.set_input(interpreter_1, pil_im1)
        common.set_input(interpreter_2, pil_im2)
        interpreter_1.invoke()
        interpreter_2.invoke()
        objs1 = get_output(interpreter_1, score_threshold=threshold, top_k=top_k)
        objs2 = get_output(interpreter_2, score_threshold=threshold, top_k=top_k)

        cv2_im1 = np.array(pil_im1)
        cv2_im2 = np.array(pil_im2)
        cv2_im1 = cv2.cvtColor(cv2_im1, cv2.COLOR_RGB2BGR)
        cv2_im2 = cv2.cvtColor(cv2_im2, cv2.COLOR_RGB2BGR)
        cv2_im1 = cv2.line(cv2_im1, (ROI, 0), (ROI, H), (0, 255, 255), 2)
        cv2_im2 = cv2.line(cv2_im2, (ROI, 0), (ROI, H), (0, 255, 255), 2)
        cv2_im1, totalCount1, direction_str1, trackableObjects1 = append_objs_to_img(cv2_im1, objs1, labels, ROI, ct1, trackableObjects1, totalCount1)
        cv2_im2, totalCount2, direction_str2, trackableObjects2 = append_objs_to_img(cv2_im2, objs2, labels, ROI, ct2, trackableObjects2, totalCount2)

        info1 = [
           ("Direction", direction_str1),
           ("Count", totalCount1),
        ]
        for (i, (k, v)) in enumerate(info1):
           text1 = "{}: {}".format(k, v)
           cv2.putText(cv2_im1, text1, (10, H - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        info2 = [
           ("Direction", direction_str2),
           ("Count", totalCount2),
        ]
        for (i, (k, v)) in enumerate(info2):
           text2 = "{}: {}".format(k, v)
           cv2.putText(cv2_im2, text2, (10, H - ((i * 20) + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if log_img:
           if direction_str1 == "In" or direction_str1 == "Out":
              cv2.imwrite(base_dir + "/detected1" + "/count-%d.jpg" % totalCount1, cv2_im1)
           if direction_str2 == "In" or direction_str2 == "Out":
              cv2.imwrite(base_dir + "/detected2" + "/count-%d.jpg" % totalCount2, cv2_im2)

        #cv2.imshow('Camera_1', cv2_im1)
        #cv2.imshow('Camera_2', cv2_im2)
        #print('counting..')
        #fps.update()
        #end_t = time.time()
        #print('frame: {:.3f}'.format(end_t - start_t))
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #   break
    #fps.stop()
    #print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    fvs1.stop()
    fvs2.stop()
    cv2.destroyAllWindows()

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

if __name__ == '__main__':
    tf_inference()
