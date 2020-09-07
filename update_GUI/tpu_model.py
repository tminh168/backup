import collections
import common
import numpy as np
import cv2
import re
from PIL import Image



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


#labels = load_labels('people_label.txt')
stream_1 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
stream_2 = 'rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
model_1 = 'detection_1_edgetpu.tflite'
model_2 = 'detection_2_edgetpu.tflite'
model = 'detection_toco_edgetpu.tflite'
base_dir = '/home/mendel/coral/DFM_counter'


class model_tpu:
    def __init__(self, model, labels):
        self.H = 480
        self.W = 640
        self.top_k = 10
        self.threshold = 0.8
        self.labels = labels

        self.interpreter = common.make_interpreter(model)
        self.interpreter.allocate_tensors()

    def detect_distance(self, frame):

        width = self.W
        height = self.H
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        pil_im = pil_im.resize((width, height), Image.NEAREST)

        common.set_input(self.interpreter, pil_im)
        self.interpreter.invoke()
        objs = get_output(
            self.interpreter, score_threshold=self.threshold, top_k=self.top_k)

        cv2_im = np.array(pil_im)
        frame = cv2.cvtColor(cv2_im, cv2.COLOR_RGB2BGR)

        pedestrian_boxes = []

        for obj in objs:
            if self.labels.get(obj.id, obj.id) == "person":
                x0, y0, x1, y1 = list(obj.bbox)
                # x0, y0, x1, y1 = int(
                #     x0*width), int(y0*height), int(x1*width), int(y1*height)

                pedestrian_boxes.append((y0, x0, y1, x1))  # TF format

        return frame, pedestrian_boxes

    def detect_count(self, frame):

        width = self.W
        height = self.H
        
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        pil_im = pil_im.resize((width, height), Image.NEAREST)

        common.set_input(self.interpreter, pil_im)
        self.interpreter.invoke()
        objs = get_output(
            self.interpreter, score_threshold=self.threshold, top_k=self.top_k)

        cv2_im = np.array(pil_im)
        cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_RGB2BGR)

        return cv2_im, objs

