from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog, QGroupBox, QComboBox,
                             QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QInputDialog, QPushButton, QVBoxLayout)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
import sys
import cv2
import imutils
import numpy as np
import time
from pyimagesearch.centroidtracker import CentroidTracker
from tpu_model import *
from track_distance import *
from main_tpu_procs import AImodel_tpu
from multiprocessing import Process, Event


class Dialog(QDialog):

    def __init__(self):
        super(Dialog, self).__init__()
        self.createFormGroupBox()
        self.exit = Event()

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.runModel)
        self.buttonBox.rejected.connect(self.abortModel)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)
        #self.resize(375, 150)
        self.setGeometry(0, 0, 480, 320)

        self.p = None
        self.setWindowTitle("DFM AI demo option")

    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Choose model option:")

        layout = QFormLayout()
        list_Model = ["SSD Custom People detection"]
        self.textModel = QLabel('Choose detection model:')
        self.optModel = QComboBox()
        self.optModel.addItems(list_Model)
        self.optModel.setCurrentIndex(
            list_Model.index('SSD Custom People detection'))
        self.input_Model = str(self.optModel.currentText())
        self.optModel.currentIndexChanged.connect(self.getModel)
        layout.addRow(self.textModel, self.optModel)

        list_Mode = ["Passenger counter", "Social distancing"]
        self.textMode = QLabel('Choose AI mode:')
        self.optMode = QComboBox()
        self.optMode.addItems(list_Mode)
        self.optMode.setCurrentIndex(list_Mode.index('Passenger counter'))
        self.input_Mode = str(self.optMode.currentText())
        self.optMode.currentIndexChanged.connect(self.getMode)
        layout.addRow(self.textMode, self.optMode)

        list_Cam = ["192.168.200.81", "192.168.200.82"]
        self.textCam = QLabel('Choose camera IP:')
        self.optCam = QComboBox()
        self.optCam.addItems(list_Cam)
        self.optCam.setCurrentIndex(list_Cam.index('192.168.200.81'))
        self.input_Cam = str(self.optCam.currentText())
        self.optCam.currentIndexChanged.connect(self.getCam)
        layout.addRow(self.textCam, self.optCam)

        list_Lim = ["1", "2", "3", "4", "5"]
        self.textLim = QLabel('Choose distance threshold(m):')
        self.optLim = QComboBox()
        self.optLim.addItems(list_Lim)
        self.optLim.setCurrentIndex(list_Lim.index('2'))
        self.input_Limit = str(self.optLim.currentText())
        self.optLim.currentIndexChanged.connect(self.getLim)
        layout.addRow(self.textLim, self.optLim)

        self.formGroupBox.setLayout(layout)

    def getModel(self, i):

        self.input_Model = str(self.optModel.currentText())
        print(self.input_Model)

    def getMode(self, i):

        self.input_Mode = str(self.optMode.currentText())
        print(self.input_Mode)

    def getCam(self, i):

        self.input_Cam = self.optCam.currentText()
        print(self.input_Cam)

    def getLim(self, i):

        self.input_Limit = self.optLim.currentText()
        print(self.input_Limit)

    def AImodel_tpu(self, input_model, input_cam, input_lim, input_pts):

        cam_81 = 'rtsp://192.168.200.81:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        cam_82 = 'rtsp://192.168.200.82:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        model_People = 'detection_toco_edgetpu.tflite'
        model_SSD = 'mobilenet_ssd_v2_edgetpu.tflite'

        if input_model == "SSD Mobilenet v2 detection":
            labels = load_labels('coco_labels.txt')
            DNN = model_tpu(model_SSD, labels)
        elif input_model == "SSD Custom People detection":
            labels = load_labels('people_label.txt')
            DNN = model_tpu(model_People, labels)

        # if input_cam == "192.168.200.81":
        #     fvs = CameraVideoStream(cam_81).start()
        # elif input_cam == "192.168.200.82":
        #     fvs = CameraVideoStream(cam_82).start()

        fvs = cv2.VideoCapture("15fps.mp4")
        limit = int(input_lim)

        # Initialize necessary variables
        ct = CentroidTracker(maxDisappeared=2, maxDistance=45)
        trackableObjects = dict()
        totalCount = 0
        countedID = 0
        ROI = 350

        if input_pts:
            total_six_feet_violations = 0
            d_thresh = np.sqrt(
                (input_pts[0][0] - input_pts[1][0]) ** 2
                + (input_pts[0][1] - input_pts[1][1]) ** 2
            )
            cv2.namedWindow("Social distancing")
            cv2.namedWindow("Passenger counter")
        else:
            cv2.namedWindow("Passenger counter", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Passenger counter",
                                cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Process each frame, until end of video
        while not self.exit.is_set():

            direction_str = "..."
            distance = "..."
            ret, frame = fvs.read()

            if frame is None:
                continue

            H = 480
            W = 640

            t_dtc = time.time()
            # Detect person and bounding boxes using DNN
            frame, pedestrian_boxes = DNN.detect_distance(frame)

            frame_ctr = cv2.line(frame, (ROI, 0),
                                (ROI, H), (0, 255, 255), 2)
            frame_ctr, countedID, totalCount, direction_str, ct, trackableObjects = append_objs_counter(
                frame_ctr, countedID, pedestrian_boxes, ROI, ct, trackableObjects, totalCount)

            info_ctr = [
                ("Direction", direction_str),
                ("Count", totalCount),
            ]
            for (i, (k, v)) in enumerate(info_ctr):
                text = "{}: {}".format(k, v)
                cv2.putText(frame_ctr, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Passenger counter", frame_ctr)
            cv2.waitKey(1)
            te_dtc = time.time()
            print('Counting: {}'.format(te_dtc - t_dtc))

            if input_pts:
                frame_dist, dist_violation = append_objs_distance(
                    frame, pedestrian_boxes, d_thresh)

                dist_count = []
                if len(dist_violation) > 0:
                    for i in range(len(dist_violation)):
                        dist_m = "{:.2f}".format(
                            dist_violation[i] / d_thresh * limit)
                        total_six_feet_violations += 1
                        dist_count.append(dist_m)
                if len(dist_count) > 0:
                    distance = ""
                    for i in range(len(dist_count)):
                        distance = distance + str(dist_count[i]) + " "

                info_dist = [
                    ("Distance(m)", distance),
                    ("Violation", total_six_feet_violations),
                ]
                for (i, (k, v)) in enumerate(info_dist):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame_dist, text, (10, H - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("Social distancing", frame_dist)
                cv2.waitKey(1)
                te_dtc = time.time()
                print('Distancing: {}'.format(te_dtc - t_dtc))

        cv2.destroyAllWindows()
        fvs.release()
        return

    def runModel(self):
        print('running..')
        self.buttonBox.button(QDialogButtonBox.Ok).setDisabled(True)

        if self.input_Mode == "Social distancing":
            mouse_pts = []

            def get_mouse_points(event, x, y, flags, param):
                # Used to mark 4 points on the frame zero of the video that will be warped
                # Used to mark 2 points on the frame zero of the video that are 6 feet away
                global mouseX, mouseY
                if event == cv2.EVENT_LBUTTONDOWN:
                    mouseX, mouseY = x, y
                    # if "mouse_pts" not in globals():
                    #    mouse_pts = []
                    mouse_pts.append((x, y))
                    print("Point detected")
                    print(mouse_pts)

            cv2.namedWindow("Click to set threshold distance")
            cv2.setMouseCallback(
                "Click to set threshold distance", get_mouse_points)

            cam_81 = 'rtsp://192.168.200.81:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
            cam_82 = 'rtsp://192.168.200.82:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'

            if self.input_Cam == "192.168.200.81":
                input_cam = cam_81
            elif self.input_Cam == "192.168.200.82":
                input_cam = cam_82
            fvs = cv2.VideoCapture("15fps.mp4")

            ret, frame = fvs.read()
            frame = imutils.resize(frame, width=640, height=480)

            while ret:
                image = frame
                cv2.imshow("Click to set threshold distance", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 3:
                    cv2.destroyWindow("Click to set threshold distance")
                    break

            fvs.release()
            two_points = mouse_pts

        else:
            two_points = None

        self.p = self.AImodel_tpu(self.input_Model, self.input_Cam, self.input_Limit, two_points)
        #self.p = Process(target=self.AImodel_tpu, args=(
        #    self.input_Model, self.input_Cam, self.input_Limit, two_points,))

        #self.p.start()
        #self.p.join()

    def abortModel(self):
        print('stopping..')

        if self.p:
            self.exit.set()
            #self.p.join()

            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

        self.restart()

    def restart(self):
        import sys
        print("argv was", sys.argv)
        print("sys.executable was", sys.executable)
        print("restart now")

        import os
        os.execv(sys.executable, ['python3'] + sys.argv)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = Dialog()
    dialog.show()

    sys.exit(app.exec_())
