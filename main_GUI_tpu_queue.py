from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog, QGroupBox, QComboBox,
                             QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QInputDialog, QPushButton, QVBoxLayout)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
import sys
import cv2
import imutils
import numpy as np
import time
from main_tpu_procs_queue import AImodel_tpu
from multiprocessing import Process, Queue
import threading


class Dialog(QDialog):

    def __init__(self):
        super(Dialog, self).__init__()
        self.createFormGroupBox()

        self.buttonBox = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.runModel)
        self.buttonBox.rejected.connect(self.abortModel)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(self.buttonBox)
        self.setLayout(mainLayout)
        #self.resize(375, 150)
        self.setGeometry(200, 200, 480, 320)

        self.p = None
        self.t = None
        self.show_flag = threading.Event()
        self.q_ctr = Queue()
        self.q_dist = Queue()
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

        list_Cam = ["192.168.200.81", "192.168.200.82", "Both camera"]
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

    def getCam(self, i):

        self.input_Cam = self.optCam.currentText()
        print(self.input_Cam)

    def getLim(self, i):

        self.input_Limit = self.optLim.currentText()
        print(self.input_Limit)

    def runModel(self):
        print('running..')
        self.buttonBox.button(QDialogButtonBox.Ok).setDisabled(True)

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

        self.p = Process(target=AImodel_tpu, args=(
            self.input_Model, "15fps.mp4", self.input_Limit, two_points, self.q_ctr, self.q_dist,))

        self.p.start()
        #self.p.join()

        self.t = threading.Thread(target=self.showCam, args=(self.q_ctr, self.q_dist,))
        self.t.start()

    def showCam(self, q_ctr, q_dist):
        cv2.namedWindow('People counting', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('People distancing', cv2.WINDOW_AUTOSIZE)
        #cv2.resizeWindow('People counting', (new_width, new_height))
        while not self.show_flag.is_set():
            frame_ctr = q_ctr.get()
            frame_dist = q_dist.get()
            if frame_ctr is None or frame_dist is None:
                continue

            cv2.imshow("People distancing", frame_dist)
            cv2.imshow("People counting", frame_ctr)
            cv2.waitKey(10)
            
        return

    def abortModel(self):
        print('stopping..')

        if self.p:
            self.p.terminate()
            self.p.join()

        if self.t:
            self.show_flag.set()
            self.t.join()

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
