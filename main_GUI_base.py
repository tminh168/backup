from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog, QGroupBox, QComboBox, QSlider,
                             QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QInputDialog, QPushButton, QVBoxLayout)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
import sys
import cv2
import imutils
import numpy as np
import time
from cameravideostream import CameraVideoStream
from main_tpu_procs import AImodel_tpu
from multiprocessing import Process


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

        self.textLine = QLabel('Choose line position(%):')
        self.sldLine = QSlider(Qt.Horizontal)
        self.sldLine.setMinimum(30)
        self.sldLine.setMaximum(70)
        self.sldLine.setValue(50)
        self.input_Line = self.sldLine.value()
        self.sldLine.setTickPosition(QSlider.TicksBelow)
        self.sldLine.setTickInterval(5)
        self.sldLine.valueChanged.connect(self.getLine)
        layout.addRow(self.textLine, self.sldLine)

        list_Left = ["Out", "In"]
        self.textLeft = QLabel('Left direction:')
        self.optLeft = QComboBox()
        self.optLeft.addItems(list_Left)
        self.optLeft.setCurrentIndex(list_Left.index("Out"))
        self.input_Left = str(self.optLeft.currentText())
        self.optLeft.currentIndexChanged.connect(self.getLeft)
        layout.addRow(self.textLeft, self.optLeft)

        list_Right = ["In", "Out"]
        self.textRight = QLabel('Right direction:')
        self.optRight = QComboBox()
        self.optRight.addItems(list_Right)
        self.optRight.setCurrentIndex(list_Right.index("In"))
        self.input_Right = str(self.optRight.currentText())
        self.optRight.currentIndexChanged.connect(self.getRight)
        layout.addRow(self.textRight, self.optRight)

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

    def getLine(self):

        self.input_Line = self.sldLine.value()
        print(self.input_Line)

    def getLeft(self):

        self.input_Left = str(self.optLeft.currentText())
        print(self.input_Left)

    def getRight(self):

        self.input_Right = str(self.optRight.currentText())
        print(self.input_Right)

    def runModel(self):
        print('running..')
        self.buttonBox.button(QDialogButtonBox.Ok).setDisabled(True)
        cv2.namedWindow("Click to set threshold distance")

        if self.input_Mode == "Social distancing":
            mouse_pts = []

            def get_mouse_points(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:

                    mouse_pts.append((x, y))
                    print("Point detected")
                    print(mouse_pts)

            cv2.setMouseCallback(
                "Click to set threshold distance", get_mouse_points)

            cam_81 = 'rtsp://192.168.200.80:555/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
            cam_82 = 'rtsp://192.168.200.82:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'

            if self.input_Cam == "192.168.200.81":
                input_cam = cam_81
            elif self.input_Cam == "192.168.200.82":
                input_cam = cam_82
            fvs = cv2.VideoCapture(input_cam)

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
            cv2.destroyWindow("Click to set threshold distance")

        self.p = Process(target=AImodel_tpu, args=(
            self.input_Model, self.input_Cam, self.input_Limit, self.input_Line, self.input_Left, self.input_Right, two_points,))

        self.p.start()
        self.p.join()

    def abortModel(self):
        print('stopping..')

        if self.p:
            self.p.terminate()
            self.p.join()

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
