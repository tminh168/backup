from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog, QGroupBox,
QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QInputDialog, QPushButton, QVBoxLayout)
from PyQt5.QtCore import pyqtSignal, pyqtSlot
import sys
import cv2
import os
import imutils
import numpy as np
from main_tpu_demo import AImodel_tpu

class Dialog(QDialog):
    NumGridRows = 3
    NumButtons = 4

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
        self.resize(375, 150)

        self.ai = None
        self.setWindowTitle("DFM AI demo option")

    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("Choose model option:")
        layout = QFormLayout()
        self.btnModel = QPushButton("Choose model")
        self.btnModel.clicked.connect(self.getModel)
        self.lnModel = QLineEdit()
        layout.addRow(self.btnModel, self.lnModel)

        self.btnCam = QPushButton("Choose camera IP")
        self.btnCam.clicked.connect(self.getCam)
        self.lnCam = QLineEdit()
        layout.addRow(self.btnCam, self.lnCam)

        self.btnLim = QPushButton("Choose distance limit")
        self.btnLim.clicked.connect(self.getLim)
        self.lnLim = QLineEdit()
        layout.addRow(self.btnLim, self.lnLim)
        self.formGroupBox.setLayout(layout)

    def getModel(self):
        items = ("SSD Mobilenet v2 detection", "SSD Custom People detection")

        item, ok = QInputDialog.getItem(self, "select model input",
                                        "detection model", items, 0, False)

        if ok and item:
            self.lnModel.setText(item)
        self.input_Model = str(self.lnModel.text())
        print(self.input_Model)

    def getCam(self):

        items = ("192.168.200.78", "192.168.200.79", "192.168.200.81")

        item, ok = QInputDialog.getItem(self, "select mode input",
                                        "camera IP selection", items, 0, False)

        if ok and item:
            self.lnCam.setText(item)
        self.input_Cam = self.lnCam.text()
        print(self.input_Cam)

    def getLim(self):
        num, ok = QInputDialog.getInt(
            self, 'Distance limit dialog', 'Enter distance limit(m):')

        if ok:
            self.lnLim.setText(str(num))
        self.input_Limit = self.lnLim.text()
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
                #if "mouse_pts" not in globals():
                #    mouse_pts = []
                mouse_pts.append((x, y))
                print("Point detected")
                print(mouse_pts)

        cv2.namedWindow("Click to set threshold distance")
        cv2.setMouseCallback("Click to set threshold distance", get_mouse_points)

        input_cam = 'rtsp://' + str(self.input_Cam) + ':556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
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

        self.ai = AImodel_tpu(self.input_Model, self.input_Cam, self.input_Limit, two_points)
        self.ai.start()
        # model_run("15fps.mp4")
        # model_run(self.input_Model, self.input_Cam, self.input_Limit)
        
    def abortModel(self):
        print('stopping..')
        if self.ai:
            self.ai.terminate()
            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

        self.restart()
        
    def restart(self):
        import sys
        print("argv was",sys.argv)
        print("sys.executable was", sys.executable)
        print("restart now")

        import os
        os.execv(sys.executable, ['python'] + sys.argv)

if __name__=="__main__":
    app = QApplication(sys.argv)
    dialog = Dialog()
    dialog.show()
    sys.exit(app.exec_())
