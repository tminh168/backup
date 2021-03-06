from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QWidget, QDialog, QGroupBox,
                             QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QInputDialog, QPushButton, QVBoxLayout)
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
import sys
import cv2
import imutils
import numpy as np
import time
from main_tpu_procs import AImodel_tpu
from multiprocessing import Process, Queue

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        #cap = cv2.VideoCapture("15fps.mp4")
        global dialog
        while self._run_flag:
            cv_ctr = dialog.q_ctr.get()
            cv_dist = dialog.q_dist.get()
            if cv_ctr and cv_dist: 
                self.change_pixmap_signal.emit(cv_ctr, cv_dist)
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class ShowCam(QWidget):
    def __init__(self):
        super(ShowCam, self).__init__()
        self.setWindowTitle("DFM AI demo camera")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_ctr = QLabel(self)
        self.image_dist = QLabel(self)
        self.image_ctr.resize(self.disply_width, self.display_height)
        self.image_dist.resize(self.disply_width, self.display_height)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_ctr)
        vbox.addWidget(self.image_dist)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray, np.ndarray)
    def update_image(self, cv_ctr, cv_dist):
        """Updates the image_label with a new opencv image"""
        qt_ctr = self.convert_cv_qt(cv_ctr)
        qt_dist = self.convert_cv_qt(cv_dist)
        self.image_ctr.setPixmap(qt_ctr)
        self.image_dist.setPixmap(qt_dist)

    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

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
        #self.resize(375, 150)
        self.setGeometry(200, 200, 800, 600)

        self.q_ctr = Queue()
        self.q_dist = Queue()
        self.p = None
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
        items = ("SSD Custom People detection", " ")

        item, ok = QInputDialog.getItem(self, "select model input",
                                        "detection model", items, 0, False)

        if ok and item:
            self.lnModel.setText(item)
        self.input_Model = str(self.lnModel.text())
        print(self.input_Model)

    def getCam(self):

        items = ("192.168.200.78", "192.168.200.81")

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
                # if "mouse_pts" not in globals():
                #    mouse_pts = []
                mouse_pts.append((x, y))
                print("Point detected")
                print(mouse_pts)

        cv2.namedWindow("Click to set threshold distance")
        cv2.setMouseCallback(
            "Click to set threshold distance", get_mouse_points)

        cam_78 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
        cam_81 = 'rtsp://192.168.200.81:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'

        if self.input_Cam == "192.168.200.78":
            input_cam = cam_78
        elif self.input_Cam == "192.168.200.81":
            input_cam = cam_81
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

        self.p = Process(target=AImodel_tpu, args=(
            self.input_Model, self.input_Cam, self.input_Limit, two_points, self.q_ctr, self.q_dist,))
        self.p.daemon = True
        self.p.start()

        self.a = ShowCam()
        self.a.show()

    def abortModel(self):
        print('stopping..')

        if self.p:
            self.p.terminate()
            self.p.join()
            time.sleep(0.5)

            self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(True)

        self.a.close()
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
