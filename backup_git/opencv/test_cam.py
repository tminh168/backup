# TEST CAMERA STREAM FROM CORAL DEV BOARD
from __future__ import print_function
from imutils.video import FileVideoStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from pyimagesearch.camerastream import CameraStream
import time
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=500,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=1,
	help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

# created a *threaded* video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from camera...")
fvs = cv2.VideoCapture('rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream')
fvs.set(cv2.CAP_PROP_FPS, 5)
#time.sleep(1.0)
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while True: #fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	start_t = time.time()
	grabbed, frame = fvs.read()
#	if frame is None:
#		continue
#	else:
	frame = imutils.resize(frame, width=600)
#		cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
#					(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	# check to see if the frame should be displayed to our screen
	#if args["display"] > 0:
	cv2.imshow("Frame", frame)
	print("screening..")
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
		#fvs.stop()
	end_t = time.time()
	interval = end_t - start_t
	print('Frame: {}'.format(interval))
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fvs.release()

