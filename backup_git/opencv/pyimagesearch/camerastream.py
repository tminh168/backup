# import the necessary packages
from threading import Thread
import sys
import cv2
import time
# import the Queue class from Python 3
#if sys.version_info >= (3, 0):
#	from queue import Queue
# otherwise, import the Queue class for Python 2.7
#else:
#	from Queue import Queue

class CameraStream:
	def __init__(self, src=0):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(src)
		self.stream.set(cv2.CAP_PROP_FPS, 15)
		(self.grabbed, self.frame) = self.stream.read()
		#self.stream.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));
		self.stopped = False
		# initialize the queue used to store frames read from
		# the video file
		#self.Q = Queue(maxsize=queueSize)

	def start(self):
		# start a thread to read frames from the file video stream
		Thread(target=self.update, args=()).start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
			# otherwise, ensure the queue has room in it
			#if not self.Q.full():
				# read the next frame from the file
			(self.grabbed, self.frame) = self.stream.read()
			#time.sleep(0.042)
			#if grabbed:
			#	self.Q.put(frame)
					#self.stop()

				# add the frame to the queue
				#self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.frame

	#def more(self):
		# return True if there are still frames in the queue
		#return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
