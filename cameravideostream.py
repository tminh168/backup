from threading import Thread
#from queue import Queue
import cv2

class CameraVideoStream:
	def __init__(self, src1=0, src2=1, name="CamearaVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream1 = cv2.VideoCapture(src1)
		self.stream2 = cv2.VideoCapture(src2)
		(self.grabbed, self.frame1) = self.stream1.read()
		(self.grabbed, self.frame2) = self.stream2.read()

		# initialize the thread name
		self.name = name
		self.thread = Thread(target=self.update, name=self.name, args=())
		self.thread.daemon = True

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				self.stream1.release()
				self.stream2.release()
				return #break

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame1) = self.stream1.read()
			(self.grabbed, self.frame2) = self.stream2.read()
		

	def read(self):
		# return the frame most recently read
		return self.frame1, self.frame2

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		self.thread.join()
