from threading import Thread
from queue import Queue
import cv2

class CameraVideoStream:
	def __init__(self, src=0, queue_size=512, name="CamearaVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)

		# initialize the thread name
		self.name = name
		self.queue_size=queue_size
		self.Q = Queue(maxsize=queue_size)
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
				self.stream.release()
				break

			if not self.Q.full():
				# otherwise, read the next frame from the stream
				(grabbed, frame) = self.stream.read()
				self.Q.put(frame)

				if self.queue_size - self.Q.qsize() <= 12:
					temp = self.Q.get()

	def read(self):
		# return the frame most recently read
		return self.Q.get()

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
