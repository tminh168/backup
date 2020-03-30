from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import FileVideoStream
from imutils.video import FPS
from utils import backbone
import tensorflow as tf
import numpy as np
import imutils
import cv2

def count_ppl(stream, out_dir):
	detection_graph, category_index = backbone.set_model('people_inference_graph', 'people_label_map.pbtxt')

	vs = FileVideoStream(stream).start()
	threshold = 0.85
	W = None
	H = None

	ct = CentroidTracker(maxDisappeared=2, maxDistance=80)
	trackers = []
	trackableObjects = {}

	totalCount = 0
	direction_str = "..."
	ROI = 300
	fps = FPS().start()

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')

			# loop over frames from the video stream
			while True:
				direction_str = "..."
				# grab the next frame and handle if we are reading from either
				# VideoCapture or VideoStream
				frame = vs.read()

				if frame is None:
					print("end of the video file...")
					break
				# resize the frame to have a maximum width of 500 pixels (the
				# less data we have, the faster we can process it), then convert
				# the frame from BGR to RGB for dlib
				frame = imutils.resize(frame, width=600)

				# if the frame dimensions are empty, set them
				if W is None or H is None:
					(H, W) = frame.shape[:2]

				rects = []
				labels = []

				# Actual detection
				input_frame = frame

				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(input_frame, axis=0)

				# Actual detection.
				(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})

				boxes = np.squeeze(boxes)
				classes = np.squeeze(classes).astype(np.int32)
				scores = np.squeeze(scores)
				
				# Sort out valid box
				for i in range(boxes.shape[0]):
					if scores[i] > threshold:
						if classes[i] in category_index.keys():
							box = tuple(boxes[i].tolist()) # valid box
							class_name = category_index[classes[i]]['name'] 
							display_str = '{}: {}%'.format(class_name,int(100*scores[i]))       

							ymin, xmin, ymax, xmax = box
							(startX, startY, endX, endY) = (int(round(xmin * W)), int(round(ymin * H)),
												int(round(xmax * W)), int(round(ymax * H)))

		                                      	# filter out detection box in the middle region of frame
							# to add to trackables objects for best accuracy
							if startX > 150 and endX < 450:				              	
								rects.append((startX, startY, endX, endY))
								labels.append((display_str, startX, startY))	              

				# draw a horizontal line in the center of the frame -- once an
				# object crosses this line we will determine whether they were
				# moving 'up' or 'down'
				cv2.line(frame, (ROI, 0), (ROI, H), (0, 255, 255), 2)

				# use the centroid tracker to associate the (1) old object
				# centroids with (2) the newly computed object centroids
				objects = ct.update(rects)

				# Draw bounding box and label for detected objects
				for (i, (startX, startY, endX, endY)) in enumerate(rects):
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
				for (i, (display_str, startX, startY)) in enumerate(labels):
					cv2.putText(frame, display_str, (startX + 10, startY + 15),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				# loop over the tracked objects
				for (objectID, centroid) in objects.items():
					# check to see if a trackable object exists for the current
					# object ID
					to = trackableObjects.get(objectID, None)

					# if there is no existing trackable object, create one
					if to is None:
						to = TrackableObject(objectID, centroid)

					# otherwise, there is a trackable object so we can utilize it
					# to determine direction
					else:		
						# check to see if the object has been counted or not
						if not to.counted:
							# if the previous centroids from one side
							# count as soon as the updated centroid reach other side
							for c in to.centroids:
								if c[0] < ROI and centroid[0] < ROI:
									direction_str = "..."
								elif c[0] < ROI and centroid[0] > ROI:
									totalCount += 1
									to.counted = True
									direction_str = "In"
									break
								elif c[0] > ROI and centroid[0] > ROI:
									direction_str = "..."
								elif c[0] > ROI and centroid[0] < ROI:
									totalCount += 1
									to.counted = True
									direction_str = "Out"
									break								
					
						# update new centroid to trackable object
						to.centroids.append(centroid)

					# store the trackable object in our dictionary
					trackableObjects[objectID] = to

					# draw both the ID of the object and the centroid of the
					# object on the output frame
					text = "ID {}".format(objectID)
					cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

				# construct a tuple of information we will be displaying on the
				# frame
				info = [
					("Direction", direction_str),
					("Count", totalCount),
				]

				# loop over the info tuples and draw them on our frame
				for (i, (k, v)) in enumerate(info):
					text = "{}: {}".format(k, v)
					cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
				if direction_str == "In" or direction_str == "Out":
					cv2.imwrite(out_dir + "/count-%d.jpg" % totalCount, frame)

				# show the output frame
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break

				fps.update()

			# stop the timer and display FPS information
			fps.stop()
			print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
			print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	
	# close any open windows
	cv2.destroyAllWindows()
	vs.stop()
