import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--frame", type=str, default='1',
	help="show counted frame")
args = vars(ap.parse_args())

count = 'count-{}.jpg'.format(args["frame"])
while True:
   image = cv2.imread(count)
   cv2.imshow('image', image)

   key = cv2.waitKey(1) & 0xFF
   if key == ord("q"):
      break
