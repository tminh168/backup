# TEST CAMERA STREAM FROM CORAL DEV BOARD
import imutils
import cv2

cap = cv2.VideoCapture()
cap.open('rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream')
#cap = cv2.VideoCapture("rtsp://admin:admin@192.168.200.79:34567")
while True:

     ret, frame = cap.read()
     if not ret:
         break

     frame = imutils.resize(frame, width=600)
     cv2.imshow('frame', frame)
     print('frame read...')

     key = cv2.waitKey(1) & 0xFF

     if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
