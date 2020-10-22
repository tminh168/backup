import cv2
import time
import calendar
import base64
from frame_submit import FrameSubmit
from cameravideostream import CameraVideoStream

cam_78 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'

fvs = CameraVideoStream(cam_78).start()

FS = FrameSubmit().start()
n = 1

while n < 50:

    frame = fvs.read()
    current_time = calendar.timegm(time.gmtime())

    totalCount = n
    direction_str = "In"

    # Convert captured image to JPG
    ret, buffer = cv2.imencode('.jpg', frame)
    # Convert to base64 encoding and show start of data
    jpg_as_text = base64.b64encode(buffer)

    FS.q_push(jpg_as_text, current_time, totalCount, direction_str, 1)

    n += 1
