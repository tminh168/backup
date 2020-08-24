import multiprocessing
import time
from main_counter import *
from cameravideostream import CameraVideoStream

manager = multiprocessing.Manager()
frame_list = manager.list()

stream_1 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
stream_2 = 'rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'


def getframe(stream_1, stream_2):
    fvs1 = CameraVideoStream(src=stream_1).start()
    fvs2 = CameraVideoStream(src=stream_2).start()

    while True:
        frame1 = fvs1.read()
        frame2 = fvs2.read()

        frame_list.clear()
        frame_list.append(frame1)
        frame_list.append(frame2)
        time.sleep(0.01)

    return frame_list

if __name__ == '__main__':
    p_stream = multiprocessing.Process(target=getframe, args=(stream_1, stream_2))
    p_count = multiprocessing.Process(target=tf_count)

    p_stream.start()

    while True:
        p_count.start()
        time.sleep(32)
        p_count.kill
        time.sleep(0.5)

    p_stream.join()
    p_count.join()
