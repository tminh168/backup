import cv2
import time
from cameravideostream import CameraVideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from tpu_model import *


# final optimized version by Minh HO
def append_objs_to_img(cv2_im, countedID, objs, labels, ROI, ct, trackableObjects, totalCount):
    height, width = cv2_im.shape[:2]
    rects = []

    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(
            x0*width), int(y0*height), int(x1*width), int(y1*height)
        if x0 < 100 or x1 > 550:
            continue
        rects.append((x0, y0, x1, y1))
        percent = int(100 * obj.score)
        label = '{}-{}%'.format(labels.get(obj.id, obj.id), percent)

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # update the current matched object IDs
    objects = ct.update(rects)
    direction_str = "..."

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # Create new obj if never counted (<countedID)
        if to is None and objectID > countedID:
            to = TrackableObject(objectID, centroid)
        # bypass obj if already counted in previous frame (=countedID) and old ID (<countedID)
        elif to is None and objectID <= countedID:
            continue
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # check to see if the object has been counted or not
            if not to.counted:
                # compare first centroid and the current one to detemine direction
                c = to.centroids[0]
                if c[0] < ROI and centroid[0] < ROI:
                    direction_str = "..."
                elif c[0] < ROI and centroid[0] > ROI:
                    totalCount += 1
                    to.counted = True
                    direction_str = "In"
                elif c[0] > ROI and centroid[0] > ROI:
                    direction_str = "..."
                elif c[0] > ROI and centroid[0] < ROI:
                    totalCount += 1
                    to.counted = True
                    direction_str = "Out"

                # to.centroids.append(centroid)

        trackableObjects[objectID] = to
        # update to in dict
        if to.counted:
            # delete from dict
            del trackableObjects[objectID]
            countedID = objectID

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(cv2_im, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(cv2_im, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return cv2_im, countedID, totalCount, direction_str, trackableObjects

def tf_count():
    with open('count.txt', 'r') as file:
        count_ckpt = file.read().split('\n')

    stream_1 = 'rtsp://192.168.200.78:556/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    stream_2 = 'rtsp://192.168.200.79:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream'
    model_1 = 'detection_1_edgetpu.tflite'
    model_2 = 'detection_2_edgetpu.tflite'
    base_dir = '/home/mendel/coral/DFM_counter'

    # Define a DNN model
    DNN_count1 = model_tpu(model_1)
    DNN_count2 = model_tpu(model_2)
    fvs1 = CameraVideoStream(src=stream_1).start()
    fvs2 = CameraVideoStream(src=stream_2).start()
    time.sleep(0.1)
    H = 400
    W = 600
    ct1 = CentroidTracker(maxDisappeared=2, maxDistance=55)
    ct2 = CentroidTracker(maxDisappeared=2, maxDistance=55)
    trackableObjects1 = dict()
    trackableObjects2 = dict()
    totalCount1 = int(count_ckpt[0])
    totalCount2 = int(count_ckpt[1])
    countedID1 = 0
    countedID2 = 0
    ROI = 350
    log_img = True

    timeout_rs = 600
    timeout_ckpt = int(time.time()) + timeout_rs
    # Process each frame, until end of video
    while True:
        t_dtc = time.time()

        direction_str1 = "..."
        direction_str2 = "..."
        frame1 = fvs1.read()
        frame2 = fvs2.read()

        if frame1 is None or frame2 is None:
            continue

        cv2_im1, objs1 = DNN_count1.detect_count(frame1)
        cv2_im2, objs2 = DNN_count2.detect_count(frame2)
        cv2_im1 = cv2.line(cv2_im1, (ROI, 0), (ROI, H), (0, 255, 255), 2)
        cv2_im2 = cv2.line(cv2_im2, (ROI, 0), (ROI, H), (0, 255, 255), 2)
        cv2_im1, countedID1, totalCount1, direction_str1, trackableObjects1 = append_objs_to_img(
            cv2_im1, countedID1, objs1, labels, ROI, ct1, trackableObjects1, totalCount1)
        cv2_im2, countedID2, totalCount2, direction_str2, trackableObjects2 = append_objs_to_img(
            cv2_im2, countedID2, objs2, labels, ROI, ct2, trackableObjects2, totalCount2)

        info1 = [
            ("Direction", direction_str1),
            ("Count", totalCount1),
        ]
        for (i, (k, v)) in enumerate(info1):
            text1 = "{}: {}".format(k, v)
            cv2.putText(cv2_im1, text1, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        info2 = [
            ("Direction", direction_str2),
            ("Count", totalCount2),
        ]
        for (i, (k, v)) in enumerate(info2):
            text2 = "{}: {}".format(k, v)
            cv2.putText(cv2_im2, text2, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if log_img:
            if direction_str1 == "In" or direction_str1 == "Out":
                cv2.imwrite(base_dir + "/detected1" + "/count-%d.jpg" %
                            totalCount1, cv2_im1)
            if direction_str2 == "In" or direction_str2 == "Out":
                cv2.imwrite(base_dir + "/detected2" + "/count-%d.jpg" %
                            totalCount2, cv2_im2)

        #cv2.imshow('Camera_1', cv2_im1)
        #cv2.imshow('Camera_2', cv2_im2)
        #cv2.waitKey(1)

        te_dtc = time.time()
        print('frame: {:.3f}'.format(te_dtc - t_dtc))

        if int(te_dtc) >= timeout_ckpt:
            with open('count.txt', 'w') as file:
                file.write(str(totalCount1) + '\n' + str(totalCount2))
            print('checkpoint saved!')
            break
            #timeout_end = int(time.time()) + timeout_rs

    return 1

if __name__ == '__main__':
    passenger_counting = tf_count()
