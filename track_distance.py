import cv2
import numpy as np
from pyimagesearch.trackableobject import TrackableObject
from scipy.spatial.distance import pdist, squareform


def append_objs_distance(frame, pedestrian_boxes, d_thresh):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    node_radius = 5
    color_node = (192, 133, 156)
    thickness_node = 5
    lineThickness = 4
    solid_back_color = (41, 41, 41)

    bird_image = np.zeros(
        (int(frame_h), int(frame_w), 3), np.uint8
    )
    bird_image[:] = solid_back_color
    center_pts = []
    for i in range(len(pedestrian_boxes)):

        mid_point_x = int(
            (pedestrian_boxes[i][1] * frame_w +
             pedestrian_boxes[i][3] * frame_w) / 2
        )
        mid_point_y = int(
            (pedestrian_boxes[i][0] * frame_h +
             pedestrian_boxes[i][2] * frame_h) / 2
        )

        #pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        pts = [mid_point_x, mid_point_y]
        #pts = [mid_point_x, mid_point_y]
        center_pts.append(pts)
        bird_image = cv2.circle(
            bird_image,
            (pts[0], pts[1]),
            node_radius,
            color_node,
            thickness_node,
        )

    if len(center_pts) > 1:
        p = np.array(center_pts)
        dist_condensed = pdist(p)
        dist = squareform(dist_condensed)

        dd = np.where(dist < d_thresh * 5 / 4)
        # six_feet_violations = len(np.where(dist_condensed < d_thresh)[0])
        # total_pairs = len(dist_condensed)
        #danger_p = []
        color_6 = (52, 92, 227)
        for i in range(int(np.ceil(len(dd[0]) / 2))):
            if dd[0][i] != dd[1][i]:
                point1 = dd[0][i]
                point2 = dd[1][i]

                #danger_p.append([point1, point2])
                cv2.line(
                    bird_image,
                    (p[point1][0], p[point1][1]),
                    (p[point2][0], p[point2][1]),
                    color_6,
                    lineThickness,
                )

    return bird_image


def append_objs_counter(frame, countedID, pedestrian_boxes, ROI, ct, trackableObjects, totalCount):
    height, width = frame.shape[:2]
    rects = []

    for i in range(len(pedestrian_boxes)):
        (ymin, xmin, ymax, xmax) = pedestrian_boxes[i]
        (x0, y0, x1, y1) = (int(round(xmin * width)), int(round(ymin * height)),
                            int(round(xmax * width)), int(round(ymax * height)))
        rects.append((x0, y0, x1, y1))
        #label = '{}-{}%'.format(labels.get(obj.id, obj.id), percent)

        frame = cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    return frame, countedID, totalCount, direction_str, ct, trackableObjects
