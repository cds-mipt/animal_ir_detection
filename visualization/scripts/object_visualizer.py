# !/usr/bin/env python

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, RegionOfInterest
import numpy as np
import cv2

from msg.ObjectArray import ObjectArray


LABELS_FILE = "../yolo-trt/data/labels_zoo.txt"
LABELS = None
N_CLASSES = None


def on_image(msg):
    on_image.images.append(msg)


def on_object_array(msg):
    on_object_array.last_object_array = msg


on_image.images = []
on_object_array.last_object_array = None


def roi_to_bbox(roi):
    return int(roi.x_offset), int(roi.y_offset), int(roi.width), int(roi.height)


def get_colors():
    idx_to_color = np.random.randint(0, 255, (N_CLASSES, 3))
    return np.array(idx_to_color, dtype=np.uint8)


def draw_objects(image, object_array, idx_to_color):
    for object in object_array.objects:
        color = tuple(map(int, idx_to_color[object.label]))

        x, y, w, h = roi_to_bbox(object.bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        cv2.putText(image, LABELS[object.label], (x + w, y), 0, 1, color, 1)
    return image


def load_labels():
    global LABELS, N_CLASSES
    with open(LABELS_FILE, "r") as f:
        LABELS = [line.strip() for line in f]
    N_CLASSES = len(LABELS)


def main():
    br = CvBridge()
    rospy.init_node('object_visualizer')

    load_labels()
    idx_to_color = get_colors()

    sub_image = rospy.Subscriber("/images", Image, on_image)
    sub_objects = rospy.Subscriber("/detection/yolo/objects", ObjectArray, on_object_array)
    pub_image_with_objects = rospy.Publisher("/visualization/yolo/objects", Image, queue_size=1)

    while not rospy.is_shutdown():
        if on_object_array.last_object_array is None:
            continue
        last_object_array = on_object_array.last_object_array
        on_object_array.last_object_array = None

        image = None
        for img in on_image.images:
            if img.header.stamp == last_object_array.header.stamp:
                image = img
        on_image.images = []
        if image is None:
            continue

        header = image.header
        img = br.imgmsg_to_cv2(image, desired_encoding="rgb8")

        img = draw_objects(img, last_object_array, idx_to_color)

        if pub_image_with_objects.get_num_connections() > 0:
            m = br.cv2_to_imgmsg(img.astype(np.uint8), encoding="rgb8")
            m.header.stamp = header.stamp
            pub_image_with_objects.publish(m)


if __name__ == "__main__":
    main()
