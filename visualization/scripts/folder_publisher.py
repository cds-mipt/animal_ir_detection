#! /usr/bin/env python

import rospy
import os
import cv2
import cv_bridge
import time
from sensor_msgs.msg import Image

RATE = 10
FOLDER = "/home/jetson/Datasets/NKBVS"


def talker():
    pub = rospy.Publisher("/images", Image, queue_size=1)
    rospy.init_node("folder_publisher")
    rate = rospy.Rate(RATE)
    br = cv_bridge.CvBridge()

    paths = sorted([os.path.join(FOLDER, filename) for filename in os.listdir(FOLDER)])
    img_idx = 0

    while not rospy.is_shutdown():
        img = cv2.imread(paths[img_idx])
        m = br.cv2_to_imgmsg(img, "bgr8")
        m.header.stamp.secs = time.time()
        m.header.stamp.nsecs = time.time() * 1e9 % 1e9
        pub.publish(m)
        img_idx = (img_idx + 1) % len(paths)
        rate.sleep()


if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
