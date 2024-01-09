# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import rospy
import cv2
import numpy as np
import time
from sensor_msgs.msg import LaserScan

def callback(data):
    i = 0
    ar = 0
    dr = np.zeros((1,68), np.float64)
    for r in data.ranges:
        i += 1
        ar += r
        if i % 10 == 0:
            dr[0][int(i/10) - 1] = ar / 10
            ar = 0
            # change infinite values to 0
            # 如果r的值是正负无穷大，归零

    print(len(data.ranges))
    print(dr)
    time.sleep(10)

if __name__ == '__main__':

    rospy.init_node('laser_listener', anonymous=True)
    while(1):

        rospy.Subscriber("/scan", LaserScan, callback, queue_size=1)


