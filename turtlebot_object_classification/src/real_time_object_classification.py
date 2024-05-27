#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import math
import os
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


class ImageFeeder():
    def __init__(self):
        # camera objects
        self.bridge_object = CvBridge()
        self.cam_topic = '/panda_camera/color/image_raw'
        self.img_sub = rospy.Subscriber(
            self.cam_topic, Image, self.cameraCallback)
        # model objects
        self.model = load_model(os.path.join(
            os.path.dirname(__file__), "cnn_model.h5"))
        print('LOADED MODEL')

    def cameraCallback(self, cam_msg):
        # can return header, height, width, encoding (rgb8), step, data
        try:
            img = self.bridge_object.imgmsg_to_cv2(
                cam_msg, desired_encoding='bgr8')
            #print('LOADED IMAGE')
        except CvBridgeError as cve:
            print('Failing to load image')
            rospy.loginfo(cve)
        h, w, ch = img.shape
        print('Image_size', h, w, ch)
        # cropping the image to only see the important stuff
        # cropped_img = img[500:1020, 900:1120]
        cropped_img = img
        cv2.imshow("RES", img)
        cv2.waitKey(1)

        self.predict(cropped_img)

    def predict(self, img_to_pred):
        img = cv2.resize(img_to_pred, dsize=(32, 32))
        img = img.reshape(1, 32, 32, 3)
        predictions = self.model.predict(img)
        y_pred = np.argmax(predictions, axis=1)
        print("Y_PRED", y_pred)
        if y_pred.size > 0:
            print('Predicted class: '+str(y_pred[0]))
        else:
            print('-------------------------')

    def clean_up(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    imagefeeder = ImageFeeder()
    rospy.init_node('image_feed__node', anonymous=True)

    rate = rospy.Rate(1)
    ctrl_c = False

    def shutdownhook():
        imagefeeder.clean_up()
        rospy.loginfo('shutdown time!')
        global ctrl_c
        ctrl_c = True

    rospy.on_shutdown(shutdownhook)
    while not ctrl_c:
        rate.sleep()
