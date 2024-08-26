#! /usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import json
import cv2
from gtts import gTTS
import os
import asyncio

class FaceDetectionCommand:
    def __init__(self):
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.pub = rospy.Publisher('human_detected', String, queue_size = 10)
        # self.pub2 = rospy.Publisher('task_status', String, queue_size=10)
        self.sub = rospy.Subscriber('final_result', String, self.handle_payload)
        self.sub2 = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=10000000)
        self.RECORD = True
        self.achieve = False
    
    def image_callback(self, msg):
        if self.RECORD:
            try:
                # 将ROS图像数据转换为OpenCV图像
                cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")

                # 调整图像大小以加快处理速度
                # cv_image = cv2.resize(cv_image, (640, 480))

                # if self.select_detect==0:
                humans = self.detect_human(cv_image)

                if len(humans) > 0:
                    rospy.sleep(1)
                    payload = json.dumps("True")
                    self.pub.publish(payload)
                    self.RECORD = False

                rospy.loginfo("Start person detection ...")

            except CvBridgeError as e:
                print(e)

    def handle_payload(self, received_payload):
        rospy.loginfo(received_payload.data)
        self.text2audio(received_payload.data)
        self.achieve = True
        # self.pub2.publish(f'True-{received_payload.data}')
        
    def detect_human(self, image):
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        (humans, _) = hog.detectMultiScale(image, winStride=(8, 8),
        padding=(16, 16), scale=1.05)
        
        return humans

    def text2audio(self, text):
        tts = gTTS(text)
        tts.save("main_audio.mp3")
        os.system("mpg321 main_audio.mp3")
        os.remove("main_audio.mp3")

    async def __call__(self):
        rospy.init_node('face_detection', anonymous=True)
        try:
            while True:
                await asyncio.sleep(0.5)
                if self.achieve == True:
                    loop = asyncio.get_running_loop()
                    loop.stop()
                    break
        except rospy.ROSInterruptException as e:
            rospy.loginfo("Face Detection Command Error:", e)
        finally:
            rospy.signal_shutdown("Shutting down ROS node.")
        