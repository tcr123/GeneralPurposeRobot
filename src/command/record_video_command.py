#! /usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import json
import cv2
from gtts import gTTS
import os
import threading
import pyaudio
import wave
import asyncio

class RecordVideoCommand:
    def __init__(self):
        self.sub = rospy.Subscriber('human_detected', String, self.handle_payload)
        self.pub = rospy.Publisher('video_recorded', String, queue_size = 10)
        
        image_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.sub2 = rospy.Subscriber(image_topic, Image, self.record_video_callback, queue_size=1, buff_size=10000000)

        self.FPS               = 10.0   # default ffmpeg FPS, do not change
        self.frame_interval = 1.0 / self.FPS
        self.FRAME_SAMPLE_RATE = 2      # samples a frame from every N frames for emotion detection
        self.FILE_NAMES = { 'mixed': '/home/mustar/catkin_ws/src/general_robot/src/assets/mixed.mp4',
                            'video': '/home/mustar/catkin_ws/src/general_robot/src/assets/vid_only.avi',
                            'audio': '/home/mustar/catkin_ws/src/general_robot/src/assets/aud_only.wav'
                          }
        self.RECORD_DURATION = 2
        self.RECORD = False 
            
    def handle_payload(self, received_payload):
        data = json.loads(received_payload.data)
        rospy.loginfo("Video recorded ...")
        if data == "True":
            try:
                self.text2audio("Hi! I am your receptionist for today! What is your name?")
                self.RECORD = True
            except:
                rospy.loginfo("Error with video and audio recording...")
    
    def record_video_callback(self, image):
        if self.RECORD:
            MIXED_VID_AUD_FILE = self.FILE_NAMES['mixed']
            TMP_VID_FILE       = self.FILE_NAMES['video']
            TMP_AUD_FILE       = self.FILE_NAMES['audio']
            
            video_thread = threading.Thread(target=self.record_video_util, args=(self.RECORD_DURATION, image, TMP_VID_FILE))
            audio_thread = threading.Thread(target=self.record_audio_util, args=(self.RECORD_DURATION, TMP_AUD_FILE))

            video_thread.start()
            audio_thread.start()

            video_thread.join()
            audio_thread.join()

            # # combine video and audio using ffmpeg
            # cmd = f'ffmpeg -y -i {TMP_VID_FILE} -i {TMP_AUD_FILE} -c:v copy -c:a aac -strict experimental {MIXED_VID_AUD_FILE}'
            # subprocess.call(cmd, shell=True)

            self.RECORD = False

            payload = {'mixed': MIXED_VID_AUD_FILE, 'video':TMP_VID_FILE, 'audio':TMP_AUD_FILE}
            payload = json.dumps(payload)
            self.pub.publish(payload)

    def record_video_util(self, duration, msg, output_file='output.mp4'):

        try:
        
            cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')   # video codec, machine specific
            
            h, w, _ = cv_image.shape

            out = cv2.VideoWriter(output_file, fourcc, self.FPS, (w,h))

        except Exception as e:
            print(f"Errorrrrrrrrrrrrrrrr: {e}")
        # out.release()

        # Start time for duration control
        start_time = cv2.getTickCount()
        frame_count = 0

        while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < duration:
            # Calculate the elapsed time
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            # Calculate the expected time for the current frame
            expected_time = frame_count * self.frame_interval
            
            # Write the frame if it's time for this frame
            if elapsed_time >= expected_time:
                out.write(cv_image)
                frame_count += 1
            
            # Display frame (optional)
            cv2.imshow('frame', cv_image)
            if cv2.waitKey(1) == ord('q'):
                break

        # Release everything if job is finished
        out.release()
        cv2.destroyAllWindows()
        
    # utility method for audio only recording
    def record_audio_util(self, duration, audio_file):
        
        CHUNK         = 1024  # record in chunks of 1024 samples
        SAMPLE_FORMAT = pyaudio.paInt16  # 16 bits per sample
        CHANNELS      = 1
        FS            = 44100  # record at 44100 samples per second

        p = pyaudio.PyAudio()

        stream = p.open(format=SAMPLE_FORMAT,
                        channels=CHANNELS,
                        rate=FS,
                        frames_per_buffer=CHUNK,
                        input=True)

        frames = []
        for _ in range(0, int(FS / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(audio_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
        wf.setframerate(FS)
        wf.writeframes(b''.join(frames))
        wf.close()

    def text2audio(self, text):
        tts = gTTS(text)
        tts.save("main_audio.mp3")
        os.system("mpg321 main_audio.mp3")
        os.remove("main_audio.mp3")
        
    async def __call__(self):
        rospy.init_node('record_video', anonymous=True)
        try:
            while True:
                await asyncio.sleep(0.5)
        except rospy.ROSInterruptException as e:
            rospy.loginfo("Video Recording Command Error:", e)
        finally:
            rospy.signal_shutdown("Shutting down ROS node.")
        