#! /usr/bin/env python

import rospy
import json
from std_msgs.msg import String
import os
import cv2
from deepface import DeepFace
import speech_recognition as sr
import dlib
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import Counter
import asyncio

def top_one_frequent_value(lst):
    count = Counter(lst)
    unique_values = list(count.keys())

    if len(unique_values) == 1:
        return unique_values
    else:
        sorted_items = count.most_common(1)
        return sorted_items[0]

def top_two_frequent_values(lst):
    count = Counter(lst)
    unique_values = list(count.keys())

    if len(unique_values) == 1:
        return unique_values
    else:
        # sorted_items = count.most_common(2)
        # top_values = [item[0] for item in sorted_items]
        # return top_values
        sorted_items = count.most_common(1)
        top_values = [item[0] for item in sorted_items]
        return top_values

class FaceRecognitionCommand:
    def __init__(self):
        self.model = YOLO('/home/mustar/catkin_ws/src/general_robot/src/models/shirt_detection.pt')
        self.sub = rospy.Subscriber('video_recorded', String, self.handle_payload)
        self.pub = rospy.Publisher('final_result', String, queue_size = 10)
        self.FRAME_SAMPLE_RATE = 2
        
    def handle_payload(self, received_payload):
        rospy.loginfo(f'Video recorded received: {received_payload.data}')
        try:
            data = json.loads(received_payload.data)
            frames = self.sample_frames(data['video'], sample_rate=self.FRAME_SAMPLE_RATE)
            rospy.loginfo(f'Processing {len(frames)} frames...')
            deepFace_result = self.detect_face_emotions(frames)
            dominant_emotion = self.process_emotions(deepFace_result)
            rospy.loginfo(f"EMOTIONS: {dominant_emotion}")
            gender = self.process_genders(deepFace_result)
            rospy.loginfo(f"GENDER: {gender}")
            age = self.process_age(deepFace_result)
            rospy.loginfo(f"AGE: {age}")
            # eye_glass = self.detect_eye_glasses(frames)
            # rospy.loginfo(f"EYE GLASSES: {eye_glass}")
            eye_glass = True
            outwear = self.detect_outwears(frames)
            rospy.loginfo(outwear)
            
            audio_text = self.audio2text(data['audio'])
            # Man and Woman
            if eye_glass and len(outwear) == 1 and gender == 'Man':
                final_result = f'Hello everyone! Introducing new guest which called {audio_text}! He is a {gender} with age around {age} and feeling {dominant_emotion} today. He is wearing a pair of eye glasses too! He is wearing a {outwear[0]} today. Please have a seat {audio_text}'
                # final_result = f'Hello {audio_text}! Seems like you are a {gender} with age around {age} and feeling {dominant_emotion}. You are wearing a pair of eye glasses too! You are wearing a {outwear[0]}.'
            elif eye_glass and len(outwear) == 1 and gender == 'Woman':
                final_result = f'Hello everyone! Introducing new guest which called {audio_text}! She is a {gender} with age around {age} and feeling {dominant_emotion} today. She is wearing a pair of eye glasses too! She is wearing a {outwear[0]} today.'
            elif eye_glass and len(outwear) == 2:
                final_result = f'Hello {audio_text}! Seems like you are a {gender} with age around {age} and feeling {dominant_emotion}. You are wearing a pair of eye glasses too! You are wearing a {outwear[0]} and {outwear[1]}.'
            elif not eye_glass and len(outwear) == 1:
                final_result = f'Hello {audio_text}! Seems like you are a {gender} with age around {age} and feeling {dominant_emotion} without any eye glasses! You are wearing a {outwear[0]} and {outwear[1]}.'
            elif not eye_glass and len(outwear) == 2:
                final_result = f'Hello {audio_text}! Seems like you are a {gender} with age around {age} and feeling {dominant_emotion} without any eye glasses! You are wearing a {outwear[0]} and {outwear[1]}.'
            else:
                final_result = 'Unknown error occured, please try again!'
                
            final_result = json.dumps(final_result)
            self.pub.publish(final_result)
        except:
            rospy.loginfo("Error with face recognition node...")
            
    def audio2text(self, audio_file):
        try:
            r = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)
                text = r.recognize_google(audio)
            rospy.loginfo(text)
            if not text:
                return "There"
            return text
        except Exception as e:
            rospy.loginfo(e)
            
    def sample_frames(self, video_file, sample_rate=2):
        cap = cv2.VideoCapture(video_file)
        frames = []
        count = 0
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if count % sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
        
        return frames
    
    def landmarks_to_np(self, landmarks, dtype="int"):
        num = landmarks.num_parts
        coords = np.zeros((num, 2), dtype=dtype)
        
        for i in range(0, num):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
            
        return coords

    def get_centers(self, img, landmarks):
        EYE_LEFT_OUTTER = landmarks[2]
        EYE_LEFT_INNER = landmarks[3]
        EYE_RIGHT_OUTTER = landmarks[0]
        EYE_RIGHT_INNER = landmarks[1]

        x = ((landmarks[0:4]).T)[0]
        y = ((landmarks[0:4]).T)[1]
        A = np.vstack([x, np.ones(len(x))]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
        x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
        LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
        RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
        
        pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
        cv2.polylines(img, [pts], False, (255,0,0), 1) 
        cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
        cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
        
        return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

    def get_aligned_face(self, img, left, right):
        desired_w = 256
        desired_h = 256
        desired_dist = desired_w * 0.5
        
        eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        dist = np.sqrt(dx*dx + dy*dy)
        scale = desired_dist / dist 
        angle = np.degrees(np.arctan2(dy,dx)) 
        M = cv2.getRotationMatrix2D(eyescenter,angle,scale)

        # update the translation component of the matrix
        tX = desired_w * 0.5
        tY = desired_h * 0.5
        M[0, 2] += (tX - eyescenter[0])
        M[1, 2] += (tY - eyescenter[1])

        aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
        
        return aligned_face

    def judge_eyeglass(self, img):
        img = cv2.GaussianBlur(img, (11,11), 0)

        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) 
        sobel_y = cv2.convertScaleAbs(sobel_y)

        edgeness = sobel_y 
        
        retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        d = len(thresh) * 0.5
        x = np.int32(d * 6/7)
        y = np.int32(d * 3/4)
        w = np.int32(d * 2/7)
        h = np.int32(d * 2/4)

        x_2_1 = np.int32(d * 1/4)
        x_2_2 = np.int32(d * 5/4)
        w_2 = np.int32(d * 1/2)
        y_2 = np.int32(d * 8/7)
        h_2 = np.int32(d * 1/2)
        
        roi_1 = thresh[y:y+h, x:x+w] 
        roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
        roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
        roi_2 = np.hstack([roi_2_1,roi_2_2])
        
        measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
        measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
        measure = measure_1*0.3 + measure_2*0.7
        
        
        if measure > 0.15:
            judge = True
        else:
            judge = False
        return judge
    
    def detect_eye_glasses(self, frames):
        predictor_path = '/home/mustar/catkin_ws/src/face_rec/src/models/shape_predictor_5_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        glasses = []

        for frame in frames:
            rospy.loginfo("Eye Glass Frame ...")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray is None or gray.size == 0:
                continue
            try:
                rospy.loginfo(f'Detector? Image shape: {gray.shape}')
                rects = detector(gray, 1)
                rospy.loginfo(f"Rects: {rects}")

                if len(rects) == 0:
                    continue

            except Exception as e:
                rospy.logerr(f"Error during face detection: {e}")
                
            for i, rect in enumerate(rects):
                try: 
                    rospy.loginfo(f"Eye Glass Frame Rectangles...: {rect}")
                    x_face = rect.left()
                    y_face = rect.top()
                    w_face = rect.right() - x_face
                    h_face = rect.bottom() - y_face
                    
                    cv2.rectangle(frame, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
                    cv2.putText(frame, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    rospy.loginfo("Setting landmarks...")
                except Exception as e:
                    rospy.logerr(f"Error during enumeration: {e}")
                landmarks = predictor(gray, rect)
                landmarks = self.landmarks_to_np(landmarks)
                for (x, y) in landmarks:
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                rospy.loginfo("Getting center...")
                LEFT_EYE_CENTER, RIGHT_EYE_CENTER = self.get_centers(frame, landmarks)
                
                rospy.loginfo('Getting aligned face...')
                
                aligned_face = self.get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)
                
                rospy.loginfo('Making decision if eye glass exists...')
                
                judge = self.judge_eyeglass(aligned_face)
                glasses.append(judge)
            
            k = cv2.waitKey(5) & 0xFF
            if k==27:   
                break

        final_answer = True
        if glasses.count(True) < (len(glasses)/2):
            final_answer = False
        
        return final_answer
    
    def detect_face_emotions(self, frames):
        emotions = []
        rospy.loginfo('DETECTING EMOTIONS ...')
        for frame in frames:
            frame_result = DeepFace.analyze(frame, actions=['emotion', 'gender', 'age'], enforce_detection = False)
            rospy.loginfo(frame_result)
            emotions.append(frame_result)
        
        return emotions
    
    def process_emotions(self, emotions):
        count = 0
        emots = {'sad':0, 
                'angry':0, 
                'surprise':0, 
                'fear':0, 
                'happy':0, 
                'disgust':0, 
                'neutral':0}
        
        for frame_result in emotions:
            if len(frame_result) > 0:
                emot = frame_result[0]['emotion']
                emots['sad'] = emots.get('sad', 0) + emot['sad']
                emots['angry'] = emots.get('angry', 0) + emot['angry']
                emots['surprise'] = emots.get('surprise', 0) + emot['surprise']
                emots['fear'] = emots.get('fear', 0) + emot['fear']
                emots['happy'] = emots.get('happy', 0) + emot['happy']
                emots['disgust'] = emots.get('disgust', 0) + emot['disgust']
                emots['neutral'] = emots.get('neutral', 0) + emot['neutral']
                count += 1
                
        # prevent zero division
        if count == 0: count = 1
        
        for i in list(emots.keys()):
            emots[i] /= (count*100)

        dominant = 'sad'
        for i in list(emots.keys()):
            if emots[i] > emots[dominant]:
                dominant = i
        emots["dominant"] = dominant
        
        return emots["dominant"]
    
    def process_genders(self, genders):
        gender_list = []
        for frame_result in genders:
            gender = frame_result[0]['dominant_gender']
            gender_list.append(gender)
        gender = top_one_frequent_value(gender_list)
        return gender[0]
    
    def process_age(self, ages):
        age_list = []
        for frame_result in ages:
            age = frame_result[0]['age']
            age_list.append(age)
        rospy.loginfo(age_list)
        total_sum = sum(age_list)
        length = len(age_list)
        age = int(total_sum/length)
        return age
    
    def detect_outwears(self, frames):
        outwears = []
        rospy.loginfo('DETECTING OUTWEARS ...')
        for frame in frames:
            detected_shirts = self.model.track(frame, persist=True)
            for detected_shirt in detected_shirts:
                try:
                    shirt_json = detected_shirt.tojson()
                    shirt_dict = json.loads(shirt_json)
                    rospy.loginfo(f'Shirt Dict: {shirt_dict}')
                    if len(shirt_dict) == 0:
                        continue
                    for result in shirt_dict:
                        rospy.loginfo(result['name']) 
                        outwears.append(result['name'])      
                except Exception as e:
                    rospy.logerr(f'Error processing detected shirt: {e}')
        
        result = top_two_frequent_values(outwears)
        
        return result
    
    async def __call__(self):
        rospy.init_node('result')
        try:
            while True:
                await asyncio.sleep(0.5)
        except rospy.ROSInterruptException as e:
            rospy.loginfo("Face Recognition Command Error:", e)
        finally:
            rospy.signal_shutdown("Shutting down ROS node.")