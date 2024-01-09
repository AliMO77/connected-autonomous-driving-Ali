from styx_msgs.msg import TrafficLight
import cv2
from ultralytics import YOLO
import numpy as np
import rospy
import sys
import os
from contextlib import redirect_stdout

class TrafficLightDetector():

    def __init__(self):
        self.model = self.load_model('/host/ros/TRAFFICLIGHTDETECTION/train6/weights/best.pt')

    def load_model(self, model_path):
        # Load and return the model
        model = YOLO(model_path)
        print("Traffic Light Detection model loaded")
        return model
    
    def preprocess_image(self, camera_frame):
        frame = camera_frame
        
        alpha = 2.5  # Contrast control (1.0-3.0)
        beta = 0    # Brightness control (0-100)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
         # Split the frame into its B, G, and R components
        B, G, R = cv2.split(frame)
        # Define the increase in the red channel
        increase = 8
        # Increase the red channel by the increase value
        # and ensure the values remain between 0 and 255
        R = cv2.add(R, increase)
        R[R > 255] = 255
        # Merge the channels back together
        processed_frame = cv2.merge([B, G, R])
        return processed_frame

    def get_traffic_light(self,camera_frame):

        frame = self.preprocess_image(camera_frame)
        
        #detection
        result = self.model.predict(
            source=frame,
            stream=True,
            max_det = 1,
            imgsz=256,
            verbose =False
        )
        
        all_labels = ["Green","Yellow","Red"]
        color = None
        for r in result:
            boxes = r.boxes.cpu().numpy()
            if len(boxes.cls) > 0:
                max_conf = boxes.conf.max()
                idx = np.where(boxes.conf == max_conf)[0][0]
                label_idx = int(boxes.cls[idx])
                color = all_labels[label_idx]
        rospy.logwarn("detected light: {}".format(color))
        return color


    def interpret_results(self, results):
        
        if results=='Red':
            return TrafficLight.RED
        elif results =='Yellow':
            return TrafficLight.YELLOW
        elif results =='Green':
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN
    
    
    def detect_state(self, camera_frame):
        # TODO
        color = self.get_traffic_light(camera_frame)
        
        state = self.interpret_results(color)

        return state
