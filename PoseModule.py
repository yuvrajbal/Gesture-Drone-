import cv2
import mediapipe as mp
import time
import math
import matplotlib.pyplot as plt
import json

class poseDetector():

    def __init__(self, mode=False,model_complexity = 1, smooth=True, enable_segmentation = False,
        smooth_segmentation = True,detectionCon=0.5, trackCon=0.5):

        # If set to False, the solution treats the input images as a video stream.  It will try to detect the most prominent person in the very first images, and upon a successful detection further localizes the pose landmark
        self.mode = mode
        self.complexity = model_complexity
        self.smooth = smooth
        self.enable_seg= enable_segmentation
        self.smooth_seg= smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.person_id = None

        #Intialize mediapipe drawing class
        self.mpDraw = mp.solutions.drawing_utils

        #Initialize mediapipe pose class
        self.mpPose = mp.solutions.pose

        #Set up the Pose function
        self.pose = self.mpPose.Pose(self.mode,self.complexity, self.smooth,self.enable_seg,self.smooth_seg,self.detectionCon, self.trackCon)
        
    
    def detectPose(self, img, pose, display=True):
        '''
        This function performs pose detection on an image.
        Args:
            image: The input image with a prominent person whose pose landmarks needs to be detected.
            pose: The pose setup function required to perform the pose detection.
            display: A boolean value that is if set to true the function displays the original input image, the resultant image,
                     and the pose landmarks in 3D plot and returns nothing.
        Returns:
            output_image: The input image with the detected pose landmarks drawn.
            landmarks: A list of detected landmarks converted into their original scale.
        '''

        # Create a copy of the input image.
        output_image = img.copy()

        # Convert the image from BGR into RGB format.
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform the Pose Detection.
        #if self.person_id is not None:
        self.results = self.pose.process(imageRGB)
        #else:
            #self.results = self.pose.process(imageRGB)

        #    if self.results.pose_landmarks:
        #        self.person_id = 0


        # Retrieve the height and width of the input image.
        height, width, _ = img.shape

        # Initialize a list to store the detected landmarks.
        landmarks = []

        # Check if any landmarks are detected.
        if self.results.pose_landmarks:

            # Draw Pose landmarks on the output image.
            self.mpDraw.draw_landmarks(output_image,self.results.pose_landmarks,
                                      self.mpPose.POSE_CONNECTIONS)

            # Iterate over the detected landmarks.
            for landmark in self.results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))

        # Return the output image and the found landmarks.
        return output_image, landmarks
    def calculateAngle( self, landmark1, landmark2, landmark3):
        '''
        This function calculates angle between three different landmarks.
        Args:
            landmark1: The first landmark containing the x,y and z coordinates.
            landmark2: The second landmark containing the x,y and z coordinates.
            landmark3: The third landmark containing the x,y and z coordinates.
        Returns:
            angle: The calculated angle between the three landmarks.

        '''

        # Get the required landmarks coordinates.
        x1, y1, _ = landmark1
        x2, y2, _ = landmark2
        x3, y3, _ = landmark3

        # Calculate the angle between the three points
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        # Check if the angle is less than zero.
        # if angle < 0:
        #     # Add 360 to the found angle.
        #     angle += 360

        angle = (angle ) % 360
        # Return the calculated angle.
        return angle
    def classifyPose(self, landmarks, img, display= False):
        label = 'Unknown Gesture'

        left_elbow_angle= self.calculateAngle(landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value])
        right_elbow_angle = self.calculateAngle( landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value],
                                           landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value])
        left_shoulder_angle=self.calculateAngle(landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value],
                                           landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value],
                                           landmarks[self.mpPose.PoseLandmark.LEFT_HIP])
        right_shoulder_angle = self.calculateAngle(landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value],
                                             landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value],
                                             landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW])

        #CHECK IF BOTH elbows ARE STRAIGHT
        if left_elbow_angle > 140 and left_elbow_angle < 195 and right_elbow_angle > 140 and right_elbow_angle < 195:
            #check shoulder angles are 90
            if left_shoulder_angle > 70 and left_shoulder_angle < 120 and right_shoulder_angle > 70 and right_shoulder_angle < 120:
                label = "Flip Forward"
            if left_shoulder_angle > 155 and left_shoulder_angle < 210 and right_shoulder_angle > 155 and right_shoulder_angle< 210:
                label = "Flip Backward"
        if left_elbow_angle > 70 and left_elbow_angle < 120 and right_elbow_angle > 140 and right_elbow_angle < 195:
            if left_shoulder_angle > 70 and left_shoulder_angle < 120 and right_shoulder_angle > 70 and right_shoulder_angle < 120:
                label= "Flip Right"
        if right_elbow_angle > 70 and right_elbow_angle < 120 and left_elbow_angle > 140 and left_elbow_angle < 195:
            if left_shoulder_angle > 70 and left_shoulder_angle < 120 and right_shoulder_angle > 70 and right_shoulder_angle < 120:
                label = "Flip Left"
        if right_shoulder_angle> 0 and right_shoulder_angle< 30 and left_shoulder_angle>0 and left_shoulder_angle<30:
            if left_elbow_angle > 70 and left_elbow_angle<120 and right_elbow_angle>70 and right_elbow_angle<120:
                label = "Stop"

        color = (0,255,255)
        cv2.putText(img,label,(10,30), cv2.FONT_HERSHEY_PLAIN,2,color,2)
    
        return img,label


        
    def findNoseLandmark(self,landmarks, img):
        lm_nose = landmarks[self.mpPose.PoseLandmark.NOSE.value]
        color = (255, 255, 255)
        #cv2.putText(img, str(lm_nose), (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        return lm_nose

    def calculate_width(self,landmarks,img):
        lm_right_shoulder = landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value]
        lm_left_shoulder = landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value]

        distance_bw_lf_right = lm_right_shoulder[0] - lm_left_shoulder[0]
        return distance_bw_lf_right
