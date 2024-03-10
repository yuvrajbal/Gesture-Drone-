import cv2
import time
from simple_pid import PID
import PoseModule as pm
# from djitellopy import Tello
import json

# tello = Tello()
# tello.connect()
# print(tello.get_battery())
# tello.streamon()
# tello.takeoff()
# tello.send_rc_control(0,0,20,0)
# time.sleep(2.2)
# Setting up Camera and window sie
with open('gesture_actions.json') as f:
  gesture_actions = json.load(f)
cap = cv2.VideoCapture(0)
cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
pTime = 0
x_ref = int(640/2)
y_ref = int(0.25*460)

#Create instance of poseDetector class
detector = pm.poseDetector()

# Drone Parameters
axis_speed = {}
pid_yaw = PID(0.25,0,0,setpoint=0,output_limits=(-100,100))
pid_throttle = PID(0.4,0,0,setpoint=0,output_limits=(-80,100))
fb =0

# F/w threshold
shoulderW_threshold= [150,200]

def calculate_yaw(x_nose):
    x_off = int(x_nose - x_ref)
    axis_speed["yaw"] = int (-pid_yaw(x_off))
    return x_off

def calculate_throttle(y_nose):
    y_off = int(y_ref -y_nose)
    axis_speed["throttle"] = int(-pid_throttle(y_off))
    return y_off

def calculate_Forward_backward(shoulderWidth):
  global fb
  if shoulderWidth > shoulderW_threshold[0] and shoulderWidth < shoulderW_threshold[1]:
    fb = 0
  if shoulderWidth > shoulderW_threshold[1]:
    fb = -20
  if shoulderWidth < shoulderW_threshold[0]:
    fb = 20
  
def printOnWindow(Img , Nose_x, Nose_y, shoulder_width):
  cv2.putText(Img, str("x,y coordinates"), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                  (255, 0, 0), 1)
  cv2.putText(Img,str(Nose_x) , (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
              (255, 0, 0), 1)
  cv2.putText(Img, "," ,(40, 100), cv2.FONT_HERSHEY_PLAIN, 1,
              (255, 0, 0), 1)
  cv2.putText(Img, str(Nose_y), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1,
              (255, 0, 0), 1)
  cv2.circle(Img, (x_ref, y_ref), 15, (250, 150, 0), 1, cv2.LINE_AA)
  cv2.arrowedLine(Img, (x_ref, y_ref), (Nose_x,Nose_y), (250, 150, 0), 6)
  cv2.putText(Img, str("Yaw and throttle"), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
              (0, 255, 255), 1)
  cv2.putText(Img, str(axis_speed["yaw"]), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
              (0, 255, 255), 1)
  cv2.putText(Img, str(axis_speed["throttle"]), (50 ,140), cv2.FONT_HERSHEY_PLAIN, 1,
              (0, 255, 255), 1)
  cv2.putText(Img, str("shoulderWidth"), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 0, 255), 1)
  cv2.putText(Img, str(shoulder_width), (140, 160), cv2.FONT_HERSHEY_PLAIN, 1,
              (0, 0, 255), 1)
  cv2.putText(Img, str(fb), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1,
              (255, 255, 255), 1)

def identifyAction(label):
  if label in gesture_actions:
      gesture_key = label
  if gesture_key:
      action = gesture_actions[gesture_key]
  return action

def performAction(action):
  if action == "action_for_flip_forward":
    tello.flip_forward()
    return True
  elif action == "action_for_flip_backward":
     tello.flip_back()
     return True
  elif action == "action_for_flip_right":
     tello.flip_right()
     return True
  elif action == "action_for_flip_left":
     tello.flip_left()
     return True
  elif action == "action_for_stop":
     tello.emergency()
     return True
  else:
     return False
   

while True:
  success, img = cap.read()
  # img = tello.get_frame_read().frame
  output_image, landmarks = detector.detectPose(img,detector.pose,display=False)

  if landmarks:
    _, gesture_str = detector.classifyPose(landmarks,output_image,True)
    action = identifyAction(gesture_str)
    # Performed = performAction(action)
    # if Performed:
    #    continue
    print(action)

    #Find coordinate of nose to calculate yaw and throttle
    x = detector.findNoseLandmark(landmarks, output_image)
    calculate_yaw(x[0])
    calculate_throttle(x[1])

    #Find the width of shoulder
    shoulderWidth = detector.calculate_width(landmarks,output_image)
    calculate_Forward_backward(shoulderWidth)

    printOnWindow(output_image, x[0],x[1], shoulderWidth)

  # tello.send_rc_control(0,fb, axis_speed["throttle"],axis_speed["yaw"])     
  cv2.imshow("Gesture Detection", output_image)
  k = cv2.waitKey(1)
  if (k==27):
    break

print(gesture_actions)
cap.release()
cv2.destroyWindow("Gesture Detection")