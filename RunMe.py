import cv2
import time
from simple_pid import PID
import PoseModule as pm
from djitellopy import Tello
import json
import time
import pygame
import threading
from Morse_Camera import CameraMorse , RollingGraph

with open('gesture_actions1.json') as f:
  gesture_actions = json.load(f)

pygame.mixer.init()
forward_sound = pygame.mixer.Sound("Audio/Forward.wav")
backward_sound = pygame.mixer.Sound("Audio/Backwards.wav")
right_sound = pygame.mixer.Sound("Audio/Right.wav")
left_sound = pygame.mixer.Sound("Audio/Left.wav")
landing_sound = pygame.mixer.Sound("Audio/Land.wav")
picture_sound = pygame.mixer.Sound("Audio/Capturing_in_2.wav")
flip_sound = pygame.mixer.Sound("Audio/Flip.wav")
Enter_fixed_distance_mode_sound = pygame.mixer.Sound("Audio/EnteredFDM.wav")
Exit_fixed_distance_mode_sound = pygame.mixer.Sound("Audio/ExitedFDM.wav")
Palm_Land_sound = pygame.mixer.Sound("Audio/PalmLanding.wav")
takeoff_sound= pygame.mixer.Sound("Audio/Takeoff.wav")
# Mode
DroneMode = True
# Drone Parameters
axis_speed = {}
pid_yaw = PID(0.25,0,0,setpoint=0,output_limits=(-100,100))
pid_throttle = PID(0.4,0,0,setpoint=0,output_limits=(-80,100))
fb =0
gesture_mode = [0,0]
fixed_distance_bool = False
palm_landing=False
is_flying = False
last_gesture_time = 0
gesture_timeout = 2 
number = 0
last_time_executed = [0]*len(gesture_actions)
# Forward/backward threshold
shoulderW_threshold= [170,220]
LRFB_speed = 30
pTime = 0
x_ref = int(920/2)
y_ref = int(0.35*460)

if DroneMode:
  tello = Tello()
  tello.connect()
  print("battery=",tello.get_battery())
  tello.streamon()
  # tello.takeoff()
  # tello.send_rc_control(0,0,20,0)
  time.sleep(2)
else:
  cap = cv2.VideoCapture(0)

cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)

detector = pm.poseDetector()

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

def timeDifference(prev_executed_time,index):
  current_time = time.time()
  global last_time_executed
  if current_time - prev_executed_time >= gesture_timeout:
    last_time_executed[index] = current_time
    return True
  else:
    False

def printOnWindow(Img , Nose_x, Nose_y, shoulder_width,Distance_fixed):
  
  if Distance_fixed:
    
    # cv2.putText(Img, str("x,y coordinates"), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1,
    #                 (255, 0, 0), 1)
    # cv2.putText(Img,str(Nose_x) , (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
    #             (255, 0, 0), 1)
    # cv2.putText(Img, "," ,(40, 100), cv2.FONT_HERSHEY_PLAIN, 1,
    #             (255, 0, 0), 1)
    # cv2.putText(Img, str(Nose_y), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1,
    #             (255, 0, 0), 1)
    cv2.circle(Img, (x_ref, y_ref), 15, (250, 150, 0), 1, cv2.LINE_AA)
    cv2.arrowedLine(Img, (x_ref, y_ref), (Nose_x,Nose_y), (0,255, 255 ), 3)
    # cv2.putText(Img, str("Yaw and throttle"), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                # (0, 255, 255), 1)
    # cv2.putText(Img, str(axis_speed["yaw"]), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
    #             (0, 255, 255), 1)
    # cv2.putText(Img, str(axis_speed["throttle"]), (50 ,140), cv2.FONT_HERSHEY_PLAIN, 1,
    #             (0, 255, 255), 1)
    
    cv2.putText(Img, str("Fixed Distance mode:"), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 255), 1)
    cv2.putText(Img, str("On") if fixed_distance_bool else str("Off"), (200, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                (57, 255, 20), 1)
    if fb>0:
      cv2.putText(Img, str("Moving Close"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 1)
    if fb<0:
      cv2.putText(Img, str("Moving Away") , (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 1)
    cv2.putText(Img, str("shoulderWidth"), (10, 480), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 255), 1)
    cv2.putText(Img, str(shoulder_width), (140, 480), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0, 255), 1)
    # cv2.putText(Img, str("Fw/Bw:"), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1,
    #             (255, 200, 200), 1)
    # cv2.putText(Img, str(fb), (100, 200), cv2.FONT_HERSHEY_PLAIN, 1,
    #             (255, 200, 200), 1)
    # if gesture_mode[0] > 0:
    #   cv2.putText(Img, str("Forward:"), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    #   cv2.putText(Img, str(gesture_mode[0]), (100, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    # elif gesture_mode[0] <0:
    #   cv2.putText(Img, str("Backward:"), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    #   cv2.putText(Img, str(-gesture_mode[0]), (100, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    # if gesture_mode[1] > 0:
    #   cv2.putText(Img, str("Left"), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    #   cv2.putText(Img, str(-gesture_mode[1]), (100, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    # elif gesture_mode[1] <0:
    #   cv2.putText(Img, str("Right"), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    #   cv2.putText(Img, str(-gesture_mode[1]), (100, 220), cv2.FONT_HERSHEY_PLAIN, 1,
    #               (0, 255, 255), 1)
    
    
  else:
    cv2.putText(Img, str("Fixed Distance mode:"), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 255), 1)
    cv2.putText(Img, str("On") if fixed_distance_bool else str("Off"), (200, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                (57, 255, 20), 1)
    if gesture_mode[0] > 0:
      cv2.putText(Img, str("Forward:"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
      cv2.putText(Img, str(gesture_mode[0]), (100, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
    elif gesture_mode[0] <0:
      cv2.putText(Img, str("Backward:"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
      cv2.putText(Img, str(-gesture_mode[0]), (100, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
    if gesture_mode[1] > 0:
      cv2.putText(Img, str("Left "), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
      cv2.putText(Img, str(gesture_mode[1]), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
    elif gesture_mode[1] <0:
      cv2.putText(Img, str("Right "), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
      cv2.putText(Img, str(-gesture_mode[1]), (70, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                  (0, 255, 255), 1)
  #   cv2.putText(Img, str("x,y coordinates"), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1,
  #                   (255, 0, 0), 1)
  #   cv2.putText(Img,str(Nose_x) , (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (255, 0, 0), 1)
  #   cv2.putText(Img, "," ,(40, 100), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (255, 0, 0), 1)
  #   cv2.putText(Img, str(Nose_y), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (255, 0, 0), 1)
    cv2.circle(Img, (x_ref, y_ref), 15, (250, 150, 0), 1, cv2.LINE_AA)
    cv2.arrowedLine(Img, (x_ref, y_ref), (Nose_x,Nose_y), (0,255, 255 ), 3)
  #   cv2.putText(Img, str("Yaw and throttle"), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (0, 255, 255), 1)
  #   cv2.putText(Img, str(axis_speed["yaw"]), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (0, 255, 255), 1)
  #   cv2.putText(Img, str(axis_speed["throttle"]), (50 ,140), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (0, 255, 255), 1)
    
  #   cv2.putText(Img, str("Fix Dis mode:"), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (200, 10, 255), 1)
  #   cv2.putText(Img, str(fixed_distance_bool), (150, 160), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (200, 10, 255), 1)
    cv2.putText(Img, str("shoulderWidth"), (10, 480), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 255), 1)
    cv2.putText(Img, str(shoulder_width), (140, 480), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0, 255), 1)
  #   cv2.putText(Img, str("Fw/Bw:"), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (255, 200, 200), 1)
  #   cv2.putText(Img, str(fb), (100, 200), cv2.FONT_HERSHEY_PLAIN, 1,
  #               (255, 200, 200), 1)

def identifyAction(label):
  # print("Inside identify action", label)
  if label in gesture_actions:
      gesture_key = label
  if gesture_key:
      action = gesture_actions[gesture_key]

  # print(action)
  return action

def performAction(action, mode_bool):
  global fixed_distance_bool,number,palm_landing
  current_time = time.time()
  # print("fixed distance bool", fixed_distance_bool)
  # if not fixed_distance_bool:
  if action == "PalmLand":
    if timeDifference(last_time_executed[8],8):
      palm_landing=True
      Palm_Land_sound.play()
      
  if action == "Come_forward" and not fixed_distance_bool:
    if timeDifference(last_time_executed[0],0):
      forward_sound.play()
      gesture_mode[0] = LRFB_speed
      # print("Forward 10cm")
      return True
  elif action == "Go_backward" and not fixed_distance_bool:
    if timeDifference(last_time_executed[1],1):
      backward_sound.play()
      gesture_mode[0]=-LRFB_speed
      # print("Backward 10cm")
      return True
  elif action == "move_right" and not fixed_distance_bool:
    if timeDifference(last_time_executed[2],2):
      right_sound.play()
      gesture_mode[1]=-LRFB_speed
      # print("Right 10 cm")
      return True
  elif action == "move_left" and not fixed_distance_bool:
    if timeDifference(last_time_executed[3],3):
      left_sound.play()
      gesture_mode[1]=+LRFB_speed
      # print("Left 10 cm")
      return True
  elif action == "action_for_stop":
    if timeDifference(last_time_executed[4],4):
      landing_sound.play()
      if mode_bool:
        # landing_sound.play()
        tello.land()
      # print(" Landing") 
      return True
  elif action == "Click picture after 2s":
    # Implement click picture
    if timeDifference(last_time_executed[5],5):
      picture_sound.play()
      capture_event.set()
      cv2.imwrite("InstantPicture.png",img)
      # number+=1
      # print("picture number",number)
    return True
  
  elif action == "Performing a front Flip":
    if timeDifference(last_time_executed[6],6):
      flip_sound.play()
      if mode_bool:
        tello.flip_forward()
        print("Doing a front Flip")
        return True

  elif action == "Entering Fixed Distance Mode":
    if timeDifference(last_time_executed[7],7):
      fixed_distance_bool = not fixed_distance_bool 
      if fixed_distance_bool:
          Enter_fixed_distance_mode_sound.play()
          print("Entering fixed distance mode")
      else:
          Exit_fixed_distance_mode_sound.play()
          print("Exiting fixed distance mode")
    
      return True
  
  else:
    gesture_mode[0]=0
    gesture_mode[1]=0
    return False
  
def clickPicture():
  time.sleep(3)
  # ret,frame =cap.read()
  frame = tello.get_frame_read().frame
  frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  cv2.imwrite("DelayImage3sec.png",frame)
# click from test camera
def clickPictureCam():
  time.sleep(3)
  ret,frame =cap.read()
  # frame = tello.get_frame_read().frame
  # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  cv2.imwrite("DelayImage3Camera.png",frame)

def enable_takeoff_event():
  takeoff_event.set()
  print("Takeoff event enabled")

def delayed_takeoff():
  global is_flying
  takeoff_sound.play()
  print("Delayed Takeoff func")
  time.sleep(3)
  tello.takeoff()
  tello.move_up(70)
  print("Sleep for 1s now")
  time.sleep(1)
  is_flying = True
  # print("Set is flying to true")


def printerfunc1():
  print("takeoff")
def printerfunc2():
  print("throw")

capture_event = threading.Event()  
takeoff_event = threading.Event()
morse = CameraMorse(display=True)
morse.define_command("---",enable_takeoff_event)
# morse.define_command("...",tello.initiate_throw_takeoff())

while DroneMode:
  # print("Indronemode loop")
  img = tello.get_frame_read().frame
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  output_image, landmarks = detector.detectPose(img,detector.pose,display=False)
  # print("Drone flying?",is_flying)
  if not is_flying:
    pressing, detected = morse.eval(img)
    if morse.is_pressed and not pressing:
      pass
    elif not morse.is_pressed and pressing:
      print("Code detected",morse.code)
    elif takeoff_event.is_set():
      takeoff_thread = threading.Thread(target = delayed_takeoff)
      takeoff_thread.start()
      # tello.send_rc_control(0,-20,40,0)
      takeoff_event.clear()
      # is_flying = True
      
  elif is_flying:
    
    # output_image, landmarks = detector.detectPose(img,detector.pose,display=False)

    if landmarks:
      _, gesture_str = detector.classifyPose(landmarks,output_image,True)
      
      #Find coordinate of nose to calculate yaw and throttle
      x = detector.findNoseLandmark(landmarks, output_image)
      calculate_yaw(x[0])
      calculate_throttle(x[1])
      shoulderWidth = detector.calculate_width(landmarks,output_image)
      
      # Find the action user has set from the JSON data
      action = identifyAction(gesture_str)

      # Perform that action
      Performed = performAction(action,DroneMode)
      
      # if palm_landing:
      #   print("in palm landing if")
      #   print(shoulderWidth)
      #   if shoulderWidth < 500:
      #     print("Shoulder width less so moving forward")
      #     print("throttle=", axis_speed["throttle"])
      #     # tello.send_rc_control(0,10,axis_speed["throttle"],axis_speed["Yaw"])
          
      #   if shoulderWidth>530:
      #     print("landing now")
      #     tello.land()
          
      #     # is_flying= False
      #     # break


      # else:
      if capture_event.is_set():
        capture_thread = threading.Thread(target=clickPicture)
        capture_thread.start()
        capture_event.clear()

        
      # Check whether fixed distance mode is on
      if fixed_distance_bool:
        #Find the width of shoulder
        # shoulderWidth = detector.calculate_width(landmarks,output_image)
        calculate_Forward_backward(shoulderWidth)
        # printOnWindow(output_image, x[0],x[1], shoulderWidth,fixed_distance_bool)
        tello.send_rc_control(0,fb, axis_speed["throttle"],axis_speed["yaw"])
      elif not fixed_distance_bool:
        if not palm_landing:
          tello.send_rc_control(gesture_mode[1],gesture_mode[0], axis_speed["throttle"],axis_speed["yaw"])
        # print("Not in fixed distance mode")
        if palm_landing:
          tello.send_rc_control(0,20, axis_speed["throttle"],axis_speed["yaw"])
          if shoulderWidth>500:
            tello.land()
            break

      # Print on Window
      printOnWindow(output_image, x[0],x[1], shoulderWidth,fixed_distance_bool)
    
    # else:
    #   # Explore to detect person
    #   tello.send_rc_control(0,0,0,20)
  
  cv2.imshow("Gesture Detection", output_image)
  k = cv2.waitKey(1)
  # Press esc for emergency
  if (k==27):
    print("Motors Off")
    tello.emergency()
    is_flying=False
    break
  # Press l for landing
  if (k==108):
    print("Landing Now")
    tello.land()
    is_flying=False
    break
while not DroneMode:
  success, img = cap.read()
  output_image, landmarks = detector.detectPose(img,detector.pose,display=False)

  # if not is_flying:
  #   pressing, detected = morse.eval(img)
  #   if morse.is_pressed and not pressing:
  #     pass
  #   elif not morse.is_pressed and pressing:
  #     print("Code detected",morse.code)
  #   elif takeoff_event.is_set():
  #     takeoff_thread = threading.Thread(target = delayed_takeoff)
  #     # takeoff_thread.start()
  #     # tello.send_rc_control(0,-20,40,0)
  #     takeoff_event.clear()
  #     is_flying = True

  if  not is_flying:
    # print("Is  flying = ",is_flying)
    if landmarks:
      _, gesture_str = detector.classifyPose(landmarks,output_image,True)
      
      #Find coordinate of nose to calculate yaw and throttle
      x = detector.findNoseLandmark(landmarks, output_image)
      calculate_yaw(x[0])
      calculate_throttle(x[1])
      shoulderWidth = detector.calculate_width(landmarks,output_image)
      
      # Find the action user has set from the JSON data
      action = identifyAction(gesture_str)
      
      # Perform that action
      Performed = performAction(action,DroneMode)
     
      if palm_landing:
          
          if shoulderWidth < 300:
            # tello.send_rc_control(0,10,0,axis_speed["Yaw"])
            print("Shoulder width less so moving forward ,sending rc command")
          if shoulderWidth>400:
            # tello.land()
            # is_flying= False
            print("landing")
            # break

      else:
        print("Not in palm landing")
        if capture_event.is_set():
          capture_thread = threading.Thread(target=clickPictureCam)
          capture_thread.start()
          capture_event.clear()

        # Check whether fixed distance mode is on 
        if fixed_distance_bool:
          calculate_Forward_backward(shoulderWidth)
          # printOnWindow(output_image, x[0],x[1], shoulderWidth,fixed_distance_bool)
            
        elif not fixed_distance_bool:
          # printOnWindow(output_image, x[0],x[1], shoulderWidth, fixed_distance_bool)
          print("Parameters passed into send_rc_control l/r and f/b=",gesture_mode[1],gesture_mode[0])
          pass
        
      printOnWindow(output_image, x[0],x[1], shoulderWidth, fixed_distance_bool)
  else:
    print("exploring")
  
  cv2.imshow("Gesture Detection", output_image)
  k = cv2.waitKey(1)
  if (k==27):
    print("Motors off")
    break
  if (k==108):
    print("Landing Now")
    break

cap.release()
# tello.land()
cv2.destroyWindow("Gesture Detection")