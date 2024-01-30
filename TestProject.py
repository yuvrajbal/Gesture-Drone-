import cv2
import time
from simple_pid import PID
import PoseModule as pm
from djitellopy import Tello

cap = cv2.VideoCapture(0)
cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
pTime = 0
detector = pm.poseDetector()
x_ref = int(640/2)
y_ref = int(0.25*460)

# Drone Parameters
axis_speed = {}
pid_yaw = PID(0.25,0,0,setpoint=0,output_limits=(-100,100))
pid_throttle = PID(0.4,0,0,setpoint=0,output_limits=(-80,100))
shoulder_threshHold= [100,200]
fb =0
while True:
    success, img = cap.read()
    output_image, landmarks = detector.detectPose(img,detector.pose,display=False)
    if landmarks:
        detector.classifyPose(landmarks,output_image,True)
        detector.findLandmark(landmarks,output_image)

        #Find coordinate of nose to calculate yaw and throttle
        x = detector.findLandmark(landmarks, output_image)
        x_nose = x[0]
        y_nose = x[1]
        x_off = int(x_nose - x_ref)
        y_off = int( y_ref- y_nose )
        axis_speed["yaw"] = int(-pid_yaw(x_off))
        axis_speed["throttle"] = int(-pid_throttle(y_off))

        #Find the width of shoulder
        shoulderWidth = detector.calculate_width(landmarks,output_image)

        if shoulderWidth > shoulder_threshHold[0] and shoulderWidth < shoulder_threshHold[1]:
            fb = 0
        if shoulderWidth > 200:
            fb =-20
        if shoulderWidth < 100:
            fb =20

        cv2.putText(output_image, str("x,y coordinates"), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1)
        cv2.putText(output_image,str(x_nose) , (10, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1)
        cv2.putText(output_image, "," ,(40, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1)
        cv2.putText(output_image, str(y_nose), (50, 100), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 0, 0), 1)
        cv2.circle(output_image, (x_ref, y_ref), 15, (250, 150, 0), 1, cv2.LINE_AA)
        cv2.arrowedLine(output_image, (x_ref, y_ref), (x_nose,y_nose), (250, 150, 0), 6)
        cv2.putText(output_image, str("Yaw and throttle"), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 1)
        cv2.putText(output_image, str(axis_speed["yaw"]), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 1)
        cv2.putText(output_image, str(axis_speed["throttle"]), (50 ,140), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 255, 255), 1)
        cv2.putText(output_image, str(shoulderWidth), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 1)
        cv2.putText(output_image, str(fb), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
    #fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime


    cv2.putText(output_image, str(int(fps)), (500, 30), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0 , 0), 1)

    # tello.send_rc_control(0,, axis_speed["throttle"],axis_speed["yaw"])
    cv2.imshow("Gesture Detection", output_image)
    k = cv2.waitKey(1)
    # Check is escape is pressed
    if (k==27):
        break
cap.release()
cv2.destroyWindow("Gesture Detection")