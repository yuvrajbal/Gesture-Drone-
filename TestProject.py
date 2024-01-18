import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture(0)
cv2.namedWindow('Gesture Detection', cv2.WINDOW_NORMAL)
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()

    output_image, landmarks = detector.detectPose(img,detector.pose,display=False)
    if landmarks:
        detector.classifyPose(landmarks,output_image,True)

    #fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(output_image, str(int(fps)), (500, 30), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0 , 0), 1)

    cv2.imshow("Gesture Detection", output_image)
    k = cv2.waitKey(1)
    # Check is escape is presed
    if (k==27):
        break
cap.release()
cv2.destroyWindow("Gesture Detection")