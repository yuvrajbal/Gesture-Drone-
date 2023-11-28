import cv2
import numpy as np
w,h = 360,240
fbRange =[6200,6800]
pid=[0.4,0.4,0]
pError=0

def FindFace(img):
    # Load the pre-trained Haar Cascade face detector
    face_cascade = cv2.CascadeClassifier("../Images/haarcascade-frontalface-default.xml")
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    myFacelist_area=[]
    myFacelist_center=[]
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=8)
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx  = x+ w//2
        cy = y+ h//2
        area=w*h
        cv2.circle(img,(cx,cy),5,(0,255,0),cv2.FILLED)
        myFacelist_area.append(area)
        myFacelist_center.append([cx,cy])
        #print(len(myFacelist_area))
    if len(myFacelist_area)!=0:
        indx=myFacelist_area.index(max(myFacelist_area))
        return img , [myFacelist_center[indx], myFacelist_area[indx]]
    else:
        return img, [[0,0],0]

def trackFace( info, w, pid, pError):
    area = info[1]
    x,y = info[0]
    fb=0
    error= x - w//2
    speed=pid[0]*error + pid[1]*(error-pError)
    speed = int(np.clip(speed,-100,100))

    if area > fbRange[0] and area < fbRange[1]:
        fb=0
    if area > fbRange[1]:
        fb=-20
    elif area < fbRange[0] and area != 0:
        fb=20
    if x==0:
        speed=0
        error=0


    print(speed,fb)
    #me.send_rc_control(0,fb,0,speed)
    return error

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (you can change it to 1, 2, etc., if you have multiple cameras)
while True:
    ret, frame = cap.read()
    frame=cv2.resize(frame,(w,h))
    frames,info=FindFace(frame)
    pError= trackFace(info,w,pid,pError)
    #print('center',info[0],'area',info[1] )
    cv2.imshow('Face Detection', frames)
    cv2.waitKey(2)



