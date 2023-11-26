import cv2

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



# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera (you can change it to 1, 2, etc., if you have multiple cameras)
while True:
    ret, frame = cap.read()
    frames,info=FindFace(frame)
    print('center',info[0],'area',info[1] )
    cv2.imshow('Face Detection', frames)
    cv2.waitKey(2)



