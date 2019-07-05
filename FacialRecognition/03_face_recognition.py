import cv2
import numpy as np
import os 
import time
import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX


id = 0


names = ['None', 'Vlad', 'Sergey', 'Alex', 'Ksenia', '5'] 
    

cam = cv2.VideoCapture(0)
cam.set(3, 1280) 
cam.set(4, 720) 


minW = 0.2*cam.get(3)
minH = 0.2*cam.get(4)

while True:

    today = datetime.datetime.today()
    #time.sleep(0.01)

    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 2,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])


        if (confidence < 100):
            id = names[id]
            int_confidence = 100 - confidence
            confidence = "  {0}%".format(round(100 - confidence))
            
            if (int_confidence > 42):
                print(today.strftime("%Y-%m-%d-%H.%M.%S") + "     [USER] " + id + "       [PROBABILITY] " + str(round(int_confidence)) + " %")
            else:
                print(today.strftime("%Y-%m-%d-%H.%M.%S") + "     [USER] UNKNOWN" + "     [PROBABILITY] " + str(round(int_confidence)) + " %")  
        
        else:
            id = names[0]
            confidence = "  {0}%".format(round(100 - confidence))
           
        
        cv2.putText(img, str(), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break


print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
