import numpy as np
import io
import cv2
import time
import picamera
from picamera.array import PiRGBArray
import os
from subprocess import call

camera=picamera.PiCamera()
# camera.start_preview()
# time.sleep(5)
# camera.stop_preview()
camera.hflip = True
camera.vflip = True
resX=1024
resY=768
camera.resolution = (resX, resY)
camera.framerate = 10

min_servo_val=70
max_servo_val=200
servo_val=130

errorX=0
errorXSum=0
errorY=0

kP=0.2
kI=0.0
kD=0

highResCap = PiRGBArray(camera)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

roi_gray=None
roi=None
template_threshold=0.9

for foo in camera.capture_continuous(highResCap, format="bgr", use_video_port=True):
    start=time.clock()
    stream = io.BytesIO()
    # capture into stream
    #camera.capture(highResCap, format='bgr', use_video_port=True)
    # convert image into numpy array
    img=highResCap.array

    highResCap.truncate()
    highResCap.seek(0)
    # print("after to img")
    # r = 1000.0 / img.shape[1]
    # dim = (1000, int(img.shape[0] * r))
    # print("Before resize")
    #
    # img=cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    ##template matching?
    if len(faces)==0:
        if roi_gray is not None:
            print("template matching")
            # Apply template Matching
            res = cv2.matchTemplate(gray, roi_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left=max_loc
            if(np.any(res>max_val)):
                faces=[[top_left[0],top_left[1],top_left[0] + roi[2], top_left[1] + roi[3]]]


    errorX=0
    #print("num faces: ",len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi=[x,y,w,h]
        # roi_color = gray[y:y+h, x:x+w]
        center=[x+(w/2),y+(h/2)]
        print("Center",center)
        errorX=(resX/2)-center[0] # we want to center on x-axis
        #print("Error x",errorX)

        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow("bla",img)
    cv2.waitKey(1)
    errorXSum+=errorX
    servo_val=servo_val+(errorX*kP)+(errorXSum*kI)
    servo_val=np.clip(servo_val,min_servo_val,max_servo_val)
    os.system("echo"+" 2="+str(servo_val)+" > /dev/servoblaster")
    print(time.clock()-start)
    #cv2.imshow('img',img)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()