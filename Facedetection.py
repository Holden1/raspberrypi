import numpy as np
import io
import cv2
import time
import picamera
from picamera.array import PiRGBArray
from subprocess import call

camera=picamera.PiCamera()
# camera.start_preview()
# time.sleep(5)
# camera.stop_preview()
camera.hflip = True
camera.vflip = True
resX=320
resY=240
camera.resolution = (resX, resY)
camera.framerate = 30

min_servo_val=70
max_servo_val=200
servo_val=130

errorX=0
errorY=0

highResCap = PiRGBArray(camera)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

for foo in camera.capture_continuous(highResCap, format="bgr", use_video_port=True):
    print("In loop")
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
    #print("Before gray")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print("Before finding faces")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #print("num faces: ",len(faces))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        center=[x+(w/2),y+(h/2)]
        print("Center",center)
        errorX=center[0]-(resX/2) # we want to center on x-axis
        print("Error x",errorX)

        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    servo_val=np.clip(servo_val+errorX,min_servo_val,max_servo_val)
    call(["echo 2="+servo_val+" > /dev/servoblaster"],shell=False)
    print(time.clock()-start)
    #cv2.imshow('img',img)
    #cv2.waitKey(1)
    #cv2.destroyAllWindows()