import numpy as np
import cv2
import time
import picamera
from picamera.array import PiRGBArray

camera=picamera.PiCamera()
# camera.start_preview()
# time.sleep(5)
# camera.stop_preview()
camera.hflip = True
camera.vflip = True
camera.resolution = (1024, 768)
camera.framerate = 30


highResCap = PiRGBArray(camera)
highResStream = camera.capture_continuous(highResCap, format="bgr", use_video_port=True)
time.sleep(2.0)
print("done warming up")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while True:
    start=time.clock()
    #cv2.imshow("preview", frame)
    hrs = highResStream.__next__()
    img=hrs.array
    highResStream.truncate(0)
    #img = cv2.imread('image.jpg')


    r = 1000.0 / img.shape[1]
    dim = (1000, int(img.shape[0] * r))
    img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = gray[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    print(time.clock()-start)
    cv2.imshow('img',img)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()