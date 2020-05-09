import cv2, time
import os
import numpy as np
from matplotlib import pyplot as plt

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# 0 is for externel camera
video= cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not video.isOpened():
    raise IOError("Cannot open webcam")

path=os.path.join(os.getcwd() , 'images\\')
a=0
n=0
while True:
   
    a=a+1
    ret, frame= video.read() 

#converting to grey scale
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    # for (x,y,w,h) in faces:
        # print(x,y,w,h)
        # pic_interested_gray=gray[y-(h):y+(h*2), x:x+(w)]
        # img="my-img5.jpg"
        # cv2.imwrite(img,pic_interested_gray)
    # resize image
    
    print (ret)
    print(frame)
    # print(resized)
    cv2.imshow("capturing", gray)
    plt.figure()
    # cv2.imshow("capturing", resized)
    # cv2.waitKey(0)
    #for video streaming
    key=cv2.waitKey(0)
    if key== ord('c'):
        for (x,y,w,h) in faces:
            print(x,y,w,h)
            # pic_interested_gray=gray[int(y-0.25*h):int(y+h*1.25), int(x-0.25*w):int(x+w*1.25)]
            pic_interested_gray=gray[y:y+h, x:x+w]
            dim = (48, 48)
            resized = cv2.resize(pic_interested_gray, dim, interpolation = cv2.INTER_AREA)
            img="img"+str(n)+".jpg"
            n=n+1
            cv2.imwrite(img,resized)
    key=cv2.waitKey(0)    
    if key== ord('x'):
        break

print (a)

#shutdown the camera
video.release()
cv2.destroyAllWindows()




   
    
    








# img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()