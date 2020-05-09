import cv2, time
import os
import numpy as np
from matplotlib import pyplot as plt

#haarcascading of the frontal face
face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# # 0 is for externel camera
img= cv2.imread('prof5.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', img)
cv2.waitKey(1)
#detecting face from the image
faces=face_cascade.detectMultiScale(img, scaleFactor=1.2,minNeighbors=2)
print(faces)
for (x,y,w,h) in faces:
    print(x,y,w,h)
    img_int=img[y:y+h, x:x+w]
    dim = (48, 48)
    resized = cv2.resize(img_int, dim)
    print(resized)
    cv2.imwrite("prof5crp.jpg",resized)
    
cv2.destroyAllWindows()
