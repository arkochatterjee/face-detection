# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:49:16 2017

@author: Arko Chatterjee
"""
#import cv2 and matplotlib
import cv2
import matplotlib.pyplot as plt

#used to read the image file
test1 = cv2.imread('F:\\PICTURES\\test.jpg')
gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray') 

#importing the Haar cascade classifier
haar_face_cascade = cv2.CascadeClassifier('F:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml')

#finding the faces
faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5);  
print('Faces found: ', len(faces))

for (x, y, w, h) in faces:     
         cv2.rectangle(test1, (x, y), (x+w, y+h), (0, 255, 0), 2) #drawing rectangle over the faces
         
         plt.imshow(test1)
       
    