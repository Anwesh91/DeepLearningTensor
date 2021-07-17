# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 00:52:58 2021

@author: AnnA
"""

#Collection of Data
#Create and Train Model
#Predection

import cv2
import numpy as np
import pandas as pd
import os

Face_Classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#init webcam
caputer_vid = cv2.VideoCapture(0)
count = 0



def face_extractor(img):
    faces = Face_Classifier.detectMultiScale(img,1.3,5)
    
    if faces is():   #this extractor will look if there is face or not in Image
        return None
    
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        crop_face = img[y:y+h+50,x:x+w+50]
    return crop_face

name = input("Enter Name :  ")
file_name = name +str(count)+'.jpg'

os.chdir('H:\\FaceDetection\\Dataset\\Test')
os.mkdir(name)

os.chdir('H:\\FaceDetection\\Dataset\\Train')
os.mkdir(name)
os.chdir('H:\\FaceDetection\\Dataset\\Train\\'+name)

while True:
    re,frame=caputer_vid.read()
    if face_extractor(frame) is not None:
       count+=1 
       face=cv2.resize(face_extractor(frame),(400,400))
       file_n = file_name +str(count)+'.jpg'
       cv2.imwrite(file_n,face)
       cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(250,0,0),2)
       cv2.imshow('Face Cropping', face)
       
      
    else:
        print("Face Not Found Please try again")
        pass
   
    
    
    if cv2.waitKey(1) == 13 or count ==80:
        print("total Train image count is : ",count)
#if cv2.waitKey(1) == 13 or count << 120:
        os.chdir('H:\\FaceDetection\\Dataset\\Test\\'+name)
        count+=1 
        face=cv2.resize(face_extractor(frame),(400,400))
        file_n = file_name +str(count)+'.jpg'
        cv2.imwrite(file_n,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(250,2,2),2)
        cv2.imshow('Face Cropping', face)    
        
    if cv2.waitKey(1) == 13 or count == 120: 
        print('We can Stop')
        break
print ('total Test and Train Image count : ', count)
    
caputer_vid.release()
cv2.destroyAllWindows()
