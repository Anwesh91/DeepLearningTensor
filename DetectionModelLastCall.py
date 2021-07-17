# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:29:51 2021

@author: AnnA
"""

from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image
model = load_model('FaceDetectionModel.h5')
print("MODEL LOADED...")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    print(" FACE EXTRACTION ...")    

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,250,250),2)
        cropped_face = img[y:y+h, x:x+w]
    print("RETURING CROPPED FACE")    
    return cropped_face

name_mapper={0:'Ashish',1:'Anwesh',2:'Aniket',3:'Suraj',4:'Avvishek',5:'Banti',6:'Tushar'}
video_capture = cv2.VideoCapture(0)
for i in range(1,500):
    _, frame = video_capture.read()
 
    
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (250, 250))
        im = Image.fromarray(face, 'RGB')

        img_array = np.array(im)
                    #Our keras model used a 4D tensor
                
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print("Prediction--",pred)
                     
        name="None matching"

        name=np.argmax(pred[0])
        cv2.putText(frame,name_mapper[name], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1)==15:
        break
video_capture.release()
cv2.destroyAllWindows()