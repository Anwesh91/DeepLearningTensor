# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:01:27 2021

@author: AnnA
"""
import tensorflow as tf
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
import keras.preprocessing
import tensorflow.keras.preprocessing.image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator




print("Imported Required Library")
Fix_Img_Sizee = [250,250]
train_path = 'Dataset/Train'
test_path = 'Dataset/Test'

vgg = VGG16(input_shape=Fix_Img_Sizee + [3], weights= 'imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False
    
folders = glob('Dataset/Train/*')

x = Flatten()(vgg.output)
x = Dense(1000,activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)
model= Model(inputs=vgg.input, outputs=prediction)
print("Model Summary-----------")
model.summary()
model.compile(
    loss='categorical_crossentropy',
    optimizer= 'adam',
    metrics=['accuracy'])

train_D = ImageDataGenerator(rescale = 1./255,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True)

test_D = ImageDataGenerator(rescale=1./255)

training= train_D.flow_from_directory('Dataset/Train',
                                       target_size = (250,250),
                                       batch_size = 32,
                                       class_mode='categorical')

testing= test_D.flow_from_directory('Dataset/Test',
                                       target_size = (250,250),
                                       batch_size = 32,
                                       class_mode='categorical')
print("Image Transformation-------")

r = model.fit_generator(
    training,validation_data = testing,
    epochs=10, steps_per_epoch=len(training),
    validation_steps=len(testing))

print("Model Training is Done...Let's Detect your people-----------")

model.save('FaceDetectionModel.h5')