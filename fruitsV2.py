#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:51:36 2020

@author: Sachit
"""


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import AveragePooling2D
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing import image

#TRAINING THE MODEL
   
model = Sequential([
       Conv2D(32,kernel_size=(3,3), activation= 'relu',input_shape = (20,20,3))
       ,MaxPooling2D((2,2))
       ,Conv2D(64,kernel_size=(3,3),activation='relu')
       ,MaxPooling2D((2,2))
       ,Flatten()
       ,Dense(128,activation='relu')
       ,Dense(6,activation='softmax')])

#model.summary()

model.compile(loss="categorical_crossentropy", optimizer= 'adam',
	metrics=["accuracy"])

SOURCE_DIR = r'../dataset/train' #DIRECTORY WHERE THE TRAINING IMAGES ARE STORED

labels = os.listdir(SOURCE_DIR)
print(labels)
EPOCHS =8
train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split


train_generator = train_datagen.flow_from_directory(
    SOURCE_DIR,
    batch_size=25,
    target_size=(20,20),
    classes=labels,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    SOURCE_DIR, # same directory as training data
    batch_size=25,
    classes=labels,
    target_size=(20,20),
    subset='validation') # set as validation data

H=model.fit_generator(
    train_generator,
    validation_data = validation_generator, 
    epochs = EPOCHS)

#PLOTTING THE TRAINING ACCURACY AND LOSS GRAPHS

# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy on Fresh vs Rotten Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")


path = '../dataset/test/rottenapples/rotated_by_15_Screen Shot 2018-06-07 at 2.34.18 PM.png' #THIS IS THE DIRECTORY AND FILE FROM WHERE THE INPUT IMAGE (FROM CAMERA?) CAN BE FEEDED IN THE MODEL.


#print(classes)

#FUNCTION WHICH RETURNS BOOLEAN VALUE.
def Fresh_or_Rotten(path):
	img = image.load_img(path, target_size=(20, 20))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	images = np.vstack([x])
	classes = model.predict(images)
	if (classes[0][1] or classes[0][3] or classes[0][4])==1:
		check= True
	else:
	        check= False
	if check==True:
    		print("Fruit is Rotten.")
	else:
    		print("Fruit is fresh.")
    
check=Fresh_or_Rotten(path)

