# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:19:01 2020

@author: anni
"""

import json
from os import listdir
from os.path import join
import os
import keras
import tensorflow as tf
import numpy as np
import PIL
import PIL.ImageOps
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
import cv2
import random
import imutils
import pickle
import sklearn.metrics as metric
from itertools import cycle
import pandas as pd
import time

script_dir = os.path.dirname(__file__)

floorplan_name = str(input("Which file from the image folder do you want to open?"))
floorplan_path = os.path.join(script_dir, "../Flurplandaten/images/{}".format(floorplan_name))

floorplan = np.array(PIL.Image.open(floorplan_path).convert('L').convert('RGB'))

net1_path = os.path.join(script_dir, "../Netze/try13_IRV2_binary.h5")
print("Loading model 1 now...")
net1 = keras.models.load_model(net1_path)
print("Loading model 1 done")

net2_path = os.path.join(script_dir, "../Netze/try10_IncResV2_randomrotation.h5")
print("Loading model 2 now...")
net2 = keras.models.load_model(net2_path)
print("Loading model 2 done")



predictions = np.empty((floorplan.shape[0], floorplan.shape[1], 2))
annotations = []
object_annotations = []
object_x_list = []
object_y_list = []

batch_size1 = 128

counter = 0
#large_counter = ((floorplan.shape[0]-100) * (floorplan.shape[1]-100))/batch_size

print("Starting prediction loop...")

start = time.time()
print("Timer started")

#prediction binary
for y in range(floorplan.shape[0]-100):
    for x in range(floorplan.shape[1]-100):
    
        #get initial x and y for this batch
        if(counter == 0):
            initial_x = x
            initial_y = y
        
        #get current annotations
        curr_annot = floorplan[y:(y+100), x:(x+100)]
        annotations.append(curr_annot)
        
        counter += 1
        
        #batch size or edge of image reached
        if((counter == batch_size1) or ((x == floorplan.shape[1]-101) and (y == floorplan.shape[0]-101))):
            
            #print("Predicting now...")
            #predict takes np.array, no list!
            prediction = net1.predict_on_batch(np.array(annotations))
            #print("Prediction done")
            
            for i in range(prediction.shape[0]):
                x_i = ((initial_x + i)%(floorplan.shape[1]-100))+50 
                y_i = initial_y + ((initial_x + i)//(floorplan.shape[1]-100))+50
                
                predictions[y_i][x_i]=prediction[i]
            
            
            '''
            for i in range(prediction.shape[0]):
                
                #if object can be seen on this
                if(prediction[i][0]>prediction[i][1]):
                    
                    #save the annotation
                    object_annotations.append(annotations[i])
                    #save their x and y coordinates
                    object_x_list.append(((initial_x + i)%(floorplan.shape[1]-100))+50)
                    object_y_list.append(initial_y + ((initial_x + i)//(floorplan.shape[1]-100))+50)
                else:
                    print("Background recognized")
                #x_i = ((initial_x + i)%(floorplan.shape[1]-100))+50 
                #y_i = initial_y + ((initial_x + i)//(floorplan.shape[1]-100))+50
                
                #predictions[y_i][x_i]=prediction[i]
                #print("current prediction = {}".format(prediction[i]))
                #print("saved current prediction = {}".format(predictions[y_i][x_i]))
            
            #large_counter -= 1
            
            #print("{} steps left".format(large_counter))
            '''
            counter = 0
            annotations.clear()
            
threshold = 0.8
#Replace all prediction values below threshold by 0
predictions[predictions < threshold] = 0

#Place to save the predictions class-wise as images and arrays
class_predictions = {}
class_predictions_arrays = {}
 
#Get prediction as images and arrays
for i in range(predictions.shape[2]):
    
    #Load floorplan, here to overwrite it each time
    floorplan1 = PIL.Image.open(os.path.join(script_dir, "../Flurplandaten/images/001.png")).convert('L').convert('RGB')
    
    #Get current prediction and expand dimensions to image
    current_prediction = predictions[:, :, i]
    current_prediction1 = np.expand_dims(current_prediction, axis = 2)
    current_prediction2 = np.insert(current_prediction1, 0, np.ones((820, 990)), axis = 2)
    current_prediction3 = np.insert(current_prediction2, 1, np.zeros((820, 990)), axis = 2)
    current_prediction4 = np.insert(current_prediction3, 2, np.zeros((820, 990)), axis = 2)
    
    #Get image values from prediction
    current_prediction5 = np.multiply(current_prediction4, 255)
    
    #Save prediction as array
    class_predictions_arrays["class"+str(i)] = current_prediction
    
    #Save prediction as Image
    image_name = "class{}.png".format(i)
    class_predictions["class"+str(i)] = PIL.Image.fromarray((current_prediction4 * 255).astype('uint8'), mode='RGBA')
    
    #Paste prediction image on top of floorplan for evaluation
    floorplan1.paste(class_predictions["class"+str(i)], (0,0), class_predictions["class"+str(i)])
    floorplan1.save(os.path.join(script_dir, "./KontrollbilderBinaryTest/{}".format(image_name)))

