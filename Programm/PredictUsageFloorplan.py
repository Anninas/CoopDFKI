# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 17:51:44 2020

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

net_path = os.path.join(script_dir, "../Netze/try10_IncResV2_randomrotation.h5")
print("Loading model now...")
net = keras.models.load_model(net_path)
print("Loading model done")

predictions = np.empty((floorplan.shape[0], floorplan.shape[1], 11))
annotations = []

batch_size = 2048

counter = 0
large_counter = ((floorplan.shape[0]-100) * (floorplan.shape[1]-100))/batch_size

print("Starting prediction loop...")

start = time.time()
print("Timer started")

for y in range(floorplan.shape[0]-100):
    for x in range(floorplan.shape[1]-100):
    
        
        if(counter == 0):
            initial_x = x
            initial_y = y
        
        curr_annot = floorplan[y:(y+100), x:(x+100)]
        annotations.append(curr_annot)
        
        counter += 1
        
        if((counter == batch_size) or ((x == floorplan.shape[1]-101) and (y == floorplan.shape[0]-101))):
            
            #print("Predicting now...")
            #predict takes np.array, no list!
            prediction = net.predict_on_batch(np.array(annotations))
            #print("Prediction done")
            for i in range(prediction.shape[0]):
                x_i = ((initial_x + i)%(floorplan.shape[1]-100))+50 
                y_i = initial_y + ((initial_x + i)//(floorplan.shape[1]-100))+50
                
                predictions[y_i][x_i]=prediction[i]
                #print("current prediction = {}".format(prediction[i]))
                #print("saved current prediction = {}".format(predictions[y_i][x_i]))
            
            large_counter -= 1
            
            #print("{} steps left".format(large_counter))
            
            counter = 0
            annotations.clear()
        

end = time.time()

print("Time for prediction loop: {}".format(end-start))

with open('predictions.json', 'w')as file:
    json.dump(predictions.tolist(), file)
    
np.save('predictions.npy', predictions)
#started 15:01



















'''
%time curr_annot = np.expand_dims(curr_annot, axis=0)
Wall time: 0 ns

%time prediction = np.array(net.predict(curr_annot)[0])
Wall time: 46 ms

with tf.device('/CPU:0'):
    %time prediction = np.array(net.predict(curr_annot)[0])
    
Wall time: 43 ms

with tf.device('/GPU:0'):
    %time prediction = np.array(net.predict(curr_annot)[0])
    
Wall time: 45 ms
'''