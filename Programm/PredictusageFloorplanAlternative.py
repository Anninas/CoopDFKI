# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:31:18 2020

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

print("Starting prediction loop...")

for y in range(floorplan.shape[0]-100):
    for x in range(floorplan.shape[1]-100):
        
        curr_annot = floorplan[y:(y+100), x:(x+100)]
        annotations.append(curr_annot)

start = time.time()
print("Timer started")                  
#print("Predicting now...")
#predict takes np.array, no list!
prediction = net.predict_on_batch(np.array(annotations))
#print("Prediction done")
end = time.time()

print("Time for prediction loop: {}".format(end-start))

for i in range(prediction.shape[0]):
    x_i = ((0 + i)%(floorplan.shape[1]-100))+50 
    y_i = 0 + ((0 + i)//(floorplan.shape[1]-100))+50
    
    predictions[y_i][x_i]=prediction[i]
    #print("current prediction = {}".format(prediction[i]))
    #print("saved current prediction = {}".format(predictions[y_i][x_i]))

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