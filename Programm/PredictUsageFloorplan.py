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

script_dir = os.path.dirname(__file__)

floorplan_name = str(input("Which file from the image folder do you want to open?"))
floorplan_path = os.path.join(script_dir, "../Flurplandaten/images/{}".format(floorplan_name))

floorplan = np.array(PIL.Image.open(floorplan_path).convert('L').convert('RGB'))

net_path = os.path.join(script_dir, "../Netze/try10_IncResV2_randomrotation.h5")
net = keras.models.load_model(net_path)


predictions = np.empty((floorplan.shape[0], floorplan.shape[1], 11))
annotations = []
counter = 0

for y in range(floorplan.shape[0]-100):
    for x in range(floorplan.shape[1]-100):
    
        
        if(counter == 0):
            initial_x = x
            initial_y = y
        
        curr_annot = floorplan[y:(y+100), x:(x+100)]
        annotations.append(curr_annot)
        
        counter += 1
        
        if(counter == 1024):
            
            #predict takes np.array, no list!
            prediction = np.array(net.predict(np.array(annotations))[0])
            
            for i in range(len(prediction)):
                x_i = ((initial_x + i)%(floorplan.shape[1]-100))+50
                y_i = initial_y + ((initial_x + i)//(floorplan.shape[0]-100))+50
                
                predictions[y_i][x_i]=prediction[i]
            
            counter = 0
            annotations.clear()
        


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