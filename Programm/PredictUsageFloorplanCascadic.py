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

net1_path = os.path.join(script_dir, "../Netze/try21_IRV2_binaryOne_256-0.0001-33.h5")
print("Loading model 1 now...")
net1 = keras.models.load_model(net1_path)
print("Loading model 1 done")

net2_path = os.path.join(script_dir, "../Netze/try10_IncResV2_randomrotation.h5")
print("Loading model 2 now...")
net2 = keras.models.load_model(net2_path)
print("Loading model 2 done")



predictions = np.empty((floorplan.shape[0], floorplan.shape[1], 11))
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
                
                #if object can be seen on this
                if(prediction[i][0]>0.8):
                    
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
            
            counter = 0
            annotations.clear()


#predicting objects
object_annotation_batch = []
batch_counter = 0    

batch_size2 = 128

large_counter2 = len(object_annotations)
#all annotations that show objects according to other net
for annotation in object_annotations:
    
    #collect batch
    object_annotation_batch.append(annotation)
    large_counter2 -= 1
    
    #batch full
    if(len(object_annotation_batch) == batch_size2 or large_counter2 < batch_size2):
        
        #count amount of full batches
        if(len(object_annotation_batch) == batch_size2):
            batch_counter += 1
        #predict batch
        prediction2 = net2.predict_on_batch(np.array(object_annotation_batch))
        
        object_annotation_batch.clear()
        
        #save prediction in image-shaped prediction matrix
        for i in range(prediction2.shape[0]):
            
            #get x and y from list from above in position of annotations (n-th batch, i-th element in it)
            x_object = object_x_list[batch_counter * batch_size2 + i]
            y_object = object_y_list[batch_counter * batch_size2 + i]
            
            predictions[y_object][x_object] = prediction2[i]

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