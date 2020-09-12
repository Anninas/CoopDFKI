# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:26:33 2020

@author: annika
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
import copy

from GetMaskSizesForNonMaximumSuppression import results

#Get from other script, sizes of masks for non max supression (60% of average sizeof symbols)
mask_sizes_x = [int(results["class"+str(x)+"width"]*(6/10)) for x in range(12)]
mask_sizes_y = [int(results["class"+str(y)+"height"]*(6/10)) for y in range(12)]

#Import prediction to post process
script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, "predictions1.json"), 'r') as file:
    initial_prediction = np.asarray(json.load(file))





#Add something like argmax if threshold smaller 0.5 bc there could be two probs of same height or a higher one    
threshold = 0.8
#Replace all prediction values below threshold by 0
initial_prediction[initial_prediction < threshold] = 0

#Place to save the predictions class-wise as images and arrays
class_predictions = {}
class_predictions_arrays = {}
 
#Get prediction as images and arrays
for i in range(initial_prediction.shape[2]):
    
    #Load floorplan, here to overwrite it each time
    floorplan1 = PIL.Image.open(os.path.join(script_dir, "../Flurplandaten/images/001.png")).convert('L').convert('RGB')
    
    #Get current prediction and expand dimensions to image
    current_prediction = initial_prediction[:, :, i]
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
    floorplan1.save(os.path.join(script_dir, "./Kontrollbilder/{}".format(image_name)))





#Maxima in lists
non_max_results_list = {}
non_max_results_images = {}

#Save a copy of the arrays to modify in Non-Max-Supression
class_predictions_nonmax = copy.deepcopy(class_predictions_arrays)

#Non-Max-Supression loop
for z in range(initial_prediction.shape[2]):
    
    #Mask iteration
    for y in range(initial_prediction.shape[0]-mask_sizes_y[z]+1):
        for x in range(initial_prediction.shape[1]-mask_sizes_x[z]+1):
            
            #Slice current mask from prediction
            current_mask = class_predictions_nonmax["class"+str(z)][y:y+mask_sizes_y[z], x:x+mask_sizes_x[z]]
            #Delete all values except for max
            current_mask[current_mask < np.max(current_mask)] = 0
            #if 0 < np.max(current_mask) < 0.8: print(np.max(current_mask))
            #Set prediction to current mask to use it for further non max
            class_predictions_nonmax["class"+str(z)][y:y+mask_sizes_y[z], x:x+mask_sizes_x[z]] = current_mask
            #if 0 < np.max(class_predictions_nonmax["class"+str(z)][y:y+mask_sizes_y[z], x:x+mask_sizes_x[z]]) < 0.8: print(np.max(class_predictions_nonmax["class"+str(z)][y:y+mask_sizes_y[z], x:x+mask_sizes_x[z]]))
    
    #Save results of non max as list
    non_max_results_list["class"+str(z)] = np.where(class_predictions_nonmax["class"+str(z)] > threshold)
    
    #Sabe results of non max as images and paste on top of floorplan for controll
    floorplan2 = PIL.Image.open(os.path.join(script_dir, "../Flurplandaten/images/001.png")).convert('L').convert('RGB')
    non_max_image = np.zeros((floorplan2.size[1], floorplan2.size[0], 4))
    
    for i in range(len(non_max_results_list["class"+str(z)][0])):
        non_max_image[non_max_results_list["class"+str(z)][0][i], non_max_results_list["class"+str(z)][1][i], 0] = 1
        non_max_image[non_max_results_list["class"+str(z)][0][i], non_max_results_list["class"+str(z)][1][i], 3] = 1
    
    non_max_results_images["class"+str(z)] = PIL.Image.fromarray((non_max_image * 255).astype('uint8'), mode='RGBA')
    
    
    image_name = "class{}.png".format(z)
    
    floorplan2.paste(non_max_results_images["class"+str(z)], (0,0), non_max_results_images["class"+str(z)])
    floorplan2.save(os.path.join(script_dir, "./KontrollbilderNonMax/{}".format(image_name)))     
    

