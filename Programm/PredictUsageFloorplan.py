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

script_dir = os.path.dirname(__file__)

floorplan_name = str(input("Which file from the image folder do you want to open?"))
floorplan_path = os.path.join(script_dir, "../Flurplandaten/images/{}".format(floorplan_name))

floorplan = np.array(PIL.Image.open(floorplan_path).convert('L').convert('RGB'))

net_path = os.path.join(script_dir, "../Netze/final_net.h5")
net = keras.models.load_model(net_path)

predictions = np.empty(100,100)

for x in range(floorplan.shape[0]):
    for y in range(floorplan.shape[1]):
        
        curr_annot = floorplan[x:(x+100), y:(y+100)]
        
        predictions[x][y] = net.predict(curr_annot)


