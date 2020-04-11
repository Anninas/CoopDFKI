# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:09:51 2020

@author: annika
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
#import opencv python as cv2
import random
import imutils

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/flo2plan_icdar_instances.json"
metadata_path = os.path.join(script_dir, rel_path)

#Load json file    
metadata = json.loads(open(metadata_path, 'r').read())

n = 0

while n < len(metadata['annotations']):
    
    bbox = metadata['annotations'][n]['bbox']
    
    if (bbox[3] < 10) or (bbox[2] < 10) or (bbox[3] > 100) or (bbox[2] > 100):
        del metadata['annotations'][n]
        
    n += 1