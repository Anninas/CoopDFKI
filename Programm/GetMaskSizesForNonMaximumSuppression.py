# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:53:18 2020

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

script_dir = os.path.dirname(__file__)

rel_path = "../Flurplandaten/floorplan_metadata_cleaned_nobidet.json"
metadata_path = os.path.join(script_dir, rel_path)   
metadata = json.loads(open(metadata_path, 'r').read())

results = {}

for i in range(12):
    results["class" + str(i)+"height"]=[]
    results["class" + str(i)+"width"]=[]

for annotation in metadata["annotations"]:
    
    try:
        results["class" + str(annotation["category_id"]-1) + "height"].append(annotation["bbox"][3])
        results["class" + str(annotation["category_id"]-1) + "width"].append(annotation["bbox"][2])
    except KeyError:
        print("Image not available -> wrong indexing")
   
for i in range(12):
    results["class" + str(i)+"height"]=np.average(results["class" + str(i)+"height"])
    results["class" + str(i)+"width"]=np.average(results["class" + str(i)+"width"])