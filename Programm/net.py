# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:19:07 2020

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

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path_annot = "../Flurplandaten/preprocessed_annotations.p"
annot_path = os.path.join(script_dir, rel_path_annot)

rel_path_categories = "../Flurplandaten/object_list_for_annotations.p"
categories_path = os.path.join(script_dir, rel_path_categories)

train_annot = np.array(pickle.load(open(annot_path, 'rb')))
train_annot_array = np.reshape(train_annot, (83832, 100, 100, 3))
train_categories = pickle.load(open(categories_path, 'rb'))



base_model = keras.applications.InceptionV3(weights = 'imagenet', include_top = False, classes = 12, input_shape = (100,100,3))

x = base_model.output
x = layers.Flatten()(x)
predictions = keras.layers.Dense(12, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

model.fit(x = train_annot_array, y = np.array(train_categories), batch_size = 128, epochs = 8)


#Change number!!!
net_path = os.path.join(script_dir, "../Netze/try4.h5")

model.save(net_path)