# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 18:17:46 2020

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
#import opencv python as cv2
import random
import imutils

###START LOADING DATA
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
metadata_path = os.path.join(script_dir, "../Flurplandaten/floorplan_metadata_cleaned_nobidet_binary_random.json")
new_path = os.path.join(script_dir, "../Flurplandaten/floorplan_metadata_binary_random.json")

#Load json file    
metadata = json.loads(open(metadata_path, 'r').read())

#List for images
imgs = []

#Dict for pairing of names and idices of images
imgs_nametoind = {}

#Path for accessing images
rel_path = "../Flurplandaten/images"
path = os.path.join(script_dir, rel_path)

#Counter to get index of image
count = 0

#Loop through elements (-->e) in directory
for e in listdir(path):
    
    link = join(path, e)                                            #Get the element's link
    curr_img = np.array(PIL.Image.open(link).convert('L'))         #Load image and transform it to B&W numpy array
    imgs.append(curr_img)                                           #Add array-shaped image to list
    
    imgs_nametoind[e] = count
    
    count+= 1

#Dict for pairing indices (meaning the place of the image in the list with the images) 
#and ids (meaning the number adressing this image in the metadata) of images
imgs_idtoind = {}

#Loop through all image entries (-->ie) in metatdata
for ie in metadata['images']:
    
    if ie['file_name'] in imgs_nametoind:                           #Check if image exists in imported image list
        
        index = imgs_nametoind[ie['file_name']]                     #Get the index to the current file name
        img_id = ie['id']                                           #Get the id of this file
        imgs_idtoind[img_id] = index                                #Link id to the index
        
print("Done")
###END LOADING DATA



n = 0        

#Check all annotations
while n < len(metadata['annotations']):
    
    image_id = metadata['annotations'][n]['image_id']
    
    #Check if image exists
    if image_id in imgs_idtoind: 
        
        #Overwrite all non-negatives as class 0
        if not(metadata['annotations'][n]['category_id'] == 12):
            metadata['annotations'][n]['category_id'] = 1
        else:
            metadata['annotations'][n]['category_id'] = 0

    n += 1

print("Done")

x = 0

print("Fixing categories...")
#Delete bidet class, let other classes follow one down
while x < len(metadata['categories']):
    if not(metadata['categories'][x]['id'] == 12):
        del metadata['categories'][x]
    else:
        metadata['categories'][x]['id'] = 0
        x += 1
        
    
    
    
metadata['categories'].append({'id':1, 'name':"object", 'supercategory':None})

print(metadata['categories'])

with open(new_path, 'w') as f:
        json.dump(metadata, f)









