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

###START LOADING DATA
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/flo2plan_icdar_instances.json"
new_path = os.path.join(script_dir, "../Flurplandaten/floorplan_metadata_cleaned_nobidet.json")
metadata_path = os.path.join(script_dir, rel_path)

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
###END LOADING DATA



###START ACTUAL CLEANING
n = 0        

#Check all annotations
while n < len(metadata['annotations']):
    
    image_id = metadata['annotations'][n]['image_id']
    
    #Check if image exists
    if image_id in imgs_idtoind:
        
        #Get image data
        bbox = metadata['annotations'][n]['bbox']
        img_width = metadata['images'][imgs_idtoind[image_id]]['width']
        img_height = metadata['images'][imgs_idtoind[image_id]]['height']
    
        #Delete image if it is larger than 100*100 or smaller than 10*10 or if it is too close to the edges of the floorplan
        if (bbox[3] < 10) or (bbox[2] < 10) or (bbox[3] > 100) or (bbox[2] > 100) or (img_height < (bbox[1] + bbox[3])) or (img_height < bbox[3])or (img_width < (bbox[0]+bbox[2])) or (img_width < bbox[2]):       
            del metadata['annotations'][n]
        
        #Overwrite all bidets as toilets and let other classes follow one down
        if(metadata['annotations'][n]['category_id'] == 5):
            metadata['annotations'][n]['category_id'] = 1
        elif(metadata['annotations'][n]['category_id']>5):
            metadata['annotations'][n]['category_id'] = (metadata['annotations'][n]['category_id'] - 1)
    n += 1

x = 0

#Delete bidet class, let other classes follow one down
while x < len(metadata['categories']):
    if(metadata['categories'][x]['id'] == 5):
        del metadata['categories'][x]
    if(metadata['categories'][x]['id'] > 5):
        metadata['categories'][x]['id'] = metadata['categories'][x]['id'] - 1
    
    x += 1
    
print(metadata['categories'])    
    
###END CLEANING

###WRITE INTO FILE AND SAVE  
with open(new_path, 'w') as f:
        json.dump(metadata, f)







