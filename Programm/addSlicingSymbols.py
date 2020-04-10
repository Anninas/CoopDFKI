# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:55:54 2020

@author: anni
"""

import json
from os import listdir
from os.path import join
import keras
import numpy as np
import PIL
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model

###START LOADING DATA

#Load json file    
metadata = json.loads(open("D:/Users/anni/Documents/KooperationsphaseHector/Flurplandaten/flo2plan-ICDAR2019/flo2plan_icdar_instances.json", 'r').read())

#List for images
imgs = []

#Dict for pairing of names and idices of images
imgs_nametoind = {}

#Path for accessing images
path = "D:/Users/anni/Documents/KooperationsphaseHector/Flurplandaten/flo2plan-ICDAR2019/images/"

#Counter to get index of image
count = 0

#Loop through elements (-->e) in directory
for e in listdir(path):
    
    link = join(path, e)                            #Get the element's link
    curr_img = np.array(PIL.Image.open(link))       #Load image and transform it to numpy array
    imgs.append(curr_img)                           #Add array-shaped image to list
    
    imgs_nametoind[e] = count
    
    count+= 1

#Dict for pairing indices (meaning the place of the image in the list with the images) 
#and ids (meaning the number adressing this image in the metadata) of images
imgs_idtoind = {}

#Loop through all image entries (-->ie) in metatdata
for ie in metadata['images']:
    
    if ie['file_name'] in imgs_nametoind:           #Check if image exists in imported image list
        
        index = imgs_nametoind[ie['file_name']]     #Get the index to the current file name
        img_id = ie['id']                           #Get the id of this file
        imgs_idtoind[img_id] = index                #Link id to the index

###END LOADING DATA

###START PREPROCESSING

#List of images
annotations = []

#Loop through all annotations
for annot in metadata['annotations']:
    bbox = annot['bbox']                             #Get bounding box
    img_id = annot['image_id']                       #Get image id
    
    #Check if the image to the id exists
    if img_id in imgs_idtoind:
        
        annot_img = imgs[imgs_idtoind[img_id]]       #Get the image
        annot_x1 = int(bbox[0])                      #Get x-coordinate of upper left corner rounded down
        annot_y1 = int(bbox[1])                      #Get y-coordinate of upper left corner rounded down
        annot_width = int(np.ceil(bbox[2]))          #Get width rounded up
        annot_height = int(np.ceil(bbox[3]))         #Get height rounded up
        annot_x2 = annot_x1 + annot_width            #Calculate x-coordinate of lower right corner 
        annot_y2 = annot_y1 + annot_height           #Calculate y-coordinate of lower right corner
        
        #Add current annotation to list
        annotations.append(annot_img[annot_y1:annot_y2, annot_x1:annot_x2])

###END PREPROCESSING 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
