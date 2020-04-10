# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 17:55:54 2020

@author: anni
"""

import json
from os import listdir
from os.path import join
import os
import keras
import numpy as np
import PIL
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
#import OpenCV as cv
import random

random.seed()




###START FUNCTION SECTION   
                
def StandardAnnotation(img, bound):
    
    annot_x1 = int(bound[0])                                     #Get x-coordinate of upper left corner rounded down
    annot_y1 = int(bound[1])                                     #Get y-coordinate of upper left corner rounded down
    annot_width = int(np.ceil(bound[2]))                         #Get width rounded up
    annot_height = int(np.ceil(bound[3]))                        #Get height rounded up
    annot_x2 = annot_x1 + annot_width                            #Calculate x-coordinate of lower right corner 
    annot_y2 = annot_y1 + annot_height                           #Calculate y-coordinate of lower right corner
    
    return img[annot_y1:annot_y2, annot_x1:annot_x2]
    
def OffsetHorizontalAnnotation(img, bound):
     
     offset = random.randint(10, 50)
     
     annot_x1 = int(bound[0]) + offset                            #Get x-coordinate of upper left corner rounded down
     annot_y1 = int(bound[1])                                     #Get y-coordinate of upper left corner rounded down
     annot_width = int(np.ceil(bound[2]))                         #Get width rounded up
     annot_height = int(np.ceil(bound[3]))                        #Get height rounded up
     annot_x2 = annot_x1 + annot_width                            #Calculate x-coordinate of lower right corner 
     annot_y2 = annot_y1 + annot_height                           #Calculate y-coordinate of lower right corner
     
     return img[annot_y1:annot_y2, annot_x1:annot_x2]
        
def OffsetVerticalAnnotation(img, bound):
     
     offset = random.randint(10, 50)
     
     annot_x1 = int(bound[0])                                     #Get x-coordinate of upper left corner rounded down
     annot_y1 = int(bound[1])  + offset                           #Get y-coordinate of upper left corner rounded down
     annot_width = int(np.ceil(bound[2]))                         #Get width rounded up
     annot_height = int(np.ceil(bound[3]))                        #Get height rounded up
     annot_x2 = annot_x1 + annot_width                            #Calculate x-coordinate of lower right corner 
     annot_y2 = annot_y1 + annot_height                           #Calculate y-coordinate of lower right corner
     
     return img[annot_y1:annot_y2, annot_x1:annot_x2]
 
def Offset2dAnnotation(img, bound):
    
     offset_x = random.randint(10, 50)
     offset_y = random.randint(10, 50)
     
     annot_x1 = int(bound[0]) + offset_y                          #Get x-coordinate of upper left corner rounded down
     annot_y1 = int(bound[1]) + offset_x                          #Get y-coordinate of upper left corner rounded down
     annot_width = int(np.ceil(bound[2]))                         #Get width rounded up
     annot_height = int(np.ceil(bound[3]))                        #Get height rounded up
     annot_x2 = annot_x1 + annot_width                            #Calculate x-coordinate of lower right corner 
     annot_y2 = annot_y1 + annot_height                           #Calculate y-coordinate of lower right corner
     
     return img[annot_y1:annot_y2, annot_x1:annot_x2]
    
#def RotateAnnotation(img, bound):                                 #Rotation of offset or standard annotation
    
    #Ausschnitt, etwas größer als Annotation, drehen
    #imutils installieren, s. Skype
    #45°, Vielfache davon
    
 #   case = random.randint(0, 3)
    
  #  if case == 0:
        
#Skalierung mit opencv.resize, interpolation cubic
  
###END FUNCTION SECTION





###START LOADING DATA
  
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/flo2plan_icdar_instances.json"
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

###START PREPROCESSING

#List of images
annotations = []

#Loop through all annotations
for annot in metadata['annotations']:
    bbox = annot['bbox']                                            #Get bounding box
    img_id = annot['image_id']                                      #Get image id
    
    #Check if the image to the id exists
    if img_id in imgs_idtoind:
        
        annot_img = imgs[imgs_idtoind[img_id]]                      #Get the image
        
        #Add current annotation to list
        annotations.append(StandardAnnotation(annot_img, bbox))
        
        #Create 20 varied Verions of this Annotation        
        for i in range(0, 20):
            
            augmentation = random.randint(0, 2)
            
            if augmentation == 0:                                   #Horizontaler Offset
                aug_annot = OffsetHorizontalAnnotation(annot_img, bbox)
                annotations.append(aug_annot)
            elif augmentation == 1:                                 #Vertikaler Offset
                aug_annot = OffsetVerticalAnnotation(annot_img, bbox)    
                annotations.append(aug_annot)
            elif augmentation == 2:                                 #2D - Offset
                aug_annot = Offset2dAnnotation(annot_img, bbox)
                annotations.append(aug_annot)
            #elif augmentation == 3:
                
            #elif augmentation == 4:
                
            #else:
        
        

###END PREPROCESSING 
    
   
    
    
    
