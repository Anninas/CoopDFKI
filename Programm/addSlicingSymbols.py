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
import PIL.ImageOps
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
import cv2
import random
import imutils
import io

random.seed()




###START FUNCTION SECTION 
def trim(value, min, max):
    
    #Keeps value between min and max
    if(value<min):
        return min
    elif(value>max):
        return max
    else:
        return value

def invert(img):
      
    #inverts an image over 255
    inverted_image = 255 - img
        
    return inverted_image
                
def StandardAnnotation(img, bound):
    
    #Cuts out the annotation as the centre of a 100x100 image
    
    #Get x-coordinate of upper left corner rounded down
    annot_x1 = trim(int((bound[0]) - (100 - bound[2])/2), 0, img.shape[1]-100) 
    #Get y-coordinate of upper left corner rounded down               
    annot_y1 = trim(int((bound[1]) - (100 - bound[3])/2), 0, img.shape[0]-100) 
    #Get width and height
    annot_width = 100
    annot_height = 100
    #Calculate x-coordinate of lower right corner
    annot_x2 = annot_x1 + annot_width
    #Calculate y-coordinate of lower right corner                             
    annot_y2 = annot_y1 + annot_height                           
    
    return img[annot_y1:annot_y2, annot_x1:annot_x2]
 
def Offset2dAnnotation(img, bound):
    
    # Calculate actual coordiantes of upper left corner of 100x100 area
    corner_100x100_no_offset_x = int(((bound[0])) - (100 - bound[2])/2)
    corner_100x100_no_offset_y = int(((bound[1])) - (100 - bound[3])/2)
    
    #Get offset (min half annotation should be still in 100x100)
    offset_x = random.randint(-50, 50)
    offset_y = random.randint(-50, 50)
    
    #Get coordinates of upper left corner (100x100) with offset
    corner_100x100_offset_x1 = trim(corner_100x100_no_offset_x + offset_x, 0, img.shape[1]-100)                       
    corner_100x100_offset_y1 = trim(corner_100x100_no_offset_y + offset_y, 0, img.shape[0]-100) 
    #Set width and height     
    width = 100                         
    height = 100 
    #Get coordinates of lower right corner (100x100)                    
    corner_100x100_offset_x2 = corner_100x100_offset_x1 + width  
    corner_100x100_offset_y2 = corner_100x100_offset_y1 + height
    
    result = img[corner_100x100_offset_y1:corner_100x100_offset_y2, corner_100x100_offset_x1:corner_100x100_offset_x2]
    
    if(result.shape[0] != 100 or result.shape[1] != 100):
        print("Size issue")
    
    return result
    
def RotateAnnotation(img, bound):                                 
    
    #Rotation of offset or standard annotation    
    case = random.randint(0, 1)
    
    if case == 0:
        rot_img = StandardAnnotation(img, bound)
    else:
        rot_img = Offset2dAnnotation(img, bound)
    
    #Get the angle (*45°) to rotate the image with
    angle = random.randint(1, 7)
    
    #Save the original scale of the image (in case it was not 100x100)
    orig_height = rot_img.shape[0]
    orig_width = rot_img.shape[1]
    
    #Rotate the image
    rotated = invert(imutils.rotate_bound(invert(rot_img), angle*45))
    
    #Crop the image back to 100*100
    y1 = int(abs((rotated.shape[0]-orig_height)/2))
    y2 = y1 + 100
    x1 = int(abs((rotated.shape[1]-orig_width)/2))
    x2 = x1 + 100
    
    result = rotated[y1:y2, x1:x2]
    
    if(result.shape[0] != 100 or result.shape[1] != 100):
        print("Size issue")
    
    return result
    

#Skalierung mit opencv.resize, interpolation cubic
    
def RescaleAnnotation(img, bound):
    
    #Cut the annotation with or without offset 
    case = random.randint(0, 1)
    
    if case == 0:
        rot_img = StandardAnnotation(img, bound)
    else:
        rot_img = Offset2dAnnotation(img, bound)
    
    #Save the original size of the image (in case it is not 100)
    orig_height = rot_img.shape[0]
    orig_width = rot_img.shape[1]
    
    #Get the scale factor
    scale = random.randint(11, 14)/10
    
    #Scale the image with the factor 
    scale_img = cv2.resize(rot_img, (int(orig_height*scale), int(orig_width*scale)), interpolation = cv2.INTER_CUBIC)
    #Cut the scaled image back to 100*100
    y1 = int(abs((scale_img.shape[0]-orig_height)/2))
    y2 = y1 + 100
    x1 = int(abs((scale_img.shape[1]-orig_width)/2))
    x2 = x1 + 100

    result = scale_img[y1:y2, x1:x2]
    
    if(result.shape[0] != 100 or result.shape[1] != 100):
        print("Size issue")
    
    return result
    
    
    
###END FUNCTION SECTION


###START LOADING DATA
  
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/floorplan_metadata_cleaned.json"
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
    
    #Get the element's link
    link = join(path, e)  
    #Load image and transform it to B&W numpy array                                          
    curr_img = np.array(PIL.Image.open(link).convert('L'))         
    #Add array-shaped image to list
    imgs.append(curr_img)                                           
    
    imgs_nametoind[e] = count
    
    count+= 1

#Dict for pairing indices (meaning the place of the image in the list with the images) 
#and ids (meaning the number adressing this image in the metadata) of images
imgs_idtoind = {}

#Loop through all image entries (-->ie) in metatdata
for ie in metadata['images']:
    
    #Check if image exists in imported image list
    if ie['file_name'] in imgs_nametoind:                           
        
        #Get the index to the current file name
        index = imgs_nametoind[ie['file_name']]   
        #Get the id of this file                  
        img_id = ie['id']   
        #Link id to the index                                        
        imgs_idtoind[img_id] = index                                

###END LOADING DATA

###START PREPROCESSING

#List of images
annotations = []

#Loop through all annotations
for annot in metadata['annotations']:
    #Get bounding box
    bbox = annot['bbox']                                            
    #Get image id
    img_id = annot['image_id']                                      
    
    #Check if the image to the id exists
    if img_id in imgs_idtoind:
        
        #Get the image
        annot_img = imgs[imgs_idtoind[img_id]]                      
        
        #Add current annotation to list
        annotations.append(StandardAnnotation(annot_img, bbox))
        
        #Create 20 varied Verions of this Annotation        
        for i in range(0, 20):
            
            #Type of augmentation
            augmentation = random.randint(0, 2)
            
            #Offset only
            if augmentation == 0:                                 
                aug_annot = Offset2dAnnotation(annot_img, bbox)
                annotations.append(aug_annot)
            #Rotate
            elif augmentation == 1:
                try:
                    aug_annot = RotateAnnotation(annot_img, bbox)
                    annotations.append(aug_annot)
                except:
                    print("Rotate Error")
            #Rescale
            elif augmentation == 2:
                try:
                    aug_annot = RescaleAnnotation(annot_img, bbox)
                    annotations.append(aug_annot)
                except:
                    print("Scale Error")
            
            #if(aug_annot.shape[0] != 100 or aug_annot.shape[1] != 100):
             #   print("Wrong Size")
        
#Save annotations to json
#https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions

memfile = io.BytesIO()
np.save(memfile, annotations)
memfile.seek(0)

script_dir = os.path.dirname(__file__)
annotation_path = os.path.join(script_dir, "../Flurplandaten/preprocessed_annotations.json")

with open(annotation_path, 'w') as f:
        json.dump(memfile.read().decode('latin-1'), f)

        

###END PREPROCESSING 
    
###NOTES:
#BBox: Wert in der Breite, Wert in der Höhe, Breite, Höhe
    
    
    
