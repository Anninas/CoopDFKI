# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:40:16 2020

@author: anni
"""

import json
from os import listdir
from os.path import join
import os
#import keras
import numpy as np
import PIL
import PIL.ImageOps
import matplotlib.pyplot as plt
#from keras import layers
#from keras.models import Model
import cv2
import random
import imutils
import pickle

random.seed()



###START FUNCTION SECTION 
def getNormalizedNumbersOfAugmentation(old_numbers):
    average = np.sum(old_numbers)/len(old_numbers)
    augmentation_numbers = np.zeros(len(old_numbers))
    for i in range(len(old_numbers)):
        if(average-60 < old_numbers[i] < average+60):
            augmentation_numbers[i] = 20
        elif(average-60 > old_numbers[i]):
            augmentation_numbers[i] = 20*(average-60)/old_numbers[i]
        elif(average+60 < old_numbers[i]):
            augmentation_numbers[i] = 20*(average+60)/old_numbers[i]
    return augmentation_numbers

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
                
def StandardAnnotation(img, points):
    
    #Cuts out the annotation out of the image by using segmentation and places it on white background
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    img = img[:,:, 0]
    
    polygon = [[]]
    
    xs = []
    ys = []
    
    i=0
    
    #Get points and sort them into polygon
    while i < len(points[0]):
        x = trim(points[0][i], 0, img.shape[1])
        y = trim(points[0][i+1], 0, img.shape[0])
        
        polygon[0].append([y, x])
        xs.append(x)
        ys.append(y)
        
        
        i+= 2
    
    #Get coordinates for cropping to 100*100 later
    mid_x = (np.max(xs)+np.min(xs))/2
    mid_y = (np.max(ys)+np.min(ys))/2
    x1 = int(mid_x - 50)
    x2 = int(mid_x + 50)
    y1 = int(mid_y - 50)
    y2 = int(mid_y + 50)
    
    
    nds = np.array(polygon)
    nds = np.int32([nds]) # Bug with fillPoly, needs explict cast to 32bit
    
    #method 1 smooth region
    cv2.drawContours(mask, nds, -1, (255, 255, 255), -1, cv2.LINE_AA)
    
    
    #cv2.fillPoly(mask, polygon, (255))
    
    res = cv2.bitwise_and(img, img, mask = mask)
 
    ## create the white background of the same size of original image
    wbg = np.ones_like(img, np.uint8)*255
    cv2.bitwise_not(wbg, wbg, mask=mask)
    
    # overlap the resulted cropped image on the white background
    dst = wbg+res
    
    #Crop to 100*100 around the annotation
    result_standard_annotation = dst[y1:y2, x1:x2]
    
    return result_standard_annotation
 
def Offset2dAnnotation(img, bound):
    
    # Calculate actual coordiantes of upper left corner of 100x100 area
    corner_100x100_no_offset_x = int(((bound[0])) - (100 - bound[2])/2)
    corner_100x100_no_offset_y = int(((bound[1])) - (100 - bound[3])/2)
    
    #Get offset (min half annotation should be still in 100x100)
    '''Lower value to test if it works better'''
    offset_x = random.randint(-20, 20)
    offset_y = random.randint(-20, 20)
    
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
    
    '''#Comment out if no offset
    #Rotation of offset or standard annotation    
    case = random.randint(0, 1)
    
    if case == 0:
        rot_img = StandardAnnotation(img, bound)
    else:
        rot_img = Offset2dAnnotation(img, bound)
    '''
    
    #Comment out if with offset
    rot_img = StandardAnnotation(img, bound)
    
    #Get the angle to rotate the image with
    angle = random.randint(1, 360)
    
    #Save the original scale of the image (in case it was not 100x100)
    orig_height = rot_img.shape[0]
    orig_width = rot_img.shape[1]
    
    #Rotate the image
    rotated = invert(imutils.rotate_bound(invert(rot_img), angle))
    
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
    
    '''#Comment out if no offset
    #Cut the annotation with or without offset 
    case = random.randint(0, 1)
    
    if case == 0:
        rot_img = StandardAnnotation(img, bound)
    else:
        rot_img = Offset2dAnnotation(img, bound)
    '''
    
    #Comment out if with offset
    rot_img = StandardAnnotation(img, bound)
    
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
    
def HotKeyEncode(int_list, num_categories):
    
    HotKeyList = []
    
    for element in int_list:
        hotkey = np.zeros(num_categories)
        hotkey[element] = 1
        HotKeyList.append(hotkey)
        
    return HotKeyList
        
    
###END FUNCTION SECTION


###START LOADING DATA
training_annotations_per_category = [651, 1326]
augmentation_numbers_for_training = getNormalizedNumbersOfAugmentation(training_annotations_per_category)
#validation_annotations_per_category = [189, 60, 61, 313, 123, 296, 48, 97, 107, 39, 66, 718]
#augmentation_numbers_for_validation = getNormalizedNumbersOfAugmentation(validation_annotations_per_category)
  
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/floorplan_metadata_binary_random.json"
metadata_path = os.path.join(script_dir, rel_path)

#Load json file    
metadata = json.loads(open(metadata_path, 'r').read())

#List for images
imgs = []

#Dict for pairing of names and idices of images
imgs_nametoind = {}

#Path for accessing images
rel_path = "../Flurplandaten/images_only_rectangular"
path = os.path.join(script_dir, rel_path)

#Counter to get index of image
count = 0

#Loop through elements (-->e) in directory
for e in listdir(path):
    
    #Get the element's link
    link = join(path, e)  
    #Load image and transform it to B&W numpy array                                          
    curr_img = np.array(PIL.Image.open(link).convert('L').convert('RGB'))         
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

#List of objects in annotations
object_categories = []

#Loop through all annotations
for annot in metadata['annotations']:
    #Get bounding box
    bbox = annot['bbox']  
    points = annot['segmentation']                                          
    #Get image id
    img_id = annot['image_id']    

    #Get object on annotation
    object_category = annot['category_id']                           
    
    #Check if the image to the id exists
    if img_id in imgs_idtoind:
        
        #Get the image
        annot_img = imgs[imgs_idtoind[img_id]]                      
        
        #Add current annotation to list
        annotations.append(StandardAnnotation(annot_img, points))
        object_categories.append(object_category)
        
        
        #Create varied Verions of this Annotation (only for training annotations, otherwise comment)
        #Change for training

        for i in range(int(augmentation_numbers_for_training[object_category-1])):
            #Type of augmentation with offset
            #augmentation = random.randint(0, 2)
            #without offset
            augmentation = random.randint(0, 1)
            
            #Offset only
            '''
            if augmentation == 0:
                             
                aug_annot = Offset2dAnnotation(annot_img, bbox)
                annotations.append(aug_annot)
                object_categories.append(object_category)
            '''
            #Rotate
            if augmentation == 0:# With offset: elif, 1; without : if, 0 

                aug_annot = RotateAnnotation(annot_img, bbox)
                annotations.append(aug_annot)
                object_categories.append(object_category)

            #Rescale
            elif augmentation == 1:#With offset: 2; without: 1

                aug_annot = RescaleAnnotation(annot_img, bbox)
                annotations.append(aug_annot)
                object_categories.append(object_category)
        

#Hot key encoding of object categories

        
#Save annotations to json
#https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions

script_dir = os.path.dirname(__file__)
annotation_path = os.path.join(script_dir, "../Flurplandaten/preprocessed__training_annotations_binary_random_nooffset_white.p")
pickle.dump(annotations, open(annotation_path, "wb"))

#object_categories_encoded = HotKeyEncode(object_categories, 2)

object_path = os.path.join(script_dir, "../Flurplandaten/object_list_for_training_annotations_binary_random_nooffset_white.p")
pickle.dump(object_categories, open(object_path, "wb"))

###END PREPROCESSING 
    
###NOTES:
#BBox: Wert in der Breite, Wert in der Höhe, Breite, Höhe
    
    
    
