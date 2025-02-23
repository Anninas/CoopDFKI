# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 19:33:55 2020

@author: anni
"""

import os
import keras
import numpy as np
import PIL
import PIL.ImageOps
import cv2
from itertools import cycle
import copy
import tensorflow as tf
import time


def bounding_box(points):

    top_left_x = min(point for point in points[1])
    top_left_y = min(point for point in points[0])
    bot_right_x = max(point for point in points[1])
    bot_right_y = max(point for point in points[0])

    return [(top_left_x, top_left_y), (bot_right_x, bot_right_y)]

def getPrediction(image):
    
    '''
    NUM_PARALLEL_EXEC_UNITS = multiprocessing.cpu_count()
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                           allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
    tf.config.optimizer.set_jit(True)
    session = tf.compat.v1.Session(config=config)

    tf.compat.v1.keras.backend.set_session(session)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
    '''
    
    tf.keras.backend.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    
    print('Loading models...')
    #Load the nets
    script_dir = os.path.dirname(__file__)    
    net1_path = os.path.join(script_dir, "./Netze/try22_IRV2_binaryOne_64-0.0001-9.h5")
    net1 = keras.models.load_model(net1_path)
    net2_path = os.path.join(script_dir, "./Netze/try10_IncResV2_randomrotation.h5")
    net2 = keras.models.load_model(net2_path)
    
    print('Loading models done...')
    
    #Prepare the floorplan
    floorplan = np.array(image.convert('L').convert('RGB'))
    
    #Prepare the empty lists and arrays to fill with the results
    predictions = np.empty((floorplan.shape[0], floorplan.shape[1], 11))
    annotations = []
    object_annotations = []
    object_x_list = []
    object_y_list = []
    
    batch_size1 = 2048
    
    counter = 0
    #large_counter = ((floorplan.shape[0]-100) * (floorplan.shape[1]-100))/batch_size
    start = time.time()
    print('Starting binary prediction...')
    
    #prediction binary
    for y in range(floorplan.shape[0]-100):
        for x in range(floorplan.shape[1]-100):
        
            #get initial x and y for this batch

            if(counter == 0):
                initial_x = x
                initial_y = y

            #get current annotations
            curr_annot = floorplan[y:(y+100), x:(x+100)]
            annotations.append(curr_annot)
            

            counter += 1
            
            #batch size or edge of image reached
            if((counter == batch_size1) or ((x == floorplan.shape[1]-101) and (y == floorplan.shape[0]-101))):
                
                #print("Predicting now...")
                #predict takes np.array, no list!
                prediction = net1.predict(np.array(annotations))
                #print("Prediction done")
                
                for i in range(prediction.shape[0]):
                    
                    #if object can be seen on this
                    if(prediction[i]>0.8):
                        
                        #save the annotation
                        object_annotations.append(annotations[i])
                        #save their x and y coordinates
                        object_x_list.append(((initial_x + i)%(floorplan.shape[1]-100))+50)
                        object_y_list.append(initial_y + ((initial_x + i)//(floorplan.shape[1]-100))+50)
                        
                    #x_i = ((initial_x + i)%(floorplan.shape[1]-100))+50 
                    #y_i = initial_y + ((initial_x + i)//(floorplan.shape[1]-100))+50
                    
                    #predictions[y_i][x_i]=prediction[i]
                    #print("current prediction = {}".format(prediction[i]))
                    #print("saved current prediction = {}".format(predictions[y_i][x_i]))
                
                #large_counter -= 1
                
                #print("{} steps left".format(large_counter))
                
                counter = 0
                annotations.clear()

    '''
    prediction = net1.predict(np.array(annotations))
    #print("Prediction done")
    for i in range(prediction.shape[0]):
        
        #if object can be seen on this
        if(prediction[i]>0.8):
            
            #save the annotation
            object_annotations.append(annotations[i])
            #save their x and y coordinates
            object_x_list.append(((0 + i)%(floorplan.shape[1]-100))+50)
            object_y_list.append(0 + ((0 + i)//(floorplan.shape[1]-100))+50)
    '''            
    
    print('Binary prediction done...')
    
    print('Starting object prediction')
    #predicting objects
    object_annotation_batch = []
    batch_counter = 0    
    
    batch_size2 = 128
    
    large_counter2 = len(object_annotations)
    #all annotations that show objects according to other net
    for annotation in object_annotations:
        
        #collect batch
        object_annotation_batch.append(annotation)
        large_counter2 -= 1
        
        #batch full
        if(len(object_annotation_batch) == batch_size2 or large_counter2 < batch_size2):
            
            #count amount of full batches
            if(len(object_annotation_batch) == batch_size2):
                batch_counter += 1
            #predict batch
            prediction2 = net2.predict_on_batch(np.array(object_annotation_batch))
            
            object_annotation_batch.clear()
            
            #save prediction in image-shaped prediction matrix
            for i in range(prediction2.shape[0]):
                
                #get x and y from list from above in position of annotations (n-th batch, i-th element in it)
                x_object = object_x_list[batch_counter * batch_size2 + i]
                y_object = object_y_list[batch_counter * batch_size2 + i]
                
                predictions[y_object][x_object] = prediction2[i]

    print('Object prediction done')
    end = time.time()

    print("Time for prediction loop: {}".format(end-start))
    
    return predictions

def getPostprocessedResults(initial_prediction, floorplan):
    
    print('Starting postprocessing')
    
    #from GetMaskSizesForNonMaximumSuppression import results
    
    results = {'class0height': 33.74259025037887, 
               'class0width': 33.39464882943144, 
               'class1height': 55.6330990685357, 
               'class1width': 56.39938841925727, 
               'class2height': 61.036729094039565, 
               'class2width': 58.72386201370594, 
               'class3height': 29.28880152591439, 
               'class3width': 29.5078879326813, 
               'class4height': 48.24540004107648, 
               'class4width': 49.09076749535626, 
               'class5height': 30.060928380419636,
               'class5width': 29.805765787760418, 
               'class6height': 33.859484788848135, 
               'class6width': 33.76672121373618, 
               'class7height': 55.18505715300597, 
               'class7width': 56.096098936678516, 
               'class8height': 35.81207212573754, 
               'class8width': 35.174171767548884, 
               'class9height': 50.957503197898326, 
               'class9width': 53.86868715957856, 
               'class10height': 53.372852117365056, 
               'class10width': 51.81429450295188, 
               'class11height': 98.86711911871882, 
               'class11width': 98.78940842883422}

        #Get from other script, sizes of masks for non max supression (60% of average sizeof symbols)
    mask_sizes_x = [int(results["class"+str(x)+"width"]*(6/10)) for x in range(12)]
    mask_sizes_y = [int(results["class"+str(y)+"height"]*(6/10)) for y in range(12)] 
    
    
    colors = cycle([(0,0,128), (64,224,208), (255,140,0), (100,149,237), (255,0,0), (0,128,0), (0,128,128), (128,0,128), (255,215,0), (173,255,47), (255,20,147), (128,128,128)])
    
    result_image = np.array(floorplan)
    result_json = {k: [] for k in range(initial_prediction.shape[2])}
    
    #Add something like argmax if threshold smaller 0.5 bc there could be two probs of same height or a higher one    
    threshold = 0.95
    necessary_hits = 150
    #Replace all prediction values below threshold by 0
    initial_prediction[initial_prediction < threshold] = 0
    
         
    #Maxima in lists
    keys = ['class0', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
    non_max_results_list = dict.fromkeys(keys)
    
    
    #Save a copy of the arrays to modify in Non-Max-Supression
    class_predictions_nonmax = copy.deepcopy(initial_prediction)
    #class_predictions_checkup = np.zeros(initial_prediction.shape[1, 2])
    bound_rects = {k: [] for k in range(initial_prediction.shape[2])}
    
    #Non-Max-Supression loop
    for n_class in range(initial_prediction.shape[2]):
        
        #Mask iteration
        for y1 in range(initial_prediction.shape[0]-mask_sizes_y[n_class]+1):
            for x1 in range(initial_prediction.shape[1]-mask_sizes_x[n_class]+1):
                
                y2 = y1+mask_sizes_y[n_class]
                x2 = x1+mask_sizes_x[n_class]
                
                #Slice current mask from prediction
                current_mask = class_predictions_nonmax[y1:y2, x1:x2, n_class]
                #Delete all values except for max
    
                current_mask[current_mask < np.max(current_mask)] = 0
    
                #if 0 < np.max(current_mask) < 0.8: print(np.max(current_mask))
                #Set prediction to current mask to use it for further non max
                class_predictions_nonmax[y1:y2, x1:x2, n_class] = current_mask
                #if 0 < np.max(class_predictions_nonmax["class"+str(z)][y:y+mask_sizes_y[z], x:x+mask_sizes_x[z]]) < 0.8: print(np.max(class_predictions_nonmax["class"+str(z)][y:y+mask_sizes_y[z], x:x+mask_sizes_x[z]]))
        
        #Save results of non max as list
        non_max_results_list["class"+str(n_class)] = [[], []]  
        non_max_results_list["class"+str(n_class)][0] = (np.where(class_predictions_nonmax[:, :, n_class] > threshold))[0].tolist()
        non_max_results_list["class"+str(n_class)][1] = (np.where(class_predictions_nonmax[:, :, n_class] > threshold))[1].tolist()
        
        i = 0
        
        #Check the results on reasonability --> Are there enough other points with probability higher threshold in the mask size?
        #Delete all points with no maximum near
        while i < len(non_max_results_list["class"+str(n_class)][0]):
            
            #Get mask size
            mask_size_y = mask_sizes_y[n_class]
            mask_size_x = mask_sizes_x[n_class]
            
            #Get x and y of current maximum
            y = non_max_results_list["class"+str(n_class)][0][i]
            x = non_max_results_list["class"+str(n_class)][1][i]
            
            #Get bbox around it
            y1 = int(y-0.5*mask_size_y)
            y2 = int(y+0.5*mask_size_y)
            x1 = int(x-0.5*mask_size_x)
            x2 = int(x+0.5*mask_size_x)
            
            #Get the current mask
            current_checkup_mask = initial_prediction[y1:y2, x1:x2, n_class]
            
            #Get the points in the mask that are above threshold
            points_in_checkup_mask = np.where(current_checkup_mask > threshold)
            
            #Delete all points where less than necessary_hits are in the mask around ist
            if not(len(points_in_checkup_mask[0]) > necessary_hits):
                
                del non_max_results_list["class"+str(n_class)][0][i]
                del non_max_results_list["class"+str(n_class)][1][i]
                
            #If there is a valid amount of hits in the mask, draw a bounding rectangle around the found object 
            #Therefore save a 100*100 piece of the initial prediction with the max in the middle in the propper place
            #of a matrix sized the image
            else:
                help_image = np.zeros(initial_prediction.shape[0:2])
                
                diff_y = mask_size_y + mask_size_y*0.5
                diff_x = mask_size_x + mask_size_x*0.5 
                
                
                y1 = int(y-diff_y)
                y2 = int(y+diff_y)
                x1 = int(x-diff_x)
                x2 = int(x+diff_x)
                
                help_image[y1:y2, x1:x2] = initial_prediction[y1:y2, x1:x2, n_class]
                current_points = np.where(help_image > 0)
                bound_rects[n_class].append(bounding_box(current_points))
                
                i+=1
            
    for n_class, color in zip(range(initial_prediction.shape[2]), colors):
        
        for i in range(len(bound_rects[n_class])):
            
            top_left = bound_rects[n_class][i][0]
            bot_right = bound_rects[n_class][i][1]
            cv2.rectangle(result_image, top_left, bot_right, color, 2)
            
            label = 'Class {}'.format(n_class)
            labelSize=cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
            
            label_x1 = top_left[0]
            label_y1 = top_left[1]
            label_x2 = top_left[0] + labelSize[0][0]
            label_y2 = top_left[1] - labelSize[0][1]
            
            cv2.rectangle(result_image,(label_x1,label_y1),(label_x2,label_y2), color, cv2.FILLED)
            cv2.putText(result_image,  label, (label_x1, label_y1), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0,0,0), 1)
            
            result_json[n_class].append({"x1":top_left[0].item(), "y1": top_left[1].item(), "x2":bot_right[0].item(), "y2": bot_right[1].item()})
            
            
                
    result_image = PIL.Image.fromarray(result_image)
     
    print('Postprocessing done')
    
    return result_image, result_json
        
   
