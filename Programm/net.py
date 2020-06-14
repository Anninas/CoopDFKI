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
import sklearn.metrics as metric
from itertools import cycle

all_categories = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

def standartize(input_array):
    
    output_array = []
    
    for array in input_array:
        output_part = np.zeros(12)
        highest = np.argmax(array, axis = 0)
        output_part[highest] = 1
        output_array.append(output_part)
        
    return np.array(output_array)    


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(12)
    plt.xticks(tick_marks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], rotation=45)
    plt.yticks(tick_marks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
def plot_precision_recall_curve(recall, precision):
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'red', 'green', 'teal', 'black', 'purple', 'gold', 'greenyellow', 'deeppink'])

    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
        
    for i, color in zip(range(12), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} '.format(i))
            
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


    plt.show()

class evaluationCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, log = None):
        print("Starting epoch {}".format(epoch+1))
    def on_epoch_end(self, epoch, log = None):
        global validation_categories
        result = model.predict(validation_annot_array)
        
        predicted_result = np.argmax(result, axis=1)
        standartized_predicted_result = standartize(result)
        true_result = np.argmax(validation_categories, axis=1)
        
        #Confusion matrix
        cm = metric.confusion_matrix(true_result, predicted_result)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print('Normalized confusion matrix')
        #print(cm_normalized)
        plt.figure()
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

        
        #F1-Score
        f1 = metric.f1_score(true_result, predicted_result, average = None)
        print("F1-Score is {}".format(f1))
        
        precision = dict()  
        recall = dict()
        
        #Category-specific metrics
        for current_category in all_categories:
        
            category_number = np.argmax(current_category, axis = 0)    
        
            #Array of 0s & one 1 showing which true results are of the current category
            y_true_aktuell = np.array([np.array_equal(current_category, t) for t in validation_categories], dtype=np.uint8)
            
        
            #Array of 0s & one 1 showing which predicted results are of the current category
            #y_pred_aktuell = predicted_results[:, category_number]
            y_pred_aktuell = np.array([np.array_equal(current_category, t) for t in standartized_predicted_result], dtype=np.uint8)
        
            #Calculating values
            f1_score = metric.f1_score(y_true_aktuell, y_pred_aktuell)
            #Threshold is missing!!! Include!!!
            #Expecting only probabilities for current class as 1-dim vector
            precision[category_number], recall[category_number], _ = metric.precision_recall_curve(y_true_aktuell, y_pred_aktuell)
            average_prec = metric.average_precision_score(y_true_aktuell, y_pred_aktuell)
        
            print('F1-Score of class {} is {}'.format(category_number, f1_score))
            print('Average Precision of class {} is {}'.format(category_number, average_prec))

        plot_precision_recall_curve(recall, precision)

###IMPORT DATASET
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
#Get training data
rel_path_train_annot = "../Flurplandaten/preprocessed__training_annotations.p"
train_annot_path = os.path.join(script_dir, rel_path_train_annot)

rel_path_train_categories = "../Flurplandaten/object_list_for_training_annotations.p"
train_categories_path = os.path.join(script_dir, rel_path_train_categories)

train_annot = np.array(pickle.load(open(train_annot_path, 'rb')))
train_annot_array = np.reshape(train_annot, (train_annot.shape[0], 100, 100, 3))
train_categories = pickle.load(open(train_categories_path, 'rb'))

#Get validation data
rel_path_validation_annot = "../Flurplandaten/preprocessed__validation_annotations.p"
validation_annot_path = os.path.join(script_dir, rel_path_validation_annot)

rel_path_validation_categories = "../Flurplandaten/object_list_for_validation_annotations.p"
validation_categories_path = os.path.join(script_dir, rel_path_validation_categories)

validation_annot = np.array(pickle.load(open(validation_annot_path, 'rb')))
validation_annot_array = np.reshape(validation_annot, (validation_annot.shape[0], 100, 100, 3))
validation_categories = pickle.load(open(validation_categories_path, 'rb'))

#Get test data
rel_path_test_annot = "../Flurplandaten/preprocessed__test_annotations.p"
test_annot_path = os.path.join(script_dir, rel_path_test_annot)

rel_path_test_categories = "../Flurplandaten/object_list_for_test_annotations.p"
test_categories_path = os.path.join(script_dir, rel_path_test_categories)

test_annot = np.array(pickle.load(open(test_annot_path, 'rb')))
test_annot_array = np.reshape(test_annot, (test_annot.shape[0], 100, 100, 3))
test_categories = pickle.load(open(test_categories_path, 'rb'))

###TRAINING
#Get model
base_model = keras.applications.InceptionV3(weights = 'imagenet', include_top = False, classes = 12, input_shape = (100,100,3))

x = base_model.output
x = layers.Flatten()(x)
predictions = keras.layers.Dense(12, activation='softmax')(x)

#Train model
opti = keras.optimizers.SGD(lr = 0.001)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer = opti, loss = "categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

model.fit(x = train_annot_array, y = np.array(train_categories), batch_size = 128, epochs = 50, callbacks = [evaluationCallback()])


#Change number!!! Save net
net_path = os.path.join(script_dir, "../Netze/try4.h5")

model.save(net_path)