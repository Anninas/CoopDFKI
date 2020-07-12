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
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Model
import tensorflow as tf
import pickle
import sklearn.metrics as metric
from itertools import cycle

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

new_path = os.path.join(os.path.dirname(__file__), "../Formal/f1_scores_automated_training_8_nobidet_Res50.json")


###GLOBAL VARIABLES
all_categories = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

trainResults = {}

###FUNCTION SECTION
def standartize(input_array):
    
    output_array = []
    
    for array in input_array:
        output_part = np.zeros(11)
        highest = np.argmax(array, axis = 0)
        output_part[highest] = 1
        output_array.append(output_part)
        
    return np.array(output_array)    


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(11)
    plt.xticks(tick_marks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], rotation=45)
    plt.yticks(tick_marks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
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
        
    for i, color in zip(range(11), colors):
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

def getModel():
    #Get model
    #ResNet 50: base_model = keras.applications.resnet50(weights = 'imagenet', include_top = False, classes = 12, input_shape = (100,100,3))
    #Ggf. vgg19
    #InceptionResnetV2
    base_model = keras.applications.ResNet50(weights = 'imagenet', include_top = False, classes = 11, input_shape = (100,100,3))
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.7)(x)
    predictions = keras.layers.Dense(11, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

error_path = os.path.join(os.path.dirname(__file__),"../Flurplandaten/FalsePredictions")

class evaluationCallback(keras.callbacks.Callback):
    
    def __init__(self, currentBatchSize, currentLearnRate):
        self.currentBatchSize = currentBatchSize
        self.currentLearnRate = currentLearnRate
    
    def on_epoch_begin(self, epoch, log = None):
        print("Starting epoch {}".format(epoch+1))
        
    def on_epoch_end(self, epoch, log = None):
        global validation_categories
        
        print("Learn Rate: {}; Batch Size: {}".format(self.currentLearnRate, self.currentBatchSize))
        
        #Result of net - exact probabilitites
        result = model.predict(validation_annot_array)
        
        #Result of net - numbers of the predicted classes
        predicted_result = np.argmax(result, axis=1)
        
        #Result of net - binary with eleven zeros and one one
        standartized_predicted_result = standartize(result)
        
        #True result - numbers of the true classes
        true_result = np.argmax(validation_categories, axis=1)
        
        #Get all annotations that were not predicted correctly to find pattern
        mask = predicted_result==true_result
        
        current_error_path = os.path.join(error_path, "{},{},{}".format(self.currentBatchSize, self.currentLearnRate, epoch))
        if(not os.path.exists(current_error_path)):
            os.makedirs(current_error_path)
        
        for number, annotation in enumerate(validation_annot_array[~mask]):
            plt.imsave(os.path.join(current_error_path, "wrong_annotation{}.png".format(number)), annotation)
            print(current_error_path)
        
        #Confusion matrix
        cm = metric.confusion_matrix(true_result, predicted_result)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

        
        #F1-Score
        f1 = metric.f1_score(true_result, predicted_result, average = None)
        print("F1-Score is {}".format(f1))
        
        #Save results only for automated training
        if(train_mode == "Y"):
                
            #Save training results for evaluation
            with open(new_path, 'w') as path:
                trainResults["SGD: {},{}, Epoch: {}".format(self.currentBatchSize, self.currentLearnRate, epoch)] = f1.tolist()
                json.dump(trainResults, path)
        
        precision = dict()  
        recall = dict()
        thresholds = dict()
        
        #Category-specific metrics
        for current_category in all_categories:
        
            category_number = np.argmax(current_category, axis = 0)    
        
            #Array of 0s & one 1 showing which true results are of the current category (1)
            y_true_aktuell = np.array([np.array_equal(current_category, t) for t in validation_categories], dtype=np.uint8)
            
            #Result of net - Array of 0s & one 1 showing if results are of current category (1) or not (0)
            #y_pred_aktuell_binary = np.array([np.array_equal(current_category, t) for t in standartized_predicted_result], dtype=np.uint8)
            
            #Result of net - exact probabilities for one class
            y_pred_aktuell = result[:, category_number]
            
            #Expecting only probabilities for current class as 1-dim vector
            precision[category_number], recall[category_number], thresholds[category_number] = metric.precision_recall_curve(y_true_aktuell, y_pred_aktuell)
            average_prec = metric.average_precision_score(y_true_aktuell, y_pred_aktuell)
        
            print('Average Precision of class {} is {}'.format(category_number, average_prec))

        plot_precision_recall_curve(recall, precision)


#Automated training of the net with varying batchSize and learn rate
        
batchSizes = [32, 64, 128, 256]
learnRates = [0.1, 0.01, 0.001, 0.0001]

def trainNet(train_annot_array, train_categories):
    
    global model
    
    for batchSize in batchSizes:
        
        for learnRate in learnRates:
                      
            #Reinstantiate model
            model = getModel()
            
            #Choose optimizer
            opti = keras.optimizers.SGD(lr = learnRate)
            
            #Compile model
            model.compile(optimizer = opti, loss = "categorical_crossentropy", metrics=["accuracy"])

            print(model.summary())
            print("We have batch size {} and learn rate {}".format(batchSize, learnRate))
            
            #Actual training with callback for evaluation
            model.fit(x = train_annot_array, y = np.array(train_categories), batch_size = batchSize, epochs = 50, callbacks = [evaluationCallback(batchSize, learnRate)]) 
    
###IMPORT DATASET
#Get path of the script to work with relative paths later
script_dir = os.path.dirname(__file__) 

#Get training data
rel_path_train_annot = "../Flurplandaten/preprocessed__training_annotations_nobidet.p"
train_annot_path = os.path.join(script_dir, rel_path_train_annot)

rel_path_train_categories = "../Flurplandaten/object_list_for_training_annotations_nobidet.p"
train_categories_path = os.path.join(script_dir, rel_path_train_categories)

train_annot = np.array(pickle.load(open(train_annot_path, 'rb')))
train_annot_array = np.reshape(train_annot, (train_annot.shape[0], 100, 100, 3))
train_categories = pickle.load(open(train_categories_path, 'rb'))

#Get validation data
rel_path_validation_annot = "../Flurplandaten/preprocessed__validation_annotations_nobidet_noaugmentation.p"
validation_annot_path = os.path.join(script_dir, rel_path_validation_annot)

rel_path_validation_categories = "../Flurplandaten/object_list_for_validation_annotations_nobidet_noaugmentation.p"
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
#Ask if training should be automated
train_mode = input("Do you want to train automatedly? Y/N: ")

if(train_mode == "Y"):
    
    print("Automated training is started.")
    
    #Train model automatedly while varying hyperparameters
    trainNet(train_annot_array, train_categories)

else:
    #Train model non-automatedly
    batchSize = int(input("You want to train non-automatedly. Enter batch size:"))
    learnRate = float(input("Now enter the learn rate:"))
    epochs = int(input("How many epochs do you want to train?"))
    opti = keras.optimizers.SGD(lr = learnRate)

    model = getModel()
    model.compile(optimizer = opti, loss = "categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    model.fit(x = train_annot_array, y = np.array(train_categories), batch_size = batchSize, epochs = epochs, callbacks = [evaluationCallback(batchSize, learnRate)])

#Save net
net_path = os.path.join(script_dir, "../Netze/try9_Res50.h5")

model.save(net_path)