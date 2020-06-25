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
import pickle
import sklearn.metrics as metric
from itertools import cycle

###GLOBAL VARIABLES
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

trainResults = {}

###FUNCTION SECTION
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
    
    def __init__(self, currentBatchSize, currentLearnRate):
        self.currentBatchSize = currentBatchSize
        self.currentLearnRate = currentLearnRate
    
    def on_epoch_begin(self, epoch, log = None):
        print("Starting epoch {}".format(epoch+1))
        
    def on_epoch_end(self, epoch, log = None):
        global validation_categories
        
        #Result of net - exact probabilitites
        result = model.predict(validation_annot_array)
        
        #Result of net - numbers of the predicted classes
        predicted_result = np.argmax(result, axis=1)
        
        #Result of net - binary with eleven zeros and one one
        standartized_predicted_result = standartize(result)
        
        #True result - numbers of the true classes
        true_result = np.argmax(validation_categories, axis=1)
        
        #Confusion matrix
        cm = metric.confusion_matrix(true_result, predicted_result)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

        
        #F1-Score
        f1 = metric.f1_score(true_result, predicted_result, average = None)
        print("F1-Score is {}".format(f1))
        trainResults["ADAM: {},{}, Epoch: {}".format(self.currentBatchSize, self.currentLearnRate, epoch)] = f1
        
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
learnRates = [0.01, 0.001, 0.0001]

def trainNet(model, train_annot_array, train_categories):
    
    for batchSize in batchSizes:
        
        for learnRate in learnRates:
            
            #Choose optimizer
            opti = keras.optimizers.Adam(lr = learnRate)
            
            #Compile model
            model.compile(optimizer = opti, loss = "categorical_crossentropy", metrics=["accuracy"])

            print(model.summary())
            
            #Actual training with callback for evaluation
            model.fit(x = train_annot_array, y = np.array(train_categories), batch_size = batchSize, epochs = 50, callbacks = [evaluationCallback(batchSize, learnRate)])

###IMPORT DATASET
#Get path of the script to work with relative paths later
script_dir = os.path.dirname(__file__) 

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

model = Model(inputs=base_model.input, outputs=predictions)

#Train model automatedly while varying hyperparameters
trainNet(model, train_annot_array, train_categories)

###SAVING RESULTS
#Convert saved training results (dict of numpy arrays) to dict of lists
new_trainResults = {}
for key in trainResults:
    new_trainResults[key] = trainResults[key].tolist()

#Save training results for evaluation
new_path = os.path.join(os.path.dirname(__file__), "../Formal/f1_scores_automated_training.json")
with open(new_path, 'w') as path:
    json.dump(new_trainResults, path)

#Save net
net_path = os.path.join(script_dir, "../Netze/try4.h5")

model.save(net_path)