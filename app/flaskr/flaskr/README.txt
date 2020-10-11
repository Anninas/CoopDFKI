How to use all these files:

Programms for Dataset Prep and Training:

The first files for chronological usage would be CleanMetadataDeleteBidetAddNegativeClass.py, CleanMetadataDeleteBidet.py and Binary Metadata. The first cleans the metadata and is the newest of the three CleanMetadata-programs, thus removing bidet and adding negative class to prepare the metadata for BinaryMetadata which makes it only include "Object" and "Background". The second one creates a file without the negative class for the creation of the object dataset for the object classification.

The next programs to use are the PreprocessDataset.py and PreprocessBinaryDataset.py files. They preprocess and augmentsthe annotations and save them in a usable shape for the net. Relevant changes to do are the first appearance of rel_path where one has to insert ones metadata name from CleanMetadataDeleteBidetAddNegativeClass.py and the third appearance fo rel_path where one has to choose the image folder depending on which part of the dataset one wants to preprocess. Further, one should comment out the for-loop starting in line 277 if the validation or test dataset is being preprocessed as this loop includes the augmentation. Remember to remove the comments for preprocession of training annotations. Last, a name has to be chosen for the annotation_path and the object_path to save ones results of preprocessing in a way one can find and identify it again.

Now, the nex programs are BuildAndTrainNet.py and BuildAndTrainBinaryNetOneOutputNeuron. These program train the two nets for the cascadic prediction and validate their performance. First, one has to change new_path and new_path2 to what one wants to save the net and its results as. If one wants to save the error predictions, one also has to change the error_path. At batchSizes and learnRates, adaptions about the hyperparameters can be made. Changes about the model can be made in def getModel(). If one wants to save the error predictions, remove the comments from line 173. From line 306 onwards, adapt the dataset paths for training, validation and test to what you chose at PreprocessDataset files. In line 362, choose a name for your net. When you run the file, it asks to choose between automated and non-automated training. For the first, just type Y, for the second type N and enter the hyperparameters as the program asks for them.


Programs in the application:

__init__.py and flaskr.py are help files to set up the app.

upload.py is the file that handles the url requests and thus calls the html templates and the PredictAndPostprocess.py file.

PredictAndPostprocess.py is the major file for the app. It includes the two necessary functions getPrediction() to have the net predict the user's input and getPostprocessedResults() to transform the prediction into the result image and result json. 

The net files were to large to fit into the zip file of the application (app.zip). They have to be trained and then placed in the Netze-folder inside app/flaskr/flaskr after training and the names should be try22_IRV2_binaryOne_64-0.0001-9.h5 for the binary net with hyperparameters batch size 64, learn rate 0.0001, optimizer adam and epochs 9 and try10_IncResV2_randomrotation.h5 for the object classification with hyperparameters batch size 128, learn rate 0.01, optimizer sgd and epochs 20. 

Requirements for running the application:
Python 2.7
Tensorflow GPU 2.3.0
Cuda 10.0
CuDNN 7.6.5 for Cuda 10.0
OpenCV
Numpy
PIL 
Itertools
Flask