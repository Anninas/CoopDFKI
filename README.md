# CoopDFKI
 Includes all programs for the cooperation project with the DFKI Kaiserslautern supvervised by Hector Seminar.

How to use all these files:

Important folders for usage are Flurplandaten and Programm. The trained net will be saved in Netze.

Dataset:

In Flurplandaten, three json files can be found. Those are different versions of the metadata. flo2plan_icdar_instances.json is the original one. floorplan_metadata_cleaned.json comes from a first cleaning step where bigger issues like annotations of non-existing images were removed. floorplan_metadata_cleaned_nobidet.json includes these corrections, the fusion of the bidet class with the toilet class and the addition of the negative class and thus creates the newest and meant for usage version of the metadata.

The four folders in Flurplandaten include the images. The images folder includes all of them. images_only rectangular contains the training dataset, images_diagonals contains the validation images and images_roundings contains the test data. 

The pickle files in Flurplandaten are comprimised preprocessed annotations and object lists corresponding to them in different stages. The newest ones are the ones with _ negative at the end. So for test and training it is obvious which one to use. For validation it is important to use the one with noaugmentation_negative as the results will be eway worse if validation data is augmented. 

Programms:

In the Programm folder, the first file for chronological usage would be CleanMetadataDeleteBidetAddNegativeClass.py. It cleans the metadata and is the newest of the three CleanMetadata-programs, thus removing bidet and adding negative class. One might want to change the variable new_path to their own understandable name of the created metadata file. 

The next program to use is the PreprocessDataset.py file. It preprocesses and augments the annotations and saves them in a usable shape for the net. Relevant changes to do are the first appearance of rel_path where one has to insert ones metadata name and the third appearance fo rel_path where one has to choose the image folder depending on which part of the dataset one wants to preprocess. Further, one should comment out the for-loop starting in line 277 if the validation or test dataset is being preprocessed as this loop includes the augmentation. Remember to remove the comments for preprocession of training annotations. Last, a name has to be chosen for the annotation_path and the object_path to save ones results of preprocessing in a way one can find and identify it again.

Now, the nex program is BuildAndTrainNet.py. This program trains the net and validates its performance. First, one has to change new_path and new_path2 to what one wants to save the net and its results as. If one wants to save the error predictions, one also has to change the error_path. At batchSizes and learnRates, adaptions about the hyperparameters can be made. Changes about the model can be made in def getModel(). If one wants to save the error predictions, remove the comments from line 173. From line 306 onwards, adapt the dataset paths for training, validation and test. In line 362, choose a name for your net. When you run the file, it asks to choose between automated and non-automated training. For the first, just type Y, for the second type N and enter the hyperparameters as the program asks for them.

To predict a file with the trained net, open PredictUsageFloorplan.py. Adapt the floorplan_path and the net_path if necessary. Also adapt the result files in lines 93 and 96. Then run the file, enter the name of the floorplan to predict.

The prediction can then be processed and checked with ClassificationToDetection.py. Here, adapt the paths in line 38, 58 and 112. The outputs can be checked in the folders Kontrollbilder and KontrollbilderNonMax at Program. 

All other Program files are tests or additions. Their functions are listed in the following:

AnnotationSizeHistogram.py - Get the sizes of the annotations displayed visually. Used to decide to delete everything larger than 100 or smaller than 10 to normalize the sizes.

GetInfoOnMetadata.py - Show numbers of annotations per dataset part visually and as numbers.

pickleTest.py - As name says it, testing pickle.

PredictusageFloorplanAllAtOnce.py - Trying to predict all image snippets at once for usage, didn't work well.
