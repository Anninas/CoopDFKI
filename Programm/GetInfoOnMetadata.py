# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:55:49 2020

@author: anni
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
import json

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/floorplan_metadata_cleaned_nobidet.json"
metadata_path = os.path.join(script_dir, rel_path)

#Load json file    
metadata = json.loads(open(metadata_path, 'r').read())

rel_path = "../Flurplandaten/images_diagonals"
path = os.path.join(script_dir, rel_path)

category_counts = np.zeros(12)

for annotation in metadata['annotations']:
    
    if(metadata['images'][(annotation['image_id'])-1]['file_name'] in listdir(path)):
    
        number = annotation['category_id']
    
        category_counts[number-1] += 1
    
print(category_counts)

plt.bar(('toilet', 'shower', 'bathtub', 'sink',  'table', 'chair', 'couch', 'armchair', 'night_table', 'bed', 'hot_plate', 'negative'), category_counts)
plt.xticks(('toilet', 'shower', 'bathtub', 'sink',  'table', 'chair', 'couch', 'armchair', 'night_table', 'bed', 'hot_plate', 'negative'), rotation = 45)
plt.show()
    