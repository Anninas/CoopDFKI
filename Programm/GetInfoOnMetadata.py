# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:55:49 2020

@author: anni
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import json

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/flo2plan_icdar_instances.json"
metadata_path = os.path.join(script_dir, rel_path)

#Load json file    
metadata = json.loads(open(metadata_path, 'r').read())



category_counts = np.zeros(12)

for annotation in metadata['annotations']:
    
    number = annotation['category_id']
    
    category_counts[number-1] += 1
    
print(category_counts)

plt.bar(('toilet', 'shower', 'bathtub', 'sink', 'bidet', 'table', 'chair', 'couch', 'armchair', 'night_table', 'bed', 'hot_plate'), category_counts)
plt.xticks(('toilet', 'shower', 'bathtub', 'sink', 'bidet', 'table', 'chair', 'couch', 'armchair', 'night_table', 'bed', 'hot_plate'), rotation = 45)
plt.show()
    