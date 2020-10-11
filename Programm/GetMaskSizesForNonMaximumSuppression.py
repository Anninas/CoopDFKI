# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 13:53:18 2020

@author: annika
"""
import json

import os

import numpy as np



script_dir = os.path.dirname(__file__)

rel_path = "../Flurplandaten/floorplan_metadata_cleaned_nobidet.json"
metadata_path = os.path.join(script_dir, rel_path)   
metadata = json.loads(open(metadata_path, 'r').read())

results = {}

for i in range(12):
    results["class" + str(i)+"height"]=[]
    results["class" + str(i)+"width"]=[]

for annotation in metadata["annotations"]:
    
    try:
        results["class" + str(annotation["category_id"]-1) + "height"].append(annotation["bbox"][3])
        results["class" + str(annotation["category_id"]-1) + "width"].append(annotation["bbox"][2])
    except KeyError:
        print("Image not available -> wrong indexing")
   
for i in range(12):
    results["class" + str(i)+"height"]=np.average(results["class" + str(i)+"height"])
    results["class" + str(i)+"width"]=np.average(results["class" + str(i)+"width"])