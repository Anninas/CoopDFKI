# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:13:44 2020

@author: annika
"""

import pickle
import os

script_dir = os.path.dirname(__file__)
annotation_path = os.path.join(script_dir, "../Flurplandaten/preprocessed_annotations.p")

annotation_test = pickle.load(open(annotation_path, "rb"))

object_path = os.path.join(script_dir, "../Flurplandaten/object_list_for_annotations.p")

object_test = pickle.load(open(object_path, "rb"))