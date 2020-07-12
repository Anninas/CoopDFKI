# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:06:52 2020

@author: anni
"""
import os
import json
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Flurplandaten/floorplan_metadata_cleaned_nobidet.json"
metadata_path = os.path.join(script_dir, rel_path)

#Load json file    
metadata = json.loads(open(metadata_path, 'r').read())

numrows = []
numcolumns = []

for annotation in metadata['annotations']:
    
    numrow = annotation['bbox'][3]
    numcolumn = annotation['bbox'][2]
    
    numrows.append(numrow)
    numcolumns.append(numcolumn)
'''
counts, edges, plot = plt.hist(numrows, bins=[0, 10, 30, 50, 100, 200, 300, 400, 500,600])
print("Rows:")
print(counts)

print("0-10: " + str(counts[0]))
print("10-30: " + str(counts[1]))
print("30-50: " + str(counts[2]))
print("50-100: " + str(counts[3]))
print("100-200: " + str(counts[4]))
print("200-300: " + str(counts[5]))
print("300-400: " + str(counts[6]))
print("400-500: " + str(counts[7]))
print("500-600: " + str(counts[8]))
'''

counts2, edges2, plot2 = plt.hist(numcolumns, bins=[0, 10, 30, 50, 100, 200, 300, 400, 500,600])
print("Columns:")
print(counts2)

print("0-10: " + str(counts2[0]))
print("10-30: " + str(counts2[1]))
print("30-50: " + str(counts2[2]))
print("50-100: " + str(counts2[3]))
print("100-200: " + str(counts2[4]))
print("200-300: " + str(counts2[5]))
print("300-400: " + str(counts2[6]))
print("400-500: " + str(counts2[7]))
print("500-600: " + str(counts2[8]))

#Ergebnis: 100*100 wählen, alles kleiner 10 raus, alles über 100 raus