# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 13:55:49 2020

@author: anni
"""

import numpy as np
import matplotlib.pyplot as plt
from addSlicingSymbols import object_categories

category_counts = np.zeros(12)

for number in object_categories:
    category_counts[number-1]+= 1
    
for i in range(12):
    category_counts[i] = category_counts[i]/21
    
print(category_counts)

counts, edges, plot = plt.hist(object_categories, bins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
print(counts)
    