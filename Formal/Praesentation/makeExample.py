# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:43:35 2020

@author: annika
"""

import json
from os import listdir
from os.path import join
import os
import numpy as np
import PIL
import PIL.ImageOps
import matplotlib.pyplot as plt



img_path = "C:/Users/annika/Documents/GitHub/CoopDFKI/Formal/Praesentation/Beispielannotation.jpg"

img = np.array(PIL.Image.open(img_path).convert('L'))

np.savetxt("beispiel2.txt", img, fmt="%d")