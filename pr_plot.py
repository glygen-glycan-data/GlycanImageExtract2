import os
import numpy as np
import cv2
import sys

import matplotlib
# matplotlib.use('tkagg')
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  

import BKGlycanExtractor.monosaccharideid as mono_finder
import modeltests as mt

try:
    configs_dir = sys.argv[1]
except IndexError:
    configs_dir = "./BKGlycanExtractor/config/"
configs_file = os.path.join(configs_dir, "configs.ini")

try:
    data_folder = sys.argv[2]
except IndexError:
    data_folder = "./data/"

'''
created find_boxes method for the YOLOMonos and KnownMonos
find_boxes() --> generates padded and unpadded boxes for YOLOMonos 
but find_objects() generates only unpadded_boxes with the entire semantic obj structure

TestModel class handles:
- reading the folder and formatting the data and configs
- getting boxes data - KnownMonos() and YOLOMonos()
- comparing the ground truth and predicted boxes based on IOU
- producing precision-recall plot
'''

pipeline_name = "BaseFinder"


if __name__ == "__main__": 
    pipeline = mt.TestModel(configs_dir, configs_file, pipeline_name, data_folder)

    pipeline.run() 
    pipeline.plotprecisionrecall() 

    print("Done")


                



