import os
import sys

import matplotlib
# matplotlib.use('tkagg')
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
import argparse

import modeltests as mt
from BKGlycanExtractor.glycanannotator import Config_Manager


'''
optinal CMD arguments:
1) known_semantics pipeline [optional, default-'KnownSemantics'] - for ground truth data
2) test pipelines [optional, default-'YOLOMonosAnnotator'] - can be a single/multiple pipelines
3) directory path [required] - where all txt files (ground truth) and png/jpg(test),etc files are stores
'''

parser = argparse.ArgumentParser(description="Start")

# optional argument
parser.add_argument(
    '--known_pipeline_name',
    type = str,
    default = 'KnownSemantics',
    help = 'Known Semantics Pipeline (default: KnownSemantics)'
)

# optional argument
parser.add_argument(
    '--pipeline_name',
    type = str,
    nargs = '+', # allows one or more values
    default = ['YOLOMonosAnnotator'],
    help = 'Test pipeline(s) (default: YOLOMonosAnnotator)'
)

# required argument
parser.add_argument(
    '--data_folder',
    type = str,
    required = True,
    help = 'Directory path where all txt files and png/jpg files are stored (required)'
)

args = parser.parse_args()
km_pipeline_name = args.known_pipeline_name
pipeline_name = args.pipeline_name
data_folder = args.data_folder


'''
created find_boxes method for the YOLOMonos and KnownMonos
find_boxes() --> unpadded/padded boxes for YOLOMonos 

Finder_Evaluator class handles:
- reading the folder and formatting the data and configs
- getting boxes data - KnownMonos() and YOLOMonos()
- comparing the ground truth and predicted boxes based on IOU
- producing precision-recall plot
'''


if __name__ == "__main__": 
    config = Config_Manager()

    # pipeline accepts one/multiple names
    pipeline = config.get(pipeline_name)
    pipeline_km = config.get(km_pipeline_name)

    evaluator = mt.Finder_Evaluator(pipeline,pipeline_km)

    # maybe generate/store x-y coordinates and keep the work of plotting for the user
    evaluator.plotprecisionrecall(data_folder)

    print("Done")






