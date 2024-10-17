import os
import sys

import matplotlib
# matplotlib.use('tkagg')
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
import argparse

# import modeltests as mt
from BKGlycanExtractor import BoxEvaluator


'''
optinal CMD arguments:
1) known_semantics pipeline [optional, default-'KnownSemantics'] - for ground truth data
2) test pipelines [optional, default-'YOLOMonosAnnotator'] - can be a single/multiple pipelines
3) directory path [required] - where all txt files (ground truth) and png/jpg(test),etc files are stores
'''

parser = argparse.ArgumentParser(description="Start")

# optional argument
parser.add_argument(
    '--base_pipeline',
    type = str,
    default = 'SingleGlycanImage-YOLOFinders',
    help = 'Base Pipeline (default: SingleGlycanImage-YOLOFinders)'
)

# optional argument
parser.add_argument(
    '--known_finder',
    type = str,
    default = 'KnownMono',
    help = 'Known Semantics Pipeline (default: KnownMono)'
)

# optional argument
parser.add_argument(
    '--pred_finder',
    type = str,
    nargs = '+', # allows one or more values
    default = ['YOLOMonosRandom'],
    # default = 'YOLOMonosRandom',
    help = 'Test pipeline(s) (default: YOLOMonosRandom)'
)

# required argument
parser.add_argument(
    '--image_folder',
    type = str,
    required = True,
    help = 'Directory path where all txt files and png/jpg files are stored (required)'
)

args = parser.parse_args()
base_pipeline = args.base_pipeline
known_finder = args.known_finder
pred_finder = args.pred_finder
image_folder = args.image_folder


if __name__ == "__main__": 
    evaluator = BoxEvaluator(pred_finder, base_pipeline=base_pipeline,known=known_finder)
    evaluator.runall(image_folder)
    evaluator.plotprecisionrecall()

    print("Done")






