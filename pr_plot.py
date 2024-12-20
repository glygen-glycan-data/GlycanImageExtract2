import os
import sys

import matplotlib
# matplotlib.use('tkagg')
matplotlib.use('Agg') 
import matplotlib.pyplot as plt  
import argparse

# import modeltests as mt
from BKGlycanExtractor import BoxEvaluator,Config_Manager,GlycanExtractorPipeline


'''
optinal CMD arguments:
1) known_semantics pipeline [optional, default-'KnownSemantics'] - for ground truth data
2) test pipelines [optional, default-'YOLOMonosAnnotator'] - can be a single/multiple pipelines
3) directory path [required] - where all txt files (ground truth) and png/jpg(test),etc files are stores
'''

parser = argparse.ArgumentParser(description="Start")

# optional argument
# parser.add_argument(
#     '--base_finder',
#     type = str,
#     default = 'SingleGlycanImage',
#     help = 'Base Finder (default: SingleGlycanImag)'
# )

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
# base_finder = args.base_finder
known_finder = args.known_finder
pred_finder = args.pred_finder
image_folder = args.image_folder

# instantiate all the finders before providing it to the BOXEvaluator
# pred_finder, base_pipeline=base_pipeline,known=known_finder
# box_padding can be taken from configs or provided by user

config = Config_Manager()

sgi = config.get_finder("SingleGlycanImage")
pipeline = GlycanExtractorPipeline()
pipeline.add_step("figure",sgi)

known = config.get_finder(known_finder)

predictors = []
for pred in pred_finder:
    predictors.append(config.get_finder(pred))


# evaluator = BoxEvaluator(pred_finder, base_pipeline=pipeline0,known=known)
evaluator = BoxEvaluator()
evaluator.runall(image_folder,pipeline,predictors,known)
# evaluator.runall(image_folder,pipeline,predictors,known, iouthr) # add IOU threshold as well
evaluator.plotprecisionrecall()

print("Done")






