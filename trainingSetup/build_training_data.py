#!../.venv/bin/python
'''
File is used to build a zip file which includes:
1) Training data: .png images and .txt files (<classid> <relative_center_x> <relative_center_y> <width> <height>)
Note: * attention: <relative_center_x> <relative_center_y> - are center of rectangle (not top-left corner)
reference: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

2) classes.txt: which contains all the labels for the training

Note: If no known finder is supplied via cmd flag (--finder), then KnownGlycanBoxes finder will be used automatically.
Else, specify a known finder like: KnownMono, KnownRoot, KnownLink...
'''

import os
import sys
import argparse
import shutil
import tempfile
import atexit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BKGlycanExtractor import Config_Manager
from BKGlycanExtractor.training_utils import build_training

parser = argparse.ArgumentParser(description="Build training data")

parser.add_argument(
    '--finder',
    type = str,
    default = "KnownGlycanBoxes",
    help = 'Finder for known boxes on images. Usually, one of KnownGlycanBoxes, KnownMono, KnownRoot, KnownLink, or KnownLinkWithInfo.'
)

parser.add_argument(
    '--label_type',
    type = str,
    required = False,
    default = None,
    help = 'Label type used to build training data. The type can be selected from the TSV file.'
)

parser.add_argument(
    '--boxpadding',
    type=int,
    required=False,
    help='Box padding on known boxes. Default: from named known finder\'s config.'
)

parser.add_argument(
    '--images',
    type = str,
    nargs = "+",
    required = True,
    help = "One or more paths - directories and/or files. Required"
)

parser.add_argument(
    '--out',
    type = str,
    default = 'images.zip',
    help = 'Filename for training data zip file, must end in .zip. Default: images.zip'
)

args = parser.parse_args()

config = Config_Manager()

build_training(
    config=config,
    finder_name=args.finder,
    images=args.images,
    out_zip=args.out,
    boxpadding=args.boxpadding,
    label_type=args.label_type
)
print("Training data is ready...")
print(args.out)
