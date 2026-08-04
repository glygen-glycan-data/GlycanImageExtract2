#!../.venv/bin/python
'''
File is used to build a zip file which includes:
1) Training data: .png images and .txt files (<classid> <relative_center_x> <relative_center_y> <width> <height>)
Note: * attention: <relative_center_x> <relative_center_y> - are center of rectangle (not top-left corner)
reference: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

2) classes.txt: which contains all the labels for the training

Label arguments are only for Glycan boxes currently, can be extended for other components but get_know_data() will have to be extended 

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
from BKGlycanExtractor.image_manager import Image_Manager, StructuredSampling

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
    help = 'Label type used to build glycan training data. The type can be selected from the TSV file.'
)

parser.add_argument(
    '--label_substitutions',
    type = str,
    required = False,
    nargs = '+',
    default = [],
    help = 'Substitute glycan label name(s) with alternative label(s). Format <current_label_name>:<new_label_name>'
)

parser.add_argument(
    '--default_label',
    type = str,
    required = False,
    default = None,
    help = 'Default glycan label used for boxes to build training data.'
)

parser.add_argument(
    '--exclude_labels',
    type = str,
    required = False,
    nargs="+",
    default = [],
    help = 'Glycan Labels to exclude from training data.'
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
    default = "images",
    help = 'Name for dataset, training data zip file will be <out>-train.zip, if test_percent is set, else <out>.zip'
)

parser.add_argument(
    '--test_percent',
    type = float,
    default = 0.0,
    help = 'Percent (%%) of images to select for testing data zip file, <out>-test.zip. Default: No testing data.'
)

parser.add_argument(
    '--split_seed',
    type = int,
    default = None,
    help = 'Random seed for train/test image split (only used when --test_percent > 0)'
)

parser.add_argument(
    '-F',
    '--force',
    action = 'store_true',
    default = False,
    help = 'Overwrite output zip files, if they exist.'
)

parser.add_argument(
    '-q',
    '--quiet',
    action = 'store_true',
    default = False,
    help = 'Run without extra output.'
)

args = parser.parse_args()

if args.out.endswith('.zip'):
    raise ValueError("Output name should not end in .zip")

if args.test_percent > 0.0:
    if os.path.exists(args.out + "-train.zip") and not args.force:
        raise AssertionError(f"Zip file {args.out}-train.zip exists")

    if os.path.exists(args.out + "-test.zip") and not args.force:
        raise AssertionError(f"Zip file {args.out}-test.zip exists")
else:
    if os.path.exists(args.out + ".zip") and not args.force:
        raise AssertionError(f"Zip file {args.out+".zip"} exists")

assert args.test_percent == 0.0 or args.test_percent >= 1.0, f"Bad testing percent: {args.test_percent}%."

config = Config_Manager()
finder = config.get_finder(args.finder)

if args.boxpadding is not None:
    finder.set_param('boxpadding', args.boxpadding)

def parse_label_substitutions(pairs):
    subs = {}
    for pair in pairs:
        if ':' not in pair:
            raise ValueError(f"Invalid label substitution {pair}; expected old:new")
        old, new = pair.split(':', 1)
        subs[old] = new
    return subs


# if a label_type was provided for substitution, then it will be picked 
# from the semantics file and substituted as the
# classlabel for the known boxes

# Note: - all exclude labels items are removed first and then label substitutions for the
# remaining data is done.
finder.set_default_label(None)
if args.default_label is not None:
    finder.set_default_label(args.default_label)

if args.label_type is not None:
    finder.set_label_type(args.label_type)

    finder.set_exclude_labels([])
    if args.exclude_labels:
        finder.set_exclude_labels(args.exclude_labels)

    if args.label_substitutions:
        finder.set_label_substitutions(parse_label_substitutions(args.label_substitutions))

images = Image_Manager(args.images,strategy=StructuredSampling())
images.exclude() # *.annotated*.{png,jpg,jpeg} by default

build_training(
    finder = finder,
    images = images,
    outname= args.out,
    test_frac=args.test_percent/100.0,
    split_seed = args.split_seed,
    quiet = args.quiet
)
