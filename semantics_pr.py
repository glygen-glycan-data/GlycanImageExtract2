#!.venv/bin/python
import os
import sys
import argparse
import logging
from BKGlycanExtractor import Image_Manager, SemanticEvaluator, Config_Manager, DebugMode
 
parser = argparse.ArgumentParser(description="Start")

parser.add_argument(
    '--finders',
    type = str,
    required = True,
    nargs = '+', # allows one or more values
    help = 'At least one glycan element finder. Required.'
)

parser.add_argument(
    '--images',
    type = str,
    required = True,
    help = 'Directory path where image files are stored. Required.'
)

# optional argument
parser.add_argument(
    '--proximity',
    type = float,
    default = 0.25,
    help = 'Enter a value between 0-1. Default: 0.25.'
)

# optional argument
parser.add_argument(
    '--wholeimage',
    action = 'store_true',
    default = False,
    help = 'Whole image PR curve'
)

# optional argument
parser.add_argument(
    '--precision',
    type = int,
    default = 8,
    help = "Precision for confidence values. Default: 8."
)

# optional argument
parser.add_argument(
    '--distproc',
    type=str,
    default  = "",
    help = "Enables distributed processing: <n0>,remote1:<n1>,remote2:<n2>. n0 is cpus on host node (optional), ni is cpus on optional remotei node."
)

# optional argument
parser.add_argument(
    '--worker',
    type=str,
    default = "",
    help = "Indicates that script should be run as a worker client for distributed processing: <n>:server. n is cpus, server is the host node."
)

# optional argument
parser.add_argument(
    '-v',
    '--verbose',
    action = 'store_true',
    default = False,
    help = 'Verbose logging.'
)

# optional argument
# parser.add_argument(
#     '-d',
#     nargs = '?', # makes the argument optional
#     const = True, # value if the flag is provided without a value
#     type = str,
#     default = False,
#     help = "Enable debug mode for additional logging and output. Provide a value b/w [1,2] for custom debug information."
# )


args = parser.parse_args()

assert os.path.isdir(args.images) or args.worker

distproc = None
if args.worker:
    distproc = ("worker",args.worker)
elif args.distproc:
    distproc = ("manager",args.distproc)

# if args.d:
#     DebugMode.debug = True
#     new_folder = DebugMode.create_unique_folder('debug_semantics')
#     DebugMode.current_folder = new_folder
#     DebugMode.glycan_folder = image_folder
#     DebugMode.json_file = os.path.join(new_folder, 'data.json')

#     if isinstance(args.d, int):
#         DebugMode.level = args.d

config = Config_Manager()


# make sure that all predictors belong to the same class
predictors = {}
fclass = None
for name in args.finders:
    finder = config.get_finder(name)
    if fclass is None:
        fclass = finder.finder_class
    elif finder.finder_class != fclass:
        sys.exit(
            f"Error: Predictors must belong to the same class. "
            f"Found conflicting classes: {fclass} and {finder.finder_class}."
        )
    predictors[name] = finder

evaluator = SemanticEvaluator(predictors,
                              workers=distproc,
                              proximity=args.proximity,
                              whole_image=args.wholeimage,
                              precision=args.precision,
                              verbose=args.verbose)
evaluator.runall(args.images)
