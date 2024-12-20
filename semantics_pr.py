import os
import sys
import argparse
import logging
from BKGlycanExtractor import Image_Manager, Evaluator, Config_Manager, DebugMode
 

parser = argparse.ArgumentParser(description="Start")

parser.add_argument(
    '--pred_finder',
    type = str,
    required = True,
    nargs = '+', # allows one or more values
    help = 'A predictor name is required'
)

parser.add_argument(
    '--image_folder',
    type = str,
    required = True,
    help = 'Directory path where all png/jpg files are stored (required)'
)

# optional argument
parser.add_argument(
    '--proximity',
    type = float,
    default = 0.25,
    help = 'Enter a value between 0-1'
)

# optional argument
parser.add_argument(
    '-p',
    nargs = '?', # makes the argument optional
    const = True, # value if the flag is provided without a value
    type = str,
    default = False,
    help = "Enables Parallel Processing"
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

pred_finder = args.pred_finder
image_folder = args.image_folder
proximity = args.proximity

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
pred_class = None

for pred in pred_finder:
    current_finder = config.get_finder(pred)
    current_class = current_finder.__class__.__name__

    if not pred_class:
        pred_class = current_class

    if pred_class == current_class:
        predictors[pred] = current_finder
        # print("current_finder",dir(current_finder))
    else:
        sys.exit(
            f"Error: Predictors must belong to the same class. "
            f"Found conflicting classes: {pred_class} and {current_class}."
        )


args = dict(parallel = args.p, proximity=proximity, semantics=True)
evaluator = Evaluator(predictors, **args)
evaluator.runall(image_folder)
