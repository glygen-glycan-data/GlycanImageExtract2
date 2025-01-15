import os
import sys
import argparse
from collections import defaultdict
from BKGlycanExtractor import Image_Manager, Evaluator, SemanticGlycanCompare, GlycanExtractorPipeline, Figure_Semantics ,Config_Manager, DebugMode, Glycan_Semantics


parser = argparse.ArgumentParser(description="Start")

# required argument
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

# parser.add_argument(
#     '-d',
#     nargs = '?', # makes the argument optional
#     const = True, # value if the flag is provided without a value
#     type = str,
#     default = False,
#     help = "Enable debug mode for additional logging and output. Provide a value b/w [1,2] for custom debug information."
# )

args = parser.parse_args()

# pipeline = args.pipeline
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


images = Image_Manager(image_folder,pattern="*.png,*.jpg")

config = Config_Manager()

base_pipeline = config.get_pipeline("SingleGlycanImage-YOLOFinders")
base_pipeline.name = "SingleGlycanImage-YOLOFinders"
known_pipeline = config.get_pipeline("KnownSemantics")


collected_results = []
unknown_iupac_results = {'FN':0, 'FP': 0}

whole_glycan = SemanticGlycanCompare(base_pipeline, known_pipeline, proximity)

whole_glycan.runall(images)



