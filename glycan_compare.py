import os
import sys
import argparse
from collections import defaultdict
from BKGlycanExtractor import Image_Manager, Evaluator, SemanticGlycanCompare, GlycanExtractorPipeline, Figure_Semantics ,Config_Manager, DebugMode

'''
Glycan compare works on the concept whether the 
IUPAC sequences of the predicted glycan and known Glycan match.
'''


def process_data(collected_results, unknown_iupac_results, pipeline_name):

    final_structure = Evaluator.process_results(collected_results,pipeline_name)

    unknown_FP = unknown_iupac_results['FP']
    unknown_FN = unknown_iupac_results['FN']

    for model_name, confidence_data in final_structure.items():
        for confidence, matches in confidence_data.items():
            TP, FP, FN = matches['TP'], matches['FP'], matches['FN']

            if FN > 0:
                final_structure[pipeline_name][confidence] = {'TP': 0, 'FP': 0, 'FN': 1}
            elif TP > 0 and FP > 0:
                final_structure[pipeline_name][confidence] = {'TP': 0, 'FP': 1, 'FN': 0}
            else:
                final_structure[pipeline_name][confidence] = {'TP': 1, 'FP': 0, 'FN': 0}

            final_structure[pipeline_name][confidence]['FP'] += unknown_FP
            final_structure[pipeline_name][confidence]['FN'] += unknown_FN

    return final_structure


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
known_pipeline = config.get_pipeline("KnownSemantics")


collected_results = []
unknown_iupac_results = {'FN':0, 'FP': 0}

for idx, image in enumerate(images):
    print("image:",idx, image)

    pred_semantics = base_pipeline.run(image)
    glycan = pred_semantics.glycans()[0]

    known_semantics = known_pipeline.run(image)
    known_glycan = known_semantics.glycans()[0]

    whole_glycan = SemanticGlycanCompare(proximity)

    known_IUPAC = known_glycan.IUPAC()
    pred_IUPAC = glycan.IUPAC()

    # known_composition = known_glycan.compstr()
    # pred_composition = glycan.compstr()

    if known_IUPAC == pred_IUPAC:
        results = whole_glycan.compare(glycan, known_glycan, pipeline_name='SingleGlycanImage-YOLOFinders')
        # print("results",results)
        collected_results.extend(results)  
    else:
        print("\nIUPAC SEQUENCE'S DID NOT MATCH FOR IMAGE:",idx)
        # results = ['FN','FP']
        unknown_iupac_results['FN'] += 1
        unknown_iupac_results['FP'] += 1
        # add FP
     

final_structure = process_data(collected_results, unknown_iupac_results, 'SingleGlycanImage-YOLOFinders')
# print("final_structure",final_structure)

Evaluator.plotprecisionrecall(final_structure,'glycan_semantics')


