#!.venv/bin/python
import os
import sys
import configparser
import io
import argparse
from BKGlycanExtractor import GlycanCompare, Image_Manager, Evaluator, Config_Manager, DebugMode
from BKGlycanExtractor import DistributedProcessing as dp
 
parser = argparse.ArgumentParser(description="Start")


# required argument
parser.add_argument(
    '--compare',
    type = str,
    required = True,
    nargs = '+', # allows one or more values
    help = 'At least one comapare type. Required. Options: composition, iupac'
)

# required argument
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
    default = [ 0.25 ],
    nargs = '+', # allows one or more values
    help = 'proximity threshold value. Default: 0.5.'
)

# optional argument
parser.add_argument(
    '--precision',
    type = int,
    default = 8,
    help = "Precision for confidence values. Default: 8."
)

dp.add_arguments(parser)

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
distproc = dp.parse_args(parser)



pred_pipelines = {}
known_pipelines = {}

cm = Config_Manager()

for compare_type in args.compare:
    kwargs = {'label_type': compare_type}
    p_pipeline = cm.get_pipeline('GlycanCompare-YOLOFinders')
    pred_finder = cm.get_finder('YOLO_Glycan',**kwargs)
    p_pipeline.add_step('glycan',pred_finder)

    k_pipeline = cm.get_pipeline('GlycanCompare-KnownFinders')
    known_finder = cm.get_finder('Known_Glycan',**kwargs)
    k_pipeline.add_step('glycan',known_finder)

    pred_pipelines[f"{compare_type}"] = p_pipeline  
    known_pipelines[f"{compare_type}"] = k_pipeline



compare_strategies = {}

cmptempl = ""
for i,proximity in enumerate(args.proximity):
    cmpstr = cmptempl%{'proximity': proximity}
    compare_strategies[cmpstr] = GlycanCompare(
        proximity=proximity,
        precision=args.precision,
        verbose=args.verbose,
    )

evaluator = Evaluator(known_pipeline=known_pipelines,
                      prediction_pipelines=pred_pipelines,
                      compare_strategies=compare_strategies,
                      workers=distproc,
                      boxeval=False,
                      verbose=args.verbose)

images = Image_Manager(args.images)
images.exclude("*._annotated.*")

evaluator.runall(images)

extra_args = {}
if len(compare_strategies) > 1 and len(args.compare) == 1:
    label = "%(comparitor)s"
    title = "%(predictor)s"
    extra_args=dict(title=title,label=label)
elif len(args.compare) > 1 and len(compare_strategies) == 1:
    label = "%(predictor)s"
    title = "%(comparitor)s"
    extra_args=dict(title=title,label=label)

evaluator.plotprecisionrecall(
    dir="glycan",
    filename="new_training",
    figsize=(10, 8),
    xlim=(0, 1),
    ylim=(0, 1),
    grid=True,
    **extra_args
)


