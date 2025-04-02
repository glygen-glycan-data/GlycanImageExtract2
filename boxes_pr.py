#!.venv/bin/python
import os
import sys
import configparser
import io
import argparse
from BKGlycanExtractor import BoxCompare, Image_Manager, Evaluator, Config_Manager, DebugMode, GlycanExtractorPipeline
from BKGlycanExtractor import DistributedProcessing as dp
 
parser = argparse.ArgumentParser(description="Start")

# required argument
parser.add_argument(
    '--finders',
    type = str,
    required = True,
    nargs = '+', # allows one or more values
    help = 'At least one glycan element finder. Required.'
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
    '--iou',
    type = float,
    default = [ 0.5 ],
    nargs = '+', # allows one or more values
    help = 'IOU Threshold value. Default: 0.5.'
)

# optional argument
parser.add_argument(
    '--class_restriction',
    type = str,
    default = [ None ],
    nargs = '+', # allows zero, one, or more values
    help = 'Class restriction. Default: No class restriction.'
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

class_restriction = []
for clsres in args.class_restriction:
    if clsres in ("","-","*","None"):
        class_restriction.append(None)
    else:
        class_restriction.append(clsres)

# if args.d:
#     DebugMode.debug = True
#     new_folder = DebugMode.create_unique_folder('debug_boxes')
#     DebugMode.current_folder = new_folder
#     DebugMode.glycan_folder = image_folder
#     DebugMode.json_file = os.path.join(new_folder, 'data.json')

#     if isinstance(args.d, int):
#         DebugMode.level = args.d

pipeline_descriptions = '''
[Monosaccharide]
figure_steps=SingleGlycanImage
glycan_steps=
known_step=KnownMono

[Root]
figure_steps=SingleGlycanImage
glycan_steps=KnownMono
known_step=KnownRoot

[Links]
figure_steps=SingleGlycanImage
glycan_steps=KnownMono
known_step=KnownLink

[Glycan]
figure_steps=
glycan_steps=
known_step=KnownGlycanBoxes
'''


config = configparser.ConfigParser()
config.read_string(pipeline_descriptions)

cm = Config_Manager()

pipelines = {}
compare_strategies = {}

for i, finder_name in enumerate(args.finders):
    pred_pipeline = GlycanExtractorPipeline()
    
    f = cm.get_finder(finder_name)
    # finder_class = f.finder_class
    finder_section = config[f.finder_class]

    figure_step = finder_section.get('figure_steps')
    glycan_step = finder_section.get('glycan_steps')

    pred_pipeline.add_step('figure', cm.get_finder(figure_step)) if figure_step else None
    pred_pipeline.add_step('glycan', cm.get_finder(glycan_step)) if glycan_step else None
    if f.finder_class == "Glycan":
        pred_pipeline.add_step('figure',f)
    else:
        pred_pipeline.add_step('glycan',f)

    # print("pred_pipeline",pred_pipeline.get_steps('figure'))
    # print("pred_pipeline",pred_pipeline.get_steps('glycan'))

    pipelines[f"{finder_name}"] = pred_pipeline    

# use clone - but if you are not sure - its okay build them seperately



for finder_name in args.finders:
    known_pipeline = GlycanExtractorPipeline()
    f = cm.get_finder(finder_name)
    # finder_class = f.finder_class
    finder_section = config[f.finder_class]

    figure_step = finder_section.get('figure_steps')
    glycan_step = finder_section.get('glycan_steps')
    known_step = finder_section.get('known_step')

    known_pipeline.add_step('figure', cm.get_finder(figure_step)) if figure_step else None
    known_pipeline.add_step('glycan', cm.get_finder(glycan_step)) if glycan_step else None
    if f.finder_class == "Glycan":
        known_pipeline.add_step('figure',cm.get_finder(known_step)) if known_step else None
    else:
        known_pipeline.add_step('glycan',cm.get_finder(known_step)) if known_step else None

    # print("known_pipeline",known_pipeline.get_steps('figure'))
    # print("known_pipeline",known_pipeline.get_steps('glycan'))
    break

if len(class_restriction) > 1 and len(args.iou) == 1:
    cmptempl = "class=%(class)s"
elif len(class_restriction) == 1 and len(args.iou) > 1:
    cmptempl = "iou=%(iou)s"
elif class_restriction[0] is None:
    cmptempl = "iou=%(iou)s"
else:
    cmptempl = "class=%(class)s, iou=%(iou)s"
    
for j,cls in enumerate(class_restriction):
  for i,iou in enumerate(args.iou):
    cmpstr = cmptempl%{'class': cls, 'iou': iou}
    restcls = None
    if cls != None:
        restcls = [ cls ]
    compare_strategies[cmpstr] = BoxCompare(
        iou=iou,
        whole_image=args.wholeimage,
        precision=args.precision,
        verbose=args.verbose,
        restrict_class = restcls
    )

evaluator = Evaluator(known_pipeline=known_pipeline,
                      prediction_pipelines=pipelines,
                      compare_strategies=compare_strategies,
                      workers=distproc,
                      boxeval=True,
                      verbose=args.verbose)

images = Image_Manager(args.images)
images.exclude("*.annotated.*")

evaluator.runall(images)

extra_args = {}
if len(compare_strategies) > 1 and len(args.finders) == 1:
    label = "%(comparitor)s"
    title = "%(predictor)s"
    extra_args=dict(title=title,label=label)
elif len(args.finders) > 1 and len(compare_strategies) == 1:
    label = "%(predictor)s"
    title = "%(comparitor)s"
    extra_args=dict(title=title,label=label)

evaluator.plotprecisionrecall(
    dir="plots",
    filename="boxpr",
    figsize=(10, 8),
    xlim=(0, 1),
    ylim=(0, 1),
    grid=True,
    **extra_args
)


