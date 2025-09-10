#!.venv/bin/python
import os
import sys
import configparser
import io
import argparse
from BKGlycanExtractor import BoxCompare, Image_Manager, Evaluator, Config_Manager, DebugMode, GlycanExtractorPipeline
from BKGlycanExtractor import runall_evaluators
from BKGlycanExtractor import DistributedProcessing as dp

# from collections import defaultdict
 
parser = argparse.ArgumentParser(description="Compute Precision-Recall")

# required argument
parser.add_argument(
    '--finders',
    type = str,
    required = True,
    nargs = '+', # allows one or more values
    help = 'At least one glycan element finder. Required.'
)

# parser.add_argument(
#     '--finder_kwargs',
#     default={},
#     type=str,
#     nargs='+',
#     help='Custom kwargs for each finder. Format: findername:key=value,key2=value2'
#     # YOLOMonosRandom:boxpadding=5 YOLOMonosBiased:boxpadding=5
# )

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
    default = [0.5],
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
    default = 4,
    help = "Precision for confidence values. Default: 4."
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
parser.add_argument(
    '-q',
    '--quiet',
    action = 'store_true',
    default = False,
    help = 'No logging.'
)

args = parser.parse_args()
distproc = dp.parse_args(parser)

class_restriction = []
for clsres in args.class_restriction:
    if clsres in ("","-","*","None"):
        class_restriction.append(None)
    else:
        class_restriction.append(clsres)

verbose = 'TQDM'
if args.verbose:
    verbose = True
elif args.quiet:
    verbose = False


# pipeline_descriptions = '''
# [Monosaccharide]
# figure_steps=SingleGlycanImage
# glycan_steps=
# known_step=KnownMono

# [Root]
# figure_steps=SingleGlycanImage
# glycan_steps=KnownMono
# known_step=KnownRoot

# [Links]
# figure_steps=SingleGlycanImage
# glycan_steps=KnownMono
# known_step=KnownLink

# [InfoLinks]
# figure_steps=SingleGlycanImage
# glycan_steps=KnownMono
# known_step=KnownLinkWithInfo

# [Glycan]
# figure_steps=
# glycan_steps=
# known_step=KnownGlycanBoxes
# '''


# config = configparser.ConfigParser()
# config.read_string(pipeline_descriptions)

cm = Config_Manager()

# add if required
# pred_kwargs = {
#     "YOLOMonosRandom": {"boxpadding":0},
#     "YOLOMonosBiased": {"boxpadding":0}
# }

# known_kwargs = {
#         "YOLOMonosRandom": {"boxpadding":2},
#         "YOLOMonosBiased": {"boxpadding":5}
#     }


for finder_name in args.finders:
    try:
        cm.get_finder(finder_name)
    except LookupError:
        print("Finder \"%s\" not found.\n\nAvailable finders:"%(finder_name,),file=sys.stderr)
        for fdname in cm.list_finders():
            print("  "+fdname,file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)

images = Image_Manager(args.images)
images.exclude("*annotated*")

evaluators = []
compare_count = 0
for i,finder_name in enumerate(args.finders):
    print(f"Building pipeline for {finder_name}")

    # ------------------------------
    # Build prediction pipeline
    # ------------------------------
    f = cm.get_finder(finder_name)
    pred_pipeline = f.finder_pipeline(cm)

    kf = f.known_finder()
    known_pipeline = kf.finder_pipeline(cm)

    pipelines = {}
    pipelines[finder_name] = (pred_pipeline,known_pipeline)


    # Set up comparison strategy
    compares = {}
    for j, cls in enumerate(class_restriction):
        for i, iou in enumerate(args.iou):
            cmp_key = f"class={cls}" if cls else f"iou={iou}"
            compares[cmp_key] = BoxCompare(
                iou=iou,
                whole_image=args.wholeimage,
                precision=args.precision,
                verbose=args.verbose,
                restrict_class=[cls] if cls else None
            )
            compare_count += 1

    # Build the evaluator...
    evaluator = Evaluator(
        pipelines=pipelines,
        compares=compares,
        boxeval=True,
        verbose=(verbose==True)
    )
    evaluators.append(evaluator)

runall_evaluators(evaluators,images,workers=distproc,verbose=verbose)

extra_args = {}
if compare_count > 1 and len(args.finders) == 1:
    label = "%(comparitor)s"
    title = "%(predictor)s"
    extra_args=dict(title=title,label=label)
elif len(args.finders) > 1 and compare_count == 1:
    label = "%(predictor)s"
    title = "%(comparitor)s"
    extra_args=dict(title=title,label=label)

# print("evaluators",evaluators)
Evaluator.plotprecisionrecall(
    evaluators,
    dir="links_plot",
    filename="links_masked",
    figsize=(8, 6),
    xlim=(0, 1),
    ylim=(0, 1),
    grid=True,
    **extra_args
)


# -----------------------------
# Old Code that works - all finder pipelines will have to use same config params
# --------------------------------

# pipelines = {}
# compare_strategies = {}

            
# for i, finder_name in enumerate(args.finders):
#     pred_pipeline = GlycanExtractorPipeline()
    
#     f = cm.get_finder(finder_name)
#     # finder_class = f.finder_class
#     finder_section = config[f.finder_class]

#     figure_step = finder_section.get('figure_steps')
#     glycan_step = finder_section.get('glycan_steps')

#     pred_pipeline.add_step('figure', cm.get_finder(figure_step)) if figure_step else None
#     pred_pipeline.add_step('glycan', cm.get_finder(glycan_step)) if glycan_step else None
#     if f.finder_class == "Glycan":
#         pred_pipeline.add_step('figure',f)
#     else:
#         pred_pipeline.add_step('glycan',f)

#     # print("pred_pipeline",pred_pipeline.get_steps('figure'))
#     # print("pred_pipeline",pred_pipeline.get_steps('glycan'))

#     pipelines[f"{finder_name}"] = pred_pipeline    

# use clone - but if you are not sure - its okay build them seperately


# instead of taking known stuff automatically for thr last known_finder/pred_finder
# maybe set it up to take args to override the config file - but for each finder - I want to add
# different configs - the known finder can be same but with different settings

# kwargs = {"boxpadding":5}
# for finder_name in args.finders:
#     known_pipeline = GlycanExtractorPipeline()
#     f = cm.get_finder(finder_name)
#     # finder_class = f.finder_class
#     finder_section = config[f.finder_class]

#     figure_step = finder_section.get('figure_steps')
#     glycan_step = finder_section.get('glycan_steps')
#     known_step = finder_section.get('known_step')

#     known_pipeline.add_step('figure', cm.get_finder(figure_step)) if figure_step else None
#     known_pipeline.add_step('glycan', cm.get_finder(glycan_step),**kwargs) if glycan_step else None

#     if f.finder_class == "Glycan":
#         known_pipeline.add_step('figure',cm.get_finder(known_step)) if known_step else None
#     else:
#         known_pipeline.add_step('glycan',cm.get_finder(known_step)) if known_step else None

#     # print("known_pipeline",known_pipeline.get_steps('figure'))
#     # print("known_pipeline",known_pipeline.get_steps('glycan'))
#     break

# if len(class_restriction) > 1 and len(args.iou) == 1:
#     cmptempl = "class=%(class)s"
# elif len(class_restriction) == 1 and len(args.iou) > 1:
#     cmptempl = "iou=%(iou)s"
# elif class_restriction[0] is None:
#     cmptempl = "iou=%(iou)s"
# else:
#     cmptempl = "class=%(class)s, iou=%(iou)s"


# for j,cls in enumerate(class_restriction):
#   for i,iou in enumerate(args.iou):
#     cmpstr = cmptempl%{'class': cls, 'iou': iou}
#     restcls = None
#     if cls != None:
#         restcls = [ cls ]
#     compare_strategies[cmpstr] = BoxCompare(
#         iou=iou,
#         whole_image=args.wholeimage,
#         precision=args.precision,
#         verbose=args.verbose,
#         restrict_class = restcls
#     )

# evaluator = Evaluator(known_pipeline=known_pipeline,
#                       prediction_pipelines=pipelines,
#                       compare_strategies=compare_strategies,
#                       workers=distproc,
#                       boxeval=True,
#                       verbose=args.verbose)



# images = Image_Manager(args.images)
# images.exclude("*._annotated.*")
# evaluator.runall(images)

# print("--->evaluator.final_structure",evaluator.final_structure)

# extra_args = {}
# if len(compare_strategies) > 1 and len(args.finders) == 1:
#     label = "%(comparitor)s"
#     title = "%(predictor)s"
#     extra_args=dict(title=title,label=label)
# elif len(args.finders) > 1 and len(compare_strategies) == 1:
#     label = "%(predictor)s"
#     title = "%(comparitor)s"
#     extra_args=dict(title=title,label=label)

# print("--->>",evaluator.final_structure)
# evaluator.plotprecisionrecall(
#     dir="plots",
#     filename="boxpr",
#     figsize=(10, 8),
#     xlim=(0, 1),
#     ylim=(0, 1),
#     grid=True,
#     **extra_args
# )
