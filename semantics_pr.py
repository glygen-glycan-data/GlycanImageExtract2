#!.venv/bin/python
import os
import sys
import configparser
import argparse
from BKGlycanExtractor import Image_Manager, Evaluator, Config_Manager, DebugMode, GlycanExtractorPipeline
from BKGlycanExtractor import runall_evaluators
from BKGlycanExtractor import DistributedProcessing as dp

parser = argparse.ArgumentParser(description="Compute Precision-Recall")

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
    '--proximity',
    type = float,
    default = [ 0.25 ],
    nargs = '+', # allows one or more values
    help = 'Proximity value. Default: 0.25.'
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

verbose = 'TQDM'
if args.verbose:
    verbose = True
elif args.quiet:
    verbose = False

cm = Config_Manager()

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
    pred_pipeline = f.finder_pipeline()

    kf = f.known_finder()
    known_pipeline = kf.finder_pipeline()

    pipelines = {}
    pipelines[finder_name] = (pred_pipeline,known_pipeline)

    # Set up comparison strategy
    compares = {}
    for j, cls in enumerate(class_restriction):
        for i, proximity in enumerate(args.proximity):
            cmp_key = f"class={cls}" if cls else f"proximity={proximity}"
            compares[cmp_key] = f.semantic_compare(
                proximity=proximity,
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
        boxeval=False,
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
    dir="presentation",
    filename="semantics",
    figsize=(8, 6),
    xlim=(0, 1),
    ylim=(0, 1),
    grid=True,
    **extra_args
)
