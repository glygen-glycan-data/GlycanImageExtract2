#!.venv/bin/python
import os
import sys
import configparser
import argparse
from collections import defaultdict
from BKGlycanExtractor import Image_Manager, Evaluator, Config_Manager, DebugMode, GlycanExtractorPipeline
from BKGlycanExtractor import runall_evaluators
from BKGlycanExtractor import DistributedProcessing as dp
from BKGlycanExtractor import MonoFinder, LinkFinder, RootFinder, GlycanFinder

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

parser.add_argument(
    '-b',
    '--boxes',
    action = 'store_true',
    default = False,
    help = 'Box-based precision recall curve.'
)

parser.add_argument(
    '-s',
    '--semantics',
    action = 'store_true',
    default = False,
    help = 'Semantics-based precision recall curve.'
)

# optional argument
parser.add_argument(
    '--proximity',
    type = float,
    nargs = '*', # allows zero or more values
    help = 'Proximity threshold. Monosaccharide semantics evaluation only. Default: 0.25.'
)

# optional argument
parser.add_argument(
    '--iou',
    type = float,
    nargs = '*', # allows zero or more values
    help = 'IOU threshold. Box-based PR curve only. Default: 0.5.'
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

if args.boxes and args.semantics:
    print("Only one of --boxes and --semantics should be specified.",file=sys.stderr)
    sys.exit(1)

if not args.boxes and not args.semantics:
    print("At least one of --boxes and --semantics should be specified.",file=sys.stderr)
    sys.exit(1)

if args.proximity and args.boxes:
    print("Proximity threshold not relevant to box-based evaluation.",file=sys.stderr)
    sys.exit(1)

if args.iou and args.semantics:
    print("IOU threshold not relevant to semantics-based evaluation.",file=sys.stderr)
    sys.exit(1)

cm = Config_Manager()

FinderTypes = (MonoFinder, LinkFinder, RootFinder, GlycanFinder)
TypeCount = defaultdict(int)

for finder_name in args.finders:
    try:
        f = cm.get_finder(finder_name)
        for ft in FinderTypes:
            if isinstance(f,ft):
                TypeCount[ft.__name__] += 1
        TypeCount[None] += 1
    except LookupError:
        print("Finder \"%s\" not found.\n\nAvailable finders:"%(finder_name,),file=sys.stderr)
        for fdname in cm.list_finders():
            print("  "+fdname,file=sys.stderr)
        print(file=sys.stderr)
        sys.exit(1)

if len(TypeCount) > 2:
    print("Mixed finder types specified.",file=sys.stderr)
    sys.exit(1)

if len(set(TypeCount.values())) != 1:
    print("Unexpected finder type specified.",file=sys.stderr)
    sys.exit(1)

if args.boxes and not args.iou:
    args.iou = [ 0.5 ]

if args.semantics and args.proximity and TypeCount['MonoFinder'] == 0:
    print("Proximity only useful for MonoFinders.",file=sys.stderr)
    sys.exit(1)

if args.semantics and not args.proximity:
    if TypeCount['MonoFinder'] > 0:
        args.proximity = [ 0.25 ]
    else:
        args.proximity = [ None ]

if args.boxes:
    levels = list(args.iou)
    levelstr = "iou"
    titlestr = "Box-based "
else:
    levels = list(args.proximity)
    levelstr = "proximity"
    titlestr = "Semantics-based "

images = Image_Manager(args.images)
images.exclude("*annotated*")

evaluators = []
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
        for i, level in enumerate(levels):
            cmp_key = []
            if cls:
                cmp_key.append(f"class={cls}")
            if level:
                cmp_key.append(f"{levelstr}={level}")
            cmp_key = ", ".join(cmp_key)
            kwargs = dict(whole_image=args.wholeimage,
                          precision=args.precision,
                          verbose=args.verbose)
            if level:
                kwargs[levelstr] = level
            if cls:
                kwargs['restrict_class'] = [cls]
            if args.boxes:
                compares[cmp_key] = f.box_compare(**kwargs)
            else:
                compares[cmp_key] = f.semantic_compare(**kwargs)            

    # Build the evaluator...
    evaluator = Evaluator(
        pipelines=pipelines,
        compares=compares,
        boxeval=args.boxes,
        verbose=(verbose==True)
    )
    evaluators.append(evaluator)

runall_evaluators(evaluators,images,workers=distproc,verbose=verbose)

if len(compares) > 1 and len(args.finders) == 1:
    label = "%(comparitor)s"
    title = titlestr + "Precision-Recall Curve (%(predictor)s)"
    extra_args=dict(title=title,label=label)
elif len(args.finders) > 1 and len(compares) == 1:
    label = "%(predictor)s"
    if list(compares)[0] == "":
        title = titlestr + "Precision-Recall Curve"
    else:
        title = titlestr + "Precision-Recall Curve (%(comparitor)s)"
    extra_args=dict(title=title,label=label)
else:
    label = "%(predictor)s, %(comparitor)s"
    title = titlestr + "Precision-Recall Curve"
extra_args=dict(title=title,label=label)

Evaluator.plotprecisionrecall(
    evaluators,
    dir="presentation",
    filename="prcurve",
    figsize=(8, 6),
    xlim=(0, 1),
    ylim=(0, 1),
    grid=True,
    **extra_args
)
