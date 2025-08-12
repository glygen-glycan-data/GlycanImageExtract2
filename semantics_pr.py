#!.venv/bin/python
import os
import sys
import configparser
import argparse
from BKGlycanExtractor import Image_Manager, Evaluator, Config_Manager, DebugMode, GlycanExtractorPipeline
from BKGlycanExtractor import runall_evaluators
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

[InfoLinks]
figure_steps=SingleGlycanImage
glycan_steps=KnownMono
known_step=KnownLinkWithInfo

[InfoLinksTopology]
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

# add if required
# pred_kwargs = {
#     "YOLOMonosRandom": {"boxpadding":0},
#     "YOLOMonosBiased": {"boxpadding":0}
# }

known_kwargs = {}


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
    comparator = f.semantic_compare

    finder_section = config[f.finder_class]
    figure_steps = finder_section.get('figure_steps')
    glycan_steps = finder_section.get('glycan_steps')

    pred_pipeline = GlycanExtractorPipeline()
    pred_pipeline.set_steps('figure', cm.get_finders(figure_steps))
    pred_pipeline.set_steps('glycan', cm.get_finders(glycan_steps))

    # Add the main finder to the right stage
    if f.finder_class == "Glycan":
        pred_pipeline.add_step('figure', f)
    else:
        pred_pipeline.add_step('glycan', f)

    # ------------------------------
    # Build known pipeline
    # ------------------------------
    known_pipeline = GlycanExtractorPipeline()
    known_pipeline.set_steps('figure', cm.get_finders(figure_steps))
    known_pipeline.set_steps('glycan', cm.get_finders(glycan_steps))

    known_step_name = finder_section.get('known_step')
    kf = cm.get_finder(known_step_name, **known_kwargs.get(finder_name,{}))
    assert set(kf.labels) >=  set(f.labels), "%s >/= %s"%(kf.labels,f.labels)
    if f.finder_class == "Glycan":
        known_pipeline.add_step('figure', kf)
    else:
        known_pipeline.add_step('glycan', kf)
    
    pipelines = {}
    pipelines[finder_name] = (pred_pipeline,known_pipeline)

    # Set up comparison strategy
    compares = {}
    for j, cls in enumerate(class_restriction):
        for i, proximity in enumerate(args.proximity):
            cmp_key = f"class={cls}" if cls else f"proximity={proximity}"
            compares[cmp_key] = comparator(
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
        verbose=args.verbose
    )
    evaluators.append(evaluator)

if args.verbose:
    runall_evaluators(evaluators,images,workers=distproc,verbose=True)
else:
    runall_evaluators(evaluators,images,workers=distproc)

for eval in evaluators:
    print("---->>>>",eval.final_structure)

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
