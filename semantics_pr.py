#!.venv/bin/python
import os
import sys
import configparser
import argparse
from BKGlycanExtractor import Image_Manager, Evaluator, Config_Manager, DebugMode, GlycanExtractorPipeline
 
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
    default = 0.25,
    help = 'Proximity value. Default: 0.25.'
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


pipeline_descriptions = '''
[Monosaccharide]
figure_steps=SingleGlycanImage
glycan_steps=
known_steps=KnownMono

[Root]
figure_steps=SingleGlycanImage
glycan_steps=KnownMono
known_steps=KnownRoot

[Links]
figure_steps=SingleGlycanImage
glycan_steps=KnownMono
known_steps=KnownLink

[Glycan]
figure_steps=
glycan_steps=
known_steps=KnownGlycan
'''


config = configparser.ConfigParser()
config.read_string(pipeline_descriptions)

cm = Config_Manager()

# # pred_pipelines = GlycanExtractorPipeline()

pipelines = {}
compare_strategies = {}



for i, finder_name in enumerate(args.finders):
    pred_pipeline = GlycanExtractorPipeline()
#     cm = Config_Manager()
    f = cm.get_finder(finder_name)
    finder_class = f.finder_class

    figure_step = config[f.finder_class].get('figure_steps')
    glycan_step = config[f.finder_class].get('glycan_steps')

    print("--->>>",cm.get_finder(config[f.finder_class].get('figure_steps')))
    pred_pipeline.add_step('figure', cm.get_finder(figure_step)) if figure_step else None
    pred_pipeline.add_step('glycan', cm.get_finder(glycan_step)) if glycan_step else None
    pred_pipeline.add_step('glycan',f)

    print("pred_pipeline",pred_pipeline.get_steps('figure'))
    print("pred_pipeline",pred_pipeline.get_steps('glycan'))

    pipelines[f"{finder_class}-{i}"] = pred_pipeline 

    compare_strategies[f"semantic_compare-{i}"] = f.semantic_compare(
        proximity=args.proximity,
        whole_image=args.wholeimage,
        precision=args.precision,
        verbose=args.verbose
    )  


for finder_name in args.finders:
    known_pipeline = GlycanExtractorPipeline()
#     cm = Config_Manager()
    f = cm.get_finder(finder_name)
    finder_class = f.finder_class

    figure_step = config[f.finder_class].get('figure_steps')
    glycan_step = config[f.finder_class].get('glycan_steps')
    known_steps = config[f.finder_class].get('known_steps')

    print("--->>>",cm.get_finder(config[f.finder_class].get('figure_steps')))
    known_pipeline.add_step('figure', cm.get_finder(figure_step)) if figure_step else None
    known_pipeline.add_step('glycan', cm.get_finder(glycan_step)) if glycan_step else None
    known_pipeline.add_step('glycan',cm.get_finder(known_steps)) if known_steps else None

    print("known_pipeline",known_pipeline.get_steps('figure'))
    print("known_pipeline",known_pipeline.get_steps('glycan'))


evaluator = Evaluator(known_pipeline=known_pipeline,
                        prediction_pipelines=pipelines,
                        compare_strategies=compare_strategies,
                        workers=distproc,
                        boxeval=False,
                        verbose=args.verbose
                    )

images = Image_Manager(args.images)
images.exclude("*.annotated.*")

evaluator.runall(images)

# predictors = {}
# fclass = None
# for name in args.finders:
#     finder = config.get_finder(name)
#     if not fclass:
#         fclass = finder.finder_class
#     elif fclass != finder.finder_class:
#         sys.exit(
#             f"Error: Predictors must belong to the same class. "
#             f"Found conflicting classes: {fclass} and {finder.finder_class}."
#         )
#     predictors[name] = finder

evaluator.plotprecisionrecall(
    dir="plots",
    filename="box_plot",
    title="Custom Precision-Recall Curve",
    figsize=(10, 8),
    legend_loc="upper right",
    xlim=(0, 1),
    ylim=(0, 1),
    grid=True,
)


