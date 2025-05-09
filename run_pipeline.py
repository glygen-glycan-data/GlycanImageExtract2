#!.venv/bin/python
import os
import sys
import argparse
import logging
from BKGlycanExtractor import Image_Manager, Config_Manager
from BKGlycanExtractor.distproc import DistributedProcessing as dp
from BKGlycanExtractor.glycanfinding import KnownGlycanBoxes
 
parser = argparse.ArgumentParser(description="Start")

parser.add_argument(
    '--pipeline',
    type = str,
    required = True,
    help = 'Pipeline to execute on images. Required.'
)

parser.add_argument(
    '--images',
    type = str,
    required = True,
    help = 'Directory path where image files are stored. Required.'
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

args = parser.parse_args()
workers = dp.parse_args(parser)

config = Config_Manager()
try:
    pipeline = config.get_pipeline(args.pipeline)
except LookupError:
    print("Pipeline \"%s\" not found.\n\nAvailable pipelines:"%(args.pipeline,),file=sys.stderr)
    for plname in config.list_pipelines():
        print("  "+plname,file=sys.stderr)
    print(file=sys.stderr)
    sys.exit(1)

images = Image_Manager(args.images)
images.exclude("*.annotated.*")
images.exclude("*.cleaned.*")

# changes specific for glycan finding make here...
kgb = config.get_finder("KnownGlycanBoxes")

for result in pipeline.runall(images,workers=workers,verbose=args.verbose):
    # print("\nMono semnatics",sem.semantics['glycans'][0].semantics)
    result.write_json()
    # sem.annotate_glycans(color=(0,0,255))
    # kgb.find_objects(sem)
    # sem.annotate_glycans(color=(0,255,0))
    # sem.write_image(extension="annotated.png")
