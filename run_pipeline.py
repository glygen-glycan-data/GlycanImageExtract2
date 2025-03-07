#!.venv/bin/python
import os
import sys
import argparse
import logging
from BKGlycanExtractor import Image_Manager, Config_Manager
from BKGlycanExtractor.distproc import DistributedProcessing as dp
 
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
pipeline = config.get_pipeline(args.pipeline)

images = Image_Manager(args.images)
images.exclude("*.annotated.*")

for sem in pipeline.runall(images,workers=workers,verbose=args.verbose):
    sem.write_json()
    sem.annotate_monos()
    sem.write_image(extension="annotated.png")
