#!.venv/bin/python
'''
File is used to build a zip file which includes:
1) Training data: .png images and .txt files (<classid> <relative_center_x> <relative_center_y> <width> <height)
2) classes.txt: which contains all the labels for the training

Note: If no known finder is supplied via cmd flag (--finder), then KnownGlycanBoxes finder will be used automatically.
Else, specify a known finder like: KnownMono, KnownRoot, KnownLink...
'''

import os
import sys
import argparse
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BKGlycanExtractor import Image_Manager, Config_Manager, GlycanExtractorPipeline

parser = argparse.ArgumentParser(description="Start")

parser.add_argument(
    '--finder',
    type = str,
    # required = True,
    help = 'Pipeline to execute on images. Optional. KnownMono, KnownRoot, KnownLink.'
)

parser.add_argument(
    '--images',
    type = str,
    required = True,
    help = 'Directory path where image files are stored. Required.'
)

parser.add_argument(
    '--zip',
    type = str,
    help = 'File name to save training data in a zip file. Default: images.zip.'
)


args = parser.parse_args()

config = Config_Manager()

folder_name = args.zip if args.zip else "images"
# If the folder exists, delete it
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
# Create a fresh new folder
os.makedirs(folder_name,exist_ok=True)

images = Image_Manager(args.images)
images.exclude("*.annotated.*")

# Building Base Pipeline
pipeline = GlycanExtractorPipeline()
finder = config.get_finder('SingleGlycanImage')
pipeline.add_step('figure',finder)

# if a known finder is specified
if args.finder:
    step = args.finder
    stage = 'glycan' 
    kwargs = {'boxpadding': 5}   # option to make boxes larger is required
    finder = config.get_finder(step,**kwargs)
    pipeline.add_step(stage,finder)


for image_path in images:
    image_filename = os.path.basename(image_path)
    base_filename = os.path.splitext(image_filename)[0]
    training_file_path = os.path.join(folder_name, base_filename + ".txt")

    result,glycan_semantics = pipeline.run_evaluation(image_path, boxesonly=True)

    with open(training_file_path, 'w') as f:
        for b in result:
            classid = b.get('classid')
            x,y,w,h = b.center_relative()
            f.write(f"{classid} {x} {y} {w} {h}\n") 

    shutil.copy(image_path, folder_name)

# create labels file
labels_file = os.path.join(folder_name, 'classes.txt')
with open(labels_file, 'w') as f:
    print("finder",finder)
    for label in finder.get_labels():
        f.write(f"{label}\n")

# Zipping the folder
shutil.make_archive(folder_name, 'zip', folder_name)
print("Training data is ready...")
print(f"{folder_name}.zip")

# delete the images directory
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)

