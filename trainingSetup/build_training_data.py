#!../.venv/bin/python
'''
File is used to build a zip file which includes:
1) Training data: .png images and .txt files (<classid> <relative_center_x> <relative_center_y> <width> <height>)
Note: * attention: <relative_center_x> <relative_center_y> - are center of rectangle (not top-left corner)
reference: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

2) classes.txt: which contains all the labels for the training

Note: If no known finder is supplied via cmd flag (--finder), then KnownGlycanBoxes finder will be used automatically.
Else, specify a known finder like: KnownMono, KnownRoot, KnownLink...
'''

import os
import sys
import argparse
import shutil
import tempfile
import atexit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BKGlycanExtractor import Image_Manager, Config_Manager, GlycanExtractorPipeline

parser = argparse.ArgumentParser(description="Build training data")

parser.add_argument(
    '--finder',
    type = str,
    required = True,
    help = 'Finder for known boxes on images. Usually, one of KnownGlycanBoxes, KnownMono, KnownRoot, KnownLink, or KnownLinkWithInfo.'
)

parser.add_argument(
    '--boxpadding',
    type=int,
    required=False,
    help='Box padding on known boxes. Default: from named known finder\'s config.'
)

parser.add_argument(
    '--images',
    type = str,
    required = True,
    help = 'Directory path where image files are stored. Required.'
)

parser.add_argument(
    '--out',
    type = str,
    default = 'images.zip',
    help = 'Filename for training data zip file, must end in .zip. Default: images.zip'
)

args = parser.parse_args()

config = Config_Manager()

if not args.out.endswith('.zip'):
    raise ValueError("Zip file filename must have .zip extension")

assert not os.path.exists(args.out), "zip file %s exists"%(args.out,)

def remove_tempdir(tempdir):
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

folder_name = tempfile.mkdtemp(prefix=".tmpdir",dir=os.getcwd())
atexit.register(remove_tempdir,folder_name)

images = Image_Manager(args.images)
images.exclude("*.annotated.*")

# Building Base Pipeline
pipeline = GlycanExtractorPipeline()
finder = config.get_finder('SingleGlycanImage')
pipeline.add_step('figure',finder)

finder = config.get_finder(args.finder)
pipeline = finder.finder_pipeline(config)

if args.boxpadding != None:
    finder.set_param('boxpadding',args.boxpadding)

for image_path in images:
    image_filename = os.path.basename(image_path)
    base_filename = os.path.splitext(image_filename)[0]
    training_file_path = os.path.join(folder_name, base_filename + ".txt")

    result,glycan_semantics = pipeline.run_evaluation(image_path, boxesonly=True)

    with open(training_file_path, 'w') as f:
        for b in result:
            b.set_image_dimensions(image_width=glycan_semantics.width(),
                                   image_height=glycan_semantics.height())
            classid = b.get('classid')
            x,y,w,h = b.center_relative()

            # <classid> <relative_center_x> <relative_center_y> <width> <height>
            f.write(f"{classid} {x} {y} {w} {h}\n") 

    shutil.copy(image_path, folder_name)

# create labels file
labels_file = os.path.join(folder_name, 'classes.txt')
finder.write_labels(labels_file)
model_file = os.path.join(folder_name, 'model.ini')
finder.write_model(args.finder,model_file)

# Zipping the folder
shutil.make_archive(args.out.rsplit('.',1)[0], 'zip', folder_name)
print("Training data is ready...")
print(args.out)

# delete the images directory
# if os.path.exists(folder_name):
#     shutil.rmtree(folder_name)

