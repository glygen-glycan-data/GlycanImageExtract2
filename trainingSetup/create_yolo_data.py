'''
Creates training data for figures that have multiple glycans.

It is required to have folder (which is provided to this program via the command line) 
containing extracted information (from Manuscripts) in TSV file and corresposnding images which is generated
during the extract figures stage. 


TODO: add more information about what is annotate_pdf, extract figures used for

File is used to build a zip file which includes:
1) Training data: .png images and .txt files (<classid> <relative_center_x> <relative_center_y> <width> <height>)
Note: * attention: <relative_center_x> <relative_center_y> - are center of rectangle (not top-left corner)
reference: https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

2) classes.txt: which contains all the labels for the training

Note: If no known finder is supplied via cmd flag (--finder), then KnownGlycanBoxes finder will be used automatically.
Else, specify a known finder like: KnownMono, KnownRoot, KnownLink...

'''


import os
import glob
import shutil
import argparse
import csv
import tempfile
import atexit
import ast
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from BKGlycanExtractor import Config_Manager
from BKGlycanExtractor.training_utils import build_training

parser = argparse.ArgumentParser(description="Build training data")

parser.add_argument(
    '--finder',
    type = str,
    required = True,
    help = 'Finder for known boxes on images. Usually, one of KnownGlycanBoxes, KnownMono, KnownRoot, KnownLink, or KnownLinkWithInfo.'
)

parser.add_argument(
    "--boxpadding", 
    default=0, 
    type=int, 
    help="Box padding on known boxes. Default: from named known finder\'s config."
)

parser.add_argument(
    "--extracted_data", 
    required=True, 
    help="Directory where images and supporting TSV file(s) is/are stored. Required."
)

parser.add_argument(
    '--images',
    type = str,
    default = 'training_data',
    help = 'Directory where you want to store images and related known data (map files) that will be created based on the extracted data from TSV files. These folder is useful for creating training data.'
)

parser.add_argument(
    "--out", 
    type = str,
    default = 'images.zip',
    help = 'Filename for training data zip file, must end in .zip. Default: images.zip'
)

args = parser.parse_args()


# maybe make this a utility method which is present in BKGlycan folder? (refer BKGlycanExtractor/training_utils.py)
def build_known_data():
    '''
    Extracts data from TSV files and prepares known data (in map file format) 
    along with corresposnding image files - which will be stored in args.images dir
    '''

    images_dir = args.images
    # Delete the directory and its contents if it exists
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)

    # Recreate the directory
    os.makedirs(images_dir)

    # collect all the TSV files from the extracted_data dir (i.e input dir)
    tsv_files = [os.path.join(args.extracted_data, f) for f in os.listdir(args.extracted_data) if f.endswith(".tsv")]

    if not tsv_files:
        raise FileNotFoundError(f"Required: TSV files.")

    # parse each tsv file and extract the annotated box info from it
    for tsv_path in tsv_files:
        current_map_file_path = None
        current_map_file = None
        with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for row in reader:
                map_file = os.path.splitext(row['figure_name'])[0] + '_map.txt'

                map_file_path = os.path.join(images_dir, map_file)

                # TODO: extract figures has a column called gly_bbox - but maybe we need to change the name
                # because if root, monos, etc are included then we need to make the columns generic 
                # and change the below key gly_box to a more generic term like bbox
                gly_bbox = list(map(int, ast.literal_eval(row["gly_bbox"])))
                x, y, w, h = gly_bbox
                
                fig_width = int(row['fig_width'])
                fig_height = int(row['fig_height'])

                # Checks if a new txt file is needed
                if map_file_path != current_map_file_path:
                    # copy the figure to the output directory and Close previous file if it exists
                    if current_map_file:
                        shutil.copy(row['figure_path'], images_dir)
                        current_map_file.close()

                    # Open a new txt file
                    current_map_file = open(map_file_path, 'w')

                    current_map_file_path = map_file_path

                    # write whole image data in map file
                    current_map_file.write(f'##### WHOLEIMAGE: {fig_height} x {fig_width} (height x width)\n')
                
                # write glycan data to map file
                current_map_file.write(f"### GLYCAN: {x} {y} {w} {h} (bbox: x y w h)\n")

                # TODO: not sure what convention we want to follow while labelling different classes (of glycans) probably comes from the TSV file?
                # current_map_file.write(f"### CLASS: 0\n")      

            # Close the last file
            if current_map_file:
                current_map_file.close()


build_known_data()


config = Config_Manager()

build_training(
    config=config,
    finder_name=args.finder,
    images=args.images,
    out_zip=args.out,
    boxpadding=args.boxpadding
)
print("Training data is ready...")
print(args.out)

