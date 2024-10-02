# import os
import sys
import argparse
from BKGlycanExtractor.glycanannotator import Config_Manager
from BKGlycanExtractor.image_manager import Image_Manager

'''
CMD arguments
1) pipeline_name [optional, default-YOLOMonosAnnotator]
2) glycan folder path [required] glob [optional *.png, *.jpg, *.pdf] - should contain path to 
the folder and optionally provide the type of files that should be accepted from the folder
'''

parser = argparse.ArgumentParser(description="Start")

# optional argument
parser.add_argument(
    '--pipeline_name',
    type = str,
    default = 'SingleGlycanImage-YOLOFinders',
    help = 'Pipeline name (default: SingleGlycanImage-YOLOFinders)'
)

# required argument
parser.add_argument(
    '--data_folder',
    type = str,
    # nargs='+',  # Accept one or more arguments
    required = True,
    help = 'Directory path where all png/jpg files are stored (required)'
)

args = parser.parse_args()
pipeline_name = args.pipeline_name
glycan_folder = args.data_folder

print("\nAnnotating using", pipeline_name)

if __name__ == '__main__':
    images = Image_Manager(glycan_folder,pattern="*.png,*.jpg")
    config = Config_Manager()
    pipeline = config.get_pipeline(pipeline_name)

    for image in images:
        obj = pipeline.run(image)
        print("\nSemantics:",obj)       











