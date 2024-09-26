import os
import sys
import argparse
from BKGlycanExtractor.glycanannotator import Config_Manager

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
    default = 'YOLOMonosAnnotator',
    help = 'Pipeline name (default: YOLOMonosAnnotator)'
)

# required argument
parser.add_argument(
    '--data_folder',
    type = str,
    nargs='+',  # Accept one or more arguments
    required = True,
    help = 'Directory path where all png/jpg files are stored (required)'
)

args = parser.parse_args()
pipeline_name = args.pipeline_name
glycan_folder = args.data_folder[0]
glob = args.data_folder[1:] if len(args.data_folder) > 1 else ['.png','.jpg','.pdf']

def match_glob(filename):
    return any(filename.name.endswith(ext) for ext in glob)


print("Annotating using", pipeline_name)

if __name__ == '__main__':
    config = Config_Manager()
    pipeline = config.get_pipeline(pipeline_name)

    for image_file in os.scandir(glycan_folder):
        if image_file.is_file() and match_glob(image_file):
            obj = pipeline.run(image_file)
            print("\nSemantics:",obj)
            











