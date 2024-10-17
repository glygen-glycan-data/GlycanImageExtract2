# import os
import sys
import argparse
from BKGlycanExtractor import Config_Manager, Image_Manager, GlycanExtractorPipeline

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
    pipeline1 = config.get_pipeline(pipeline_name)

    sgi = config.get_finder("SingleGlycanImage")
    sm =  config.get_finder("KnownMono")
    pipeline0 = GlycanExtractorPipeline()
    pipeline0.add_step("figure",sgi)
    pipeline0.add_step("glycan",sm)

    for image in sorted(images):
        obj0 = pipeline0.run(image)
        obj1 = pipeline1.run(image)
        for gly0,gly1 in zip(obj0.glycans(),obj1.glycans()):
            comp0 = gly0.compstr()
            comp1 = gly1.compstr()
            print(image,"GOOD" if comp0 == comp1 else "BAD",comp0,comp1)
    #         # print("\nSemantics:",gly0.tojson())       
    #         # print("\nSemantics:",gly1.tojson())   

        # for gly1 in obj1.glycans():
        #     print("\nSemantics:",gly1.tojson())  
        #     pass












