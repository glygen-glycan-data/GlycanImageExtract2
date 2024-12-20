import os
import sys
import argparse
from BKGlycanExtractor import Config_Manager, Image_Manager, GlycanExtractorPipeline, Image_Data, DebugMode



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
    '--image_folder',
    type = str,
    required = True,
    help = 'Directory path where all png/jpg files are stored (required)'
)

parser.add_argument(
    '-d',
    nargs = '?', # makes the argument optional
    const = True, # value if the flag is provided without a value
    type = str,
    default = False,
    help = "Enable debug mode for additional logging and output. Provide a value b/w [1,2] for custom debug information."
)


args = parser.parse_args()
pipeline_name = args.pipeline_name
glycan_folder = args.image_folder

if args.d:
    DebugMode.debug = True
    new_folder = DebugMode.create_unique_folder('main_file_logs')
    DebugMode.current_folder = new_folder
    DebugMode.glycan_folder = glycan_folder
    DebugMode.json_file = os.path.join(new_folder, 'data.json')

    if isinstance(args.d, int):
        DebugMode.level = args.d



print("\nAnnotating using", pipeline_name)

# Converting SVG - to PNG and txt
Image_Data(glycan_folder)

images = Image_Manager(glycan_folder,pattern="*.png,*.jpg")
config = Config_Manager()
pipeline1 = config.get_pipeline(pipeline_name)

sgi = config.get_finder("SingleGlycanImage")
sm =  config.get_finder("KnownMono")
sl = config.get_finder("KnownLink")
sr = config.get_finder("KnownRoot")
pipeline0 = GlycanExtractorPipeline()
pipeline0.add_step("figure",sgi)
pipeline0.add_step("glycan",sm)
pipeline0.add_step("glycan",sl)
pipeline0.add_step("glycan",sr)


for image in sorted(images):

    print("\nImage:",image)
    obj0 = pipeline0.run(image)
    obj1 = pipeline1.run(image)

    name = os.path.basename(image).split('.')[0]
    obj1.label_image(obj1.image().copy(), name = name + '_yolo')
    obj0.label_image(obj0.image().copy(), name = name + '_known')

    for gly0,gly1 in zip(obj0.glycans(),obj1.glycans()):
        # comp0 = gly0.compstr()
        # comp1 = gly1.compstr()
        # print("known comp",comp0)
        # print("pred comp", comp1)
        # print(image,"GOOD" if comp0 == comp1 else "BAD",comp0,comp1)
        
        IUPAC0 = gly0.IUPAC()
        IUPAC1 = gly1.IUPAC()
        
        print("Known IUPAC: ",IUPAC0)
        print("Detected IUPAC: ",IUPAC1)
        print("\nDo the IUPAC sequences match?", IUPAC0==IUPAC1)

        # print("\nSemantics:",gly0.tojson())       
        # print("\nSemantics:",gly1.tojson())   















