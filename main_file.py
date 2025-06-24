import os
import sys
import argparse
from BKGlycanExtractor import Config_Manager, Image_Manager, GlycanExtractorPipeline, Image_Data, DebugMode, Glycan_Semantics



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

# finders = {
#     "KnownRoot": ["YOLORootFinder"],
#     "KnownLink": ["ConnectYOLO", "ConnectYOLOBig"]
# }

# last_finder_name = 'YOLOMonosRandom'

# # provide kwargs depending on the last step
# config = Config_Manager()
# sgi = config.get_finder("SingleGlycanImage")
# sm =  config.get_finder("KnownMono")
# sl = config.get_finder("KnownLink")
# sr = config.get_finder("KnownRoot")

# known_pipeline = GlycanExtractorPipeline()
# known_pipeline.add_step("figure",sgi)
# known_pipeline.add_step("glycan",sm)
# # known_pipeline.add_step("glycan",sl)
# # known_pipeline.add_step("glycan",sr)


# # pred_pipeline = known_pipeline.clone()
# # print("figure steps",known_pipeline.get_steps('figure'))
# # print("glycan steps",known_pipeline.get_steps('glycan'))

# prediction_pipeline = known_pipeline.clone()
# last_pred_step = config.get_finder(last_finder_name)
# prediction_pipeline.add_step('glycan',last_pred_step)

# print("prediction_pipeline",prediction_pipeline.get_steps('glycan'))

# for known, pred in finders.items():
#     print("known",known)
#     if last_finder_name in pred:
#         known_pipeline.add_step('glycan',config.get_finder(known))

# # but what if I want to add more than 1 final_step types?
# print("prediction_pipeline",prediction_pipeline.get_steps('glycan'))
# print("known_pipeline",known_pipeline.get_steps('glycan'))


# if user specifies - last pred name - use the above code

# sgi, sm will be used everywhere by deafault
# so I create a map for rest of the steps?
# map = {
    # "KnownRoot": ["YOLORootFinder","AnotherRootFinder"],
    # "KnownLink": ["ConnectYOLO", "ConnectYOLOBig"]
# }




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
    # obj0 = pipeline0.run(image)
    # obj1 = pipeline1.run(image)

#     name = os.path.basename(image).split('.')[0]
#     # Annotate Image
#     # obj1.label_image(obj1.image().copy(), name = name + '_yolo')
#     # obj0.label_image(obj0.image().copy(), name = name + '_known')

    for gly0,gly1 in zip(obj0.glycans(),obj1.glycans()):

        # print("\nsemantics",gly1.semantics)
        # gly1.undirected_links()
        # print("directed_links",gly1.semantics)
        # gly1.directed_links()
        print("directed_links",gly1.semantics)

        # comp0 = Glycan_Semantics.compstr(gly0)
        # comp1 = Glycan_Semantics.compstr(gly1)
#         # print("known comp",comp0)
#         # print("pred comp", comp1)
        # print(image,"GOOD" if comp0 == comp1 else "BAD",comp0,comp1)

#     # #     # print("-->",gly0.semantics)
        
        IUPAC0 = Glycan_Semantics.IUPAC(gly0) 
        IUPAC1 = Glycan_Semantics.IUPAC(gly1) 
        print("Known IUPAC: ",IUPAC0) 
        print("Detected IUPAC: ",IUPAC1 if IUPAC1 else 'No sequence detected') 
#         print("\nDo the IUPAC sequences match?", IUPAC0==IUPAC1)


        # print("\nSemantics:",gly0.tojson())       
        # print("\nSemantics:",gly1.tojson())   

    # for gly1 in obj1.glycans():
    #     IUPAC1 = Glycan_Semantics.IUPAC(gly1)
    #     print("Detected IUPAC: ",IUPAC1 if IUPAC1 else 'No sequence detected')

    #     print("composition",Glycan_Semantics.compstr(gly1.monosaccharides()))

        # print("\nSemantics:",gly1.tojson())  
















