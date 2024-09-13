import os
import shutil
import sys
import time
# from BKGlycanExtractor.semantics import Image_Semantics, Figure_Semantics, Glycan_Semantics
import BKGlycanExtractor.glycanannotator as ga

glycan_folder = os.path.abspath(sys.argv[1])

try:
    configs_dir = sys.argv[3]
except IndexError:
    configs_dir = "./BKGlycanExtractor/config/"
configs_file = os.path.join(configs_dir, "configs.ini")

try:
    pipeline_name = sys.argv[2]
except IndexError:
    pipeline_name = "YOLOMonosAnnotator"
if os.path.isdir(pipeline_name):
    pipeline_name = "YOLOMonosAnnotator"

print("Annotating using", pipeline_name)
    

if __name__ == '__main__':
    pipeline = ga.Annotator()

    # pipeline = config.read_pipeline_configs(configs_dir, configs_file, pipeline_name)
    configs = pipeline.read_pipeline_configs(configs_dir, configs_file, pipeline_name)

    for image in os.scandir(glycan_folder):
        obj = pipeline.run(configs,image)
        print("\nSemantics:",obj)









# glycanfinder = pipeline.get("glycanfinder")
# monosfinder = pipeline.get("mono_id")
# linkfinder = pipeline.get("connector")
# rootfinder = pipeline.get("rootfinder") 
# builder = pipeline.get("builder")
# searches = pipeline.get("search")


# print("glycanfinder",glycanfinder)
# print("monosfinder",monosfinder)
# print("linkfinder",linkfinder)
# print("rootfinder",rootfinder)


# img_semantics = Image_Semantics()
# glycan_semantics = Glycan_Semantics()

# for image in os.scandir(glycan_folder):

#     # call the semantics class and pass image, so that the skeleton can be filled with some data
#     figure_semantics = Figure_Semantics(image)
    
#     # Image_Semantics has all the common data that is stored in an obj - JSON format
#     # img_semantics = Image_Semantics()
#     # img_semantics.create_image_semantics(figure_semantics)

#     glycanfinder.find_objects(figure_semantics)

#     for gly_obj in figure_semantics.glycans():
#         monosfinder.find_objects(gly_obj)
#         linkfinder.find_objects(gly_obj)
#         rootfinder.find_objects(gly_obj)

#         # print("\ngly_obj",gly_obj)


#     # print("\nfigure_semantics",figure_semantics.glycans())


#     # print("\n---->>>>",glycan_semantics.mono_links())










