import os
import shutil
import sys
import time
# from BKGlycanExtractor.semantics import Image_Semantics, Figure_Semantics, Glycan_Semantics
import BKGlycanExtractor.glycanannotator as ga

# args: 1) Annotator-pipelinename (if user doesnt enter print the entire list of types of Annotators),
# 2) folder_path/list of file_names
# 3) remove the configs file arg

# the constructor should be able to get the configs folder directly without explicit mention

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
    # pipeline = config.get(“BaseFinder”) remove configs_dir, configs_file
    pipeline = ga.Annotator(configs_dir, configs_file, pipeline_name)

    # configs = pipeline.read_pipeline_configs(configs_dir, configs_file, pipeline_name)

    for image in os.scandir(glycan_folder):
        obj = pipeline.run(image)
        print("\nSemantics:",obj)











