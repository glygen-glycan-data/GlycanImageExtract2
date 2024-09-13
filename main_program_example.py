# -*- coding: utf-8 -*-
"""
annotate all files in a folder using one pipeline, start-to-finish
"""

import os
import shutil
import sys
import time

import BKGlycanExtractor.glycanannotator as glycanannotator

glycan_folder = os.path.abspath(sys.argv[1])
# print(glycan_folder)
try:
    configs_dir = sys.argv[3]
except IndexError:
    configs_dir = "./BKGlycanExtractor/config/"
configs_file = os.path.join(configs_dir, "configs.ini")

try:
    pipeline = sys.argv[2]
except IndexError:
    pipeline = "YOLOMonosAnnotator"
if os.path.isdir(pipeline):
    pipeline = "YOLOMonosAnnotator"

print("Annotating using", pipeline)

annotator = glycanannotator.Annotator()
methods = annotator.read_pipeline_configs(configs_dir, configs_file, pipeline)

for file in os.scandir(glycan_folder):
    glycan_file = os.path.abspath(file)
    # print(glycan_file)
    glycan_name = os.path.basename(glycan_file)
    base_glycan_name = glycan_name.rsplit('.', 1)[0]

    #initialize directory to save output
    workdir = os.path.join("./files", base_glycan_name+"_"+pipeline)
    if os.path.isdir(os.path.join(workdir)):
        shutil.rmtree(os.path.join(workdir))
        time.sleep(5)
    os.makedirs(os.path.join(workdir))
    os.makedirs(os.path.join(workdir, "input"))
    os.makedirs(os.path.join(workdir, "output")) 
    
    shutil.copyfile(glycan_file, os.path.join(workdir, "input", glycan_name))  
    
    annotator.set_loggers(workdir, file, methods)

    results = annotator.annotate_file(glycan_file, methods)

    print("results:",results)

    # annotator.save_results(workdir, glycan_file, results, annotation)
    # annotator.close_logger(glycan_file)
    
print("Finished.")