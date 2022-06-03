import BKGLycanExtractor.SystemInteraction as si
from BKGLycanExtractor.inputhandler import InputHandler

import sys, logging
import numpy as np


logger = logging.getLogger("search")
logger.setLevel(logging.INFO)

item = InputHandler(sys.argv[1])
base_configs = "./BKGLycanExtractor/configs/"

print(item, "Start")

weight=base_configs+"Glycan_300img_5000iterations.weights"
coreyolo=base_configs+"coreyolo.cfg"
colors_range=base_configs+"colors_range.txt"

color_range_file = open(colors_range)
color_range_dict = {}
for line in color_range_file.readlines():
    line = line.strip()
    name = line.split("=")[0].strip()
    color_range = line.split("=")[1].strip()
    color_range_dict[name] = np.array(list(map(int, color_range.split(","))))

configs = {
    "weights" : weight,
    "net" : coreyolo,
    "colors_range": color_range_dict}

framework = si.FindGlycans(str(item))
framework.initialize_directory()

workdir = framework.get_directory()


framework.log()

logger = logging.getLogger(str(item))

if item.is_file:
    framework.make_copy()
    extn = item.ext
else:
    extn = "Not a file"
    
item.annotate(configs)

output = item.results
if hasattr(item, "annotated_image"):
    output_png = item.annotated_image
else: 
    output_png = None
if hasattr(item, "annotated_doc"):
    output_pdf = item.annotated_doc
else:
    output_pdf = None
    
framework.get_output(output, doc = output_pdf, image = output_png)


framework.save_results_json(extn)

logger.info("%s Finished", str(item))
