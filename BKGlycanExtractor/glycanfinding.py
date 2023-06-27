# -*- coding: utf-8 -*-
"""
class for glycan locating methods.

all subclasses need a find_glycans method
which takes an image 
and returns bounding boxes for all glycans in the image.

any classes for detection need to return Detected bounding boxes, 
with confidence values.

bounding boxes are laid out in boundingboxes.py.
they require the image the glycan was found in, 
some set of coordinates, and confidence of detection.
YOLO format is not required; coordinates can be absolute or relative, 
center/w/h, 4 corners, etc.

to avoid later errors, 
use the boundingbox coordinate conversion functions
to completely fill out the coordinate system 
during the initial definition
(relative center_x/y, relative width/height, 
 absolute center_x/y/width/height, 4 corners)

"""

import logging

from .boundingboxes import BoundingBox
from .yolomodels import YOLOModel 

class FoundGlycan(BoundingBox):
    # class definitions for glycan bounding boxes
    # keep consistent, but can be added to for multiclass model use
    class_dictionary = {
        0: "glycan",
        }
    
    def __init__(self, **kw):
        super().__init__(**kw)

class GlycanFinder:  
    def __init__(self, **kw):
        # for debugging
        # print('start GlycanFinder.__init__')
        # super().__init__()
        # print('end GlycanFinder.__init__')
        pass
    
    def find_glycans(self, image, **kw):
        raise NotImplementedError
        
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanfinding')


# YOLO based glycan finder
# allows minimum confidence thresholding, to restrict returns
# also allows requesting padding of glycan borders (off by default)
# confidences for YOLO detection are stored in the bounding box
       
class YOLOGlycanFinder(YOLOModel, GlycanFinder):
    
    def __init__(self, configs):
        # for debugging
        # print('start YOLOGlycanFinder.__init__')
        super().__init__(configs)
        # print('end YOLOGlycanFinder.__init__')
        
    def find_glycans(self, image, **kw):
        
        threshold = kw.get('threshold', 0.0)
        
        padding = kw.get('request_padding', False)
        
        image_dict = self.format_image(image)
        
        glycans = self.get_YOLO_output(
            image_dict, threshold, request_padding=padding
            )
        
        glycans = [FoundGlycan(boundingbox=glycan) for glycan in glycans]
        
        for glycan in glycans:
            printstr = f'Glycan found at: {glycan.to_list()}'
            self.logger.info(printstr)
        
        return glycans
    
"""  
if __name__ == "__main__":
    config_dict = {
        "weights": "./config/largerboxes_plusindividualglycans.weights",
        "config": "./config/coreyolo.cfg"}
    glycanfinder = YOLOGlycanFinder(config_dict)
"""
