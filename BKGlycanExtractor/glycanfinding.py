# -*- coding: utf-8 -*-
"""
class for glycan locating methods.

all subclasses need a find_objects method
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

# Need a way to take padding/crop/threshold before the find_objects()

# create an obj which stores information about the glycan(s) which can be passed to
# other classes

import logging

from . bbox import BoundingBox
from . yolomodels import YOLOModel 
from . glycanannotator import Config

# Base class
class GlycanFinder(object):  

    def execute(self, obj):
        self.find_objects(obj)

    def find_objects(self, obj):
        raise NotImplementedError
        
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanfinding')


# YOLO based glycan finder
# allows minimum confidence thresholding, to restrict returns
# also allows requesting padding of glycan borders (off by default)
# confidences for YOLO detection are stored in the bounding box
class YOLOGlycanFinder(YOLOModel,GlycanFinder):

    defaults = {
        'threshold': 0.0,
        'boxpadding': 0,
    }

    def __init__(self,**kwargs):
        params = dict(
           boxpadding = Config.get_param('boxpadding', Config.FLOAT, kwargs, self.defaults),
           threshold = Config.get_param('threshold', Config.FLOAT, kwargs, self.defaults),
           config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
           weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
        )
        YOLOModel.__init__(self,params)
    #    assert self.classes == 1
        GlycanFinder.__init__(self)
        
    def execute(self, figure_semantics):
        self.find_objects(figure_semantics)

    def find_boxes(self, image):
        return self.get_YOLO_output(image)

    def find_objects(self, figure_semantics):
        image = figure_semantics.image()

        boxes = self.find_boxes(image)

        figure_semantics.clear_glycans()
        for box in boxes:
            figure_semantics.add_glycan(box=box)
    
class SingleGlycanImage(GlycanFinder):

    defaults = {
        'crop': False,
        'padding': 0
    }

    def __init__(self,**kwargs):
       self.crop = Config.get_param('crop', Config.BOOL, kwargs, self.defaults)
       self.padding = Config.get_param('padding', Config.FLOAT, kwargs, self.defaults)
       super().__init__()

    def find_objects(self, obj):
        obj.clear_glycans()
        boxes = self.find_boxes(obj.image())
        obj.add_glycan(box=boxes[0],image_path=obj.image_path())

    def find_boxes(self, image):
        #implement crop and padding?
        height, width, _ = image.shape
        return [ BoundingBox(image=image, x=0, y=0, width=width, height=height) ]

        

