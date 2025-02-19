# -*- coding: utf-8 -*-
"""
class for glycan locating methods.

all subclasses need a find_objects method
which takes an image and returns semantic data for all glycans in the image.

any classes for detection need to return Detected bounding boxes, 
with confidence values.

bounding boxes are laid out in bbox.py.
they require the image the glycan was found in, 
some set of coordinates, and confidence of detection.
"""

import logging
import os
import json
from . bbox import BoundingBox
from . yolomodels import YOLOModel 
from . glycanannotator import Config
from BKGlycanExtractor import DebugMode

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
        'conf_threshold': 0.5,
        'boxpadding': 0,
        'iou_threshold': 0.5
    }

    def __init__(self,**kwargs):
        params = dict(
           boxpadding = Config.get_param('boxpadding', Config.FLOAT, kwargs, self.defaults),
           conf_threshold = Config.get_param('conf_threshold', Config.FLOAT, kwargs, self.defaults),
           iou_threshold = Config.get_param('iou_threshold', Config.FLOAT, kwargs, self.defaults),
           config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
           weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
        )
        YOLOModel.__init__(self,params)
        assert self.classes == 1
        GlycanFinder.__init__(self)

    def find_boxes(self, image):
        return self.get_YOLO_output(image)

    def find_objects(self, obj):
        image = obj.image()
        print("Beofr",image.shape)

        # do expand_image in init instead of here
        image = self.expand_image(image, expand=200)
        obj.set_image(image)
        
        boxes = self.find_boxes(image)
        print("find_objects",image.shape)

        print("semantics_image",obj.image().shape)

        obj.clear_glycans()
        for box in boxes:
            obj.add_glycan(box=box)
    
    
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

        

