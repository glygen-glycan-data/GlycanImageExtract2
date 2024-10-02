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

from .boundingboxes import BoundingBox
from .yolomodels import YOLOModel 
from .config_data import ConfigData

class FoundGlycan(BoundingBox):
    # class definitions for glycan bounding boxes
    # keep consistent, but can be added to for multiclass model use
    class_dictionary = {
        0: "glycan",
        }
    
    def __init__(self, **kw):
        super().__init__(**kw)

# Base class
class GlycanFinder:  
    def __init__(self,**kw):
        pass

    def find_objects(self, image, **kw):
        raise NotImplementedError
        
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanfinding')


# YOLO based glycan finder
# allows minimum confidence thresholding, to restrict returns
# also allows requesting padding of glycan borders (off by default)
# confidences for YOLO detection are stored in the bounding box
class YOLOGlycanFinder(GlycanFinder,YOLOModel,ConfigData):
    def __init__(self,config,**kwargs):
        self.threshold = kwargs.get('threshold',0.0)
        self.padding = kwargs.get('padding',False)

        self.defaults = {
            'weights': './BKGlycanExtractor/config/largerboxes_plusindividualglycans.weights',
            'config_net': './BKGlycanExtractor/config/coreyolo.cfg'
        }
        
        GlycanFinder.__init__(self)
        ConfigData.__init__(self,config,self.defaults,class_name=self.__class__.__name__)
        self.weights = self.get_param('weights',**kwargs)
        self.config_net = self.get_param('config',**kwargs)

        current_config = {
            'weights':self.weights,
            'config': self.config_net
        }

        YOLOModel.__init__(self,current_config)

        
    def execute(self, figure_semantics):
        self.find_objects(figure_semantics)

    def find_objects(self, figure_semantics):
        image = figure_semantics.semantics['image']

        # threshold = kw.get('threshold', 0.0)
        # padding = kw.get('request_padding', False)
        threshold = self.threshold
        padding = self.padding

        glycans = self.get_YOLO_output(image, threshold, request_padding=padding)

        glycans = [FoundGlycan(boundingbox=glycan) for glycan in glycans]

        for idx,glycan in enumerate(glycans):
            glycan_obj = self.save_object(idx,glycan,image)
            figure_semantics.semantics['glycans'].append(glycan_obj)
        return glycans

    
    def save_object(self,idx,glycan,image):
        (x, y), (x2, y2) = glycan.to_image_coords()
        glycan_img = image[y:y2, x:x2].copy()
        height, width, _ = glycan_img.shape

        single_glycan = {
            'id': idx,
            'image': glycan_img,
            'width': width,
            'height': height,
            'bbox': [x,y,width,height],
            'box': glycan,
            'monos': [],
        }
        return single_glycan

