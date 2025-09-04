# -*- coding: utf-8 -*-
"""
class for monosaccharide finding methods.
"""
import cv2
import logging
import math
import numpy as np
import os
import sys
import json
from PIL import Image

from .bbox import BoundingBox
from .yolomodels import YOLOModel
from .glycanannotator import Config, Config_Manager, GlycanExtractorPipeline
from .finder import Finder, YOLOFinder, KnownFinder
from .compareboxes import CompareBoxes
from .semantics import MonoSemantics
from BKGlycanExtractor import MonosCompare, DebugMode, FilterOverlaps

# We are inheriting YOLOFinder in the heruristic finders are well - might not be a good approach
# the Heuristic finders have their own find_object and find_boxes (they shouldnt use the base class methods) 
class MonoFinder: 
    '''
    Base class 'Finder' requires that labels should be defined.
    Either set labels in this class or other child classes inherting from MonoFinder can set it.
    
    It is preferred to have the child class inheriting from MonoFinder set the labels 
    from the .labels file that was used for training,
    so that multiple different finders trained on different labels can be used
    collectively at once.

    Note: Some models were trained with/without Xylose (Xyl), hence it 
    required to refer to the .labels file if using a specific training model.
    '''

    def set_results(self, obj, accepted, rejected):
        obj.set_monos(accepted, rejected)

    def finder_pipeline(self,config_manager):
        pipeline = GlycanExtractorPipeline()
        pipeline.set_steps('figure', config_manager.get_finders("SingleGlycanImage"))
        pipeline.set_steps('glycan', [self])
        return pipeline

    def semantic_compare(self,**kwargs):
        return MonosCompare(**kwargs)

    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.monosaccharideid')
    
class YOLOMonos(YOLOFinder,MonoFinder):

    filters = [ FilterOverlaps() ]
   
    def __init__(self,**kwargs):   
        YOLOFinder.__init__(self,**kwargs)
        MonoFinder.__init__(self)

    def box_to_object(self,box,obj):
        symbol = box.get('classlabel')
        return MonoSemantics(symbol=symbol,box=box,**box.items())

class KnownMono(MonoFinder,KnownFinder):

    def __init__(self,**kwargs):
        KnownFinder.__init__(self,**kwargs)
        MonoFinder.__init__(self)

    # map_dict structure is present in KnownFinder class
    def create_boxes(self, map_dict):
        boxes = []

        for id, data in map_dict['monos'].items():

            symbol = data['symbol']

            box = BoundingBox(
                x1=data['x_min'], y1=data['y_min'], 
                x2=data['x_max'], y2=data['y_max'], 
                symbol=symbol,
                classid=self.get_label_index(symbol),
                classlabel=symbol,
                id=id
            )
            boxes.append(box)

        return boxes

    def box_to_object(self,box,obj):
        return MonoSemantics(box=box,**box.items())

    
