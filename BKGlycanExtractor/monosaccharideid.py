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
from .glycanannotator import Config, Config_Manager
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

    labels = None
    # labels = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]
    finder_class = 'Monosaccharide'

    semantic_compare = MonosCompare

    def set_results(self, obj, accepted, rejected):
        obj.set_monos(accepted, rejected)
    
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.monosaccharideid')
    


class YOLOMonos(YOLOFinder,MonoFinder):

    filters = [FilterOverlaps()]
   
    defaults = {
        'conf_threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0,
        'iou_threshold': 0.4
    }

    def __init__(self,**kwargs):

        self.params = dict(
            config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
            weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
            conf_threshold = Config.get_param('conf_threshold', Config.FLOAT, kwargs, self.defaults),
            iou_threshold = Config.get_param('iou_threshold', Config.FLOAT, kwargs, self.defaults),
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
            expandimage = Config.get_param('expandimage', Config.INT, kwargs, self.defaults)
        )   

        YOLOModel.__init__(self,self.params)
        MonoFinder.__init__(self)


    def box_to_object(self,box,obj):
        symbol = box.get('classlabel')
        return MonoSemantics(symbol=symbol,box=box,**box.items())


class KnownMono(MonoFinder,KnownFinder):

    # Need to be able to support any monosaccharide symbol in generated code
    # maybe allow users to add their own known monos labels?
    # maybe create a function in finder which accepts labels text file?
    labels = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc","Xyl"]
    defaults = {
        'boxpadding': 0,
    }

    def __init__(self,**kwargs):

        # config file created from training data
        # maybe make the first 4 lines a part of the class data member?
        model_ini = Config.get_param('model', Config.CONFIGFILE, kwargs, self.defaults)
        cm = Config_Manager(config_filename=model_ini)
        finder = cm.list_finders()[0]
        secondary_config = cm.get_config(f"Finder:{finder}")
        kwargs['__secondary_config__'] = secondary_config

        self.params = dict(
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
        )
        MonoFinder.__init__(self)
        self.set_labels(self.labels)


    # map_dict structure is present in KnownFinder class
    def create_boxes(self, map_dict):
        boxes = []

        for id, data in map_dict['monos'].items():

            symbol = data['symbol']

            box = BoundingBox(x1=data['x_min'], y1=data['y_min'], 
                x2=data['x_max'], y2=data['y_max'], 
                symbol=symbol,
                classid=self.get_label_index(symbol),
                classlabel=symbol,
                id=id
            )

            box.pad(self.params['boxpadding']) # known data is absolute
            boxes.append(box)

        return boxes

    def box_to_object(self,box,obj):
        return MonoSemantics(box=box,**box.items())

    