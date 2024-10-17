# -*- coding: utf-8 -*-
"""
class for various methods of identifying the root monosacharide
"""
import logging

import numpy as np

from .bbox import BoundingBox
from .yolomodels import YOLOModel
from .glycanannotator import Config
from .compareboxes import CompareBoxes
            
            
class RootFinder:
    orientation_type = ["left_right","right_left","top_bottom","bottom_top"]
    mono_type = ["root_mono","nonroot"]

    def execute(self, obj):
        self.find_objects(obj)

    def find_objects(self, obj):
        raise NotImplementedError
        
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.rootmonofinding')
        
# Base class for orientation finders   
class OrientationRootFinder(RootFinder):
    # base class for routes that rely on orientation
    # all subclasses need a get_orientation method
    # returns orientation and its confidence
    # 0 (left-right), 1 (right-left), 2 (top-bottom), 3 (bottom-top)
    # potential to add other orientations (diagonal?) with numbers >3

    def get_orientation(self, obj):
        raise NotImplementedError


    def find_objects(self, obj):
        orientation, confidence = self.get_orientation(obj)
        mono_boxes = obj.mono_boxes()
        
        # left-right
        if orientation == 0:
            mono_boxes.sort(
                key=lambda mono: mono.center()[0], 
                reverse=False
                )
            
        # right-left
        elif orientation == 1:
            mono_boxes.sort(
                key=lambda mono: mono.center()[0],
                reverse=True
                )
            
        # top-bottom
        elif orientation == 2:
            mono_boxes.sort(
                key=lambda mono: mono.center()[1],
                reverse=False
                )
            
        # bottom-top
        elif orientation == 3:
            mono_boxes.sort(
                key=lambda mono: mono.center()[1],
                reverse=True
                )

        # iterate mono_boxes --> check which box matches with semantic_boxes  --> once you find
        # the match, if the symbol in the semantics is not Fuc --> add it as your root, else root is None
        for mono in mono_boxes:
            for mono_semantics in obj.monosaccharides():
                if mono_semantics['box'] == mono and mono_semantics['symbol'] != 'Fuc':
                    obj.add_root(mono_semantics['id'])
                    break



# this class needs links to work before root finding
class DefaultOrientationRootFinder(OrientationRootFinder):    
    
    def get_orientation(self, obj):
        
        h_count = 0
        v_count = 0
        
        for mono in obj.monosaccharides():
            aX, aY = mono['center']
            ID = mono['id']
            for mono2 in obj.get_links(ID):
                for x in obj.monosaccharides():
                    if x['id'] == mono2:
                        mono2 = x
                        break

                bX, bY = mono2['center']
                
                xdiff = abs(aX - bX)
                ydiff = abs(aY - bY)
                
                if xdiff > ydiff:
                    h_count += 1
                elif ydiff > xdiff:
                    v_count += 1
                    
        if h_count >= v_count:
            return 1, 1        # right-left, confidence value - what should we set this to?
        else:
            return 3, 1        # bottom-top, confidence value
 

class YOLOOrientationRootFinder(YOLOModel, OrientationRootFinder):
    defaults = {
        'threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0,
    }

    def __init__(self,**kwargs):

        params = dict(
            config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
            weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
            threshold = Config.get_param('threshold', Config.FLOAT, kwargs, self.defaults),
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
            expandimage = Config.get_param('expandimage', Config.INT, kwargs, self.defaults)
        )

        OrientationRootFinder.__init__(self)
        YOLOModel.__init__(self,params)

    def get_boxes(self,image):
        return self.get_YOLO_output(image)
 
    def get_orientation(self, obj):
        image = obj.image()
                
        oriented_glycans = self.get_boxes(image)
        confidences = [box.get('confidence') for box in oriented_glycans]
        
        try:
            best_index = np.argmax(confidences)
        except ValueError:
            return None, None
        
        oriented_glycan = oriented_glycans[best_index]

        return oriented_glycan.get('classid'), oriented_glycan.get('confidence')


class YOLORootFinder(YOLOModel, RootFinder):
    defaults = {
        'threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0,
    }

    def __init__(self,**kwargs):

        params = dict(
            config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
            weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
            threshold = Config.get_param('threshold', Config.FLOAT, kwargs, self.defaults),
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
            expandimage = Config.get_param('expandimage', Config.INT, kwargs, self.defaults)
        )

        RootFinder.__init__(self)
        YOLOModel.__init__(self,params)


    def find_boxes(self, image):
        return self.get_YOLO_output(image)
    
    def find_objects(self, obj):
        image = obj.image()

        mono_boxes = self.find_boxes(image)
        
        root_monos = [x for x in mono_boxes if x.get('classid') == 0]
        if len(root_monos) == 0:
            return None
        elif len(root_monos) == 1:
            root_mono = root_monos[0]
        else:
            confidences = [mono.get('confidence') for mono in root_monos]
            best_index = np.argmax(confidences)
            root_mono = root_monos[best_index]

        # compare root box with all the boxes' of monosaccharides - the best match will be the root ID
        semantic_monos = list(obj.monosaccharides())
        
        if semantic_monos == []:
            return None

        comparison_alg = CompareBoxes()
        
        intersection_list = [0]*len(semantic_monos)

        for i, mono in enumerate(semantic_monos):
            if comparison_alg.have_intersection(mono['box'], root_mono):
                intersection_list[i] = comparison_alg.intersection_area(mono['box'], root_mono)
                
        max_int_idx = np.argmax(intersection_list)
        
        box = semantic_monos[max_int_idx]['box']
        t_area = box.area()
        d_area = root_mono.area()
        
        inter = intersection_list[max_int_idx]
        
        if ((inter == t_area and comparison_alg.training_contained(box, root_mono))
        or (inter == d_area and comparison_alg.detection_sufficient(box, root_mono))
        or comparison_alg.is_overlapping(box, root_mono)):
            root = semantic_monos[max_int_idx]
            obj.add_root(root.get('id'))
    