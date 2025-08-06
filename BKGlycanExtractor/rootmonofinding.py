# -*- coding: utf-8 -*-
"""
class for various methods of identifying the root monosacharide
"""
import logging

import numpy as np
import math

from .bbox import BoundingBox
from .yolomodels import YOLOModel
from .glycanannotator import Config
from .finder import Finder,YOLOFinder,KnownFinder
from .compareboxes import CompareBoxes
from BKGlycanExtractor import RootCompare, DebugMode, FilterAlternativeRoots
from .semantics import RootSemantics

            
class RootFinder(Finder):

    labels = ['redend','not_redend']
    finder_class = 'Root'

    semantic_compare = RootCompare
    
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
            for mono_semantics in obj.monos():
                if mono_semantics.box() == mono and mono_semantics.symbol() != 'Fuc':
                    root_semantics = RootSemantics(mono_id=mono_semantics.get('id'),box=mono_semantics.box())
                    obj.set_root(root_semantics)
                    break
                else:
                    obj.reset_root()

        return [ obj.root() ]

# this class needs links to work before root finding
class DefaultOrientationRootFinder(OrientationRootFinder):    
    
    def get_orientation(self, obj):
        
        h_count = 0
        v_count = 0
        
        for mono in obj.monosaccharides():
            aX, aY = mono.get('center')
            ID = mono.get('id')
            for mono2 in obj.links(ID):
                for x in obj.monosaccharides():
                    if x.get('id') == mono2:
                        mono2 = x
                        break

                bX, bY = mono2.get('center')
                
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
 

class YOLOOrientationRootFinder(OrientationRootFinder):
    defaults = {
        'threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0,
    }
    labels = ["left_right","right_left","top_bottom","bottom_top"]

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

    def find_boxes(self,image):
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


class YOLORootFinder(YOLOFinder, RootFinder):
    
    filters = [FilterAlternativeRoots()]
    # filters = []

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

        self.cb = CompareBoxes()
        YOLOModel.__init__(self,self.params)
        RootFinder.__init__(self)
        
    
    def find_objects(self, obj):
        obj_list = []

        boxes = self.find_boxes(obj)
        for box in boxes:
            root_obj = self.box_to_object(box,obj)
            if root_obj:
                obj_list.append(root_obj)

        accepted, rejected = self.filter_objects(obj_list)

        # if accepted is empty - no root found
        if len(accepted) == 1:
            # obj.set_root(accepted[0]["mono_id"], **{k: v for k, v in accepted[0].items() if k != "mono_id"})
            obj.set_roots(accepted[0],rejected)

        elif len(accepted) == 0:       # no root
            obj.set_roots(None,rejected)
            obj.add_glycan_error("Unable to find root")
            obj.log("Unable to find root")
        
        else:
            raise ValueError("More than one accepted root.")

        # add a step to process the rejected monos?
        # so idea here is to add the alternative monos present in the rejected list

        # print("found root", obj.root())
        
        
        return [ obj.root() ]

    
    def box_to_object(self,box,obj):
        '''
        checks if the detected root_box has any mono which is close enough to match with it
        '''
        if self.box_label(box) != "redend":
            return None
        
        normalized_dist, selected_mono = self.match_root_to_mono(obj,box)

        if normalized_dist <= 0.5: 
            return RootSemantics(
                mono_id=selected_mono.id(),
                confidence=box.get('confidence'),
                classlabel=self.box_label(box),
                box=box
            )

        return None


    def match_root_to_mono(self, obj, root_box):
        semantic_monos = list(obj.monos())
            
        if semantic_monos == []:
            return None
        
        intersection_list = [0]*len(semantic_monos)

        for i, mono in enumerate(semantic_monos):
            if self.intersect(mono, root_box):
                intersection_list[i] = self.intersection_area(mono, root_box)
                
        max_int_idx = np.argmax(intersection_list)
        
        mono_box = semantic_monos[max_int_idx].box()
        # this is the monosaccharide which matched with the root
        selected_mono = semantic_monos[max_int_idx]

        euclidean_distance = self.dist(mono_box,root_box)
        

        x1,y1,w1,h1 = mono_box.bbox()
        x2,y2,w2,h2 = root_box.bbox()
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2

        avg_object_size = math.sqrt((avg_width**2 + avg_height**2))  # diagonal

        # helps determine if the two detected boxes (monos and root) are close enough to be considered the same object
        normalized_dist = euclidean_distance / avg_object_size

        return normalized_dist, selected_mono



class KnownRoot(KnownFinder,RootFinder):

    labels = ['redend','not_redend']

    defaults = {
        'boxpadding': 0,
    }

    def __init__(self,**kwargs):
        self.params = dict(
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
        )

        RootFinder.__init__(self)

    def find_boxes(self, obj):
        image_path = obj.image_path()
        boxes = []
        box_dict = {}

        map_dict = self.get_known_data(image_path)

        root_id = map_dict['root']
        for mono_id, mono_details in map_dict['monos'].items():

            classid = 0 if root_id == mono_id else 1

            box = BoundingBox(
                x1=mono_details['x_min'], y1=mono_details['y_min'],
                x2=mono_details['x_max'], y2=mono_details['y_max'],
                mono_id=mono_id,
                classid=classid,
                classlabel=self.get_label(classid)
            )
            box.pad(self.params['boxpadding']) # known data is absolute
            boxes.append(box)
        return boxes

    def find_objects(self, obj):
        boxes = self.find_boxes(obj)
        
        for box in boxes:
            if box.get('classid') == 0:
                root = RootSemantics(mono_id=box.get('mono_id'),classlabel=box.get('classlabel'))
                obj.set_roots(root)
                break

        return [ obj.root() ]

