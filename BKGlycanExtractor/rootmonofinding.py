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
from .finder import Finder
from .compareboxes import CompareBoxes
from BKGlycanExtractor import RootCompare, DebugMode

            
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
            for mono_semantics in obj.monosaccharides():
                if mono_semantics['box'] == mono and mono_semantics['symbol'] != 'Fuc':
                    obj.set_root(mono_semantics['id'])
                    break
                else:
                    obj.no_root()

        return [ obj.root() ]

# this class needs links to work before root finding
class DefaultOrientationRootFinder(OrientationRootFinder):    
    
    def get_orientation(self, obj):
        
        h_count = 0
        v_count = 0
        
        for mono in obj.monosaccharides():
            aX, aY = mono['center']
            ID = mono['id']
            for mono2 in obj.links(ID):
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


class YOLORootFinder(YOLOModel, RootFinder):
    defaults = {
        'conf_threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0,
        'iou_threshold': 0.4
    }


    def __init__(self,**kwargs):

        params = dict(
            config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
            weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
            conf_threshold = Config.get_param('conf_threshold', Config.FLOAT, kwargs, self.defaults),
            iou_threshold = Config.get_param('iou_threshold', Config.FLOAT, kwargs, self.defaults),
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
            expandimage = Config.get_param('expandimage', Config.INT, kwargs, self.defaults)
        )

        # YOLOModel gets the labels from the current Class and you can use this to set classlabels in the YOLOclass
        self.cb = CompareBoxes()
        YOLOModel.__init__(self,params)
        RootFinder.__init__(self)
        

    def find_boxes(self, obj):
        image = obj.image()
        boxes = self.get_YOLO_output(image)

        if DebugMode.debug:
            DebugMode.log_data(
                identifier = DebugMode.curr_image,
                data = {'root':[len(boxes)]},
                image_path = DebugMode.image_path,
            )

        return boxes

    def match_root_to_mono(self, obj, root):
        semantic_monos = list(obj.monosaccharides())
            
        if semantic_monos == []:
            return None
        
        intersection_list = [0]*len(semantic_monos)

        for i, mono in enumerate(semantic_monos):
            if self.cb.have_intersection(mono['box'], root):
                intersection_list[i] = self.cb.intersection_area(mono['box'], root)
                
        max_int_idx = np.argmax(intersection_list)
        
        box = semantic_monos[max_int_idx]['box']
        # this is the monosaccharide which matched with the root
        selected_mono = semantic_monos[max_int_idx]

        euclidean_distance = self.cb.euclidean_distance(box,root)

        x1,y1,w1,h1 = box.bbox()
        x2,y2,w2,h2 = root.bbox()
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2

        avg_object_size = math.sqrt((avg_width**2 + avg_height**2))  # diagonal

        # helps determine if the two detected boxes (monos and root) are close enough to be considered the same object
        normalized_dist = euclidean_distance / avg_object_size

        return normalized_dist, selected_mono

    
    def find_objects(self, obj):
        boxes = self.find_boxes(obj)

        root_boxes = []
        root = None

        for box in boxes:
            if box.get('classid') == 0:
                root_boxes.append(box)
        
        if len(root_boxes) > 1:
            print("Log data: Multiple Roots were detected")
            obj.log("Multiple roots (%d) were detected"%(len(root_boxes),))
            confidences = [mono.get('confidence') for mono in root_boxes]
            best_index = np.argmax(confidences)
            root = root_boxes[best_index]
        elif len(root_boxes) == 0:
            print("Log: Unable to find root")
            obj.log("Unable to find root")
            obj.glycan_error("Unable to find root")
            obj.no_root()
        else:
            root = root_boxes[0]

        if root:

            normalized_dist, selected_mono = self.match_root_to_mono(obj,root)

            # likely the same object
            if normalized_dist <= 0.5:
                obj.set_root(selected_mono.get('id'),**{'confidence':float(root.get('confidence')),'classlabel': root.get('classlabel')})

                # add alternate root
                # this may or may not intersect the first best root (optionally if we want to
                # add only the second best root - just add a break statement in the for loop)
                if len(root_boxes) > 1:
                    root_boxes_sorted = sorted(root_boxes, key=lambda mono: mono.get('confidence', 0), reverse=True)
                    roots_taken = set()
                    roots_taken.add(selected_mono.get('id'))

                    for i in range(1,len(root_boxes_sorted)):
                        normalized_dist, selected_mono = self.match_root_to_mono(obj,root_boxes_sorted[i])
                        if normalized_dist <= 0.5 and selected_mono.get('id') not in roots_taken:
                            altr = {'classlabel': root_boxes_sorted[i].get('classlabel'),'confidence':float(root_boxes_sorted[i].get('confidence')), "mono_id": selected_mono.get('id')}
                            obj.add_alternative_root(altr)
            else:
                obj.no_root()  
                obj.glycan_error("Unable to find root")
                obj.log("Unable to find root")

        return [ obj.root() ]

class KnownRoot(RootFinder):

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
        box_dict = {}

        image_data = image_path.rsplit('.',1)[0] + "_map.txt"
        with open(image_data, 'r') as file:
            root_id = float('inf')

            # root_id = None
            for line in file:
                if line.startswith('m'):
                    data_points = line.split()
                    # if int(data_points[1]) < root_id:
                    root_id = min(root_id, int(data_points[1]))
                    data_points = line.split()
                    # root_id = int(data_points[1])
                    mono_id = data_points[1]
                    name = data_points[2]

                    x_coords = []
                    y_coords = []

                    for coords in data_points[4:]:
                        if ',' in coords:
                            x,y = map(int,coords.split(','))
                            x_coords.append(x)
                            y_coords.append(y)
                    
                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords) 
                
                    # Note: YOLO predicts 0 or 1 as the classid for roots, 
                    # known_items will also have classid = 0 for root and 1 for rest of the monos
                    box = BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, symbol=name,classid=1,classlabel=self.get_label(1),id=int(mono_id),image=obj.image())
                    box.pad(self.params['boxpadding']) # known data is absolute
                    # boxes.append(box)
                    box_dict[int(mono_id)] = box
                    
            # setting the class_id for root as 0
            box_dict[int(root_id)].set('classid',0)            
            box_dict[int(root_id)].set('classlabel',self.get_label(0)) 

        if DebugMode.debug:
            DebugMode.log_data(
                identifier = DebugMode.curr_image,
                data = {'root_known':len(box_dict.values())},
                image_data = DebugMode.image_data,
            )

        return list(box_dict.values())

    def find_objects(self, obj):
        boxes = self.find_boxes(obj)
        
        for box in boxes:
            if box.get('classid') == 0:
                obj.set_root(box.get('id'),classlabel=box.get('classlabel'))
                break

        return [ obj.root() ]

