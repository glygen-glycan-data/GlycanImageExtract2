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
from BKGlycanExtractor import RootCompare, BoxCompare, DebugMode

            
class RootFinder:
    orientation_type = ["left_right","right_left","top_bottom","bottom_top"]
    mono_type = ["root_mono","nonroot"]
    mono_syms = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]
    

    def execute(self, obj):
        self.find_objects(obj)

    def find_objects(self, obj):
        raise NotImplementedError

    @staticmethod
    def box_components(iou):
        return BoxCompare(iou)
        
    @staticmethod
    def semantic_components(proximity):
        return RootCompare(proximity)

    @staticmethod
    def known_predictor():
        return KnownRoot()
        
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
                else:
                    obj.add_root(None)




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

        RootFinder.__init__(self)
        YOLOModel.__init__(self,params)

        # YOLO detects two classes for roots - either 0 (root) or 1 (non-root)
        assert self.classes == 2


    def find_boxes(self, image, **kwargs):

        boxes = self.get_YOLO_output(image)

        if DebugMode.debug:
            DebugMode.log_data(
                identifier = DebugMode.curr_image,
                data = {'root':[len(boxes)]},
                image_path = DebugMode.image_path,
            )

        return boxes
    

    def find_objects(self, obj, **kwargs):
        image = obj.image()

        boxes = self.find_boxes(image)

        root_boxes = []
        root_mono = None


        for box in boxes:
            if box.get('classid') == 0:
                root_boxes.append(box)

        if len(root_boxes) > 1:
            print("Log data: Multiple Roots were detected")
            # print("Log data: Multiple Roots were detected", [(mono.get('confidence'), dir(mono)) for mono in root_boxes])
            confidences = [mono.get('confidence') for mono in root_boxes]
            best_index = np.argmax(confidences)
            root_mono = root_boxes[best_index]
        elif len(root_boxes) == 0:
            print("Log: No root was detected")
            obj.add_root(-1) 
        else:
            root_mono = root_boxes[0]

        # assert root_mono is not None  # remove it because it should not break the whole semantics PR
        # treat is as FN on the sequence - for PR curves of knownSemantics
        if root_mono:
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
                obj.add_root(root.get('id'), root_mono.get('confidence'))
                box.set('confidence',root_mono.get('confidence'))
            else:
                obj.add_root(-1)  

        root_id = obj.root()

        return obj


class KnownRoot(RootFinder):

    defaults = {
        'boxpadding': 0,
    }

    def __init__(self,**kwargs):
        self.params = dict(
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
        )

    def find_boxes(self, image):
        
        box_dict = {}

        image_path = image.rsplit('.',1)[0] + "_map.txt"
        with open(image_path, 'r') as file:
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

                    for coords in data_points[3:]:
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
                    box = BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, symbol=name,classid=1,id=int(mono_id))
                    box.pad(self.params['boxpadding']) # known data is absolute
                    # boxes.append(box)
                    box_dict[int(mono_id)] = box

            box_dict[int(root_id)].set('classid',0)            

        if DebugMode.debug:
            DebugMode.log_data(
                identifier = DebugMode.curr_image,
                data = {'root_known':len(box_dict.values())},
                image_path = DebugMode.image_path,
            )

        return list(box_dict.values())
                                      
    def find_objects(self, obj):
        image_path = obj.image_path()
        boxes = self.find_boxes(image_path)

        for box in boxes:
            classid = box.get('classid')
            if classid == 0:
                obj.add_root(box.get('id'))

            box.set('classid',self.mono_syms.index(box.get('symbol')))

        return obj
