# -*- coding: utf-8 -*-
"""
class for various methods of identifying the root monosacharide
"""
import logging

import numpy as np

from .boundingboxes import BoundingBox
from . import compareboxes
from .yolomodels import YOLOModel


class FoundOrientedGlycan(BoundingBox):
    class_dictionary = {
        0: "left_right",
        1: "right_left",
        2: "top_bottom",
        3: "bottom_top",
        }
    
    def __init__(self, **kw):
        super().__init__(**kw)
            
            
class FoundRootMonosaccharide(BoundingBox):
    class_dictionary = {
        0: "root_mono",
        1: "nonroot"
        }
    
    def __init__(self, **kw):
        super().__init__(**kw)
            
            
class RootFinder:
    def __init__(self, **kw):
        pass
    def find_objects(self, mono_info, **kw):
        raise NotImplementedError
        
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.rootmonofinding')
        
        
class OrientationRootFinder(RootFinder):
    # base class for routes that rely on orientation
    # all subclasses need a get_orientation method
    # returns orientation and its confidence
    # 0 (left-right), 1 (right-left), 2 (top-bottom), 3 (bottom-top)
    # potential to add other orientations (diagonal?) with numbers >3
    def find_objects(self, mono_info, **kw):
        orientation, confidence = self.get_orientation(mono_info, **kw)
        # print(orientation)
        monos_list = mono_info.get_monos()
        
        # left-right
        if orientation == 0:
            monos_list.sort(
                key=lambda mono: mono.get_center_point()[0], 
                reverse=False
                )
            
        # right-left
        elif orientation == 1:
            monos_list.sort(
                key=lambda mono: mono.get_center_point()[0],
                reverse=True
                )
            
        # top-bottom
        elif orientation == 2:
            monos_list.sort(
                key=lambda mono: mono.get_center_point()[1],
                reverse=False
                )
            
        # bottom-top
        elif orientation == 3:
            monos_list.sort(
                key=lambda mono: mono.get_center_point()[1],
                reverse=True
                )

        for mono in monos_list:
            if mono.get_ID().find("Fuc") == -1:
                mono.set_root()
                break 
        
        mono_info.set_monos(monos_list)
        return confidence
    
    def get_orientation(self, **kw):
        raise NotImplementedError

class DefaultOrientationRootFinder(OrientationRootFinder):    
    def get_orientation(self, obj):
        monos = [mono['box'] for mono in obj['monos']]
        # monos = obj.get_monos()
        
        h_count = 0
        v_count = 0
        
        for mono in monos:
            aX, aY = [mono.cen_x, mono.cen_y]
            for mono2 in mono.get_linked_monos():
                for x in monos:
                    if x.get_ID() == mono2:
                        mono2 = x
                        break
                bX, bY = mono2.get_center_point()
                
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
    def __init__(self, configs):
        super().__init__(configs)
        
    def get_orientation(self, mono_info, **kw):
        glycanimage = mono_info.get_image()
        
        threshold = kw.get('threshold', 0.0)
        image_dict = self.format_image(glycanimage)
        
        oriented_glycans = self.get_YOLO_output(image_dict, threshold)
        
        oriented_glycans = [
            FoundOrientedGlycan(boundingbox=glycan) 
            for glycan in oriented_glycans
            ]
        
        confidences = [box.get_confidence() for box in oriented_glycans]
        
        try:
            best_index = np.argmax(confidences)
        except ValueError:
            return None, None
        
        oriented_glycan = oriented_glycans[best_index]
        
        return oriented_glycan.get_class(), oriented_glycan.get_confidence()
    
class YOLORootFinder(YOLOModel, RootFinder):
    # finds the root monosaccharide directly
    def __init__(self, configs):
        super().__init__(configs)
    
    def find_objects(self, obj, **kw):
        glycanimage = obj['image']
        threshold = kw.get('threshold', 0.0)
        
        # image_dict = self.format_image(glycanimage)
        
        monosaccharides = self.get_YOLO_output(glycanimage, threshold)
        
        monosaccharides = [
            FoundRootMonosaccharide(boundingbox=mono) 
            for mono in monosaccharides
            ]
        
        root_monos = [x for x in monosaccharides if x.get_class() == 0]
        if len(root_monos) == 0:
            return None
        elif len(root_monos) == 1:
            root_mono = root_monos[0]
        else:
            confidences = [mono.get_confidence() for mono in root_monos]
            best_index = np.argmax(confidences)
            root_mono = root_monos[best_index]

        monos_list, monos_id, mono_boxes =  zip(*[(mono['box'], mono['id'], mono['box']) for mono in obj['monos']])
        # monos_list = mono_info.get_monos()
        if monos_list == []:
            return None
        comparison_alg = compareboxes.CompareBoxes()
        
        intersection_list = [0]*len(monos_list)

        for i, mono in enumerate(monos_list):
            if comparison_alg.have_intersection(mono, root_mono):
                intersection_list[i] = comparison_alg.intersection_area(
                                       mono, root_mono
                                       )
                
        # print(intersection_list)
        # here check if identified root corresponds to identified mono
        max_int_idx = np.argmax(intersection_list)
        
        box = monos_list[max_int_idx]
        t_area = box.area()
        d_area = root_mono.area()
        
        inter = intersection_list[max_int_idx]
        
        if ((inter == t_area 
                and comparison_alg.training_contained(box, root_mono))
            or (inter == d_area
                and comparison_alg.detection_sufficient(box, root_mono))
            or comparison_alg.is_overlapping(box, root_mono)):
            # assign root to the appropriate monosaccharide
            # print("\n monos_list",monos_list[max_int_idx],max_int_idx,monos_list)
            pred_root_id = monos_id[max_int_idx]
            # pred_box = mono_boxes[max_int_idx]
            # print("pred_root_id",pred_root_id,root_mono.to_new_list(),pred_box.to_new_list())
            # monos_list[max_int_idx].set_root()
            obj['root'] = pred_root_id
            # mono_info.set_monos(monos_list)
            return root_mono.get_confidence()
        else:
            return None
