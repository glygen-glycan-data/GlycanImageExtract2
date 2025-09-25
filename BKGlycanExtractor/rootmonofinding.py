# -*- coding: utf-8 -*-
"""
class for various methods of identifying the root monosacharide
"""
import logging

import numpy as np
import math

from .bbox import BoundingBox
from .yolomodels import YOLOModel
from .glycanannotator import Config, Config_Manager, GlycanExtractorPipeline
from .finder import Finder,YOLOFinder,KnownFinder
from BKGlycanExtractor import RootCompare, DebugMode, RootFilter
from .semantics import RootSemantics
from .object_filters import FilterOverlaps, SingleBest, DiscardClass

            
class RootFinder:

    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.rootmonofinding')

    def set_results(self, obj, accepted, rejected):
        if len(accepted) > 0:
            obj.set_roots(accepted[0],rejected)
        else:
            obj.set_roots(None,rejected)

    def log_error(self,obj,accepted,rejected):
        # if no root is present, log it as an error
        if len(accepted) < 1:
            obj.add_glycan_error(f"Couldn't find a reducing end for the glycan")

    def finder_pipeline(self,config_manager):
        pipeline = GlycanExtractorPipeline()
        pipeline.set_steps('figure', config_manager.get_finders("SingleGlycanImage"))
        pipeline.set_steps('glycan', config_manager.get_finders("KnownMono")+[self])
        return pipeline

    def semantic_compare(self,**kwargs):
        return RootCompare(**kwargs)


class YOLORootFinder(YOLOFinder, RootFinder):

    filters = [ FilterOverlaps(maxiou=0.2,discard=True),
                DiscardClass(todiscard=["not_redend"]), 
                SingleBest() ]

    def __init__(self,**kwargs):
        YOLOFinder.__init__(self,**kwargs)
        RootFinder.__init__(self)

    def box_to_object(self,box,obj):
        '''
        checks if the detected root_box has any mono which is close enough to match with it
        '''
        monos = obj.monos()

        normalized_dist, selected_mono = self.match_root_to_mono(monos,box)

        if normalized_dist <= 0.5: 
            return RootSemantics(mono_id=selected_mono.get('id'), box=box, **box.items())
        return None

    def match_root_to_mono(self, monos, root_box):
        if monos == []:
            return 1e+20, None
        
        intersection_list = [0]*len(monos)

        for i, mono in enumerate(monos):
            if self.intersect(mono, root_box):
                intersection_list[i] = self.intersection_area(mono, root_box)
                
        max_int_idx = np.argmax(intersection_list)
        
        mono_box = monos[max_int_idx].box()
        # this is the monosaccharide which matched with the root
        selected_mono = monos[max_int_idx]
        euclidean_distance = self.dist(mono_box,root_box)
        
        x1,y1,w1,h1 = mono_box.bbox()
        x2,y2,w2,h2 = root_box.bbox()
        avg_width = (w1 + w2) / 2
        avg_height = (h1 + h2) / 2

        avg_object_size = math.sqrt((avg_width**2 + avg_height**2))  # diagonal

        # helps determine if the two detected boxes (monos and root) are close enough to be considered the same object
        normalized_dist = euclidean_distance / avg_object_size

        return normalized_dist, selected_mono

class KnownRoot(RootFinder,KnownFinder):

    filters = [ DiscardClass(todiscard=["not_redend"]) ]

    def __init__(self,**kwargs):
        KnownFinder.__init__(self,**kwargs)
        RootFinder.__init__(self)

    # map_dict structure is present in KnownFinder class
    def create_boxes(self, map_dict):
        boxes = []

        root_mono_id = map_dict['root']
        for id, data in map_dict['monos'].items():
            if id == root_mono_id:
                classlabel = "redend"
            else:
                classlabel = "not_redend"
            classid = self.get_label_index(classlabel)

            box = BoundingBox(x1=data['x_min'], y1=data['y_min'],
                x2=data['x_max'], y2=data['y_max'],
                classid=classid,
                classlabel=classlabel,
                mono_id=id
            )

            boxes.append(box)

        return boxes

    def box_to_object(self, box, obj):
        return RootSemantics(box=box, **box.items())
        

class YOLORootPlusAnomerFinder(YOLORootFinder):
    def box_to_object(self,box,obj):
        rootobj = super().box_to_object(box,obj)
        if rootobj is None:
            return rootobj
        classlabel = self.box_label(box)
        if classlabel == "redenda":
            rootobj.set("anomer","a")
        if classlabel == "redendb":
            rootobj.set("anomer","b")
        return rootobj

class KnownRootPlusAnomer(KnownRoot):
    
    def create_boxes(self, map_dict):
        boxes = []

        root_mono_id = map_dict['root']
        for id, data in map_dict['monos'].items():
            anomer = data.get('anomer','?')
            if anomer == "?":
                anomer = "x"
            if id == root_mono_id:
                classlabel = "redend"+anomer
            else:
                classlabel = "not_redend"
            classid = self.get_label_index(classlabel)

            box = BoundingBox(x1=data['x_min'], y1=data['y_min'],
                x2=data['x_max'], y2=data['y_max'],
                classid=classid,
                classlabel=classlabel,
                mono_id=id,
                anomer=anomer
            )
            boxes.append(box)

        return boxes

