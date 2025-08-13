# -*- coding: utf-8 -*-
"""
class for various methods of identifying the root monosacharide
"""
import logging

import numpy as np
import math

from .bbox import BoundingBox
from .yolomodels import YOLOModel
from .glycanannotator import Config, Config_Manager
from .finder import Finder,YOLOFinder,KnownFinder
from BKGlycanExtractor import RootCompare, DebugMode, RootFilter
from .semantics import RootSemantics

            
class RootFinder:

    labels = ['redend','not_redend']
    finder_class = 'Root'

    semantic_compare = RootCompare
    
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.rootmonofinding')

    def set_results(self, obj, accepted, rejected):
        if len(accepted) > 0:
            obj.set_roots(accepted[0],rejected)
        else:
            obj.set_roots(None,rejected)
        

class YOLORootFinder(YOLOFinder, RootFinder):
    # filters = [RootFilter()]

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
        RootFinder.__init__(self)


    def box_to_object(self,box,obj):
        '''
        checks if the detected root_box has any mono which is close enough to match with it
        '''
        monos = obj.monos()

        normalized_dist, selected_mono = self.match_root_to_mono(monos,box)

        if normalized_dist <= 0.5: 
            classlabel = box.get('classlabel')
            return RootSemantics(mono_id=selected_mono.get('id'), box=box, **box.items())
        return None

    def match_root_to_mono(self, monos, root_box):
        if monos == []:
            return None
        
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

    labels = ['redend','not_redend']

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

        RootFinder.__init__(self)
        self.set_labels(self.labels)

    # map_dict structure is present in KnownFinder class
    def create_boxes(self, map_dict):
        root_mono_id = map_dict['root']
        mono_details = map_dict['monos'][root_mono_id]
        
        box = BoundingBox(x1=mono_details['x_min'], y1=mono_details['y_min'], 
                x2=mono_details['x_max'], y2=mono_details['y_max'], 
                mono_id=root_mono_id,
                classlabel=self.get_label(0)
            )

        box.pad(self.params['boxpadding']) # known data is absolute
        return [box]


    def box_to_object(self, box, obj):
        return RootSemantics(box=box, **box.items())
        

