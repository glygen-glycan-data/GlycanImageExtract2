# -*- coding: utf-8 -*-
"""
With monosaccharides already located and classed, connect them.
Returns undirected links.

all subclasses need a connect method
which takes a GlycanMonoInfo object as defined in monosaccharideid.py
and returns an  list of connected Monosaccharide objects
"""

import collections
import cv2
import logging
import math
import numpy as np
from collections import defaultdict
from .yolomodels import YOLOModel
from .glycanannotator import Config, Config_Manager, GlycanExtractorPipeline
from .bbox import BoundingBox
from .finder import Finder,YOLOFinder, KnownFinder
from BKGlycanExtractor import LinksCompare, DebugMode, FilterTreeLinks, FilterRepeatedLinks, RemapLinkLabels
from .semantics import UndirectedLinkSemantics


class LinkFinder:

    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanconnections')

    def set_results(self, obj, accepted, rejected):
        obj.set_undirected_links(accepted,rejected)
    
    def finder_pipeline(self,config_manager):
        pipeline = GlycanExtractorPipeline()
        pipeline.set_steps('figure', config_manager.get_finders("SingleGlycanImage"))
        pipeline.set_steps('glycan', config_manager.get_finders("KnownMono")+[self])
        return pipeline

    def semantic_compare(self,**kwargs):
        return LinksCompare(**kwargs)

class ConnectYOLO(YOLOFinder,LinkFinder):
    filters = [ FilterRepeatedLinks(), 
                FilterTreeLinks() ]

    def __init__(self,**kwargs):
        YOLOFinder.__init__(self,**kwargs)
        LinkFinder.__init__(self)
    
    def box_to_object(self, box, obj):
        ''' Process a single detected box: find the linked monos in the box, 
        return link info if valid. 
        
        Note: Pairs might repeat themselves.
        Idea here is to return all the pairs as it is. 
        The repeated pairs need to be filtered out - this will be done in the object filtering step
        '''

        
        linked_monos = []
        x1, y1, x2, y2 = box.corners()

        for mono in obj.monos(): 
            x_cen, y_cen = mono.get('center')

            if x_cen > x1 and x_cen < x2 and y_cen > y1 and y_cen < y2:
                linked_monos.append(mono)

        mono_pair = None
        if len(linked_monos) == 2:
            mono_pair = (linked_monos[0], linked_monos[1])
        elif 2 < len(linked_monos) <= 4:    # more than 2 monos present in the same detected box
            farthest_pair = self.find_farthest_pair(linked_monos)
            if not farthest_pair:
                return None
            mono_pair = farthest_pair

        if mono_pair:
            return self.make_link(mono_pair,box)

        return None

    def find_farthest_pair(self, monos):
        ''' Find farthest pair. Ignore Fuc '''
        max_distance = 0
        farthest_pair = None

        for i in range(len(monos)):
            for j in range(i + 1, len(monos)):
                if monos[i].symbol() != 'Fuc' and monos[j].symbol() != 'Fuc':
                    dist = self.dist(monos[i], monos[j])
                    if dist > max_distance:
                        max_distance = dist
                        farthest_pair = (monos[i], monos[j])

        return farthest_pair


    def make_link(self, mono_pair, box):
        link = UndirectedLinkSemantics(mono_id1=mono_pair[0].id(), mono_id2=mono_pair[1].id(), box=box, **box.items())

        if len(link.classlabel()) == 2:
            cl = link.classlabel()
            if cl[0] != "x":
                link.set('anomer',cl[0])
            if cl[1] != "x":
                link.set('parent_bond',int(cl[1]))
        return link

    

class KnownLink(LinkFinder,KnownFinder):

    def __init__(self,**kwargs):
        KnownFinder.__init__(self,**kwargs)
        LinkFinder.__init__(self)

    def create_boxes(self, map_dict):
        boxes = []

        for (mono_id1, mono_id2), data in map_dict['links'].items():

            classlabel = "link"
            box = BoundingBox(x1=data['x_min'], y1=data['y_min'], 
                x2=data['x_max'], y2=data['y_max'], 
                classlabel=classlabel,
                classid=self.get_label_index(classlabel),
                mono_id1=mono_id1,
                mono_id2=mono_id2,
            )

            boxes.append(box)

        return boxes

    def box_to_object(self, box, obj):
        return UndirectedLinkSemantics(box=box, **box.items())

class KnownLinkWithInfo(KnownLink):
    
    def create_boxes(self, map_dict):
        boxes = []

        for (mono_id1, mono_id2), data in map_dict['links'].items():

            classlabel = f"{map_dict['monos'][mono_id2]['anomer']}{data['carbon_number']}"
            classlabel = classlabel.replace("?","x")
            classid = self.get_label_index(classlabel)

            box = BoundingBox(x1=data['x_min'], y1=data['y_min'], 
                x2=data['x_max'], y2=data['y_max'], 
                classlabel=classlabel,
                classid=classid,
                mono_id1=mono_id1,
                mono_id2=mono_id2,
                carbon_number=data['carbon_number'],
                anomer=map_dict['monos'][mono_id2]['anomer']
            )
            boxes.append(box)

        return boxes

    def box_to_object(self, box, obj):
        link = UndirectedLinkSemantics(box=box, **box.items())

class ConnectYOLOInfo(ConnectYOLO):

    def box_to_object(self, box, obj):
        link = super().box_to_object(box,obj)
        classlabel = box.get('classlabel')
        if len(classlabel) == 2:
            if classlabel[0] in ('a','b'):
                link.set('anomer',classlabel[0])
            if classlabel[1] not in ('?', 'x'):
                link.set('parent_bond',int(classlabel[1]))
        return link


class ConnectYOLOInfoLabel(ConnectYOLO):
    '''
    class to mask all the different labels (eg. ax, bx, etc) to 'link'.
    '''
    filters = ConnectYOLO.filters + [RemapLinkLabels()]