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
from BKGlycanExtractor import LinksCompare, DebugMode, FilterTreeLinks, FilterRepeatedLinks, RemapLinkLabels, LabelMap, FilterLabels, FilterOverlaps
from .semantics import UndirectedLinkSemantics


class LinkFinder:

    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanconnections')


    def set_results(self, obj, accepted, rejected):
        obj.set_undirected_links(accepted,rejected)

    def log_error(self,obj,accepted,rejected):
        # check if num_monos - 1 == num_links
        monos_count = len(obj.monos())
        if monos_count - 1 != len(accepted):
            obj.add_glycan_error(f"Count of monos: {monos_count}, count of links: {len(accepted)}")

    
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
        
        # print(x1,y1,x2,y2,len(linked_monos),[(m.id(),m.symbol()) for m in linked_monos])
        
        mono_pairs = None

        if len(linked_monos) == 2:
            mono_pairs = [(linked_monos[0], linked_monos[1])]

        elif len(linked_monos) > 2:    # more than 2 monos present in the same detected box
            mono_pairs = self.closest_to_diag_corners(box,linked_monos)
            
        if mono_pairs:
            ulinks = [ self.make_link(mp,box) for mp in mono_pairs ]
            # print([ (mp[0].id(),mp[0].symbol(),mp[1].id(),mp[1].symbol()) for mp in mono_pairs ])
            if len(ulinks) == 1:
                return ulinks[0]
            return ulinks
        
        return None

    # each second chance box has more than two links in it. Ideally, decide, using 
    # all the cleanly identified links, which of the two to pick. Last resort, terminal 
    # residues (Fuc, Xyl, NeuAc, NeuGc)
    def second_chance_boxes_to_objects(self,scboxes,obj,obj_list):
        local_obj_list = list(obj_list) # avoid modifying obj_list
        print("number of second-chance boxes:",len(scboxes))
        resolved = set()
        anyresolved = True
        while anyresolved:
            print("start cycle elimination pass...")
            anyresolved = False
            for i,(box,alts) in enumerate(scboxes):
                if i in resolved:
                    continue
                keep = []
                for alt in alts:
                    if not alt.creates_cycle(local_obj_list):
                        keep.append(alt)
                if len(keep) == 1:
                    local_obj_list.append(keep[0])
                    resolved.add(i)
                    anyresolved = True
                    print("resolved second-chance box",i+1)
        
        print("number resolved:",len(resolved))

        terminal_symbols = ["Fuc", "Xyl", "NeuAc", "NeuGc"]

        anyresolved = True
        while anyresolved:
            print("start terminal residue degree test pass...")
            anyresolved = False
            for i,(box,alts) in enumerate(scboxes):
                if i in resolved:
                    continue
        
                degree = defaultdict(int)
                for link in obj_list:
                    m1,m2 = link.mono_ids()
                    degree[m1]+=1
                    degree[m2]+=1

                keep = []
                for alt in alts:
                    skip = False
                    for mi in alt.mono_ids():
                        if obj.mono(mi).symbol() in terminal_symbols and degree[mi] > 0:
                            skip = True
                            break
                    if not skip:
                        keep.append(alt)
                if len(keep) == 1:
                    local_obj_list.append(keep[0])
                    resolved.add(i)
                    anyresolved = True
                    print("resolved second-chance box",i+1)

        return local_obj_list

    def closest_to_diag_corners(self, box, monos):
        meanw = sum(m.width() for m in monos)/len(monos)
        meanh = sum(m.height() for m in monos)/len(monos)
        mwh = (meanw+meanh)/2
        # print(mwh)
        
        x1, y1, x2, y2 = box.corners()
        anchors = dict()
        anchors["TL"] = (x1+mwh/2,y1+mwh/2)
        anchors["TR"] = (x2-mwh/2,y1+mwh/2)
        anchors["BL"] = (x1+mwh/2,y2-mwh/2)
        anchors["BR"] = (x2-mwh/2,y2-mwh/2)

        # each mono is assigned to its closest corner if <= mwh
        corners = defaultdict(list)
        for m in monos:
            dists = {}
            for k,v in anchors.items():
                dists[k] = self.dist(anchors[k],m)
            dists = sorted(dists.items(),key=lambda t: t[1])
            if dists[0][1] <= mwh:
                corners[dists[0][0]].append((dists[0][1],m))
        
        # each corner chooses its closest monosaccharide
        cms = defaultdict(lambda: None)
        for k in corners:
            if len(corners[k]) > 0:
                corners[k].sort(key=lambda t: t[0])
                cms[k] = corners[k][0]
        
        # diagonally opposite monosaccharides can't be too far from their anchors
        if cms["TL"] and cms["BR"] and (cms["TL"][0]+cms["BR"][0])/2>=mwh/2:
            cms["TL"] = None; cms["BR"] = None
        if cms["TR"] and cms["BL"] and (cms["TR"][0]+cms["BL"][0])/2>=mwh/2:
            cms["TR"] = None; cms["BL"] = None

        # print(cms)
        # for k in cms:
        #     if cms[k]:
        #         print(k,cms[k][0],cms[k][1].id(),cms[k][1].symbol())

        if cms["TL"] is not None and cms["BR"] is not None:
            if cms["TR"] is None or cms["BL"] is None:
                # only one pair is good
                return [(cms["TL"][1],cms["BR"][1])]
            else:
                # if one pair is significantly closer to the anchors than the other
                if (cms["TL"][0]+cms["BR"][0])*3 < (cms["TR"][0]+cms["BL"][0]):
                    return [(cms["TL"][1],cms["BR"][1])]
                elif (cms["TR"][0]+cms["BL"][0])*3 < (cms["TL"][0]+cms["BR"][0]):
                    return [(cms["TR"][1],cms["BL"][1])]
                else:
                    # can't decide which one, need more context...
                    # print("altenatives")
                    return [(cms["TL"][1],cms["BR"][1]),(cms["TR"][1],cms["BL"][1])]  
        elif cms["TR"] is not None and cms["BL"] is not None:
            # only one pair is good
            return [(cms["TR"][1],cms["BL"][1])]
        # no good solutions
        # print("No good solutions")
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

        # Note: map_dict data structure can store multiple glycans, but the current use-case is for SGI only
        links = map_dict['glycans'][0]['links']
        for (mono_id1, mono_id2), data in links.items():
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

class KnownLinkNoLink(LinkFinder,KnownFinder):

    def __init__(self,**kwargs):
        KnownFinder.__init__(self,**kwargs)
        LinkFinder.__init__(self)

    def create_boxes(self, map_dict):
        boxes = []

        # Note: map_dict data structure can store multiple glycans, but the current use-case is for SGI only
        linklabelid = self.get_label_index("link")
        monos = map_dict['glycans'][0]['monos']
        links = map_dict['glycans'][0]['links']
        for m1 in monos:
          for m2 in monos:
            if m1 >= m2:
              continue
            if (m1,m2) in links:
              classlabel = "link"
              mono_id1 = m1
              mono_id2 = m2
            elif (m2,m1) in links:
              classlabel = "link"
              mono_id1 = m2
              mono_id2 = m1
            else:
              classlabel = "nolink"
              mono_id1 = m1
              mono_id2 = m2
            box = BoundingBox(
                x1=min(monos[m1]['x_min'],monos[m2]['x_min']),
                y1=min(monos[m1]['y_min'],monos[m2]['y_min']),
                x2=max(monos[m1]['x_max'],monos[m2]['x_max']),
                y2=max(monos[m1]['y_max'],monos[m2]['y_max']),
                classlabel=classlabel,
                classid=self.get_label_index(classlabel),
                mono_id1=m1,
                mono_id2=m2,
            )
            boxes.append(box)

        toremove = []
        for b1 in boxes:
            for b2 in boxes:
                if b1 == b2:
                    continue
                if b1.contains(b2) and b1.get('classlabel') == "nolink":
                    toremove.append(b1)
                    break

        for b1 in toremove:
            boxes.remove(b1)
       
        return boxes

    def box_to_object(self, box, obj):
        if box.get('classlabel') == "link":
            return UndirectedLinkSemantics(box=box, **box.items())
        return None

class KnownLinkWithInfo(KnownLink):
    
    def create_boxes(self, map_dict):
        boxes = []

        # Note: map_dict data structure can store multiple glycans, but the current use-case is for SGI only
        links = map_dict['glycans'][0]['links']

        for (mono_id1, mono_id2), data in links.items():
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
        return link

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
    filters =  [ RemapLinkLabels() ] + ConnectYOLO.filters

class ConnectYOLOwNoLink(ConnectYOLO):

    filters = [ FilterOverlaps(maxiou=0.8), FilterLabels(keep=["link"]) ] + ConnectYOLO.filters