# -*- coding: utf-8 -*-
"""
methods to compare bounding boxes
assess ovelap, intersection over union value, class comparison, etc
"""

import math

from . bbox import BoundingBox

class CompareBoxes:

    def __init__(self, **kw):
        self.detection_threshold = kw.get("detection_threshold", 0.5)
        self.overlap_threshold = kw.get("overlap_threshold", 0.5)
        self.containment_threshold = kw.get("containment_threshold", 0.5)
        
    def compare_class(self, known, detected):
        if known.get('classid',-1) == detected.get('classid',-2):
            return True
        return False
        
    def detection_sufficient(self, training, detected):
        if CompareBoxes.iou(training, detected) > self.detection_threshold:
            return True
        else:
            return False

    @staticmethod                                                                                                            
    def euclidean_distance_points(p1,p2):                                                                                    
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)                                                            
                                                                                                                             
    @staticmethod                                                                                                            
    def euclidean_distance(x,y):                                                                                             
        if isinstance(x,BoundingBox):                                                                                        
            x = x.center()                                                                                                  
        if isinstance(y,BoundingBox):                                                                                        
            y = y.center()                                                                                                  
        return CompareBoxes.euclidean_distance_points(x,y)

    @staticmethod
    def proximity(known_box,pred_box):
        distance = CompareBoxes.euclidean_distance(known_box, pred_box)
        x,y,w,h = known_box.bbox()
        return distance/min(w, h)

    @staticmethod    
    def have_intersection(training, detected):
        t_x, t_y, t_x2, t_y2 = training.corners()
        d_x, d_y, d_x2, d_y2 = detected.corners()
        assert t_x <= t_x2
        assert d_x <= d_x2
        assert t_y <= t_y2
        assert d_y <= d_y2
        
        if d_x > t_x2:
            return False
        if d_x2 < t_x:
            return False
        if d_y > t_y2:
            return False
        if d_y2 < t_y:
            return False
        return True   

    @staticmethod
    def intersection_area(training, detected):
        t_x, t_y, t_x2, t_y2 = training.corners()
        d_x, d_y, d_x2, d_y2 = detected.corners()
        xA = max(t_x, d_x)
        yA = max(t_y, d_y)
        xB = min(t_x2, d_x2)
        yB = min(t_y2, d_y2)
        return (xB - xA + 1)*(yB - yA + 1)
    
    @staticmethod
    def iou(training, detected):
        if CompareBoxes.have_intersection(training, detected):
            i = CompareBoxes.intersection_area(training, detected)
            u = CompareBoxes.union_area(training, detected)
            iou = float(i/u)
            # assert float('-inf') <= iou <= 1
            return iou
        return 0.0
    
    def is_overlapping(self, training, detected):
        if CompareBoxes.iou(training, detected) > self.overlap_threshold:
            return True
        else:
            return False
        
    def match_to_training(self, tboxes, dboxes):
        
        compare_dict = {}
        for idx, tbox in enumerate(tboxes):
            name = str(idx)
            tbox.set_name(name)
            
            intersecting_boxes = []
            for d_idx, dbox in enumerate(dboxes):
                dbox_name = str(d_idx)
                dbox.set_name(dbox_name)
                dbox.corners()
                if CompareBoxes.have_intersection(tbox, dbox):
                    iou = CompareBoxes.iou(tbox, dbox)
                    intersecting_boxes.append((iou, dbox))
            
            # sort by first element (iou)
            intersecting_boxes.sort(key=lambda x: (x[0], x[1].get_confidence()), reverse=True)
            
            matched_boxes = []
            iou_conf = 0
            for iou, dbox in intersecting_boxes:
                if not self.compare_class(tbox, dbox):
                    continue
                else:
                    t_area = tbox.area()
                    d_area = dbox.area()
                    inter = CompareBoxes.intersection_area(tbox, dbox)
                    if inter == 0:
                        break
                    elif inter == t_area:
                        if not self.training_contained(tbox, dbox):
                            continue
                    elif inter == d_area:
                        if not self.detection_sufficient(tbox, dbox):
                            continue
                    else:
                        if not self.is_overlapping(tbox, dbox):
                            continue
                # in descending order of iou
                conf = dbox.get_confidence()
                if conf > iou_conf: 
                    iou_conf = conf
                    matched_boxes.append(dbox)
                    
            compare_dict[name] = (tbox, matched_boxes)
            
        
        # compare_dict: { tboxname : (tbox, [matched list])}
        # matched list: [dbox_1, ..., dbox_n]
        # where iou_1 > iou_2 > ... > iou_n
        # and conf_1 < conf_2 < ... < conf_n
            
        return compare_dict
                    
        
    def training_contained(self, training, detected):
        if (CompareBoxes.iou(training, detected) > self.containment_threshold):
            return True
        else:
            return False

    @staticmethod   
    def union_area(training, detected):
        d_area = detected.area()
        t_area = training.area()
        intersection = CompareBoxes.intersection_area(training, detected)
        return float(d_area + t_area - intersection)
        
