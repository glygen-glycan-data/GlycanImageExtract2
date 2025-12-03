# -*- coding: utf-8 -*-
"""
methods to compare bounding boxes
assess ovelap, intersection over union value, class comparison, etc
"""

import math

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
        try:
            x = x.center()
        except AttributeError:
            pass
        try:
            y = y.center()
        except AttributeError:
            pass                                                                                                                                                                                        
        return CompareBoxes.euclidean_distance_points(x,y)

    @staticmethod
    def proximity(known_box,pred_box):
        distance = CompareBoxes.euclidean_distance(known_box, pred_box)
        x,y,w,h = known_box.bbox()
        return distance/min(w, h)

    @staticmethod
    def is_contained_in(b1,b2):   
        # return true only if b1 in contained inside b2                                                                        
        b1x1,b1y1,b1x2,b1y2 = b1.corners()                                                                
        b2x1,b2y1,b2x2,b2y2 = b2.corners()                                                                
        if b1x1 < b2x1:                                                                                   
            return False                                                                                  
        if b1x2 > b2x2:                                                                                   
            return False                                                                                  
        if b1y1 < b2y1:                                                                                   
            return False                                                                                  
        if b1y2 > b2y2:                                                                                   
            return False                                                                                  
        assert CompareBoxes.intersection_area(b1,b2) == b1.area()                                                      
        return True

    @staticmethod
    def get_containment(b1,b2):
        # returns (contained box, container box)
        if CompareBoxes.is_contained_in(b1,b2):
            return (b1,b2)
        elif CompareBoxes.is_contained_in(b2,b1):
            return (b2,b1)
        return None


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

    @staticmethod
    def union_boxes(bbox1, bbox2):
        x1,y1,w1,h1 = bbox1.bbox()
        x2,y2,w2,h2 = bbox2.bbox()

        # finding the bbox which contains both the boxes i.e merging both boxes into one box
        x_min = min(x1,x2)
        y_min = min(y1,y2)
        x_max = max(x1+w1-1, x2+w2-1)
        y_max = max(y1+h1-1, y2+h2-1)
        
        return (x_min, y_min, x_max-x_min, y_max-y_min)     # x,y,w,h 

    @staticmethod
    def union_pdf_boxes(bbox1, bbox2):
        # IMP - this method is only meant to merge pdf boxes
        x1, y1, x2, y2 = bbox1.bbox()
        _x1, _y1, _x2, _y2 = bbox2.bbox()
        
        # finding the bbox which contains both the boxes i.e merging both boxes into one box
        x_min = min(x1, _x1)  # leftmost x
        y_min = min(y1, _y1)  # bottommost y (or topmost depending on coordinate system)
        x_max = max(x2, _x2)  # rightmost x
        y_max = max(y2, _y2)  # topmost y (or bottommost depending on coordinate system)
        
        return [x_min, y_min, x_max, y_max]  # x1, y1, x2, y2

        
    
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
        
