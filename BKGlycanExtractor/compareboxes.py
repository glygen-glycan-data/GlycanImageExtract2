# -*- coding: utf-8 -*-
"""
methods to compare 2 bounding boxes
assess ovelap, intersection over union value, class comparison, etc
"""

class CompareBoxes:
    def __init__(self, **kw):
        self.detection_threshold = kw.get("detection_threshold", 0.5)
        self.overlap_threshold = kw.get("overlap_threshold", 0.5)
        self.containment_threshold = kw.get("containment_threshold", 0.5)
        
    def compare_class(self, training, detected):
        if detected.get_class() == training.get_class():
            return True
        else:
            return False
        
    def detection_sufficient(self, training, detected):
        if self.iou(training, detected) > self.detection_threshold:
            return True
        else:
            return False
        
    def have_intersection(self, training, detected):
        (t_x, t_y), (t_x2, t_y2) = training.to_image_coords()
        (d_x, d_y), (d_x2, d_y2) = detected.to_image_coords()
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
    
    def intersection_area(self, training, detected):
        (t_x, t_y), (t_x2, t_y2) = training.to_image_coords()
        (d_x, d_y), (d_x2, d_y2) = detected.to_image_coords()
        xA = max(t_x, d_x)
        yA = max(t_y, d_y)
        xB = min(t_x2, d_x2)
        yB = min(t_y2, d_y2)
        return (xB - xA + 1)*(yB - yA + 1)
    
    def iou(self, training, detected):
        i = self.intersection_area(training, detected)
        u = self.union_area(training, detected)
        return float(i/u)
    
    def is_overlapping(self, training, detected):
        if self.iou(training, detected) > self.overlap_threshold:
            return True
        else:
            return False
        
    def training_contained(self, training, detected):
        if (self.iou(training, detected) > self.containment_threshold):
            return True
        else:
            return False
        
    def union_area(self, training, detected):
        d_area = detected.area()
        t_area = training.area()
        intersection = self.intersection_area(training, detected)
        return float(d_area + t_area - intersection)
        
class ComparePaddedBox(CompareBoxes):
    def __init__(self, **kw):
        detection_threshold = kw.get("detection_threshold", 0.95)
        overlap_threshold = kw.get("overlap_threshold", 0.95)
        containment_threshold = kw.get("containment_threshold",0.25)
        super().__init__(
            detection_threshold=detection_threshold,
            overlap_threshold=overlap_threshold, 
            containment_threshold=containment_threshold
            )
        
class CompareRawBox(CompareBoxes):
    def __init__(self, **kw):
        detection_threshold = kw.get("detection_threshold", 0.95)
        overlap_threshold = kw.get("overlap_threshold", 0.95)
        containment_threshold = kw.get("containment_threshold",0.5)
        super().__init__(
            detection_threshold=detection_threshold, 
            overlap_threshold=overlap_threshold, 
            containment_threshold=containment_threshold
            )
        
class CompareDetectedClass(CompareBoxes):
    def __init__(self,**kw):
       detection_threshold = kw.get("detection_threshold", 0.5)
       overlap_threshold = kw.get("overlap_threshold", 0.5)
       containment_threshold = kw.get("containment_threshold",0.5)
       super().__init__(
           detection_threshold=detection_threshold, 
           overlap_threshold=overlap_threshold, 
           containment_threshold=containment_threshold)
