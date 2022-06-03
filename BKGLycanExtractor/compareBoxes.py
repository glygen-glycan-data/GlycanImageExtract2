#from BKGLycanExtractor.boundingboxes import BoundingBox

class CompareBoxes:
    def __init__(self,dbox,tbox):
        self.dbox = dbox
        self.tbox = tbox
    def have_intersection(self):
        training = self.tbox
        detected = self.dbox
        assert training.x < training.x2
        assert detected.x < detected.x2
        assert training.y < training.y2
        assert detected.y < detected.y2
        
        if detected.x > training.x2:
            return False
        if detected.x2 < training.x:
            return False
        if detected.y > training.y2:
            return False
        if detected.y2 < training.y:
            return False
        return True
    def intersection_area(self):
        training = self.tbox
        detected = self.dbox
        xA = max(training.x, detected.x)
        yA = max(training.y, detected.y)
        xB = min(training.x2, detected.x2)
        yB = min(training.y2, detected.y2)
        return (xB - xA + 1)*(yB - yA + 1)
    def iou(self):
        i = self.intersection_area()
        u = self.union_area()
        return float(i/u)
    def union_area(self):
        d_area = self.dbox.area()
        t_area = self.tbox.area()
        intersection = self.intersection_area()
        return float(d_area + t_area - intersection)
        
class ComparePaddedBox(CompareBoxes):
    def detection_sufficient(self):
        if self.iou() > 0.75:
            return True
        else:
            return False
    def is_overlapping(self):
        if self.iou() > 0.5:
            return True
        else:
            return False
    def training_contained(self):
        if self.iou() > 0.25:
            return True
        else:
            return False
        
class CompareRawBox(CompareBoxes):
    def detection_sufficient(self):
        if self.iou() > 0.75:
            return True
        else:
            return False
    def is_overlapping(self):
        if self.iou() > 0.5:
            return True
        else:
            return False
    def training_contained(self):
        if self.iou() > 0.5:
            return True
        else:
            return False