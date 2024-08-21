#from BKGlycanExtractor.boundingboxes import BoundingBox

class CompareBoxes:
    def __init__(self,**kwargs):
        self.detection_threshold = kwargs.get("detection_threshold", 0.5)
        self.overlap_threshold = kwargs.get("overlap_threshold", 0.5)
        self.containment_threshold = kwargs.get("containment_threshold",0.5)
    def compare_class(self,training,detected):
        if detected.class_ == training.class_:
            return True
        else:
            return False
    def detection_sufficient(self, training, detected):
        if self.iou(training, detected) > self.detection_threshold:
            return True
        else:
            return False
    def have_intersection(self,training,detected):
        assert training.x <= training.x2
        #print(detected.x,detected.x2)
        assert detected.x <= detected.x2
        assert training.y <= training.y2
        assert detected.y <= detected.y2
        
        if detected.x > training.x2:
            return False
        if detected.x2 < training.x:
            return False
        if detected.y > training.y2:
            return False
        if detected.y2 < training.y:
            return False
        return True
    def intersection_area(self, training, detected):
        xA = max(training.x, detected.x)
        yA = max(training.y, detected.y)
        xB = min(training.x2, detected.x2)
        yB = min(training.y2, detected.y2)
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
        #print(self.iou(training,detected), self.containment_threshold)
        if (self.iou(training, detected) > self.containment_threshold):
            #print("hello")
            return True
        else:
            return False
    def union_area(self, training, detected):
        d_area = detected.area()
        t_area = training.area()
        intersection = self.intersection_area(training, detected)
        return float(d_area + t_area - intersection)
        
class ComparePaddedBox(CompareBoxes):
    def __init__(self,**kwargs):
        detection_threshold = kwargs.get("detection_threshold", 0.95)
        overlap_threshold = kwargs.get("overlap_threshold", 0.95)
        containment_threshold = kwargs.get("containment_threshold",0.25)
        super().__init__(detection_threshold = detection_threshold,
                         overlap_threshold = overlap_threshold, 
                         containment_threshold = containment_threshold)
        
class CompareRawBox(CompareBoxes):
    def __init__(self,**kwargs):
        detection_threshold = kwargs.get("detection_threshold", 0.95)
        overlap_threshold = kwargs.get("overlap_threshold", 0.7)
        containment_threshold = kwargs.get("containment_threshold",0.5)
        super().__init__(detection_threshold = detection_threshold, 
                         overlap_threshold = overlap_threshold, 
                         containment_threshold = containment_threshold)
        
class CompareDetectedClass(CompareBoxes):
    def __init__(self,**kwargs):
       detection_threshold = kwargs.get("detection_threshold", 0.5)
       overlap_threshold = kwargs.get("overlap_threshold", 0.5)
       containment_threshold = kwargs.get("containment_threshold",0.5)
       super().__init__(detection_threshold = detection_threshold, 
                        overlap_threshold = overlap_threshold, 
                        containment_threshold = containment_threshold)
