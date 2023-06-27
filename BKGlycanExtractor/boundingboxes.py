"""
classes to describe YOLO bounding boxes:
training box descriptions, boxes returned by YOLO model detection,
and pseudo-bounding box objects which are created by the program
starting reference: 
https://github.com/rafaelpadilla/review_object_detection_metrics/blob/56d8969739d4774b4bab5b8122870e7e4c833021/src/bounding_box.py
"""
import cv2

class BoundingBox: 
    # initialize box. requires an image
    # other arguments are optional 
    # but it should be initialised with some appropriate set of:
    # x/y/w/h or x/x2/y/y2
    def __init__(self, **kw):
        bounding_box = kw.pop("boundingbox", None)
        if bounding_box is not None:
            bounding_box.copy_to(self)
        else:
            image = kw["image"]
            height, width, channels = image.shape
            self.imwidth = width
            self.imheight = height
            self.rel_cen_x = kw.get("rel_cen_x", None)
            self.rel_cen_y = kw.get("rel_cen_y", None)
            self.rel_w = kw.get("rel_w", None)
            self.rel_h = kw.get("rel_h", None)
            self.cen_x = kw.get("cen_x", None)
            self.cen_y = kw.get("cen_y", None)
            self.w = kw.get("width", None)
            self.h = kw.get("height", None)
            self.x = kw.get("x", None)
            self.y = kw.get("y", None)
            self.x2 = kw.get("x2", None)
            self.y2 = kw.get("y2", None)
            self.class_ = kw.get("class_", 0)
            self.class_name = kw.get("class_name", '')
            self.whitespace = kw.get("white_space", 0)
        
    # convert an absolute bounding box into a relative bounding box
    def abs_to_rel(self):
        assert self.cen_x is not None
        assert self.w is not None
        self.rel_cen_x = float(self.cen_x/self.imwidth)
        self.rel_cen_y = float(self.cen_y/self.imheight)
        self.rel_w = float(self.w/self.imwidth)
        self.rel_h = float(self.h/self.imheight) 
        
    def annotate(self, image, defaulttext=''):
        self.to_four_corners()
        p1 = (self.x, self.y)
        p2 = (self.x2, self.y2)
        cv2.rectangle(image, p1, p2, (0,255,0), 3)
        if self.class_dictionary is not None:
            text = self.class_dictionary[self.get_class()]
        else:
            text = defaulttext
        cv2.putText(image, text, p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
        return image

    def area(self):
        assert self.w is not None
        return (self.w+1) * (self.h+1)
    
    def center_to_corner(self):
        assert self.cen_x is not None
        assert self.w is not None
        self.x = int(self.cen_x - self.w/2)
        self.y = int(self.cen_y - self.h/2)
        
    def copy_to(self, box2):
        for attr, value in vars(self).items():
            setattr(box2, attr, value)
    
    def corner_to_center(self):
        assert self.x is not None
        assert self.w is not None
        self.cen_x = int(self.x + self.w/2)
        self.cen_y = int(self.y + self.h/2)
        
    # adjust image for whitespace padding
    def fix_image(self):
        assert self.cen_x is not None
        assert self.w is not None
        half_white_space = int(self.whitespace/2)
        self.cen_x = self.cen_x - half_white_space
        self.cen_y = self.cen_y - half_white_space
        self.rel_cen_x = float(self.cen_x/self.imwidth)
        self.rel_cen_y = float(self.cen_y/self.imheight)
        self.rel_w = float(self.w/self.imwidth)
        self.rel_h = float(self.h/self.imheight)
        
    def get_center_point(self):
        return self.cen_x, self.cen_y
    
    def get_class(self):
        return self.class_
    
    def get_confidence(self):
        return self.confidence
        
    # convert relative bounding box to absolute bounding box
    def rel_to_abs(self):
        assert self.rel_cen_x is not None
        assert self.rel_w is not None
        self.cen_x = int(self.rel_cen_x * (self.imwidth+self.whitespace))
        self.cen_y = int(self.rel_cen_y * (self.imheight+self.whitespace))
        self.w = int(self.rel_w * (self.imwidth+self.whitespace))
        self.h = int(self.rel_h * (self.imheight+self.whitespace))
    
    def set_class(self, classnum):
        self.class_ = classnum
    
    def set_name(self, name):
        self.name = name
    
    def to_four_corners(self):
        assert self.x is not None
        assert self.w is not None
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h
        
    def to_image_coords(self):
        p1 = (self.x, self.y)
        p2 = (self.x2, self.y2)
        return p1, p2
        
    # convert bounding box to a list
    # types are detected (with confidence) and training (with class)
    # takes an optional list_type argument, 
    # or relies on the type of the bounding box
    def to_list(self, list_type=None):
        if list_type is None:
            list_type = self.type
        if list_type == "detected":
            [x,y,w,h,confidence] = [
                self.x, self.y, self.w, self.h, self.confidence
                ]
            return [x, y, w, h, confidence]
        elif list_type == "training":
            [class_,rel_cen_x,rel_cen_y,rel_w,rel_h] = [
                self.class_, self.rel_cen_x, 
                self.rel_cen_y, self.rel_w, self.rel_h
                ]
            return [class_, rel_cen_x, rel_cen_y, rel_w, rel_h]
    
    # compatible with new openCV versions
    def to_new_list(self, list_type=None):
        if list_type is None:
            list_type = self.type
        if list_type == "detected":
            [x,y,w,h] = [self.x, self.y, self.w, self.h]
            return [x,y,w,h]
        elif list_type == "training":
            [class_,rel_cen_x,rel_cen_y,rel_w,rel_h] = [
                self.class_, self.rel_cen_x, 
                self.rel_cen_y, self.rel_w, self.rel_h
                ]
            return [class_, rel_cen_x, rel_cen_y, rel_w, rel_h]
        
    # create coordinates for pdf placement
    def to_pdf_coords(self):
        assert self.x is not None
        assert self.x2 is not None
        self.x0 = self.x/self.imwidth
        self.y0 = self.y/self.imheight
        self.x1 = self.x2/self.imwidth
        self.y1 = self.y2/self.imheight
        return [self.x0, self.y0, self.x1, self.y1]
        
    #string format, to be printed as in a file
    def __str__(self):
        printdict = {
            "x0": str(self.x), "x2": str(self.x2), 
            "y0": str(self.y), "y2": str(self.y2)
            }
        printstr = "[ "
        for key, value in printdict.items():
            printstr += key + ": "
            printstr += value + "    "
        printstr += "]"
        return printstr
        
    
# subclass intended for bounding boxes detected by YOLO models
# can be used to manually create a bounding box 
# with the qualities expected of a detected object bounding box        
class Detected(BoundingBox):
    # must be initialised with image and a confidence score for the box
    def __init__(self, confidence, **kw):
        super().__init__(**kw)
        self.confidence = confidence
        self.type = "detected"
        self.class_options = kw.get(
            "class_options", {str(self.class_), self.confidence}
            )
    # check if the bounding box is too large
    def is_entire_image(self):
        assert self.w is not None
        if self.w*self.h > 0.8*0.8*self.imwidth*self.imheight:
            self.x = 0
            self.y = 0
            self.w = self.imwidth
            self.h = self.imheight
        else:
            pass 
        
    # fix borders 
    # so box cannot be outside image boundaries 
    # once whitespace is removed
    def fix_borders(self):
        assert self.x is not None
        assert self.w is not None
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x+self.w >= self.imwidth:
            self.w = int(self.imwidth-self.x)
        if self.y+self.h >= self.imheight:
            self.h = int(self.imheight-self.y)
            
    # pad borders by 20% to protect from cropping issues
    def pad_borders(self):
        assert self.x is not None
        assert self.w is not None
        self.x = self.x - int(0.2*self.w)
        self.y = self.y - int(0.2*self.h)
        self.w = int(1.4*self.w)
        self.h = int(1.4*self.h) 
        
    # this is the same as calling self.to_list(list_type="training") 
    # new programs could use either
    def to_relative_list(self):
        [classid,relcenx,relceny,relw,relh] = [
            self.class_, self.rel_cen_x, 
            self.rel_cen_y, self.rel_w, self.rel_h
            ]
        return [classid, relcenx, relceny, relw, relh]


# class for bounding boxes from training data
# or self-created boxes that want to be used as training data        
class Training(BoundingBox):
    #must be initialised with the image
    def __init__(self, **kw):
        super().__init__(**kw)
        self.type = "training"
        
    #fix borders to prevent boxes outside image boundary
    def fix_borders(self):
        assert self.x is not None
        assert self.w is not None
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x+self.w >= self.imwidth:
            self.w = int(self.imwidth-self.x)
        if self.y+self.h >= self.imheight:
            self.h = int(self.imheight-self.y)
            
    #set bounding box to contain the entire image
    def is_entire_image(self):
        self.rel_cen_x = 0.5
        self.rel_cen_y = 0.5
        self.rel_w = 1
        self.rel_h = 1
        
    #pad borders by a factor of 10%
    def pad_borders(self):
        assert self.x is not None
        assert self.w is not None
        self.x = self.x - int(0.1*self.w)
        self.y = self.y - int(0.1*self.h)
        self.w = int(1.2*self.w)
        self.h = int(1.2*self.h)
        
    #add whitespace to image
    def reset_image(self,white_space):
        assert self.cen_x is not None
        self.white_space = white_space
        half_white_space = int(white_space/2)
        self.cen_x = self.cen_x + half_white_space
        self.cen_y = self.cen_y + half_white_space
        self.imwidth = self.imwidth + self.white_space
        self.imheight = self.imheight + self.white_space
        
    def set_dummy_confidence(self,confidence):
        self.confidence = confidence
