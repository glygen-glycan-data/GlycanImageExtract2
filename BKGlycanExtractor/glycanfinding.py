# -*- coding: utf-8 -*-
"""
class for glycan locating methods.

all subclasses need a find_objects method
which takes an image and returns semantic data for all glycans in the image.

any classes for detection need to return Detected bounding boxes, 
with confidence values.

bounding boxes are laid out in bbox.py.
they require the image the glycan was found in, 
some set of coordinates, and confidence of detection.
"""

import logging
import os
import json
import cv2
import numpy as np
from . bbox import BoundingBox
from . yolomodels import YOLOModel 
from . glycanannotator import Config
from . finder import Finder
from BKGlycanExtractor import DebugMode

# Base class
class GlycanFinder(Finder):  

    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanfinding')


# YOLO based glycan finder
# allows minimum confidence thresholding, to restrict returns
# also allows requesting padding of glycan borders (off by default)
# confidences for YOLO detection are stored in the bounding box
class YOLOGlycanFinder(YOLOModel,GlycanFinder):

    defaults = {
        'conf_threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 200,
        'iou_threshold': 0.5
    }
    labels = [ 'glycan' ]

    def __init__(self,**kwargs):
        params = dict(
           boxpadding = Config.get_param('boxpadding', Config.FLOAT, kwargs, self.defaults),
           expandimage = Config.get_param('expandimage', Config.FLOAT, kwargs, self.defaults),
           conf_threshold = Config.get_param('conf_threshold', Config.FLOAT, kwargs, self.defaults),
           iou_threshold = Config.get_param('iou_threshold', Config.FLOAT, kwargs, self.defaults),
           config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
           weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
        )
        YOLOModel.__init__(self,params)
        GlycanFinder.__init__(self)

    def find_boxes(self, obj):
        image = obj.image()
        return self.get_YOLO_output(image)

    def find_objects(self, obj):
        boxes = self.find_boxes(obj)
        obj.clear_glycans()
        for box in boxes:
            obj.add_glycan(box=box)
    
    
class SingleGlycanImage(GlycanFinder):

    defaults = {
        'crop': False,
        'padding': 0
    }

    def __init__(self,**kwargs):
       self.crop = Config.get_param('crop', Config.BOOL, kwargs, self.defaults)
       self.padding = Config.get_param('padding', Config.FLOAT, kwargs, self.defaults)
       super().__init__()

    def find_objects(self, obj):
        obj.clear_glycans()
        boxes = self.find_boxes(obj)
        obj.add_glycan(box=boxes[0],image_path=obj.image_path())
        return obj.glycans()

    def find_boxes(self, obj):
        #implement crop and padding?
        image = obj.image()
        height, width, _ = image.shape
        return [ BoundingBox(image=image, x=0, y=0, width=width, height=height) ]

        
# handles one/many glycans 
class CleanGlycanImage(GlycanFinder):

    def __init__(self):
        super().__init__()

    def find_boxes(self, obj):
        print("\nCLEAN IMAGE")
        boxes = []
        for gly in obj.glycans():
            img = gly.image()
            cleaned_img, (x, y, w, h) = self.process_image(img)
            box = gly.get('box') 
            # print("box",box)

            new_box = box.clone()
            new_box.update_bbox(x=x,y=y,w=w,h=h)
            # print("new_box",new_box)

            box.set('image',cleaned_img)

            # box = BoundingBox(image=cleaned_img,x=x,y=y,w=w,h=h)
            # cleaned_image_dimensions={'x':x,'y':y,'w':w,'h':h}
            box_details = dict(id=gly.get('id'), box=box, image=cleaned_img)
            boxes.append(box_details)

        return boxes

    def find_objects(self, obj):
        boxes = self.find_boxes(obj)
        for gly in obj.glycans():
            gly_id = gly.get('id')

            for box_details in boxes:
                if box_details['id'] == gly_id:
                    gly.set_image(box_details['image'])
                    # gly.set("cleaned_image_dimensions",box_details['cleaned_image_dimensions'])
                    # gly.set("box", box_details['box'])
                    
        return obj.glycans()

    def image_contour(self,img):
        # Convert to grayscale and apply Binary inverse thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            largest_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
            return contours, largest_index

        return contours, None

    # Crop and clean the largest detected component in the image
    def process_image(self,img):
        contours, largest_index = self.image_contour(img)

        if largest_index is None:
            return img

        # crop image - offset (x, y) and the cropped region size (w, h)
        # need to store this information - required when we annotate details on the entire image
        x, y, w, h = cv2.boundingRect(contours[largest_index])
        cropped_image = img[y:y+h, x:x+w]

        # clean image
        contours, largest_index = self.image_contour(cropped_image)
        out = np.zeros_like(cropped_image)
        cv2.drawContours(out, contours, largest_index, (255, 255, 255), -1)
        _, out = cv2.threshold(out, 230, 255, cv2.THRESH_BINARY_INV)
        cleaned_image = cv2.bitwise_or(out, cropped_image)
        return cleaned_image, (x, y, w, h)

    # def save_cleaned_image(self, obj, img):
    #     print("-->image_path",obj.image_path())
    #     img_path = os.path.splitext(obj.image_path()) + ".cleaned.png"
    #     cv2.imwrite(img_path, img)
    #     return img_path

