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
from . semantics import GlycanSemantics
from collections import Counter
from BKGlycanExtractor import DebugMode

# Base class
class GlycanFinder(Finder):  

    finder_class = 'Glycan'
    labels = ['glycan']

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
        GlycanFinder.__init__(self)
        YOLOModel.__init__(self,params)

    def find_boxes(self, obj):
        return sorted(self.get_YOLO_output(obj.image()),key=lambda b: b.bbox())

    def find_objects(self, figure_obj):
        figure_obj.reset_glycans()
        for box in self.find_boxes(figure_obj):
            glyobj = GlycanSemantics(figure=figure_obj,box=box,classlabel=self.boxlabel(box))
            figure_obj.add_glycan(glyobj)
        return figure_obj.glycans()
    
class SingleGlycanImage(GlycanFinder):
    
    labels = ["glycan"]
    defaults = {
        'crop': False,
        'padding': 0
    }

    def __init__(self,**kwargs):
       self.crop = Config.get_param('crop', Config.BOOL, kwargs, self.defaults)
       self.padding = Config.get_param('padding', Config.FLOAT, kwargs, self.defaults)
       super().__init__()

    def find_objects(self, obj):
        obj.reset_glycans()
        boxes = self.find_boxes(obj)
        box = boxes[0]
        glyobj = GlycanSemantics(figure=obj,box=box,image_path=obj.image_path(),classlabel=self.get_label(0))
        obj.add_glycan(glyobj)
        return obj.glycans()

    def find_boxes(self, obj):
        #implement crop and padding?
        image = obj.image()
        height, width, _ = image.shape
        return [ BoundingBox(image=image,x=0,y=0,width=width,height=height,classlabel=self.get_label(0)) ]

class KnownGlycanBoxes(GlycanFinder):

    labels = ["glycan"]
    defaults = {
        'boxpadding': 0,
    }

    def __init__(self,**kwargs):
        self.params = dict(
            boxpadding = Config.get_param('boxpadding', Config.FLOAT, kwargs, self.defaults),
        )
        self.boxpadding = self.params['boxpadding']
        super().__init__()

    def find_boxes(self, obj):
        yoloannot = obj.image_path().rsplit('.',1)[0] + ".txt"
        image = obj.image()
        boxes = []
        for l in open(yoloannot):
            classid,rcx,rcy,rw,rh = map(float,l.split())
            classid = int(classid)
            box = BoundingBox(rcx=rcx,rcy=rcy,rw=rw,rh=rh,image=image,
                              classid=classid,classlabel=self.get_label(classid))
            if 0 < self.boxpadding <= 1:
                box.pad_relative(self.boxpadding)
            elif 1 < self.boxpadding:
                box.pad(self.boxpadding)
            boxes.append(box)
        return boxes

    def find_objects(self, obj):
        boxes = self.find_boxes(obj)
        obj.clear_glycans()
        for box in boxes:
            glyobj = GlycanSemantics(figure=obj,box=box,classlabel=self.getlabel(0))
            obj.add_glycan(glyobj)
        return obj.glycans()
        
# handles one/many glycans 
class CleanGlycanImage(GlycanFinder):

    def __init__(self,**kwargs):
        super().__init__()

    def find_boxes(self, obj):
        # print("\nCLEAN IMAGE")
        boxes = []
        for gly in obj.glycans():
            img = gly.image()
            cropped_img, cleaned_img = self.process_image(img)
            box = gly.get('box') 

            # new_box = box.clone()
            # new_box.update_bbox(x=x,y=y,w=w,h=h)


            # saving the cleaned_image and original seperately - so that we have access to both the 
            # original and cleaned image for the webpage, but question is whether I should update the
            # coordinates (offset it) of the bounding boxes wrt to the cleaned image?
            box.set('image',cleaned_img)
            box.set('extracted_image', img)
            box_details = dict(id=gly.get('id'), box=box, image=cleaned_img, extracted_image=img)
            boxes.append(box_details)

        return boxes

    def find_objects(self, obj):
        boxes = self.find_boxes(obj)
        for gly in obj.glycans():
            gly_id = gly.get('id')

            for box_details in boxes:
                if box_details['id'] == gly_id:
                    gly.set_image(box_details['image'])
                    gly.set('extracted_image', box_details['extracted_image'])
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


        # Estimate background color
        bg_color = self.get_dominant_background_color(img)

        # pad the cropped and cleaned image with a white background to resize the extracted image to 
        # its original dimensions
        # Create background using dominant color
        background_cropped = np.full_like(img, bg_color)
        background_cleaned = np.full_like(img, bg_color)

        # Overlay cropped and cleaned images
        background_cropped[y:y+h, x:x+w] = cropped_image
        background_cleaned[y:y+h, x:x+w] = cleaned_image
        
        return background_cropped, background_cleaned


    
    def get_dominant_background_color(self,img):
        # Resize to speed up color counting
        small_img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)

        # Reshape to a list of pixels
        pixels = small_img.reshape(-1, 3)

        # Convert to tuple for Counter
        pixels = [tuple(p) for p in pixels]

        # Count pixel frequencies
        most_common_color = Counter(pixels).most_common(1)[0][0]
        return np.array(most_common_color, dtype=np.uint8)
