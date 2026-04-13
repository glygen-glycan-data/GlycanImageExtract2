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
from . glycanannotator import Config, GlycanExtractorPipeline
from . finder import YOLOFinder, KnownFinder, Finder
from . semantics import GlycanSemantics
from collections import Counter
from BKGlycanExtractor import DebugMode, GlycanCompare

# Base class
class GlycanFinder:  

    def set_results(self, obj, accepted, rejected):
        obj.set_glycans(accepted, rejected)

    def log_error(self, obj, accepeted, rejected):
        pass

    def finder_pipeline(self,config_manager=None):
        pipeline = GlycanExtractorPipeline()
        pipeline.set_steps('figure', [self])
        return pipeline

    def semantic_compare(self,**kwargs):
        return GlycanCompare(**kwargs)

    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanfinding')

# YOLO based glycan finder
# allows minimum confidence thresholding, to restrict returns
# also allows requesting padding of glycan borders (off by default)
# confidences for YOLO detection are stored in the bounding box
class YOLOGlycanFinder(YOLOFinder,GlycanFinder):

    # must provide all these values...since we are overriding those in YOLOFinder
    defaults = {
        'conf_threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 200,
        'iou_threshold': 0.5
    }

    def __init__(self,**kwargs):
        YOLOFinder.__init__(self,**kwargs)
        GlycanFinder.__init__(self)


    def box_to_object(self,box,obj):
        return GlycanSemantics(figure=obj.image(),box=box,**box.items())

    
class SingleGlycanImage(Finder,GlycanFinder):
    
    defaults = {
        'crop': False,
        'padding': 0
    }

    def __init__(self,**kwargs):
        self.crop = Config.get_param('crop', Config.BOOL, kwargs, self.defaults)
        self.padding = Config.get_param('padding', Config.FLOAT, kwargs, self.defaults)
        
        Finder.__init__(self)
        GlycanFinder.__init__(self)

    def box_to_object(self,box,obj):
        return GlycanSemantics(figure=obj.image(),box=box,**box.items())

    def find_boxes(self, obj):
        #implement crop and padding?
        classid=self.get_label_index("glycan")
        return [ BoundingBox(classid=classid,classlabel="glycan",x=0,y=0,**obj) ]

class KnownGlycanBoxes(KnownFinder,GlycanFinder):

    def __init__(self,**kwargs):
        KnownFinder.__init__(self,**kwargs)
        GlycanFinder.__init__(self)

    # method adapted to handle single/multiple glycan data
    def create_boxes(self, map_dict):
        boxes = []
        fig_height = map_dict['figure']['height']
        fig_width = map_dict['figure']['width']

        # Note: map_dict data structure can store single/multiple glycans.
        for glycan in map_dict['glycans']:
            
            # classlabel can be updated via the semantics file - if the optionally provided label_type matches a key in the semantics file
            if self.label_type:
                classlabel = glycan[self.label_type]
            else:   # use default label - "glycan"
                classlabel = glycan['classlabel']
            classid = self.get_label_index(classlabel)
            gly_bbox = glycan['bbox']

            x,y,w,h = gly_bbox
            gly_box = BoundingBox(classid=classid,classlabel=classlabel,x=x,y=y,w=w,h=h,image_width=fig_width, image_height=fig_height)
            boxes.append(gly_box)

        return boxes

    def box_to_object(self,box,obj):
        return GlycanSemantics(figure=obj.image(),box=box,**box.items())
        
# handles one/many glycans
# Still need to work on this 
class CleanGlycanImage(Finder,GlycanFinder):

    def __init__(self,**kwargs):
        Finder.__init__(self)
        GlycanFinder.__init__(self)

    def find_boxes(self, obj):
        raise NotImplementedError
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
        for gly in obj.glycans():
            img = gly.image()
            cropped_img, cleaned_img = self.process_image(img)
            gly.set_image(cleaned_img)
            gly.set('extracted_image',img)
                    
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
