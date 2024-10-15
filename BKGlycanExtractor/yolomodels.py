# -*- coding: utf-8 -*-
"""
YOLOModel is superclass for all YOLO models
__init__ is the same for all YOLO models, 
requires weights file and YOLO .cfg file

all models need a get_YOLO_output method 
which takes an image as input and returns a list of boundingbox objects
but implementation may differ by class

YOLOTrainingData processes training.txt files
"""

import os
import math
import cv2
import numpy as np

from .bbox import BoundingBox


class YOLOModel:
    
    def __init__(self, **kwargs):

        self.threshold = kwargs.get('threshold',0.0)
        self.expandimage = kwargs.get('expandimage',100)
        self.boxpadding = kwargs.get("boxpadding", 0.0)
        self.multiclass = kwargs.get("multiclass", False)

        weights = kwargs.get("weights")
        net = kwargs.get("config")
        
        if weights is None:
            raise ValueError("required argument weights missing")
        if weights is not None and (not os.path.isfile(weights)):
            raise FileNotFoundError()
        if net is None:
            raise ValueError("required argument config missing")
        if net is not None and (not os.path.isfile(net)):
            raise FileNotFoundError()

        inyolo = False
        for l in open(net):
            if l.strip() == "[yolo]":
                inyolo = True
            elif l.strip().startswith('['):
                inyolo = False
            elif inyolo and 'classes' in l:
                sl = [ s.strip() for s in l.split('=') ]
                assert(sl[0]) == 'classes'
                self.classes = int(sl[1])
                break
        
        self.net = cv2.dnn.readNet(weights,net)
        
        layer_names = self.net.getLayerNames()
        #compatibility with new opencv versions
        try:
            self.output_layers = [layer_names[i[0] - 1] 
                                  for i in self.net.getUnconnectedOutLayers()]
        except IndexError:
            self.output_layers = [layer_names[i - 1] 
                                  for i in self.net.getUnconnectedOutLayers()]

    def get_YOLO_output(self, image):
        original_image = image.copy()
        if self.expandimage != 0:
            image = self.expand_image(image)
        blob = self.format_image(image)
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        boxes = []
        for out in outs:
            for detection in out:

                if not any(math.isnan(x) for x in detection):
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if self.multiclass:
                    
                        scores[class_id] = 0.
                        class_options = {class_id: confidence}
                    
                        while np.any(scores):
                            classopt = np.argmax(scores)
                            confopt = scores[classopt]
                            scores[classopt] = 0.
                            class_options[classopt] = confopt
                    
                        if len(class_options) > 1:
                            message = "WARNING: More than one class possible: " \
                                    + str(class_options)
                            print(message)                           
                    
                    box = BoundingBox(image=image,
                                      rcx=detection[0], rcy=detection[1], 
                                      rw=detection[2], rh=detection[3],
                                      classid=class_id, confidence=confidence)

                    if self.expandimage != 0:
                        box.set_image_dimensions(image=original_image)
                        box.shift(-self.expandimage,-self.expandimage)

                    if float(self.boxpadding) != 0.0:
                        if 0 < self.boxpadding < 1:
                            box.pad_relative(self.boxpadding)
                        else:
                            box.pad(self.boxpadding)

                    boxes.append(box)

        boxesfornms = [box.bbox() for box in boxes]
        confidences = [box.get('confidence') for box in boxes]
                
        indexes = cv2.dnn.NMSBoxes(
            boxesfornms, confidences, self.threshold, 0.4
        )

        return [boxes[i] for i in indexes]

    def format_image(self, image):
        return cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    def expand_image(self, image, expand=100):
        height, width, channels = image.shape

        # add expand pixels to top, bottom, left, and right
        bigwhite = np.zeros([height+(2*expand), width+(2*expand), 3], dtype=np.uint8)

        # white background...
        bigwhite.fill(255)

        # put image in the middle/center
        bigwhite[expand:(height+expand), expand:(width+expand)] = image

        return bigwhite


