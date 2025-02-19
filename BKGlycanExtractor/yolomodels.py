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
from collections import defaultdict
from .bbox import BoundingBox
from .debug_methods import DebugMode

class YOLOModel:
    
    def __init__(self, config):
        weights = config.get("weights",None)
        net = config.get("config",None)

        self.conf_threshold = config.get('conf_threshold')
        self.iou_threshold = config.get('iou_threshold')
        self.expandimage = config.get('expandimage',0)
        self.boxpadding = config.get('boxpadding',0)
        
        if weights is not None and (not os.path.isfile(weights)):
            raise FileNotFoundError()
        if net is not None and (not os.path.isfile(net)):
            raise FileNotFoundError()
        
        self.net = cv2.dnn.readNet(weights,net)
        self.classes = self.get_num_classes(net)

        
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
        blob = self.format_image(image)
                
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        confidences = []
        boxes = []
        class_boxes = defaultdict(list)

        for out in outs:
            detections = out[~np.isnan(out).any(axis=1)] 

            for detection in detections:
 
                # if not any(math.isnan(x) for x in detection):
                scores = detection[5:]
                
                for class_id, confidence in enumerate(scores):
                    if confidence >= self.conf_threshold:

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

                        class_boxes[class_id].append(box)

        for class_id in class_boxes:
            boxesfornms = [box.bbox() for box in class_boxes[class_id]]
            confidences = [box.get('confidence') for box in class_boxes[class_id]]

            indexes = cv2.dnn.NMSBoxes(
                boxesfornms, confidences, self.conf_threshold, self.iou_threshold 
            )

            if len(confidences) != len(indexes):
                DebugMode.info = "Runner up boxes were rejected"
                # print("Log: Runner up boxes were rejected")

            # flatten - converts a n-Dimensional array into a 1D flat array,
            # but if no boxes satisfy the threshold NMSBoxes return an empty tuple(()) 
            try:
                boxes.extend([class_boxes[class_id][i] for i in indexes.flatten()])
            except:
                print("Log: No boxes passed NMS")

        return boxes

    def get_num_classes(self, config_path):
        # Parse the config file to get the number of classes
        with open(config_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            if 'classes=' in line:
                num_classes = int(line.split('=')[1].strip())
                return num_classes
        raise ValueError("Number of classes not found in the config file.")



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


