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
    
    def __init__(self, defaults ,multicore=False):

        # need to remove the below and make changes to all the functions that are usign it
        # all details are now stored in self (Note: I will make the transition after testing things out)
        weights = self.weights
        net = self.config
        user_labels = self.labels
        file_labels = net.replace(".cfg",".labels")
        # weights = config.get("weights",None)
        # net = config.get("config",None)
        # user_labels = config.get("labels",None)
        # file_labels = net.replace(".cfg",".labels")
        # known_finder = net.replace(".cfg",".model")

        print('\nknown_finder',self.known_finder)

        self.conf_threshold = self.known_finder.get('conf_threshold',defaults['conf_threshold'])
        self.iou_threshold = self.known_finder.get('iou_threshold',defaults['iou_threshold'])
        self.expandimage = self.known_finder.get('expandimage',defaults['expandimage'])
        self.boxpadding = self.known_finder.get('boxpadding',defaults['boxpadding'])
        
        if not os.path.isfile(weights):
            raise FileNotFoundError()
        if not os.path.isfile(net):
            raise FileNotFoundError()
        if not os.path.isfile(file_labels):   # maybe .labels file should exist irrespective of - if the user provides their own labels or not, so that there is some record of the the true labels used during during training 
            raise FileNotFoundError()

        if isinstance(user_labels, list) and len(user_labels) > 0:
            self.labels = user_labels
        else:
            self.labels = [ l.strip() for l in open(file_labels).read().split() ]

        # get known_finder + other args - but notice that other args might be obtained
        # from the config file as well - so which one should get higher priority
        # with open(known_finder, 'r') as f:
        #     self.known_finder = f.read().strip()

        if not multicore:
            cv2.setNumThreads(1)

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
        if self.expandimage > 0:
            image = self.expand_image(image,self.expandimage)
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
                            classid=class_id, confidence=confidence, classlabel=self.get_label(class_id))

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


    # below method are use to make comparsions between boxes predicted by the YOLO model

    @staticmethod
    def euclidean_distance(box1,box2):
        obj1_cen_x, obj1_cen_y = box1.center()
        obj2_cen_x, obj2_cen_y = box2.center()
        return math.sqrt((obj1_cen_x - obj2_cen_x)**2 + (obj1_cen_y - obj2_cen_y)**2) 

    @staticmethod
    def proximity(known_box,pred_box):
        distance = CompareBoxes.euclidean_distance(known_box, pred_box)
        x,y,w,h = known_box['bbox']
        return distance/min(w, h)

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
    def union_area(training, detected):
        d_area = detected.area()
        t_area = training.area()
        intersection = CompareBoxes.intersection_area(training, detected)
        return float(d_area + t_area - intersection)


