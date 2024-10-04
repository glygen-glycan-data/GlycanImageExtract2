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

from BKGlycanExtractor.boundingboxes import BoundingBox


class YOLOModel:
    
    def __init__(self, config):
        weights = config.get("weights",None)
        net = config.get("config",None)
        
        if weights is not None and (not os.path.isfile(weights)):
            raise FileNotFoundError()
        if net is not None and (not os.path.isfile(net)):
            raise FileNotFoundError()
        
        self.net = cv2.dnn.readNet(weights,net)
        
        layer_names = self.net.getLayerNames()
        #compatibility with new opencv versions
        try:
            self.output_layers = [layer_names[i[0] - 1] 
                                  for i in self.net.getUnconnectedOutLayers()]
        except IndexError:
            self.output_layers = [layer_names[i - 1] 
                                  for i in self.net.getUnconnectedOutLayers()]

    def get_YOLO_output(self, image, threshold, **kw):

        image_dict = self.format_image(image)
        
        request_padding = kw.get("request_padding", False)
        multi_class = kw.get("class_options", False)
        pr_test = kw.get("pr_test",False)
        
        blob = image_dict["formatted_blob"]
        origin_image = image_dict["origin_image"]
        white_space = image_dict["whitespace"]
        
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        unpadded_boxes = []
        padded_boxes = []

        for out in outs:
            for detection in out:

                if not any(math.isnan(x) for x in detection):
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if multi_class:
                    
                        scores[class_id] = 0.
                        class_options = {str(class_id): confidence}
                    
                        while np.any(scores):
                            classopt = np.argmax(scores)
                            confopt = scores[classopt]
                            scores[classopt] = 0.
                            class_options[str(classopt)] = confopt
                    
                        if len(class_options) > 1:
                            message = "WARNING: More than one class possible: " \
                                    + str(class_options)
                            print(message)                           
                    
                    box = BoundingBox(confidence=confidence, image=origin_image, class_=class_id,
                    white_space=white_space, rel_cen_x=detection[0],
                    rel_cen_y=detection[1], rel_w=detection[2],
                    rel_h=detection[3]
                    )
                    # generates boxes which are common for all the different components (monos, root, connector) that use YOLO detector
                    box.rel_to_abs()
                    
                    box.fix_image()
                    
                    box.center_to_corner()
                    
                    if request_padding:
                        box.pad_borders()
                    
                    box.fix_borders()
                    
                    box.is_entire_image()

                    if box.y<0:
                        box.y = 0

                    box.to_four_corners()
                    
                    boxes.append(box)

                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # returns boxes - which are common for all the different components (monos, root, connector) that use YOLO detector 
        boxesfornms = [bbox.to_list() for bbox in boxes]
                
        try:
            indexes = cv2.dnn.NMSBoxes(
                boxesfornms, confidences, threshold, 0.4
                )
        except TypeError:
            boxesfornms = [bbox.to_new_list() for bbox in boxes]
            indexes = cv2.dnn.NMSBoxes(
                boxesfornms, confidences, threshold, 0.4
                )
        try:
            indexes = [index[0] for index in indexes]
        except IndexError:
            pass

        boxes = [boxes[i] for i in indexes]
        confidences = [confidences[i] for i in indexes]
        class_ids = [class_ids[i] for i in indexes]
        return boxes


    def format_image(self, image):
        origin_image = image.copy()
        height, width, channels = image.shape


        white_space = 200
        bigwhite = np.zeros(
            [image.shape[0]+white_space, image.shape[1]+white_space, 3],
            dtype=np.uint8
            )
        bigwhite.fill(255)
        half_white_space = int(white_space/2)
        bigwhite[half_white_space : (half_white_space+image.shape[0]), 
                 half_white_space : (half_white_space+image.shape[1])] = image
        image = bigwhite.copy()

        blob = cv2.dnn.blobFromImage(
            image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
            )
        
        image_dict = {
            "origin_image": origin_image,
            "whitespace": white_space,
            "formatted_blob": blob
            }
        
        return image_dict


