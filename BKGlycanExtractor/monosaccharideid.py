# -*- coding: utf-8 -*-
"""
class for monosaccharide finding methods.
"""
import cv2
import logging
import math
import numpy as np
import os
import sys
import json

from .bbox import BoundingBox
from .yolomodels import YOLOModel
from .glycanannotator import Config
from .finder import Finder
from .compareboxes import CompareBoxes
from BKGlycanExtractor import MonosCompare, BoxCompare, DebugMode


class MonoID(Finder): 
    
    labels = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]
    finder_class = 'Monosaccharide'

    @staticmethod
    def box_components(*args,**kwargs):
        return BoxCompare(*args,**kwargs)

    @staticmethod
    def semantic_components(*args,**kwargs):
        return MonosCompare(*args,**kwargs)

    @staticmethod
    def known_predictor():
        return KnownMono()

    def crop_largest(self, image):
        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        contours_list, _ = cv2.findContours(
            gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        
        area_list = []
        for i, contour in enumerate(contours_list):
            area = cv2.contourArea(contour)
            area_list.append((area, i))
        (_, largest_index) = max(area_list)
        out = np.zeros_like(img)
        cv2.drawContours(
            out, contours_list, largest_index, (255, 255, 255), -1
            )
        _, out = cv2.threshold(out, 230, 255, cv2.THRESH_BINARY_INV)

        out2 = cv2.bitwise_or(out, img)
        return out2
    
    def resize_image(self, img):
        bigwhite = np.zeros(
            [img.shape[0] + 30, img.shape[1] + 30, 3], dtype=np.uint8
            )
        bigwhite.fill(255)
        bigwhite[15:15 + img.shape[0], 15:15 + img.shape[1]] = img
        img = bigwhite.copy()
        mag = 84000 / (img.shape[0] * img.shape[1])
        if mag <= 1:
            mag = 1
        img = cv2.resize(img, None, fx=mag, fy=mag)
        return img
    
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.monosaccharideid')
    
class HeuristicMonos(MonoID):

    defaults = {
        'colors_range': 'colors_range.txt'
    }
    
    def __init__(self, **kwargs):

        self.color_range = Config.get_param('color_range', Config.CONFIGFILE, kwargs, self.defaults)
    
        color_range_file = open(self.color_range)
        color_range_dict = {}
        for line in color_range_file.readlines():
            line = line.strip()
            name = line.split("=")[0].strip()
            color_range = line.split("=")[1].strip()
            color_range_dict[name] = np.array(
                list(map(int, color_range.split(",")))
                )
        color_range_file.close()
        self.color_range = color_range_dict

        MonoID.__init__(self)

    def compare_to_img(self, img1, img2):
        if img1.shape == img2.shape:
            pass
        else:
            return -1
        score = 0
        diff = cv2.absdiff(img1, img2)
        r, g, b = cv2.split(diff)
        score = cv2.countNonZero(g) / (img1.shape[0] * img1.shape[1])
        return 1 - score

    def find_objects(self, obj):
        # split into find_boxes and semantics?

        image = obj.get('image')
        img = self.crop_largest(image)

        #save original image, and then format it for masking
        origin_image = img.copy()
        img = self.smooth_and_blur(img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_height, img_width, _ = img.shape
        final = img.copy()  # final annotated pieces

        mask_array, mask_array_name, mask_dict = self.get_masks(hsv)
        
        obj.clear_monos()
        
        count = 0
        for color in mask_array_name:
            if color == "black_mask":
                continue
            contours_list, _ = cv2.findContours(
                mask_dict[color], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            
            for contour in contours_list:
                x, y, w, h = cv2.boundingRect(contour)

                area = cv2.contourArea(contour)
                
                squareness = abs(math.log(float(w)/float(h),2))
                arearatio = 1e6*float(area)/(img_height*img_width)
                arearatio1 = 1000*area/float(w*h)
                if squareness < 2 and arearatio > 100 and arearatio1 > 200:
                    if (squareness > 0.25 
                            or arearatio < 1000.0 
                            or arearatio1 < 500):
                        # self.logger.info("BAD")
                        continue
                    box = BoundingBox(x=x, y=y, width=w, height=h)
                    
                    if color == "red_mask":
                        mono = 'Fuc'                        

                    elif color == "purple_mask":
                        mono = 'NeuAc'
                        
                    elif color == "blue_mask":
                        white = np.zeros([h, w, 3], dtype=np.uint8)
                        white.fill(255)
                        this_blue_img = mask_dict["blue_mask"][y:y+h, x:x+w]
                        this_blue_img = cv2.cvtColor(
                            this_blue_img, cv2.COLOR_GRAY2BGR
                            )
                        score = self.compare_to_img(white, this_blue_img)
                        if score >= 0.8:  # is square
                            mono = 'GlcNAc'

                        elif 0.5 < score < 0.8: # is circle
                            mono = 'Glc'

                        else:
                            mono = '??? score='+score

                    elif color == "green_mask":
                        mono = "Man"

                    elif color == "yellow_mask":
                        white = np.zeros([h, w, 3], dtype=np.uint8)
                        white.fill(255)
                        yellow_img = mask_dict["yellow_mask"][y:y+h, x:x+w]
                        yellow_img = cv2.cvtColor(
                            yellow_img, cv2.COLOR_GRAY2BGR
                            )

                        score = self.compare_to_img(white, yellow_img)
                        if score > 0.9:  # is square
                            mono = "GalNAc"

                        elif 0.5 < score < 0.9: # is circle
                            mono = "Gal"
                        else:
                            mono = "??? score="+str(score)
                else:
                    continue
                if "???" not in mono:
                    classid = self.get_label_index(mono)
                    box.set('classid',classid)
                    # box.set('symbol',mono)
                    obj.add_mono(classid=classid,symbol=mono,box=box)

        return obj.monosaccharides()
        
    def get_masks(self, hsv_image):
        color_range_dict = self.color_range
        hsv = hsv_image

        # create mask for each color
        yellow_mask = cv2.inRange(
            hsv, color_range_dict['yellow_lower'], 
            color_range_dict['yellow_upper']
            )
        purple_mask = cv2.inRange(
            hsv, color_range_dict['purple_lower'], 
            color_range_dict['purple_upper']
            )
        red_mask_l = cv2.inRange(
            hsv, color_range_dict['red_lower_l'], 
            color_range_dict['red_upper_l'])
        red_mask_h = cv2.inRange(
            hsv, color_range_dict['red_lower_h'], 
            color_range_dict['red_upper_h']
            )
        red_mask = red_mask_l + red_mask_h
        green_mask = cv2.inRange(
            hsv, color_range_dict['green_lower'], 
            color_range_dict['green_upper']
            )
        blue_mask = cv2.inRange(
            hsv, color_range_dict['blue_lower'], 
            color_range_dict['blue_upper']
            )
        black_mask = cv2.inRange(
            hsv, color_range_dict['black_lower'], 
            color_range_dict['black_upper']
            )

        # store these mask into array
        mask_array = (
            red_mask, yellow_mask, green_mask, 
            blue_mask, purple_mask, black_mask
            )
        mask_array_name = (
            "red_mask", "yellow_mask", "green_mask", 
            "blue_mask", "purple_mask", "black_mask"
            )
        mask_dict = dict(zip(mask_array_name, mask_array))
        return mask_array,mask_array_name,mask_dict 
    
    def smooth_and_blur(self, img):
    
        img = cv2.GaussianBlur(img, (11, 11), 0)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img

class YOLOMonos(YOLOModel,MonoID):

    defaults = {
        'conf_threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0,
        'iou_threshold': 0.4
    }

    def __init__(self,**kwargs):
 
        params = dict(
            config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
            weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
            conf_threshold = Config.get_param('conf_threshold', Config.FLOAT, kwargs, self.defaults),
            iou_threshold = Config.get_param('iou_threshold', Config.FLOAT, kwargs, self.defaults),
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
            expandimage = Config.get_param('expandimage', Config.INT, kwargs, self.defaults)
        )        

        self.name = Config.get_finder_name(kwargs)
        self.cb = CompareBoxes()
        YOLOModel.__init__(self,params)
        MonoID.__init__(self)

    def find_objects(self, obj):
        mono_boxes = self.find_boxes(obj)
        obj.clear_monos()

        for id, box in enumerate(mono_boxes):
            classid = box.get('classid')
            conf = float(box.get('confidence'))
            symbol = self.get_label(classid)
            box.set('id', id)
            box.set('symbol', symbol)
            box.set('classlabel', symbol)
            obj.add_mono(classid=classid,classlabel=symbol,symbol=symbol,box=box,id=id,confidence=conf)

        # check for overlaps, necessarily with different classes, keep
        # highest confidence as primary - do not expect bad
        # cases, predictions are not expected to partially overlap
        sortedmono = sorted(obj.monosaccharides(),key=lambda m: -m['confidence'])
        removed = set()
        for i1 in range(0,len(sortedmono)-1):
            if i1 in removed:
                continue
            m1 = sortedmono[i1]
            for i2 in range(i1+1,len(sortedmono)):
                if i2 in removed:
                    continue
                m2 = sortedmono[i2]
                if self.cb.have_intersection(m1.get('box'),m2.get('box')):
                    obj.monosaccharide(m2['id'])['iou'] = self.cb.iou(m1.get('box'),m2.get('box'))
                    obj.make_alternative_mono(m1['id'],m2['id'])
                    removed.add(i2)

        return obj.monosaccharides()

    def find_boxes(self, obj):
        image = obj.image()
        boxes = self.get_YOLO_output(image)
        
        if DebugMode.debug:
            DebugMode.log_data(
            identifier= DebugMode.curr_image,
            data={'monos':[len(boxes)]},
            image_path = DebugMode.image_path,
            )

            DebugMode.info = None

        return boxes


class KnownMono(MonoID):

    # Need to be able to support any monosaccharide symbol in generated code
    labels = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc","Xyl"]
    defaults = {
        'boxpadding': 0,
    }

    def __init__(self,**kwargs):
        self.params = dict(
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
        )
        MonoID.__init__(self)
    
    def find_objects(self, obj):
        mono_boxes = self.find_boxes(obj)
        obj.clear_monos()
        for box in mono_boxes:
            box.set_image_dimensions(image_width=obj.width(),image_height=obj.height())
            obj.add_mono(classid=self.get_label_index(box.get('symbol')),symbol=box.get('symbol'),box=box,id=box.get('id'))

        return obj.monosaccharides()

    def find_boxes(self, obj):
        image_path = obj.image_path()
        assert image_path, "KnownMono can only run on SingleGlycanImage glycan finder semantics objects"
        boxes = []
        image_path = image_path.rsplit('.',1)[0] + "_map.txt"
        with open(image_path, 'r') as file:
            for line in file:
                if line.startswith('m'):
                    data_points = line.split()
                    mono_id = data_points[1]
                    name = data_points[2]
                    x_coords = []
                    y_coords = []

                    for coords in data_points[3:]:
                        if ',' in coords:
                            x,y = map(int,coords.split(','))
                            x_coords.append(x)
                            y_coords.append(y)

                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    box = BoundingBox(x1=x_min,y1=y_min,x2=x_max,y2=y_max,symbol=name,classid=self.get_label_index(name),id=int(mono_id))
                    box.pad(self.params['boxpadding']) # known data is absolute
                    boxes.append(box)

        if DebugMode.debug:
            DebugMode.log_data(
                identifier= DebugMode.curr_image,
                data={'monos_known':len(boxes)},
                image_path = DebugMode.image_path,
            )

        return boxes
