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

from .bbox import BoundingBox
from .yolomodels import YOLOModel
from .glycanannotator import Config

class MonoID(object): 
    
    mono_syms = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]

    def get_mono_sym(self, index):
        return self.mono_syms[index]

    def get_mono_index(self, name):
        assert name in self.mono_syms
        return self.mono_syms.index(name)

    def execute(self, obj):
        self.find_objects(obj)

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
    
    def find_objects(self, obj):
        raise NotImplementedError
        
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

        # MonoID has no constructor...

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
        
        monos = []
        
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
                    obj.add_mono(symbol=mono,box=box)
        
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
        'threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0
    }

    def __init__(self,**kwargs):
 
        params = dict(
            config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
            weights = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults),
            threshold = Config.get_param('threshold', Config.FLOAT, kwargs, self.defaults),
            boxpadding = Config.get_param('boxpadding', Config.INT, kwargs, self.defaults),
            expandimage = Config.get_param('expandimage', Config.INT, kwargs, self.defaults)
        )
        YOLOModel.__init__(self,params)
        # assert self.classes == len(self.mono_syms)
        MonoID.__init__(self)

    def find_objects(self, obj):
        image = obj.image()
        mono_boxes = self.find_boxes(image)
        obj.clear_monos()
        for box in mono_boxes:
            sym = self.get_mono_sym(box.get("classid"))
            # id is being added to the semantics and not bounding-box
            obj.add_mono(symbol=sym,box=box)

    def find_boxes(self, image, **kwargs):
        return self.get_YOLO_output(image,**kwargs,class_options=True)

class KnownMono(MonoID):
    def __init__(self,**kwargs):
        pass
    
    def find_objects(self, obj):
        image_path = obj.image_path()
        assert image_path, "KnownMono can only run on SingleGlycanImage glycan finder semantics objects"
        mono_boxes = self.find_boxes(image_path)
        obj.clear_monos()
        for box in mono_boxes:
            box.set_image_dimensions(image_width=obj.width(),image_height=obj.height())
            obj.add_mono(symbol=self.get_mono_sym(box.get('classid')),box=box,id=box.get('id'))

    def find_boxes(self, image_path, **kwargs):
        boxpadding = kwargs.get('boxpadding',0)
        
        monos = []
        image_path = image_path.rsplit('.',1)[0] + "_map.txt"
        with open(image_path, 'r') as file:
            for line in file:
                if line.startswith('m'):
                    data_points = line.split()
                    mono_id = data_points[1]
                    name = data_points[2]
                    x_coords = []
                    y_coords = []

                    for coords in data_points[3:-1]:
                        x,y = map(int,coords.split(','))
                        x_coords.append(x)
                        y_coords.append(y)

                    x_min = min(x_coords)
                    y_min = min(y_coords)
                    x_max = max(x_coords)
                    y_max = max(y_coords)

                    box = BoundingBox(x1=x_min, y1=y_min, x2=x_max, y2=y_max, classid=self.get_mono_index(name), id=int(mono_id))
                    box.pad(boxpadding) # known data is absolute
                    monos.append(box)

        return monos
