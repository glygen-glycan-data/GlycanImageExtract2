# -*- coding: utf-8 -*-
"""
class for monosaccharide finding methods.

all need a find_monos method, 
which takes an image (should be cropped to single glycan)
and returns a GlycanMonoInfo instance consisting of: 
image(formatted for searching), image(annotated), 
dictionary of monosaccharide symbols and their counts, 
list of monosaccharide objects, 
string describing monosaccharide composition.
"""
import cv2
import logging
import math
import numpy as np

from .boundingboxes import BoundingBox
from .yolomodels import YOLOModel

# class_dictionary = {
#     0: "GlcNAc",
#     1: "NeuAc",
#     2: "Fuc",
#     3: "Man",
#     4: "GalNAc",
#     5: "Gal",
#     6: "Glc",
#     7: "NeuGc",
#     }

class_list = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]

# class FoundMonosaccharide(BoundingBox):
#     # constrain type to refer to these
#     contour_based = 1
#     bounding_box_based = 2
#     content_types = [contour_based, bounding_box_based]

#     # class_list = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]
    
#     # backwards_class_dictionary = {
#     #     "GlcNAc": 0,
#     #     "NeuAc": 1,
#     #     "Fuc": 2,
#     #     "Man": 3,
#     #     "GalNAc": 4,
#     #     "Gal": 5,
#     #     "Glc": 6,
#     #     "NeuGc": 7,
#     #     }
    
#     # class_dictionary = {
#     #     0: "GlcNAc",
#     #     1: "NeuAc",
#     #     2: "Fuc",
#     #     3: "Man",
#     #     4: "GalNAc",
#     #     5: "Gal",
#     #     6: "Glc",
#     #     7: "NeuGc",
#     #     }
    
#     id_beginnings = tuple(class_list)
    
#     def __init__(self, monoid, type_, **kw):
#         super().__init__(**kw)
#         assert monoid.startswith(self.id_beginnings)
#         self.monoID = monoid
#         assert type_ in self.content_types
#         self.type_ = type_
#         self.linked_monos = []
#         self.root = False
        
#     def add_linked_monos(self, mono_list):
#         for mono in mono_list:
#             if mono not in self.linked_monos:
#                 self.linked_monos.append(mono)
        
#     def get_ID(self):
#         return self.monoID
        
#     def get_linked_monos(self):
#         return self.linked_monos
    
#     def is_root(self):
#         return self.root
    
#     def remove_linked_mono(self, monoID):
#         self.linked_monos.remove(monoID)
        
#     def set_root(self):
#         self.root = True
        
#     def __str__(self):
#         printstr = f"{{ id: {self.monoID}, box: {self.to_new_list()}, "
#         printstr += f"image dimensions: w:{self.imwidth}, h:{self.imheight},\n"
#         printstr += "linked monos: "
#         printstr += ','.join([monoid for monoid in self.linked_monos])
#         printstr += f",\nroot: {self.root} }}"
        
#         return printstr
        

# monos is a list of Monosaccharide objects
# class GlycanMonoInfo:
#     def __init__(self, **kw):
#         self.original_img = kw["original"]
#         self.comp_dict = kw.get("count_dict",)
#         self.monos = kw.get("monos", [])
#         self.annotated_img = kw["annotated"]
#         self.comp_string = kw.get("comp_str", '')
#         self.boxes = kw.get('boxes',[])
    
#     def get_annotated_image(self):
#         return self.annotated_img
    
#     def get_composition(self):
#         return self.comp_dict
    
#     def get_composition_string(self):
#         return self.comp_string
    
#     def get_image(self):
#         return self.original_img
    
#     def get_monos(self):
#         return self.monos
    
#     def set_monos(self, monos_list):
#         self.monos = monos_list

#     def get_boxes(self):
#         return self.boxes
        
#     def __str__(self):
#         printstr = f"{{ composition_dict:\n{self.comp_dict}\nmonos: "
#         printstr += ','.join([mono.monoID for mono in self.monos])
#         printstr += f"\ncompostion count: {self.comp_string},"
#         height, width, channels = self.original_img.shape
#         printstr += f"\nimage dimensions: w:{width}, h:{height} }}"
            
#         return printstr

class MonoID: 
    def __init__(self, **kw):
        super().__init__()
        
    # def compstr(self, counts):
    #     s = ""
    #     for sym,count in sorted(counts.items()):
    #         if count > 0:
    #             s += "%s(%d)"%(sym,count)
    #     return s
    
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
    
    def find_objects(self, image):
        raise NotImplementedError
        # return mono_info, an instance of GlycanMonoInfo
        
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
    def __init__(self, configs,resize_image=False):
        #read in color ranges for mono id
        color_range = configs.get("color_range",)
        color_range_file = open(color_range)
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

        self.img_resize = resize_image
        super().__init__()
        
    def compare_to_img(self, img1, img2):
        # return similarity between two image
        if img1.shape == img2.shape:
            pass
        else:
            return -1
        score = 0
        diff = cv2.absdiff(img1, img2)
        r, g, b = cv2.split(diff)
        score = cv2.countNonZero(g) / (img1.shape[0] * img1.shape[1])

        # cv2.imshow("different", diff)
        # cv2.waitKey(0)
        return 1 - score
        

    def find_objects(self, obj):
        # monos = {}
        img_resize = self.img_resize
        # test = kw.get("test", False)
        image = obj.get('image')
        img = self.crop_largest(image)

        if not img_resize:
            img = self.resize_image(img)

        #save original image, and then format it for masking
        origin_image = img.copy()
        # monos["original"] = origin_image
        img = self.smooth_and_blur(img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_width = img.shape[1]
        img_height = img.shape[0]

        # read color range in config folder
        #origin = img.copy()
        final = img.copy()  # final annotated pieces

        mask_array, mask_array_name, mask_dict = self.get_masks(hsv)
        
        monoCount_dict = {
            "GlcNAc": 0, 
            "NeuAc": 0, 
            "Fuc": 0, 
            "Man": 0, 
            "GalNAc": 0, 
            "Gal": 0, 
            "Glc": 0, 
            "NeuGc": 0,
            }

        monos = []
        
        count = 0
        for color in mask_array_name:
            if color == "black_mask":
                continue
            contours_list, _ = cv2.findContours(
                mask_dict[color], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            
            for contour in contours_list:

                # approx = cv2.approxPolyDP(
                #     contour, 0.035 * cv2.arcLength(contour, True), True
                #     )
                x, y, w, h = cv2.boundingRect(contour)

                area = cv2.contourArea(contour)
                
                squareness = abs(math.log(float(w)/float(h),2))
                arearatio = 1e6*float(area)/(img_height*img_width)
                arearatio1 = 1000*area/float(w*h)
                if squareness < 2 and arearatio > 100 and arearatio1 > 200:
                    if (squareness > 0.25 
                            or arearatio < 1000.0 
                            or arearatio1 < 500):
                        self.logger.info("BAD")
                        continue
                    box = BoundingBox(
                        image=img, x=x, y=y, width=w, height=h
                        )
                    
                    box.corner_to_center()
                    box.abs_to_rel()
                    box.to_four_corners()
                    
                    # p1 = (x,y)
                    # p2 = (x+w, y+h)
                    # cv2.rectangle(final, p1, p2, (0, 255, 0), 1)
                    # cv2.drawContours(final, [approx], 0, (0, 0, 255), 1)
                    
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
                            mono = "??? score="+score
                else:
                    continue
                if "???" not in mono:
                    # box.set_class(FoundMonosaccharide.backwards_class_dictionary[mono])
                    box.set_class(class_list.index(mono))
                    box.set_dummy_confidence(1)

                    # monosaccharide = FoundMonosaccharide(
                    #     monoid=mono+str(count), type_=1, boundingbox=box
                    #     )

                    count += 1
                    # monos.append(monosaccharide)

                    mono_info = {
                        'id': mono+str(count),
                        'type': mono,
                        'bbox': box.to_new_list(),
                        'center': [box.cen_x, box.cen_y],
                        'box': box
                    }

                    obj['monos'].append(mono_info)

                    # final = monosaccharide.annotate(final)
                # else:
                #     final = box.annotate(final, mono)
                
                # self.logger.info(mono)
                if mono in monoCount_dict:
                    monoCount_dict[mono] += 1
                    
        # c = self.compstr(monoCount_dict)
                    
        # mono_info = GlycanMonoInfo(
        #     original=origin_image, count_dict=monoCount_dict, 
        #     monos=monos, annotated=final, comp_str=c
        #     )
        
        return obj  
        
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
    
    
class YOLOMonos(YOLOModel, MonoID):
    def __init__(self, configs,threshold=0.5,resize_image=False):
        super().__init__(configs)
        self.threshold = threshold
        self.img_resize = resize_image

    def find_objects(self, obj):
        
        # threshold = kw.get('threshold', 0.5)
        # resize_image = kw.get('resize_image', False)
        threshold = self.threshold
        img_resize = self.img_resize

        image = obj.get('image')
        
        # monosaccharide finding formatting
        img = self.crop_largest(image)
        
        if not img_resize:
            img = self.resize_image(img)
        
        origin_image = img.copy()

        # print("img",img)
        
        # YOLO formatting
        # image_dict = self.format_image(img)
        
        mono_boxes = self.get_YOLO_output(img, threshold,class_options=True)

        # Contours: monos_boxes is monos_list which was saved as monos["contours"] = monos_list  in link-prediction code

        final = origin_image.copy()
        
        monoCount_dict = {
            "GlcNAc": 0, 
            "NeuAc": 0, 
            "Fuc": 0, 
            "Man": 0, 
            "GalNAc": 0, 
            "Gal": 0, 
            "Glc": 0, 
            "NeuGc": 0,
            }
        
        monos = []
        count = 0
        for mono in mono_boxes:
            
            # print("mono",mono.cen_x, mono.cen_y)
            class_id = mono.class_
            # mononame = class_dictionary[class_id]
            mononame = class_list[class_id]

            monoCount_dict[mononame] += 1

            # self.logger.info(mononame)
            # monosaccharide = FoundMonosaccharide(
            #     monoid=mononame+str(count), type_=2, boundingbox=mono
            #     )
            # final = monosaccharide.annotate(final)

            # monos.append(monosaccharide)


            count += 1

            mono_info = {
                'id': mononame+str(count),
                'type': mononame,
                'bbox': mono.to_new_list(),
                'center': [mono.cen_x,mono.cen_y],
                'box': mono
            }

            obj['monos'].append(mono_info)
                
        return obj
        
                
        
