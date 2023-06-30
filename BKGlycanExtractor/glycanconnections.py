# -*- coding: utf-8 -*-
"""
With monosaccharides already located and classed, connect them.
Returns undirected links.

all subclasses need a connect method
which takes a GlycanMonoInfo object as defined in monosaccharideid.py
and returns an  list of connected Monosaccharide objects
"""

import cv2
import logging
import numpy as np

class GlycanConnector:
    def __init__(self):
        pass
    
    def connect(self, **kw):
        raise NotImplementedError
        
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanconnections')
        

class HeuristicConnector(GlycanConnector):
    def __init__(self, configs):
        #read in color ranges for masking
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
        
        super().__init__()
        
    def connect(self, mono_info):
        origin_image = mono_info.get_image()
        monos_list = mono_info.get_monos()
        
        image = origin_image.copy()
        
        all_masks, black_masks = self.get_masks(image)
        
        
        list_center_point = []
        
        for mono in monos_list:

            [x, y, w, h] = mono.to_new_list(list_type="detected")

            centerX, centerY = mono.get_center_point()
            
            list_center_point.append((centerX, centerY))

            p11 = (int(x*0.985), int(y*0.985))
            p22 = (int((x + w)*1.015), int((y + h)*1.015))
            cv2.rectangle(black_masks, p11, p22, (0, 0, 0), -1)
            
        avg_distance = self.get_average_mono_distance(list_center_point)
        
        monos_list = self.link_monos(
            black_masks, monos_list, average=avg_distance
            )
        
        printstr = "Linked monosaccharides:\n" + str([mono.get_ID() for mono in monos_list])
        
        self.logger.info(printstr)
        
        mono_info.set_monos(monos_list)
        
        self.logger.info(mono_info)
        
        # for mono in mono_info.get_monos():
        #     printstr = f"{mono.get_ID()}: linked monos {mono.get_linked_monos()}"
        #     self.logger.info(printstr)
        
            
    def get_average_mono_distance(self, center_point_list):
        # find median distance between mono default = 100
        avg_mono_distance = 100

        for point in center_point_list:
            length_list = []
            for point2 in center_point_list:
                aux_len = self.length_line(point, point2)
                length_list.append(aux_len)
            length_list.sort()
            length_list = length_list[1:]
            if length_list!=[]:
                avg_mono_distance += length_list[0]
        if len(center_point_list)!=0:
            avg_mono_distance = avg_mono_distance / len(center_point_list)
        return avg_mono_distance            
    
    def get_masks(self, image):
        
        img = self.smooth_and_blur(image)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        
        color_range_dict = self.color_range
        hsv = hsv

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
        
        all_masks = list(mask_array_name)

        all_masks_no_black = all_masks.copy()
        all_masks_no_black.remove("black_mask")

        all_masks_no_black = sum([mask_dict[a] for a in all_masks_no_black])
        all_masks = sum([mask_dict[a] for a in all_masks])

        black_masks = mask_dict["black_mask"]

        empty_mask = np.zeros(
            [black_masks.shape[0], black_masks.shape[1], 1], dtype=np.uint8
            )
        empty_mask.fill(0)

        all_masks = cv2.cvtColor(all_masks, cv2.COLOR_GRAY2BGR)
        all_masks_no_black = cv2.cvtColor(
            all_masks_no_black, cv2.COLOR_GRAY2BGR
            )
        return all_masks, black_masks
    
    def interaction_line_line(self, A, B, C, D):
        Ax, Ay, Bx, By, Cx, Cy, Dx, Dy = \
            A[0], A[1], B[0], B[1], C[0], C[1], D[0], D[1]

        if ((Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay)) != 0: 
            # line is horizontal or vertical
            cont1 = (
                (Dx - Cx) * (Ay - Cy) - (Dy - Cy) * (Ax - Cx)
                ) / (
                    (Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay)
                    )
            cont2 = (
                (Bx - Ax) * (Ay - Cy) - (By - Ay) * (Ax - Cx)
                ) / (
                    (Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay)
                    )
            if (0 <= cont1 <= 1 and 0 <= cont2 <= 1):
                # intersec_X = Ax + (cont1 * (Bx - Ax))
                # intersec_Y = Ay + (cont1 * (By - Ay))
                # print(intersec_X, intersec_Y)
                return True
        return False
    
    def interaction_line_rect(self, line, rect):
        # line two points
        A, B = line[0], line[1]
        # rect x,y,w,h
        x, y, w, h = rect
        top = ((x, y), (x + w, y))
        bottom = ((x, y + h), (x + w, y + h))
        right = ((x + w, y), (x + w, y + h))
        left = ((x, y), (x, y + h))
        if (self.interaction_line_line(A, B, top[0], top[1]) 
            or self.interaction_line_line(A, B, bottom[0], bottom[1]) 
            or self.interaction_line_line(A, B, right[0], right[1]) 
            or self.interaction_line_line(A, B, left[0], left[1])):
                    
            return True
        return False
    
    
    def length_line(self, A, B):
        Ax, Ay, Bx, By = A[0], A[1], B[0], B[1]
        l = ((Ax - Bx) ** 2 + (By - Ay) ** 2) ** 0.5
        return l
    
    def link_monos(self, binary_img, monos_list, **kw):
        
        avg_mono_distance = kw.get("average", 100)
        diff = binary_img
        imheight, imwidth, *channels = diff.shape

        for mono in monos_list:
            
            if mono.type_ == 2:
                cropfactor = 1.2
            elif mono.type_ == 1:
                cropfactor = 1.5
            
            [x, y, w, h] = mono.to_new_list(list_type="detected")
            cir_radius = int((((h ** 2 + w ** 2) ** 0.5) / 2) * cropfactor)
            centerX, centerY = mono.get_center_point()
            
            y1 = centerY - cir_radius
            y2 = centerY + cir_radius
            x1 = centerX - cir_radius
            x2 = centerX + cir_radius
            if y1 < 0:
                y1 = 0
            if x1 < 0:
                x1 = 0
            if y2 > imheight:
                y2 = imheight
            if x2 > imwidth:
                x2 = imwidth
            crop = diff[y1:y2,
                   x1:x2]
            

            contours_list, _ = cv2.findContours(
                crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
            
            linked_monos = mono.get_linked_monos()

            for contour in contours_list:
                stop=0
                point_centerX2 = 0
                point_centerY2 = 0
                for point in contour:
                    point_centerX2 += point[0][0]
                    point_centerY2 += point[0][1]
                point_centerX2 = int(point_centerX2 / len(contour))
                point_centerY2 = int(point_centerY2 / len(contour))


                Ax = centerX
                Ay = centerY

                Bx = centerX - cir_radius + point_centerX2
                By = centerY - cir_radius + point_centerY2
                #################### length adjustable
                for i in range(1, 200, 5):
                    i = i / 100
                    length = avg_mono_distance * i
                    lenAB = ((Ax - Bx) ** 2 + (Ay - By) ** 2) ** 0.5
                    if lenAB==0:
                        lenAB=1
                    Cx = int(Bx + (Bx - Ax) / lenAB * length)
                    Cy = int(By + (By - Ay) / lenAB * length)
                    for mono2 in monos_list:
                        
                        [x, y, w, h] = mono2.to_new_list(list_type="detected")
                        
                        linked_monos2 = mono2.get_linked_monos()
                        
                        rectangle = (x, y, w, h)

                        line = ((Ax, Ay), (Cx, Cy))
                        if (self.interaction_line_rect(line, rectangle) 
                            and mono2.get_ID() != mono.get_ID()):
                            if mono2.get_ID() not in linked_monos:
                                linked_monos.append(mono2.get_ID())
                            if mono.get_ID() not in linked_monos2:
                                linked_monos2.append(mono.get_ID())
                                mono2.add_linked_monos(linked_monos2)
                            stop = 1
                            break
                    if stop == 1:
                        break
            mono.add_linked_monos(linked_monos)
            
        return monos_list
  
    
    def smooth_and_blur(self, img):
    
        img = cv2.GaussianBlur(img, (11, 11), 0)
        # _,img_file=cv2.threshold(img_file,140,255,cv2.THRESH_BINARY)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img