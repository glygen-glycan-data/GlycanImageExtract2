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
    
    def find_objects(self, **kw):
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

    #expects a monosaccharide connection dictionary and 2 monosaccharides
    #connects the monosaccharides to each other in the connection dictionary and returns it
    def append_links(self, obj, mono1, mono2):
        for mono in obj['monos']:
            if mono1 == mono['id']:
                if 'links' in mono:
                    if mono2 not in mono['links']:
                        mono['links'].append(mono2)
                else:
                    mono['links'] = [mono2] 

            elif mono2 == mono['id']:
                if 'links' in mono:
                    if mono1 not in mono['links']:
                        mono['links'].append(mono1)
                else:
                    mono['links'] = [mono1]

        return obj

    def find_objects(self, obj):
        origin_image = obj.get('image')
        # monos_list = obj.get_monos()
        
        image = origin_image.copy()
        
        all_masks, black_masks = self.get_masks(image)

        mono_boxes = [mono['box'] for mono in obj['monos']]

        black_masks = self.fill_mono_dict(mono_boxes, black_masks)

        average_mono_distance = self.get_average_mono_distance(obj)

        obj, v_count, h_count = self.link_monos(black_masks, obj, average_mono_distance)

        # print("\nmono_dict",obj)

        # for mono_id, mono_details in mono_dict.items():
        #     print("\n-->>",obj['monos'][0])

        # dont need to find root - there is a seperate module that does it.
        # mono_dict = self.find_root(mono_dict, v_count, h_count, origin, orientation_method)

        # printmonodict = "{\n"
        # for key, value in mono_dict.items():
        #     printmonodict+=key+": "
        #     for x in value[1:6]:
        #         printmonodict += str(x) + ", "
        #     printmonodict += "\n"
        # printmonodict += "}"

        # print("\nprintmonodict",printmonodict)

        
        # self.logger.info(f"Monosaccharide Connection Dictionary:\n{printmonodict}")

        return obj


    def fill_mono_dict(self,monos_list,black_masks):
        raise NotImplementedError
    #method to get the average distance betwen monosaccharides
    #requires a partial connection dictionary from fill_mono_dict
    #returns the average distance (number)
    def get_average_mono_distance(self,obj):
        # find median distance between mono default = 100
        average_mono_distance = 100
        # list_center_point = [mono_dict[id_][1] for id_ in mono_dict.keys()]
        list_center_point = [mono['center'] for mono in obj['monos']]
        # print(list_center_point)
        for point in list_center_point:
            length_list = []
            for point2 in list_center_point:
                aux_len = self.length_line(point, point2)
                length_list.append(aux_len)
            length_list.sort()
            length_list = length_list[1:]
            if length_list!=[]:
                average_mono_distance += length_list[0]
        if len(list_center_point)!=0:
            average_mono_distance = average_mono_distance / len(list_center_point)
        return average_mono_distance
    
    
    
    #method to get color-based masks
    #requires the mask dictionary contained in the monosaccharide dictionary returned from the monosaccharideid class
    #returns all masks and black masks, separately
    def get_masks(self, image):

        img = self.smooth_and_blur(image)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        color_range_dict = self.color_range
        hsv = hsv

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # create mask for each color
        yellow_mask = cv2.inRange(hsv, color_range_dict['yellow_lower'], color_range_dict['yellow_upper'])
        purple_mask = cv2.inRange(hsv, color_range_dict['purple_lower'], color_range_dict['purple_upper'])
        red_mask_l = cv2.inRange(hsv, color_range_dict['red_lower_l'], color_range_dict['red_upper_l'])
        red_mask_h = cv2.inRange(hsv, color_range_dict['red_lower_h'], color_range_dict['red_upper_h'])
        red_mask = red_mask_l + red_mask_h
        green_mask = cv2.inRange(hsv, color_range_dict['green_lower'], color_range_dict['green_upper'])
        blue_mask = cv2.inRange(hsv, color_range_dict['blue_lower'], color_range_dict['blue_upper'])
        black_mask = cv2.inRange(hsv, color_range_dict['black_lower'], color_range_dict['black_upper'])

        # store these mask into array
        mask_array = (red_mask, yellow_mask, green_mask, blue_mask, purple_mask, black_mask)
        mask_array_name = ("red_mask", "yellow_mask", "green_mask", "blue_mask", "purple_mask", "black_mask")
        mask_dict = dict(zip(mask_array_name, mask_array))

        all_masks = list(mask_dict.keys())
        #print(all_masks)
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
    
        
    #method to find locations of heuristically identified monosaccharides
    # takes a list of monosaccharides extracted from the monosaccharideid class monosaccharide dictionary
    # takes the black masks from get_masks
    #returns a dictionary of monosaccharide keys; contour, location, and radius values; also returns a black-masked image with monosaccharide areas removed
    def heuristic_mono_finder(self, monos_list, black_masks):
        class_dict = {
            str(0): "GlcNAc",
            str(1): "NeuAc",
            str(2): "Fuc",
            str(3): "Man",
            str(4): "GalNAc",
            str(5): "Gal",
            str(6): "Glc",
            str(7): "NeuGc",
        }
        # mono_dict = {}
        count = 0

        for box in monos_list:
            count += 1
            mono_name = class_dict[str(box.class_)]
            monoID = mono_name + str(count)

            x, y, w, h = box.x, box.y, box.w, box.h
            p1 = (x, y)
            p2 = (x + w, y + h)
        
            # calculate center& radius
            # mo = cv2.moments(box)
            # centerX = int(mo["m10"] / mo["m00"])
            # centerY = int(mo["m01"] / mo["m00"])
            # cir_radius = int(((h ** 2 + w ** 2) ** 0.5) / 2)
            # mono_dict[monoID] = [box, (centerX, centerY), cir_radius, (x, y, w, h)]

            # remove mono
            #cv2.circle(black_masks, (centerX, centerY), int(cir_radius * 0.12) + cir_radius, (0, 0, 0), -1)
            p11 =(int(x*0.985), int(y*0.985 ))
            p22=(int((x + w)*1.015), int((y + h)*1.015))
            cv2.rectangle(black_masks, p11, p22, (0, 0, 0), -1)
            
        return black_masks

        # for i in range(len(monos_list)):

        #     count += 1
        #     monoID = monos_list[i][0] + str(count)
        #     contour = monos_list[i][1]

        #     print("NAME", monoID)

        #     x, y, w, h = cv2.boundingRect(contour)
        #     p1 = (x, y)
        #     p2 = (x + w, y + h)
        #     # cv2.rectangle(origin, p1, p2, (0, 255, 0), 1)

        #     # cv2.putText(origin, monoID[:2] + monoID[-2:], (p1[0] - 5, p1[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
        #     #             (0, 0, 255), thickness=1)

        #     # calculate center& radius
        #     mo = cv2.moments(contour)
        #     centerX = int(mo["m10"] / mo["m00"])
        #     centerY = int(mo["m01"] / mo["m00"])
        #     cir_radius = int(((h ** 2 + w ** 2) ** 0.5) / 2)
        #     mono_dict[monoID] = [contour, (centerX, centerY), cir_radius, (x, y, w, h)]

        #     # cv2.circle(origin, (centerX, centerY), 7, (0, 0, 0),
        #     #            -1)  # img,point, radius,color last value -1 for fill else its thickness
        #     # cv2.circle(black_masks, (centerX, centerY), 7, (255, 0, 255), -1)
        #     #cv2.circle(all_masks_no_black, (centerX, centerY), 7, (255, 0, 255), -1)

        #     # visual only
        #     #cv2.circle(all_masks_no_black, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)

        #     # remove mono
        #     #cv2.circle(black_masks, (centerX, centerY), int(cir_radius * 0.12) + cir_radius, (0, 0, 0), -1)
        #     p11 =(int(x*0.985), int(y*0.985 ))
        #     p22=(int((x + w)*1.015), int((y + h)*1.015))
        #     cv2.rectangle(black_masks, p11, p22, (0, 0, 0), -1)
            
        #     # circle to detect lines
        #     #cv2.circle(visual, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)  # 9 is thickness
        #     #cv2.circle(empty_mask, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)
        # return mono_dict, black_masks        

    #method to detect if two lines intersect
    #takes the endpoints of the lines AB and CD
    #returns true for intersection or false for none
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
    
    #method to link monosaccharides that should be connected
    #takes a binary image with monosaccharides removed
    #takes the started connection dictionary
    #takes the average monosaccharide distance
    #returns the linked connection dictionary, and vertical and horizontal line count
    def link_monos(self, binary_img, obj, avg_mono_distance):
        diff = binary_img
        imheight, imwidth, *channels = diff.shape
        # loop through all mono to find connection
        v_count = 0  # count vertical link vs horizontal
        h_count = 0

        for mono in obj['monos']:
            #print(id)
            boundaries = mono['box']
            
            x, y, w, h = mono['bbox']
            cir_radius = int((((h ** 2 + w ** 2) ** 0.5) / 2) * self.cropfactor)
            centerX,centerY = mono['center']
            # radius = mono[2]
            
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
            
            #cv2.imshow('fig1',crop)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()  

            #crop_origin = ext_origin[centerY - cir_radius:centerY + cir_radius,
                          #centerX - cir_radius:centerX + cir_radius]
            contours_list, _ = cv2.findContours(crop,
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #aux = crop.copy()

            for contour in contours_list:
                #print(contour)
                point_mo = cv2.moments(contour)
                stop=0
                point_centerX2 = 0
                point_centerY2 = 0
                for point in contour:
                    point_centerX2 += point[0][0]
                    point_centerY2 += point[0][1]
                point_centerX2 = int(point_centerX2/len(contour))
                point_centerY2 = int(point_centerY2/len(contour))


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
                    for mono2 in obj['monos']:
                        
                        rectangle = mono2['bbox']
                        
                        # need function to detect first hit
                        

                        # cv2.circle(crop, (Cx, Cy), 4, (0, 0, 255), -1)

                        # cv2.putText(origin, (id[-2:] + id_2[-2:]), (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))

                        line = ((Ax, Ay), (Cx, Cy))
                        if self.interaction_line_rect(line, rectangle) and mono2['id'] != mono['id']:
                            # cv2.line(origin, (Ax, Ay), (Cx, Cy),
                            #              (0, 0, 255), 1, 1, 0)
                            # cv2.circle(origin, (Cx, Cy), 4, (0, 0, 255), -1)
                            obj = self.append_links(obj, mono['id'], mono2['id'])
                            # cv2.putText(origin, (id_[-2:] + id_2[-2:]), (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                            #                 (0, 0, 0))
                            if (abs(Ax - Cx) > abs(Ay - Cy)):
                                    h_count += 1
                            else:
                                    v_count += 1
                            stop=1
                            break
                    if stop ==1:
                        break

                    # DEMO!!! this mess with the crop image use for demo only
                    #cv2.line(aux, (int(crop.shape[0] / 2), int(crop.shape[1] / 2)), (point_centerX2, point_centerY2),
                             #(255, 255, 0), 1, 1, 0)
        for mono in obj['monos']:
            if 'links' not in mono:
                mono['links'] = []

        return obj, v_count, h_count


    def smooth_and_blur(self, img):
    
        img = cv2.GaussianBlur(img, (11, 11), 0)
        # _,img_file=cv2.threshold(img_file,140,255,cv2.THRESH_BINARY)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img 
        



### Subclass of HeuristicConnect; for connecting heuristically-identified monosaccharides    
class OriginalConnector(HeuristicConnector):
    
    def __init__(self, configs,cropfactor=1.5):
        super().__init__(configs)
        self.cropfactor = cropfactor

    #method to start creating the connection dictionary; calls the heuristic_mono_finder method from the superclass HeuristicConnector
    #takes a list of monosaccharides from the dictionary returned by the monosaccharideid class
    #takes the black masks from get_masks
    #returns the new connection dictionary and new black masks
    def fill_mono_dict(self, monos_list, black_masks):
        black_masks = self.heuristic_mono_finder(monos_list, black_masks)
        return black_masks
    
# class to connect monosaccharides found with YOLO models
class ConnectYOLO(HeuristicConnector):
    
    def __init__(self, configs):
        super().__init__(configs)
        self.cropfactor = 1.2
        
    #method to start the connection dictionary
    #takes the monosaccharide list from monosaccharideid class returns
    #takes black_masks from get_masks
    #returns the start of the connection dictionary
    def fill_mono_dict(self, monos_list, black_masks):
        class_dict = {
            str(0): "GlcNAc",
            str(1): "NeuAc",
            str(2): "Fuc",
            str(3): "Man",
            str(4): "GalNAc",
            str(5): "Gal",
            str(6): "Glc",
            str(7): "NeuGc",
            }
        
        # mono_dict = {}
        
        count = 0
        for box in monos_list:
            count += 1
            mono_name = class_dict[str(box.class_)]
            monoID = mono_name + str(count)
            #print("NAME", monoID)
            x, y, w, h = box.x, box.y, box.w, box.h
            p1 = (x, y)
            p2 = (x + w, y + h)
            #cv2.rectangle(origin, p1, p2, (0, 255, 0), 1)

            # cv2.putText(origin, monoID[:2] + monoID[-2:], (p1[0] - 5, p1[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
            #             (0, 0, 255), thickness=1)

            # calculate center& radius
            centerX = box.cen_x
            centerY = box.cen_y

            # radius = int(((h ** 2 + w ** 2) ** 0.5) / 2)
            # mono_dict[monoID] = [box, (centerX, centerY), radius, (x, y, w, h)]
            #print(box, (centerX, centerY), radius, (x, y, w, h))

            # cv2.circle(origin, (centerX, centerY), 7, (0, 0, 0),
            #            -1)  # img,point, radius,color last value -1 for fill else its thickness
            # cv2.circle(black_masks, (centerX, centerY), 7, (255, 0, 255), -1)
            #cv2.circle(all_masks_no_black, (centerX, centerY), 7, (255, 0, 255), -1)

            # visual only
            #cv2.circle(all_masks_no_black, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)

            # remove mono
            #cv2.circle(black_masks, (centerX, centerY), int(cir_radius * 0.12) + cir_radius, (0, 0, 0), -1)
            p11 =(int(x), int(y))
            p22=(int((x + w)), int((y + h)))
            cv2.rectangle(black_masks, p11, p22, (0, 0, 0), -1)
            
            # circle to detect lines
            #cv2.circle(visual, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)  # 9 is thickness
            #cv2.circle(empty_mask, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)
        return black_masks



    #method to define the rectangle of a monosacharide
    #requires the monosaccharide from the connection dictionary, image height and image width
    #returns a rectangle which can be superimposed on the image
    # def get_mono_rect(self, mono, imheight, imwidth):
    #     box = mono[0]
    #     radius = mono[2]
    #     x, y, w, h = mono[3]
    #     #centerX, centerY = mono[1]
    #     #print(centerX, centerY, radius)
    #     if x < 0:
    #         x = 0
    #     if y < 0:
    #         y = 0
    #     x_end = int((x+w)*1.05)
    #     y_end = int((y+h)*1.05)
    #     x = int(x*0.95)
    #     y = int(y*0.95)
    #     w = int(x_end-x)
    #     h = int(y_end-y)

    #     if y+h > imheight:
    #         h = imheight-y
    #     if x+w > imwidth:
    #         w = imwidth-x
    #     mono_rect = (x,y,w,h)
    #     return mono_rect