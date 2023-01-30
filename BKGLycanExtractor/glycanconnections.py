import numpy as np
import cv2
import BKGlycanExtractor.boundingboxes as boundingboxes


#### Class for connecting monosaccharides extracted from an image. all connectors should be subclasses of GlycanConnector
#### all subclasses need method connect which connects the monosaccharides and returns a connection dictionary
class GlycanConnector:
    def __init__(self):
        pass
    def connect(self,**kw):
        raise NotImplementedError
        
#Subclass which uses masking to identify black lines in the figure and uses them to connect monosaccharides
# all subclasses of this class need method fill_mono_dict to start the connection dictionary, and method link_monos to connect monosaccharides
class HeuristicConnector(GlycanConnector):
    
    #expects a monosaccharide connection dictionary and 2 monosaccharides
    #connects the monosaccharides to each other in the connection dictionary and returns it
    def append_links(self, mono_dict, mono1, mono2):
        if len(mono_dict[mono1]) == 4:
            mono_dict[mono1].append([mono2])
        elif len(mono_dict[mono1]) == 5:
            if mono2 not in mono_dict[mono1][4]:
                mono_dict[mono1][4].append(mono2)
        if len(mono_dict[mono2]) == 4:
            mono_dict[mono2].append([mono1])
        elif len(mono_dict[mono2]) == 5:
            if mono1 not in mono_dict[mono2][4]:
                mono_dict[mono2][4].append(mono1)
        return mono_dict
    
    #method to connect a dictionary of monosaccharides returned from a monosaccharideid class
    #requires an image, a monosaccharide dictionary, and an orientation method to be used for finding the root monosaccharide
    # returns a dictionary of connected monosaccharides
    def connect(self, image, monos, orientation_method):

        mask_dict = monos.get("mask_dict",{})
        contours = monos.get("contours",{})
        origin = image
        mono_dict = {}  # mono id = contour, point at center, radius, bounding rect, linkages, root or child
        all_masks, black_masks = self.get_masks(mask_dict)

        mono_dict, black_masks = self.fill_mono_dict(contours, black_masks)

        average_mono_distance = self.get_average_mono_distance(mono_dict)

        mono_dict, v_count, h_count = self.link_monos(black_masks, mono_dict, average_mono_distance)

        mono_dict = self.find_root(mono_dict, v_count, h_count, origin, orientation_method)
        
        #print(mono_dict)
        # DEMO!!!
        # cv2.imshow('e', cv2.resize(origin, None, fx=1, fy=1))
        # cv2.waitKey(0)
        return mono_dict
    def fill_mono_dict(self,monos_list,black_masks):
        raise NotImplementedError
        
    #method to determine glycan orientation and use it to find the root monosaccharide
    #requires a partially filled connection dictionary, vertical and horizontal line counts from link_monos
    #requires a glycan image and an instance of the requested orientation class
    #returns a complete connection dictionary with root monosaccharide chosen
    def find_root(self, mono_dict, v_count, h_count, image, orientation_method):
        ###### find root ##########
        from operator import itemgetter
        aux_list = []
        # mono id = contour, point at center, radius, bounding rect, linkages, root or child
        root = None
        
        #use orientation method to get glycan orientation
        orientation = orientation_method.get_orientation(glycanimage=image, horizontal_count=h_count, vertical_count=v_count)
        #left-right
        if orientation == 0:
            aux_list = sorted([(mono_id, mono_dict[mono_id][1][0]) for mono_id in mono_dict.keys()], key=itemgetter(1),
                              reverse=False)
        #right-left
        elif orientation == 1:
            aux_list = sorted([(mono_id, mono_dict[mono_id][1][0]) for mono_id in mono_dict.keys()], key=itemgetter(1),
                              reverse=True)
        #top-bottom
        elif orientation == 2:
            aux_list = sorted([(mono_id, mono_dict[mono_id][1][1]) for mono_id in mono_dict.keys()], key=itemgetter(1),
                              reverse=False)
        #bottom-top
        elif orientation == 3:
            aux_list = sorted([(mono_id, mono_dict[mono_id][1][1]) for mono_id in mono_dict.keys()], key=itemgetter(1),
                              reverse=True)
        for mono in aux_list:
            if mono[0].find("Fuc") == -1:
                root = mono[0]
                break 
        #print(aux_list)
        #print(f"root = {root}")

        for mono_id in mono_dict.keys():
            #print(mono_id)
            if mono_id == root:
                mono_dict[mono_id].append("root")
            else:
                mono_dict[mono_id].append("child")
            #print(mono_id, mono_dict[mono_id][1:])   
        return mono_dict
    
    #method to get the average distance betwen monosaccharides
    #requires a partial connection dictionary from fill_mono_dict
    #returns the average distance (number)
    def get_average_mono_distance(self,mono_dict):
        # find median distance between mono default = 100
        average_mono_distance = 100
        list_center_point = [mono_dict[id_][1] for id_ in mono_dict.keys()]
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
    
    #method to get the points corresponding to limits of each contour, also count vertical and horizontal lines
    #not in use - meant for the new connector which was worse at its job
    #takes a contour and returns a line corresponding to it, and interim vertical and horizontal line counts
    # def get_contour_limits(self, contour):
    #     int_v_count = 0
    #     int_h_count = 0
    #     #print(contour)
    #     contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)  
    #     if contour_w == 1:
    #         int_v_count += 1
    #     if contour_h == 1:
    #         int_h_count += 1
    #     if len(contour) == 2:
    #         contour_point_1 = contour[0][0]
    #         contour_point_2 = contour[1][0]
    #     else:
    #         for point in contour:
    #             [[point_x,point_y]] = point
    #             if point_x == contour_x:
    #                 contour_point_1 = [point_x,point_y]
    #                 if point_y in range(contour_y - 2, contour_y + 3):
    #                     contour_point_2 = [point_x+contour_w,point_y+contour_h]
    #                     break
    #                 else:
    #                     for point2 in contour:
    #                         [[point2_x,point2_y]] = point2
    #                         if point2_y == contour_y:
    #                             contour_point_2 = [point2_x, point2_y]
    #                             break
    #     line = ((contour_point_1[0], contour_point_1[1]), (contour_point_2[0], contour_point_2[1])) 
    #     return line, int_v_count, int_h_count
    
    #method to get color-based masks
    #requires the mask dictionary contained in the monosaccharide dictionary returned from the monosaccharideid class
    #returns all masks and black masks, separately
    def get_masks(self,mask_dict):
        all_masks = list(mask_dict.keys())
        #print(all_masks)
        all_masks_no_black = all_masks.copy()
        all_masks_no_black.remove("black_mask")

        all_masks_no_black = sum([mask_dict[a] for a in all_masks_no_black])
        all_masks = sum([mask_dict[a] for a in all_masks])
        #cv2.imshow("all",all_masks)
        black_masks = mask_dict["black_mask"]
        # cv2.imshow('image',black_masks)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #visual = black_masks.copy()

        empty_mask = np.zeros([black_masks.shape[0], black_masks.shape[1], 1], dtype=np.uint8)
        empty_mask.fill(0)
        #cv2.imshow("empty",empty_mask)

        #print(len(contours))
        #cv2.imshow('origin', origin)
        # cv2.waitKey(0)
        # print(all_masks)
        all_masks = cv2.cvtColor(all_masks, cv2.COLOR_GRAY2BGR)
        all_masks_no_black = cv2.cvtColor(all_masks_no_black, cv2.COLOR_GRAY2BGR)
        return all_masks, black_masks
        
        
    #method to find locations of heuristically identified monosaccharides
    # takes a list of monosaccharides extracted from the monosaccharideid class monosaccharide dictionary
    # takes the black masks from get_masks
    #returns a dictionary of monosaccharide keys; contour, location, and radius values; also returns a black-masked image with monosaccharide areas removed
    def heuristic_mono_finder(self, monos_list, black_masks):
        mono_dict = {}
        count = 0
        for i in range(len(monos_list)):
            count += 1
            monoID = monos_list[i][0] + str(count)
            contour = monos_list[i][1]
            #print("NAME", monoID)
            x, y, w, h = cv2.boundingRect(contour)
            p1 = (x, y)
            p2 = (x + w, y + h)
            # cv2.rectangle(origin, p1, p2, (0, 255, 0), 1)

            # cv2.putText(origin, monoID[:2] + monoID[-2:], (p1[0] - 5, p1[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
            #             (0, 0, 255), thickness=1)

            # calculate center& radius
            mo = cv2.moments(contour)
            centerX = int(mo["m10"] / mo["m00"])
            centerY = int(mo["m01"] / mo["m00"])
            cir_radius = int(((h ** 2 + w ** 2) ** 0.5) / 2)
            mono_dict[monoID] = [contour, (centerX, centerY), cir_radius, (x, y, w, h)]

            # cv2.circle(origin, (centerX, centerY), 7, (0, 0, 0),
            #            -1)  # img,point, radius,color last value -1 for fill else its thickness
            # cv2.circle(black_masks, (centerX, centerY), 7, (255, 0, 255), -1)
            #cv2.circle(all_masks_no_black, (centerX, centerY), 7, (255, 0, 255), -1)

            # visual only
            #cv2.circle(all_masks_no_black, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)

            # remove mono
            #cv2.circle(black_masks, (centerX, centerY), int(cir_radius * 0.12) + cir_radius, (0, 0, 0), -1)
            p11 =(int(x*0.985), int(y*0.985 ))
            p22=(int((x + w)*1.015), int((y + h)*1.015))
            cv2.rectangle(black_masks, p11, p22, (0, 0, 0), -1)
            
            # circle to detect lines
            #cv2.circle(visual, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)  # 9 is thickness
            #cv2.circle(empty_mask, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)
        return mono_dict, black_masks        

    #method to detect if two lines intersect
    #takes the endpoints of the lines AB and CD
    #returns true for intersection or false for none
    def interaction_line_line(self, A, B, C, D):
        Ax, Ay, Bx, By, Cx, Cy, Dx, Dy = A[0], A[1], B[0], B[1], C[0], C[1], D[0], D[1]
        # function determine whereas AB intersect with CD

        if ((Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay)) != 0:  # line is horrizontal or vertical
            cont1 = ((Dx - Cx) * (Ay - Cy) - (Dy - Cy) * (Ax - Cx)) / ((Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay))
            cont2 = ((Bx - Ax) * (Ay - Cy) - (By - Ay) * (Ax - Cx)) / ((Dy - Cy) * (Bx - Ax) - (Dx - Cx) * (By - Ay))
            if (0 <= cont1 <= 1 and 0 <= cont2 <= 1):
                # intersec_X = Ax + (cont1 * (Bx - Ax))
                # intersec_Y = Ay + (cont1 * (By - Ay))
                # print(intersec_X, intersec_Y)
                return True
        return False
    #method to detect if a line and a rectangle intersect
    #returns true for intersection and false for none
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
    
    #method to calculate the length of a line
    def length_line(self,A, B):
        Ax, Ay, Bx, By = A[0], A[1], B[0], B[1]
        l = ((Ax - Bx) ** 2 + (By - Ay) ** 2) ** 0.5
        return l
    
    def link_monos(self, binary_img, mono_dict, avg_mono_distance):
        raise NotImplementedError
        
    ### unused - intended to be a new linkage method but was worse than the original
    # def new_linker(self, binary_img, mono_dict, avg_mono_distance):
    #     diff = binary_img
    #     imheight, imwidth, *channels = diff.shape
        
    #     # loop through all connecting lines to find monos
    #     v_count = 0  # count vertical link vs horizontal
    #     h_count = 0
    #     contours_list, _ = cv2.findContours(diff,
    #                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(all_masks,contours_list,-1, (0,255,0), 10)
    #     # cv2.imshow('empty',all_masks)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()        
    #     for contour in contours_list:
    #         line, interim_v_count, interim_h_count = self.get_contour_limits(contour)
    #         v_count += interim_v_count
    #         h_count += interim_h_count
                        
    #         for id_,mono_dict_values in mono_dict.items():
    #             #print(id)
    #             mono_rect = self.get_mono_rect(mono_dict_values,imheight,imwidth)
    #             #print(id_,mono_rect)
    #             if self.interaction_line_rect(line, mono_rect):
    #                 #print("hello")
    #                 for id_2,mono_dict_values2 in mono_dict.items():
    #                     if id_2 == id_:
    #                         continue
    #                     mono_rect2 = self.get_mono_rect(mono_dict_values2, imheight,imwidth)

    #                     if self.interaction_line_rect(line, mono_rect2):
                            
    #                         mono_dict = self.append_links(mono_dict, id_, id_2)

    #     for i, mono in mono_dict.items():
    #         if len(mono) == 4:
    #             mono.append([])
    #         #print(i,mono[4])
    #     #print(f"horizontal:{h_count}\nvertical:{v_count}")
    #     return mono_dict, v_count, h_count
    # def original_linker(self,binary_img, mono_dict, avg_mono_distance):
    #     diff = binary_img
    #     imheight, imwidth, *channels = diff.shape
    #     # loop through all mono to find connection
    #     v_count = 0  # count vertical link vs horizontal
    #     h_count = 0
    #     for id_, mono in mono_dict.items():
    #         #print(id)
    #         boundaries = mono[0]
            
    #         x, y, w, h = mono[3]
    #         cir_radius = int((((h ** 2 + w ** 2) ** 0.5) / 2) * 1.5)
    #         (centerX,centerY) = mono[1]
    #         radius = mono[2]
            
    #         y1 = centerY - cir_radius
    #         y2 = centerY + cir_radius
    #         x1 = centerX - cir_radius
    #         x2 = centerX + cir_radius
    #         if y1 < 0:
    #             y1 = 0
    #         if x1 < 0:
    #             x1 = 0
    #         if y2 > imheight:
    #             y2 = imheight
    #         if x2 > imwidth:
    #             x2 = imwidth
    #         crop = diff[y1:y2,
    #                x1:x2]

    #         #crop_origin = ext_origin[centerY - cir_radius:centerY + cir_radius,
    #                       #centerX - cir_radius:centerX + cir_radius]
    #         contours_list, _ = cv2.findContours(crop,
    #                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         #aux = crop.copy()

    #         for contour in contours_list:
    #             #print(contour)
    #             point_mo = cv2.moments(contour)
    #             stop=0
    #             point_centerX2 = 0
    #             point_centerY2 = 0
    #             for point in contour:
    #                 point_centerX2 += point[0][0]
    #                 point_centerY2 += point[0][1]
    #             point_centerX2 = int(point_centerX2/len(contour))
    #             point_centerY2 = int(point_centerY2/len(contour))


    #             Ax = centerX
    #             Ay = centerY

    #             Bx = centerX - cir_radius + point_centerX2
    #             By = centerY - cir_radius + point_centerY2
    #             #################### length adjustable
    #             for i in range(1, 200, 5):
    #                 i = i / 100
    #                 length = avg_mono_distance * i
    #                 lenAB = ((Ax - Bx) ** 2 + (Ay - By) ** 2) ** 0.5
    #                 if lenAB==0:
    #                     lenAB=1
    #                 Cx = int(Bx + (Bx - Ax) / lenAB * length)
    #                 Cy = int(By + (By - Ay) / lenAB * length)
    #                 for id_2 in mono_dict.keys():

    #                     rectangle = mono_dict[id_2][3]

    #                     # need function to detect first hit


    #                     # cv2.circle(crop, (Cx, Cy), 4, (0, 0, 255), -1)

    #                     # cv2.putText(origin, (id[-2:] + id_2[-2:]), (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))

    #                     line = ((Ax, Ay), (Cx, Cy))
    #                     if self.interaction_line_rect(line, rectangle) and id_2 != id_:
    #                         # cv2.line(origin, (Ax, Ay), (Cx, Cy),
    #                         #              (0, 0, 255), 1, 1, 0)
    #                         # cv2.circle(origin, (Cx, Cy), 4, (0, 0, 255), -1)
    #                         mono_dict = self.append_links(mono_dict, id_, id_2)
    #                         # cv2.putText(origin, (id_[-2:] + id_2[-2:]), (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
    #                         #                 (0, 0, 0))
    #                         if (abs(Ax - Cx) > abs(Ay - Cy)):
    #                                 h_count += 1
    #                         else:
    #                                 v_count += 1
    #                         stop=1
    #                         break
    #                 if stop ==1:
    #                     break

    #                 # DEMO!!! this mess with the crop image use for demo only
    #                 #cv2.line(aux, (int(crop.shape[0] / 2), int(crop.shape[1] / 2)), (point_centerX2, point_centerY2),
    #                          #(255, 255, 0), 1, 1, 0)
    #     for i, mono in mono_dict.items():
    #         if len(mono) == 4:
    #             mono.append([])
    #     return mono_dict, v_count, h_count 


### Subclass of HeuristicConnect; for connecting heuristically-identified monosaccharides    
class OriginalConnector(HeuristicConnector):
    
    #method to start creating the connection dictionary; calls the heuristic_mono_finder method from the superclass HeuristicConnector
    #takes a list of monosaccharides from the dictionary returned by the monosaccharideid class
    #takes the black masks from get_masks
    #returns the new connection dictionary and new black masks
    def fill_mono_dict(self, monos_list, black_masks):
        mono_dict, black_masks = self.heuristic_mono_finder(monos_list, black_masks)
        return mono_dict, black_masks
    
    #method to connect monosaccharides
    #takes a binary image, started connection dictionary from fill_mono_dict, and average monosaccharide distance
    #returns dictionary with connections
    #returns vertical and horizontal line count
    def link_monos(self, binary_img, mono_dict, avg_mono_distance):
        diff = binary_img
        imheight, imwidth, *channels = diff.shape
        # loop through all mono to find connection
        v_count = 0  # count vertical link vs horizontal
        h_count = 0
        for id_, mono in mono_dict.items():
            #print(id)
            boundaries = mono[0]
            
            x, y, w, h = mono[3]
            cir_radius = int((((h ** 2 + w ** 2) ** 0.5) / 2) * 1.5)
            (centerX,centerY) = mono[1]
            radius = mono[2]
            
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
                    for id_2 in mono_dict.keys():

                        rectangle = mono_dict[id_2][3]

                        # need function to detect first hit


                        # cv2.circle(crop, (Cx, Cy), 4, (0, 0, 255), -1)

                        # cv2.putText(origin, (id[-2:] + id_2[-2:]), (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))

                        line = ((Ax, Ay), (Cx, Cy))
                        if self.interaction_line_rect(line, rectangle) and id_2 != id_:
                            # cv2.line(origin, (Ax, Ay), (Cx, Cy),
                            #              (0, 0, 255), 1, 1, 0)
                            # cv2.circle(origin, (Cx, Cy), 4, (0, 0, 255), -1)
                            mono_dict = self.append_links(mono_dict, id_, id_2)
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
        for i, mono in mono_dict.items():
            if len(mono) == 4:
                mono.append([])
        return mono_dict, v_count, h_count 
        return mono_dict, v_count, h_count


# class to connect monosaccharides found with YOLO models
class ConnectYOLO(HeuristicConnector):

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
        
        mono_dict = {}
        
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

            radius = int(((h ** 2 + w ** 2) ** 0.5) / 2)
            mono_dict[monoID] = [box, (centerX, centerY), radius, (x, y, w, h)]
            #print(box, (centerX, centerY), radius, (x, y, w, h))

            # cv2.circle(origin, (centerX, centerY), 7, (0, 0, 0),
            #            -1)  # img,point, radius,color last value -1 for fill else its thickness
            # cv2.circle(black_masks, (centerX, centerY), 7, (255, 0, 255), -1)
            #cv2.circle(all_masks_no_black, (centerX, centerY), 7, (255, 0, 255), -1)

            # visual only
            #cv2.circle(all_masks_no_black, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)

            # remove mono
            #cv2.circle(black_masks, (centerX, centerY), int(cir_radius * 0.12) + cir_radius, (0, 0, 0), -1)
            p11 =(int(x*0.985), int(y*0.985 ))
            p22=(int((x + w)*1.015), int((y + h)*1.015))
            cv2.rectangle(black_masks, p11, p22, (0, 0, 0), -1)
            
            # circle to detect lines
            #cv2.circle(visual, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)  # 9 is thickness
            #cv2.circle(empty_mask, (centerX, centerY), int(cir_radius * 0.13) + cir_radius, (255, 0, 255), 6)
        return mono_dict, black_masks

    #method to define the rectangle of a monosacharide
    #requires the monosaccharide from the connection dictionary, image height and image width
    #returns a rectangle which can be superimposed on the image
    def get_mono_rect(self, mono, imheight, imwidth):
        box = mono[0]
        radius = mono[2]
        x, y, w, h = mono[3]
        #centerX, centerY = mono[1]
        #print(centerX, centerY, radius)
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        x_end = int((x+w)*1.05)
        y_end = int((y+h)*1.05)
        x = int(x*0.95)
        y = int(y*0.95)
        w = int(x_end-x)
        h = int(y_end-y)

        if y+h > imheight:
            h = imheight-y
        if x+w > imwidth:
            w = imwidth-x
        mono_rect = (x,y,w,h)
        return mono_rect
    
    #method to link monosaccharides that should be connected
    #takes a binary image with monosaccharides removed
    #takes the started connection dictionary
    #takes the average monosaccharide distance
    #returns the linked connection dictionary, and vertical and horizontal line count
    def link_monos(self, binary_img, mono_dict, avg_mono_distance):
        diff = binary_img
        imheight, imwidth, *channels = diff.shape
        # loop through all mono to find connection
        v_count = 0  # count vertical link vs horizontal
        h_count = 0
        for id_, mono in mono_dict.items():
            #print(id)
            boundaries = mono[0]
            
            x, y, w, h = mono[3]
            cir_radius = int((((h ** 2 + w ** 2) ** 0.5) / 2) * 1.5)
            (centerX,centerY) = mono[1]
            radius = mono[2]
            
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

                Bx = centerX - radius + point_centerX2
                By = centerY - radius + point_centerY2
                #################### length adjustable
                for i in range(1, 200, 5):
                    i = i / 100
                    length = avg_mono_distance * i
                    lenAB = ((Ax - Bx) ** 2 + (Ay - By) ** 2) ** 0.5
                    if lenAB==0:
                        lenAB=1
                    Cx = int(Bx + (Bx - Ax) / lenAB * length)
                    Cy = int(By + (By - Ay) / lenAB * length)
                    for id_2 in mono_dict.keys():

                        rectangle = mono_dict[id_2][3]

                        # need function to detect first hit


                        # cv2.circle(crop, (Cx, Cy), 4, (0, 0, 255), -1)

                        # cv2.putText(origin, (id[-2:] + id_2[-2:]), (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))

                        line = ((Ax, Ay), (Cx, Cy))
                        if self.interaction_line_rect(line, rectangle) and id_2 != id_:
                            # cv2.line(origin, (Ax, Ay), (Cx, Cy),
                            #              (0, 0, 255), 1, 1, 0)
                            # cv2.circle(origin, (Cx, Cy), 4, (0, 0, 255), -1)
                            mono_dict = self.append_links(mono_dict, id_, id_2)
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
        for i, mono in mono_dict.items():
            if len(mono) == 4:
                mono.append([])
        return mono_dict, v_count, h_count   

    
### unused - intended for new connection method which was worse
# class NewConnector(HeuristicConnector):
#     def fill_mono_dict(self, monos_list, black_masks):
#         mono_dict, black_masks = self.heuristic_mono_finder(monos_list, black_masks)
#         return mono_dict, black_masks  
#     def get_mono_rect(self, mono, imheight, imwidth):
#         contour = mono[0]
#         mo = cv2.moments(contour)
#         cir_radius = mono[2]
#         centerX = int(mo["m10"] / mo["m00"])
#         centerY = int(mo["m01"] / mo["m00"])
        
#         y1 = centerY - cir_radius
#         y2 = centerY + cir_radius
#         x1 = centerX - cir_radius
#         x2 = centerX + cir_radius
        
#         y1 = int(y1*0.95)
#         y2 = int(y2*1.05)
#         x1 = int(x1*0.95)
#         x2 = int(x2*1.05)
        
#         if y1 < 0:
#             y1 = 0
#         if x1 < 0:
#             x1 = 0
#         if y2 > imheight:
#             y2 = imheight
#         if x2 > imwidth:
#             x2 = imwidth
        
#         w = x2-x1
#         h = y2-y1
#         mono_rect = (x1, y1, w, h)
#         return mono_rect 
#     def link_monos(self, binary_img, mono_dict, avg_mono_distance):
#         mono_dict, v_count, h_count = self.new_linker(binary_img, mono_dict, avg_mono_distance)
#         return mono_dict, v_count, h_count         

### also unused
# class OldConnectionYOLO(ConnectYOLO):
#     def link_monos(self, binary_img, mono_dict, avg_mono_distance):
#         mono_dict, v_count, h_count = self.original_linker(binary_img, mono_dict, avg_mono_distance)
#         return mono_dict, v_count, h_count 
