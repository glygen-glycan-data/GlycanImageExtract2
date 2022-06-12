import numpy as np
import cv2

class GlycanConnector:
    def __init__(self):
        pass
    def connect(self,**kw):
        raise NotImplementedError
class HeuristicConnector(GlycanConnector):
    def connect(self,image = None,monos = None):
        mask_dict = monos.get("mask_dict",{})
        contours = monos.get("contours",{})
        origin = image
        mono_dict = {}  # mono id = contour, point at center, radius, bounding rect, linkages, root or child
        all_masks = list(mask_dict.keys())
        #print(all_masks)
        all_masks_no_black = all_masks.copy()
        all_masks_no_black.remove("black_mask")

        all_masks_no_black = sum([mask_dict[a] for a in all_masks_no_black])
        all_masks = sum([mask_dict[a] for a in all_masks])
        # cv2.imshow("all",cv2.resize(all_masks, None,fx=1,fy=1))
        black_masks = mask_dict["black_mask"]
        #visual = black_masks.copy()

        empty_mask = np.zeros([black_masks.shape[0], black_masks.shape[1], 1], dtype=np.uint8)
        empty_mask.fill(0)

        #print(len(contours))
        #cv2.imshow('origin', origin)
        # cv2.waitKey(0)
        # print(all_masks)

        all_masks = cv2.cvtColor(all_masks, cv2.COLOR_GRAY2BGR)
        all_masks_no_black = cv2.cvtColor(all_masks_no_black, cv2.COLOR_GRAY2BGR)
        # black_masks=cv2.cvtColor(black_masks, cv2.COLOR_GRAY2BGR)

        count = 0
        for i in range(len(contours)):
            count += 1
            monoID = contours[i][0] + str(count)
            contour = contours[i][1]
            #print("NAME", monoID)
            x, y, w, h = cv2.boundingRect(contour)
            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(origin, p1, p2, (0, 255, 0), 1)

            cv2.putText(origin, monoID[:2] + monoID[-2:], (p1[0] - 5, p1[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                        (0, 0, 255), thickness=1)

            # calculate center& radius
            mo = cv2.moments(contour)
            centerX = int(mo["m10"] / mo["m00"])
            centerY = int(mo["m01"] / mo["m00"])
            cir_radius = int(((h ** 2 + w ** 2) ** 0.5) / 2)
            mono_dict[monoID] = [contour, (centerX, centerY), cir_radius, (x, y, w, h)]

            cv2.circle(origin, (centerX, centerY), 7, (0, 0, 0),
                       -1)  # img,point, radius,color last value -1 for fill else its thickness
            cv2.circle(black_masks, (centerX, centerY), 7, (255, 0, 255), -1)
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
        #ext_origin = origin.copy()
        diff =black_masks
        #diff = cv2.bitwise_and(black_masks, empty_mask)
        # diff=cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        # DEMO!!!
        # cv2.imshow('a', cv2.resize(origin, None, fx=1, fy=1))
        #cv2.imshow('b', cv2.resize(all_masks_no_black, None, fx=1, fy=1))
        #cv2.imshow('c', cv2.resize(black_masks, None, fx=1, fy=1))
        # cv2.imshow('d', cv2.resize(empty_mask, None, fx=1, fy=1))

        # cv2.imshow('e', cv2.resize(diff, None, fx=1, fy=1))
        # cv2.imshow('visual', cv2.resize(visual, None, fx=1, fy=1))
        #cv2.waitKey(0)

        # find median distance between mono default = 100
        average_mono_distance = 100
        list_center_point = [mono_dict[id][1] for id in mono_dict.keys()]
        # print(list_center_point)
        for point in list_center_point:
            length_list = []
            for point2 in list_center_point:
                aux_len = self.lengthLine(point, point2)
                length_list.append(aux_len)
            length_list.sort()
            length_list = length_list[1:]
            if length_list!=[]:
                average_mono_distance += length_list[0]
        if len(list_center_point)!=0:
            average_mono_distance = average_mono_distance / len(list_center_point)

        # loop through all mono to find connection
        v_count = 0  # count vertical link vs horizontal
        h_count = 0
        for id in mono_dict.keys():
            #print(id)
            contour = mono_dict[id][0]
            mo = cv2.moments(contour)
            x, y, w, h = cv2.boundingRect(contour)
            cir_radius = int((((h ** 2 + w ** 2) ** 0.5) / 2) * 1.5)
            centerX = int(mo["m10"] / mo["m00"])
            centerY = int(mo["m01"] / mo["m00"])

            crop = diff[centerY - cir_radius:centerY + cir_radius,
                   centerX - cir_radius:centerX + cir_radius]
            #crop_origin = ext_origin[centerY - cir_radius:centerY + cir_radius,
                          #centerX - cir_radius:centerX + cir_radius]
            contours_list, _ = cv2.findContours(crop,
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            linked_monos = []
            #aux = crop.copy()

            for contour in contours_list:
                point_mo = cv2.moments(contour)

                if point_mo["m00"] != 0:
                    stop=0
                    point_centerX2 = int(point_mo["m10"] / (point_mo["m00"]))
                    point_centerY2 = int(point_mo["m01"] / (point_mo["m00"]))

                    Ax = centerX
                    Ay = centerY

                    Bx = centerX - cir_radius + point_centerX2
                    By = centerY - cir_radius + point_centerY2
                    #################### length adjustable
                    for i in range(1, 200, 5):
                        i = i / 100
                        length = average_mono_distance * i
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
                            if self.interactionLineRect(line, rectangle) and id_2 != id:
                                cv2.line(origin, (Ax, Ay), (Cx, Cy),
                                             (0, 0, 255), 1, 1, 0)
                                cv2.circle(origin, (Cx, Cy), 4, (0, 0, 255), -1)
                                linked_monos.append(id_2)
                                cv2.putText(origin, (id[-2:] + id_2[-2:]), (Cx, Cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5,
                                                (0, 0, 0))
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
            # DEMO!!!
            #cv2.imshow('visual', aux)
            #cv2.imshow('visual2', crop_origin)
            #cv2.waitKey(0)
            mono_dict[id].append(linked_monos)
            #print(mono_dict[id])

            #print(linked_monos)
        #print(f"horizontal:{h_count}\nvertical:{v_count}")

        ###### find root ##########
        from operator import itemgetter
        aux_list = []
        # mono id = contour, point at center, radius, bounding rect, linkages, root or child
        root = None
        if h_count > v_count:
            aux_list = sorted([(mono_id, mono_dict[mono_id][1][0]) for mono_id in mono_dict.keys()], key=itemgetter(1),
                              reverse=True)
            for mono in aux_list:
                if mono[0].find("Fuc") == -1:
                    root = mono[0]
                    break
        else:
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
        #print(mono_dict)
        # DEMO!!!
        # cv2.imshow('e', cv2.resize(origin, None, fx=1, fy=1))
        # cv2.waitKey(0)
        return mono_dict
    
    def interactionLineLine(self, A, B, C, D):
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
    def interactionLineRect(self,line, rect):
        # line two points
        A, B = line[0], line[1]
        # rect x,y,w,h
        x, y, w, h = rect
        top = ((x, y), (x + w, y))
        bottom = ((x, y + h), (x + w, y + h))
        right = ((x + w, y), (x + w, y + h))
        left = ((x, y), (x, y + h))
        if self.interactionLineLine(A, B, top[0], top[1]) or self.interactionLineLine(A, B, bottom[0],
                                                                            bottom[1]) or self.interactionLineLine(A, B,
                                                                                                              right[0],
                                                                                                              right[
                                                                                                                  1]) or self.interactionLineLine(
            A, B, left[0], left[1]):
            return True
        return False
    
    def lengthLine(self, A, B):
        Ax, Ay, Bx, By = A[0], A[1], B[0], B[1]
        l = ((Ax - Bx) ** 2 + (By - Ay) ** 2) ** 0.5
        return l
    