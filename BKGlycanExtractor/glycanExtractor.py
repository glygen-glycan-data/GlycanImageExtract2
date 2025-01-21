import cv2, sys, os
import numpy as np
import time
from .pygly.MonoFactory import MonoFactory
from .pygly.Glycan import Glycan
from .pygly.Monosaccharide import Linkage
from .pygly.Monosaccharide import Anomer
#import win32api,win32con,win32gui
import cv2
#import mss
import numpy as np
import PIL,PIL.Image
import time
import math


def reorientedGlycan(img):
    new_img = img.copy()
    return new_img


def recordoperation(x):
    pass



def compare2img(img1, img2):
    # return similarity between two image
    if img1.shape == img2.shape:
        # print("samesize")
        # print(img1.shape)
        pass
    else:
        print("img1,img2,not same size")
        print(img1.shape)
        print(img2.shape)
        return -1
    score = 0
    diff = cv2.absdiff(img1, img2)
    r, g, b = cv2.split(diff)
    score = cv2.countNonZero(g) / (img1.shape[0] * img1.shape[1])

    # cv2.imshow("different", diff)
    # cv2.waitKey(0)
    return 1 - score

def croplargest(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    contours_list, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    area_list = []
    for i, contour in enumerate(contours_list):
        area = cv2.contourArea(contour)
        area_list.append((area, i))
    (_, largest_index) = max(area_list)
    out = np.zeros_like(img)
    cv2.drawContours(out, contours_list, largest_index, (255, 255, 255), -1)
    _, out = cv2.threshold(out, 230, 255, cv2.THRESH_BINARY_INV)

    out2 = cv2.bitwise_or(out, img)
    return out2

def countcolors(img_file,base_configs,log=None):

    return_dict = {}
    # process image input
    img_file = croplargest(img_file)

    # print(img_file.shape[0]*img_file.shape[1])
    bigwhite = np.zeros([img_file.shape[0] + 30, img_file.shape[1] + 30, 3], dtype=np.uint8)
    bigwhite.fill(255)
    bigwhite[15:15 + img_file.shape[0], 15:15 + img_file.shape[1]] = img_file
    img_file = bigwhite.copy()



    mag = 84000 / (img_file.shape[0] * img_file.shape[1])
    # print(mag)
    if mag <= 1:
        mag = 1
    img_file = cv2.resize(img_file, None, fx=mag, fy=mag)
    img_file = cv2.GaussianBlur(img_file, (11, 11), 0)
    # _,img_file=cv2.threshold(img_file,140,255,cv2.THRESH_BINARY)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img_file = cv2.filter2D(img_file, -1, kernel)
    hsv = cv2.cvtColor(img_file, cv2.COLOR_BGR2HSV)
    img_file_width = img_file.shape[0]
    img_file_height = img_file.shape[1]

    # read color range in config folder
    origin = img_file.copy()
    final = img_file.copy()  # final annotated pieces

    d = {}
    colors_range = base_configs + "colors_range.txt"
    
    color_range_file = open(colors_range)
    color_range_dict = {}
    for line in color_range_file.readlines():
        line = line.strip()
        name = line.split("=")[0].strip()
        color_range = line.split("=")[1].strip()
        color_range_dict[name] = np.array(list(map(int, color_range.split(","))))

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
    # all_mask = sum(mask_array)

    # loop through each countors
    monoCount_dict = {"GlcNAc": 0, "NeuAc": 0, "Fuc": 0, "Man": 0, "GalNAc": 0, "Gal": 0,"Glc": 0, }
    yellows_contours = []
    return_contours = []
    for color in mask_array_name:
        if color == "black_mask":
            continue
        contours_list, _ = cv2.findContours(mask_dict[color],
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        for contour in contours_list:

            approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            squareness = abs(math.log(float(w)/float(h),2))
            arearatio = 1e6*float(area)/(img_file_height*img_file_width)
            arearatio1 = 1000*area/float(w*h)
            if squareness < 2 and arearatio > 100 and arearatio1 > 200:
                if log:
                    print(x,y,w,h,area,round(squareness,2),round(arearatio,2),round(arearatio1,2),color,file=log,end=" ")
                if squareness > 0.25 or arearatio < 1000.0 or arearatio1 < 500:
                    if log:
                        print("BAD",file=log)
                    continue
                p1 = (x, y)
                p2 = (x + w, y + h)
                contours.append((p1, p2))
                cv2.rectangle(final, p1, p2, (0, 255, 0), 1)
                cv2.drawContours(final, [approx], 0, (0, 0, 255), 1)

                if color == "red_mask":
                    cv2.putText(final, "Fuc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                    monoCount_dict["Fuc"] += 1
                    return_contours.append(("Fuc", contour))
                    if log:
                        print("Fuc",file=log)

                elif color == "purple_mask":
                    cv2.putText(final, "NeuAc", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1,
                                (0, 0, 255))
                    monoCount_dict["NeuAc"] += 1
                    return_contours.append(("NeuAc", contour))
                    if log:
                        print("NeuAc",file=log)

                elif color == "blue_mask":
                    white = np.zeros([h, w, 3], dtype=np.uint8)
                    white.fill(255)
                    this_blue_img = blue_mask[y:y + h, x:x + w]
                    this_blue_img = cv2.cvtColor(this_blue_img, cv2.COLOR_GRAY2BGR)
                    score = compare2img(white, this_blue_img)
                    if score >= 0.8:  # is square
                        cv2.putText(final, "GlcNAc", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        monoCount_dict["GlcNAc"] += 1
                        return_contours.append(("GlcNAc", contour))
                        if log:
                            print("GlcNAc",file=log)

                    elif 0.5 < score < 0.8:
                        cv2.putText(final, "Glc", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        monoCount_dict["Glc"] += 1
                        return_contours.append(("Glc", contour))
                        if log:
                            print("Glc",file=log)

                    else:
                        cv2.putText(final, "?", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        if log:
                            print("???",file=log)
                    #cv2.imshow("blue_mask",this_blue_img)
                    #cv2.imshow("origin_mask", origin[y:y + h, x:x + w])
                    #cv2.waitKey(0)
                elif color == "green_mask":
                    cv2.putText(final, "Man", (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (0, 0, 255))
                    monoCount_dict["Man"] += 1
                    return_contours.append(("Man", contour))
                    if log:
                        print("Man",file=log)

                elif color == "yellow_mask":

                    yellows_contours.append(contour)
                    white = np.zeros([h, w, 3], dtype=np.uint8)
                    white.fill(255)
                    this_yellow_img = yellow_mask[y:y + h, x:x + w]
                    # this_yellow_img = cv2.resize(this_yellow_img, None, fx=1, fy=1)
                    this_yellow_img = cv2.cvtColor(this_yellow_img, cv2.COLOR_GRAY2BGR)

                    score = compare2img(white, this_yellow_img)
                    if score > 0.9:  # is square
                        cv2.putText(final, "GalNAc", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        monoCount_dict["GalNAc"] += 1
                        return_contours.append(("GalNAc", contour))
                        if log:
                            print("GalNAc",file=log)

                    elif 0.5 < score < 0.9:
                        cv2.putText(final, "Gal", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        monoCount_dict["Gal"] += 1
                        return_contours.append(("Gal", contour))
                        if log:
                            print("Gal",file=log)

                    else:
                        cv2.putText(final, "?", (approx.ravel()[0], approx.ravel()[1]),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                    (0, 0, 255))
                        if log:
                            print("???",score,file=log)

    #     pass
    # print("herte",yellows_contours)
    # cv2.imshow("yellow_mask",all_mask)
    # cv2.imshow("final", final)
    # cv2.waitKey(0)
    return monoCount_dict, final, origin, mask_dict, return_contours


def lengthLine(A, B):
    Ax, Ay, Bx, By = A[0], A[1], B[0], B[1]
    l = ((Ax - Bx) ** 2 + (By - Ay) ** 2) ** 0.5
    return l


def interactionLineRect(line, rect):
    # line two points
    A, B = line[0], line[1]
    # rect x,y,w,h
    x, y, w, h = rect
    top = ((x, y), (x + w, y))
    bottom = ((x, y + h), (x + w, y + h))
    right = ((x + w, y), (x + w, y + h))
    left = ((x, y), (x, y + h))
    if interactionLineLine(A, B, top[0], top[1]) or interactionLineLine(A, B, bottom[0],
                                                                        bottom[1]) or interactionLineLine(A, B,
                                                                                                          right[0],
                                                                                                          right[
                                                                                                              1]) or interactionLineLine(
        A, B, left[0], left[1]):
        return True
    return False


def interactionLineLine(A, B, C, D):
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


def extractGlycanTopology(mask_dict, contours, origin):
    # declare variables
    mono_dict = {}  # mono id = contour, point at center, radius, bounding rect, linkages, root or child
    all_masks = list(mask_dict.keys())
    all_masks_no_black = all_masks.copy()
    all_masks_no_black.remove("black_mask")

    all_masks_no_black = sum([mask_dict[a] for a in all_masks_no_black])
    all_masks = sum([mask_dict[a] for a in all_masks])
    # cv2.imshow("all",cv2.resize(all_masks, None,fx=1,fy=1))
    black_masks = mask_dict["black_mask"]
    visual = black_masks.copy()

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
    ext_origin = origin.copy()
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
            aux_len = lengthLine(point, point2)
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
        contour = mono_dict[id][0]
        mo = cv2.moments(contour)
        x, y, w, h = cv2.boundingRect(contour)
        cir_radius = int((((h ** 2 + w ** 2) ** 0.5) / 2) * 1.5)
        centerX = int(mo["m10"] / mo["m00"])
        centerY = int(mo["m01"] / mo["m00"])

        crop = diff[centerY - cir_radius:centerY + cir_radius,
               centerX - cir_radius:centerX + cir_radius]
        crop_origin = ext_origin[centerY - cir_radius:centerY + cir_radius,
                      centerX - cir_radius:centerX + cir_radius]
        contours_list, _ = cv2.findContours(crop,
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        linked_monos = []
        aux = crop.copy()

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
                        if interactionLineRect(line, rectangle) and id_2 != id:
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
                cv2.line(aux, (int(crop.shape[0] / 2), int(crop.shape[1] / 2)), (point_centerX2, point_centerY2),
                         (255, 255, 0), 1, 1, 0)
        # DEMO!!!
        #cv2.imshow('visual', aux)
        #cv2.imshow('visual2', crop_origin)
        #cv2.waitKey(0)
        mono_dict[id].append(linked_monos)

        #print(linked_monos)
    #print(f"horizontal:{h_count}\nvertical:{v_count}")

    ###### find root ##########
    from operator import itemgetter, attrgetter
    aux_list = []
    # mono id = contour, point at center, radius, bounding rect, linkages, root or child
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
        if mono_id == root:
            mono_dict[mono_id].append("root")
        else:
            mono_dict[mono_id].append("child")
        #print(mono_id, mono_dict[mono_id][1:])
    # print(mono_dict)
    # DEMO!!!
    # cv2.imshow('e', cv2.resize(origin, None, fx=1, fy=1))
    # cv2.waitKey(0)
    return mono_dict, origin, ext_origin


def buildglycan(mono_dict):
    # print("MONO_DICT",mono_dict)
    # mono id = contour, point at center, radius, bounding rect, linkages, root or child

    backup_dict = mono_dict.copy()

    mf = MonoFactory()

    # Identify the root
    root_id = None
    for mono_id, details in mono_dict.items():
        if isinstance(mono_id, int) and details["classid"] == 0:  # Root condition
            root_id = mono_id
            break

    if root_id is None:
        raise ValueError("No root found in the mono_dict.")

    # aux_list = [(mono_id, mono_dict[mono_id][4:5]) for mono_id in mono_dict.keys()]

    # root_id = [mono_id for mono_id in mono_dict.keys() if mono_dict[mono_id][5] == "root"][0]
    #print("aux_list", aux_list)

    # Create the root node
    root_symbol = mono_dict[root_id]["symbol"]
    print(f"root id: {root_id}, type: {root_symbol}, child = {mono_dict[root_id]['links']}")
    root_node = mf.new(root_symbol)

    # root_type = "".join([c for c in root_id if c.isalpha()])
    # #print(f"root id: {root_id}, type: {root_type}, child = {mono_dict[root_id][4]}")
    # root_node = mf.new(root_type)

    # child_list = mono_dict[root_id][4]
    #print("##########################")
    # need stop recursion here #####################
    fail_safe = 0

    print("BEFORE", root_symbol)
    print("root_node",root_node)

    build_tree = buildtree(mono_dict, root_id, root_node, fail_safe)
    print("build_tree",build_tree)
    root_node = build_tree[2]
    #unknonw root properties

    if root_node != None:
        #print(root_node)
        root_node.set_anomer(Anomer.missing)
        #root_node.set_ring_start(None)
        #root_node.set_ring_end(None)
        g = Glycan(root_node)
        glycoCT =g.glycoct()
    elif root_node ==None:
        #print("Error in glycan structure")
        glycoCT = None

    return glycoCT


# def buildglycan(mono_dict):
#     print("MONO_SICT",mono_dict)
#     # mono id = contour, point at center, radius, bounding rect, linkages, root or child

#     backup_dict = mono_dict.copy()

#     mf = MonoFactory()
#     aux_list = [(mono_id, mono_dict[mono_id][4:5]) for mono_id in mono_dict.keys()]

#     root_id = [mono_id for mono_id in mono_dict.keys() if mono_dict[mono_id][5] == "root"][0]
#     #print("aux_list", aux_list)

#     root_type = "".join([c for c in root_id if c.isalpha()])
#     #print(f"root id: {root_id}, type: {root_type}, child = {mono_dict[root_id][4]}")
#     root_node = mf.new(root_type)

#     child_list = mono_dict[root_id][4]
#     #print("##########################")
#     # need stop recursion here #####################
#     fail_safe=0
#     root_node=buildtree(mono_dict, root_id,root_node,fail_safe)[2]
#     #unknonw root properties

#     if root_node != None:
#         #print(root_node)
#         root_node.set_anomer(Anomer.missing)
#         #root_node.set_ring_start(None)
#         #root_node.set_ring_end(None)
#         g = Glycan(root_node)
#         glycoCT =g.glycoct()
#     elif root_node ==None:
#         #print("Error in glycan structure")
#         glycoCT = None

#     return glycoCT


def buildtree(mono_dict, root, root_node, fail_safe):
    # mono_dict[mono id] = {contour, point at center, radius, bounding rect, linkages, root or child}
    # variables:
    fail_safe += 1
    #print(f"Current:{fail_safe} " + root, end=" ")
    mf = MonoFactory()
    # current_type = "".join([c for c in mono_dict[root]['symbol'] if c.isalpha()])
    # child_list = list(set(mono_dict[root][4]))

    # child_list consists a list if link_id's
    # child_list = mono_dict[root]['links']
    child_list = [link_id for link_id, conf in mono_dict[root]['links']]
    # print("Child_list", child_list)

#     ---->>>child_list ['GlcNAc7']
#     ------>>>>>>child_id GlcNAc7

    # case 0: no child at all return build mono
    if fail_safe > len(mono_dict.values())-1:
        return None, None, None, fail_safe
        
    if child_list == []:
        # child_mono = mf.new("".join([c for c in mono_dict[root]['symbol'] if c.isalpha()]))
        child_mono = mf.new(mono_dict[root]['symbol'])
        child_mono.set_anomer(Anomer.missing)
        #print(f"adding leaves: {root}")
        root_node.add_child(child_mono,parent_type=Linkage.oxygenPreserved,child_pos=1,
                  child_type=Linkage.oxygenLost)

        return mono_dict, root, root_node, fail_safe

    # case 1: there are child, do for loop
    if child_list != []:
        print("child_list",child_list, root)

        for child_id in child_list:
            # remove parent link from child
            if root in child_list:
                mono_dict[child_id]['links'] = [
                    nested_list for nested_list in mono_dict[child_id]['links']
                    if not (isinstance(nested_list, list) and len(nested_list) > 0 and nested_list[0] == root)
                ]

            # name_temp = "".join([c for c in mono_dict[child_id]['symbol'] if c.isalpha()])
            name_temp = mono_dict[child_id]['symbol']
            child_mono = mf.new(name_temp)
            child_mono.set_anomer(Anomer.missing)

            if mono_dict[child_id]['links'] != []:
                _ ,_ ,child_mono, fail_safe = buildtree(mono_dict, child_id, child_mono, fail_safe)

            if fail_safe > len(mono_dict.values())-1 or child_mono == None or root_node == None:
                return None, None, None, fail_safe

            if name_temp in ("NeuAc"):      #("Glc", "Gal", "GlcNAc"):
                root_node.add_child(child_mono, parent_type=Linkage.oxygenPreserved, child_pos=2,
                      child_type=Linkage.oxygenLost)
            else:
                root_node.add_child(child_mono, parent_type=Linkage.oxygenPreserved, child_pos=1,
                                    child_type=Linkage.oxygenLost)

    if fail_safe > len(mono_dict.values())-1 or child_mono == None or root_node == None:
        print("fail safe activated")
        return None, None, None, fail_safe
    else:
        return mono_dict, root, root_node, fail_safe


# path = sys.argv[1]
# img_file = cv2.imread(path)
# countmono(path)
# print(path)
# capturescreen()
# monocount_dict = countcolors(img_file)
# print(monocount_dict)
'''
def screenshot(bbox):
    if bbox==None:
        bbox = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        #bbox =(0,0,1920,1080)
    else:
        pass

    screen = mss.mss().grab(bbox) #take screen shot
    #conver tto opencv
    screen=np.array(PIL.Image.frombytes('RGB', screen.size, screen.bgra, 'raw', 'BGRX')).reshape(screen.size[1],screen.size[0],3)
    screen=cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    return screen
'''
'''
if __name__ == '__main__':
    #problematic case G38637SA
    #path = sys.argv[1]
    path="000001.png"
    path="3 - Copy.png"
    path="7.png"
    path="G38637SA.png"
    path = "G53916LT.png"
    #path="42.png"
    img_file = cv2.imread(path)
    monoCount_dict, final, origin, mask_dict, return_contours = countcolors(img_file)
    mono_dict, a, b = extractGlycanTopology(mask_dict, return_contours, origin)
    for mono_id in mono_dict.keys():
        print(mono_id, mono_dict[mono_id][4])
    print(a.shape)
    cv2.imshow('a', cv2.resize(a, None, fx=1, fy=1))
    cv2.waitKey(0)
    cv2.imshow('b', cv2.resize(b, None, fx=2, fy=2))
    cv2.waitKey(0)
    print("Condensed GlycoCT:\n", buildglycan(mono_dict))
    #screen = screenshot({'top': 300, 'left': 300, 'width': 900, 'height': 900})
    bbox={'top': 300, 'left': 300, 'width': 600, 'height': 600}

    last_time=time.time()
    while(1):
        last_time = time.time()
        screen = mss.mss().grab(bbox) #take screen shot
        #conver tto opencv
        screen=np.array(PIL.Image.frombytes('RGB', screen.size, screen.bgra, 'raw', 'BGRX')).reshape(screen.size[1],screen.size[0],3)
        screen=cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        print(f'FPS: {(1 / float(format(time.time() - last_time)))} ')
        cv2.imshow('screenshot', screen)
        try:
            monoCount_dict, final, origin, mask_dict, return_contours = countcolors(screen)
            mono_dict, a, b = extractGlycanTopology(mask_dict, return_contours, origin)
            cv2.imshow('a', cv2.resize(a, None, fx=1, fy=1))
        except (ValueError,ZeroDivisionError,UnboundLocalError,IndexError,TypeError):
            cv2.putText(screen, "Glycan not found", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 0, 255),2)
            cv2.imshow('a', cv2.resize(screen, None, fx=1, fy=1))

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
'''
