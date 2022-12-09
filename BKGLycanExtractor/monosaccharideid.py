import cv2, math, logging
import numpy as np

import BKGLycanExtractor.boundingboxes as boundingboxes

logger = logging.getLogger("search")

class MonoID:
    def __init__(self, configs):
        #read in color ranges for mono id
        color_range = configs.get("colors_range",)
        color_range_file = open(color_range)
        color_range_dict = {}
        for line in color_range_file.readlines():
            line = line.strip()
            name = line.split("=")[0].strip()
            color_range = line.split("=")[1].strip()
            color_range_dict[name] = np.array(list(map(int, color_range.split(","))))
        color_range_file.close()
        self.color_range = color_range_dict
    def compstr(self,counts):
        s = ""
        for sym,count in sorted(counts.items()):
            if count > 0:
                s += "%s(%d)"%(sym,count)
        return s
    def find_monos(self,image = None, **kw):
        raise NotImplementedError
    def format_monos(self,image = None, **kw):
        raise NotImplementedError
    def get_masks(self,hsv_image=None):
        color_range_dict = self.color_range
        hsv = hsv_image
        # color_range_file = open(colors_range)
        # color_range_dict = {}
        # for line in color_range_file.readlines():
        #     line = line.strip()
        #     name = line.split("=")[0].strip()
        #     color_range = line.split("=")[1].strip()
        #     color_range_dict[name] = np.array(list(map(int, color_range.split(","))))

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
        return mask_array,mask_array_name,mask_dict        
class HeuristicMonos(MonoID):
    def __init__(self, configs, box = False):
        super().__init__(configs)
        self.box_flag = box
    def compare2img(self,img1,img2):
        # return similarity between two image
        if img1.shape == img2.shape:
            # print("samesize")
            # print(img1.shape)
            pass
        else:
            # print("img1,img2,not same size")
            # print(img1.shape)
            # print(img2.shape)
            return -1
        score = 0
        diff = cv2.absdiff(img1, img2)
        r, g, b = cv2.split(diff)
        score = cv2.countNonZero(g) / (img1.shape[0] * img1.shape[1])

        # cv2.imshow("different", diff)
        # cv2.waitKey(0)
        return 1 - score
    
    def crop_largest(self,image = None):
        img = image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
        contours_list, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print(contours_list)
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
    def find_monos(self,image = None):
        monos = {}
        image = self.crop_largest(image)
        image = self.resize_image(image)
        monos["original"] = image
        img = self.format_image(image)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_width = img.shape[1]
        img_height = img.shape[0]

        # read color range in config folder
        #origin = img.copy()
        final = img.copy()  # final annotated pieces

        #d = {}
        mask_array, mask_array_name, mask_dict = self.get_masks(hsv)

        # loop through each countors
        boxes = []
        return_contours = []
        for color in mask_array_name:
            if color == "black_mask":
                continue
            contours_list, _ = cv2.findContours(mask_dict[color],
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_list:

                #approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                squareness = abs(math.log(float(w)/float(h),2))
                arearatio = 1e6*float(area)/(img_height*img_width)
                arearatio1 = 1000*area/float(w*h)
                if squareness < 2 and arearatio > 100 and arearatio1 > 200:
                    #logger.info(f"{x,y,w,h,area,round(squareness,2),round(arearatio,2),round(arearatio1,2),color} ")
                    if squareness > 0.25 or arearatio < 1000.0 or arearatio1 < 500:
                        #logger.info("BAD")
                        continue
                    box = boundingboxes.Training(img,x = x, y = y, width = w, height = h)
                    box.corner_to_center()
                    box.abs_to_rel()
                    if color == "red_mask":
                        return_contours.append(("Fuc", contour))
                        box.set_class(2)
                        boxes.append(box)

                    elif color == "purple_mask":
                        return_contours.append(("NeuAc", contour))
                        box.set_class(1)
                        boxes.append(box)
                    elif color == "blue_mask":
                        white = np.zeros([h, w, 3], dtype=np.uint8)
                        white.fill(255)
                        this_blue_img = mask_dict["blue_mask"][y:y + h, x:x + w]
                        this_blue_img = cv2.cvtColor(this_blue_img, cv2.COLOR_GRAY2BGR)
                        score = self.compare2img(white, this_blue_img)
                        if score >= 0.8:  # is square
                            return_contours.append(("GlcNAc", contour))
                            box.set_class(0)
                            boxes.append(box)
                        elif 0.5 < score < 0.8:

                            return_contours.append(("Glc", contour))
                            box.set_class(6)
                            boxes.append(box)

                        else:
                            logger.info("???")

                    elif color == "green_mask":

                        return_contours.append(("Man", contour))
                        box.set_class(3)
                        boxes.append(box)

                    elif color == "yellow_mask":

                        #yellows_contours.append(contour)
                        white = np.zeros([h, w, 3], dtype=np.uint8)
                        white.fill(255)
                        this_yellow_img = mask_dict["yellow_mask"][y:y + h, x:x + w]
                        # this_yellow_img = cv2.resize(this_yellow_img, None, fx=1, fy=1)
                        this_yellow_img = cv2.cvtColor(this_yellow_img, cv2.COLOR_GRAY2BGR)

                        score = self.compare2img(white, this_yellow_img)
                        if score > 0.9:  # is square

                            return_contours.append(("GalNAc", contour))
                            box.set_class(4)
                            boxes.append(box)

                        elif 0.5 < score < 0.9:

                            return_contours.append(("Gal", contour))
                            box.set_class(5)
                            boxes.append(box)
                        else:

                            logger.info("???",score)

        monos["annotated"] = final
        monos["mask_dict"] = mask_dict
        if self.box_flag:
            monos["contours"] = boxes
        else:
            monos["contours"] = return_contours
        return monos
    def format_image(self,img = None):

        img = cv2.GaussianBlur(img, (11, 11), 0)
        # _,img_file=cv2.threshold(img_file,140,255,cv2.THRESH_BINARY)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img
    def format_monos(self,image= None,monos_dict = None):
        monos = monos_dict
        monos_list = monos["contours"]
        final = monos["annotated"]

        # loop through each countors
        monoCount_dict = {"GlcNAc": 0, "NeuAc": 0, "Fuc": 0, "Man": 0, "GalNAc": 0, "Gal": 0,"Glc": 0, "NeuGc": 0,}

        for monopair in monos_list:
            mono = monopair[0]
            contour = monopair[1]

            approx = cv2.approxPolyDP(contour, 0.035 * cv2.arcLength(contour, True), True)
            x, y, w, h = cv2.boundingRect(contour)
            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(final, p1, p2, (0, 255, 0), 1)
            cv2.drawContours(final, [approx], 0, (0, 0, 255), 1)
            cv2.putText(final, mono, (approx.ravel()[0], approx.ravel()[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        (0, 0, 255))
            monoCount_dict[mono] += 1

        monos["count_dict"] = monoCount_dict
        monos["annotated"] = final
        return monos
    def resize_image(self, img = None):
        # print(img_file.shape[0]*img_file.shape[1])
        bigwhite = np.zeros([img.shape[0] + 30, img.shape[1] + 30, 3], dtype=np.uint8)
        bigwhite.fill(255)
        bigwhite[15:15 + img.shape[0], 15:15 + img.shape[1]] = img
        img = bigwhite.copy()
        mag = 84000 / (img.shape[0] * img.shape[1])
        # print(mag)
        if mag <= 1:
            mag = 1
        img = cv2.resize(img, None, fx=mag, fy=mag)
        return img

class BoxMonos(HeuristicMonos):
    def __init__(self, configs):
        super().__init__(configs, box = True) 
    def format_monos(self, image= None, monos_dict = None):
        monos = monos_dict
        monos_boxes = monos["contours"]
        monoCount_dict = {"GlcNAc": 0, "NeuAc": 0, "Fuc": 0, "Man": 0, "GalNAc": 0, "Gal": 0,"Glc": 0, "NeuGc": 0,}
        
        img = self.crop_largest(image)
        #final_boxes = []
        final = img.copy()
        for mono in monos_boxes:
            # new_x = mono.rel_cen_x
            # new_y = mono.rel_cen_y
            # new_w = mono.rel_w
            # new_h = mono.rel_h
            new_class = mono.class_
            
            # finalbox = boundingboxes.Training(final,class_ = new_class, rel_cen_x = new_x, rel_cen_y = new_y, rel_w = new_w, rel_h = new_h, white_space = 30)
            
            # finalbox.rel_to_abs()
            
            # finalbox.fix_image()
            
            # finalbox.center_to_corner()

            mono.pad_borders()


            mono.fix_borders()
            mono.corner_to_center()
            mono.abs_to_rel()
            # final_boxes.append(finalbox)
            
            # p1 = (finalbox.x,finalbox.y)
            # p2 = (finalbox.x2,finalbox.y2)            
            # cv2.rectangle(final, p1, p2, (0, 255, 0), 1)
            # cv2.putText(final, new_class, p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
            #             (0, 0, 255))
            if new_class == 0:
                monoCount_dict["GlcNAc"] += 1
            elif new_class == 1:
                monoCount_dict["NeuAc"] += 1
            elif new_class == 2:
                monoCount_dict["Fuc"] += 1    
            elif new_class == 3:
                monoCount_dict["Man"] += 1 
            elif new_class == 4:
                monoCount_dict["GalNAc"] += 1 
            elif new_class == 5:
                monoCount_dict["Gal"] += 1  
            elif new_class == 6:
                monoCount_dict["Glc"] += 1                 
        monos["count_dict"] = monoCount_dict
        monos["annotated"] = final
        monos["contours"] = monos_boxes
        
        return monos
class YOLOMonos(MonoID):
    def __init__(self,configs):
        super().__init__(configs)
        weights=configs.get("monoid_weights",)
        net=configs.get("monoid_cfg",)
        
        self.net = cv2.dnn.readNet(weights,net)
        
        layer_names = self.net.getLayerNames()
        #print(layer_names)
        #compatibility with new opencv versions
        #print(self.net.getUnconnectedOutLayers())
        try:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            #print(self.output_layers)
        except IndexError:
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        

    def find_monos(self,image=None, threshold = 0.5):
        #extract location of all glycan from image
        monos = {}
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_array, mask_array_name, mask_dict = self.get_masks(hsv)
        monos["mask_dict"] = mask_dict
        
        origin_image = image.copy()
        monos["original"] = origin_image
        # cv2.imshow('image',self.origin_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        height, width, channels = image.shape
        ############################################################################################
        #fix issue with
        ############################################################################################
        white_space = 200
        bigwhite = np.zeros([image.shape[0] +white_space, image.shape[1] +white_space, 3], dtype=np.uint8)
        bigwhite.fill(255)
        half_white_space = int(white_space/2)
        bigwhite[half_white_space:(half_white_space + image.shape[0]), half_white_space:(half_white_space+image.shape[1])] = image
        image = bigwhite.copy()
        #detected_glycan = image.copy()
        # cv2.imshow("bigwhite", bigwhite)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ############################################################################################
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        #print(outs)
        # loop through results and print them on images
        class_ids = []
        confidences = []
        monos_list = []
        for out in outs:
            #print(out)
            for detection in out:
                #print(detection)
                scores = detection[5:]
                #print(scores)
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                box = boundingboxes.Detected(origin_image, confidence, class_=class_id,white_space=white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])

                box.rel_to_abs()
                
                box.fix_image()
                
                box.center_to_corner()
                
                box.fix_borders()
                
                box.to_four_corners()

    #BOXES FORMAT IS A PROBLEM
                monos_list.append(box)

                confidences.append(float(confidence))
                class_ids.append(class_id)
        #cv.dnn.NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]])
        #print(boxes)
        boxesfornms = [bbox.to_list() for bbox in monos_list]
        #print(unpaddedboxesfornms)
        
        indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, threshold, 0.4)
        #print(unpadded_indexes)
        
        indexes = [index[0] for index in indexes]

        #print(f"\nGlycan detected: {len(boxes)}")
        #cv2.imshow("Image", detected_glycan)
        #cv2.waitKey(0)
        monos_list = [monos_list[i] for i in indexes]
        confidences = [confidences[i] for i in indexes]
        class_ids = [class_ids[i] for i in indexes]
        
        monos["contours"] = monos_list
        return monos
    
    def format_monos(self, image = None, monos_dict = None,conf_threshold = 0.0):
        monos_list = monos_dict["contours"]
        
        for mono in monos_list:
            if mono.confidence < conf_threshold:
                monos_list.remove(mono)
                
        monos_dict["contours"] = monos_list

        # read color range in config folder
        #origin = img.copy()
        final = image.copy()  # final annotated pieces



               # loop through each countors
        monoCount_dict = {"GlcNAc": 0, "NeuAc": 0, "Fuc": 0, "Man": 0, "GalNAc": 0, "Gal": 0,"Glc": 0, "NeuGc": 0}
        for mono in monos_list:
            #print(mono)
            class_id = mono.class_
            p1 = (mono.x,mono.y)
            p2 = (mono.x2,mono.y2)
            if class_id == 0:

                cv2.rectangle(final,p1,p2,(255,0,0),3)
                cv2.putText(final, "GlcNAc", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 0, 0))
                monoCount_dict["GlcNAc"] += 1
                logger.info("GlcNAc")
            elif class_id == 1:
                cv2.rectangle(final,p1,p2,(128,0,128),3)
                cv2.putText(final, "NeuAc", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (128, 0, 128))
                monoCount_dict["NeuAc"] += 1
                logger.info("NeuAc")
            elif class_id == 2:
                cv2.rectangle(final,p1,p2,(0,0,255),3)
                cv2.putText(final, "Fuc", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 0, 255))
                monoCount_dict["Fuc"] += 1
                logger.info("Fuc")
            elif class_id == 3:
                cv2.rectangle(final,p1,p2,(0,255,0),3)
                cv2.putText(final, "Man", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 255, 0))
                monoCount_dict["Man"] += 1
                logger.info("Man")
            elif class_id == 4:
                cv2.rectangle(final,p1,p2,(0,255,255),3)
                cv2.putText(final, "GalNAc", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 255, 255))
                monoCount_dict["GalNAc"] += 1
                logger.info("GalNAc")
            elif class_id == 5:
                cv2.rectangle(final,p1,p2,(0,255,255),3)
                cv2.putText(final, "Gal", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (0, 255, 255))
                monoCount_dict["Gal"] += 1
                logger.info("Gal")
            elif class_id == 6:
                cv2.rectangle(final,p1,p2,(255,0,0),3)
                cv2.putText(final, "Glc", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255,0,0))
                monoCount_dict["Glc"] += 1
                logger.info("Glc")
            elif class_id == 7:
                cv2.rectangle(final,p1,p2,(255,0,0),3)
                cv2.putText(final, "NeuGc", p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255,0,0))
                monoCount_dict["NeuGc"] += 1
                logger.info("NeuGc")


               #     pass
               # print("herte",yellows_contours)
               # cv2.imshow("yellow_mask",all_mask)
               # cv2.imshow("final", final)
               # cv2.waitKey(0)
        monos_dict["count_dict"] = monoCount_dict
        monos_dict["annotated"] = final

        return monos_dict