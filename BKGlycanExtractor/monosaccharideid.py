import os
import cv2, math, logging
import numpy as np

import BKGlycanExtractor.boundingboxes as boundingboxes


### base class, all classes to find and identify monosaccharides should be subclasses of MonoID
### all subclasses expect to be initialised with a file describing the range of colors which define yellow/purple/redl/redh/red/green/blue/black
### base class has a compstr method to create a formatted string of monosaccharide composition
### all subclasses should have a find_monos method which is followed by a format_monos method to return a formatted dictionary of monosaccharids, color masks, and composition
### base class has a get_masks method to use the color ranges to create cv2 masks and save them into an array and dictionary
class MonoID:
    def __init__(self, configs):
        #read in color ranges for mono id
        color_range = configs.get("color_range",)
        color_range_file = open(color_range)
        color_range_dict = {}
        for line in color_range_file.readlines():
            line = line.strip()
            name = line.split("=")[0].strip()
            color_range = line.split("=")[1].strip()
            color_range_dict[name] = np.array(list(map(int, color_range.split(","))))
        color_range_file.close()
        self.color_range = color_range_dict
    #expects a dictionary of monosaccharide counts, returns a formatted string
    def compstr(self,counts):
        s = ""
        for sym,count in sorted(counts.items()):
            if count > 0:
                s += "%s(%d)"%(sym,count)
        return s
    #expects an image and leaves just the largest continuous contour, removing extraneous aspects of the image
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
    def find_monos(self,image = None, **kw):
        raise NotImplementedError
    def format_monos(self,image = None, **kw):
        raise NotImplementedError
    #expects an hsv-formatted image, returns masks based on the image and color ranges
    def get_masks(self,hsv_image=None):
        color_range_dict = self.color_range
        hsv = hsv_image

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
    #resizes image for optimised monosaccharide id
    #takes an image, returns it resized
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
    
# class to use heuristic color matching methods to find and identify monosaccharides in glycan images
# expects to be initiated with a configs dictionary containing the color range file
# the box flag should not be used directly - box=True when you want to return bounding boxes based on the contours of the monosaccharides
#   in this case you should be using the BoxMonos subclass which sets box=True itself
#   this can be used for creating training data but should not be used in a glycan extraction pipeline
#find_monos should be used before format_monos, which takes the value returned from find_monos
class HeuristicMonos(MonoID):
    def __init__(self, configs, box = False):
        super().__init__(configs)
        self.box_flag = box
    #expects 2 images, returns a similarity score for comparison
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

    
    #method to find monosaccharides in a glycan image
    #requires an image and the name of the parent logger
    #returns a dictionary containing a mask dictionary, monosaccharide boundaries and identifications, and the cropped image with monosaccharides annotated
    def find_monos(self,image = None, logger_name='', **kw):
        logger = logging.getLogger(logger_name+'.monosaccharideid')
        monos = {}
        if not self.box_flag:
            image = self.crop_largest(image)
            image = self.resize_image(image)
        #save original image, and then format it for masking
        monos["original"] = image
        img = self.format_image(image)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_width = img.shape[1]
        img_height = img.shape[0]

        # read color range in config folder
        #origin = img.copy()
        final = img.copy()  # final annotated pieces

        mask_array, mask_array_name, mask_dict = self.get_masks(hsv)

        # loop through each countors
        # sets up contours and bounding boxes, chooses at the end based on the box_flag which should be set automatically by class choice
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
                        logger.info("BAD")
                        continue
                    box = boundingboxes.Training(img,x = x, y = y, width = w, height = h)
                    box.corner_to_center()
                    box.abs_to_rel()
                    if color == "red_mask":
                        return_contours.append(("Fuc", contour))
                        logger.info("Fuc")
                        box.set_class(2)
                        boxes.append(box)

                    elif color == "purple_mask":
                        return_contours.append(("NeuAc", contour))
                        logger.info("NeuAc")
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
                            logger.info("GlcNAc")
                            box.set_class(0)
                            boxes.append(box)
                        elif 0.5 < score < 0.8: # is circle

                            return_contours.append(("Glc", contour))
                            logger.info("Glc")
                            box.set_class(6)
                            boxes.append(box)

                        else:
                            logger.info("???", score)

                    elif color == "green_mask":

                        return_contours.append(("Man", contour))
                        logger.info("Man")
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
                            logger.info("GalNAc")
                            box.set_class(4)
                            boxes.append(box)

                        elif 0.5 < score < 0.9: # is circle

                            return_contours.append(("Gal", contour))
                            logger.info("Gal")
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
    
    #formats image for mask usage
    #expects an image, returns a blurred and filtered image
    def format_image(self,img = None):

        img = cv2.GaussianBlur(img, (11, 11), 0)
        # _,img_file=cv2.threshold(img_file,140,255,cv2.THRESH_BINARY)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        return img
    
    #method to format the monosaccharide contours returned by find_monos
    #requires an image and the dictionary returned by find_monos
    #returns a formatted dictionary with monosaccharide counts and an annotated final image
    def format_monos(self, monos_dict = None, **kw):
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


#Subclass of HeuristicMonos which creates bounding boxes for monosaccharides found through heuristic image-matching methods
#Can be used to create training data for YOLO models
#expects to be initialised with a configs dictionary containig the color range file
#uses the HeuristicMonos find_monos method, has its own format_monos method which overrides the superclass method
class BoxMonos(HeuristicMonos):
    def __init__(self, configs):
        super().__init__(configs, box = True) 
    
    #format the monosaccharide dictionary returned by find_monos by counting monosaccharides
    #requires an image and the find_monos output
    #returns the formatted monosaccharide dictionary
    def format_monos(self, monos_dict = None, image= None, **kw):
        monos = monos_dict
        monos_boxes = monos["contours"]
        monoCount_dict = {"GlcNAc": 0, "NeuAc": 0, "Fuc": 0, "Man": 0, "GalNAc": 0, "Gal": 0,"Glc": 0, "NeuGc": 0,}
        
        #img = self.crop_largest(image)
        #final_boxes = []
        final = image.copy()
        for mono in monos_boxes:
            new_class = mono.class_
            

            mono.pad_borders()


            mono.fix_borders()
            mono.corner_to_center()
            mono.abs_to_rel()


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
    
### Subclass to use a trained YOLO / darknet model to find and classify monosaccharides
#expects to be initiated with a configs dictionary containing the color_range file and the weights and .cfg file for the YOLO model
class YOLOMonos(MonoID):
    def __init__(self,configs, **kw):
        super().__init__(configs)
        weights=configs.get("weights",)
        net=configs.get("config",)
        if not os.path.isfile(weights):
            raise FileNotFoundError()
        if not os.path.isfile(net):
            raise FileNotFoundError()
        
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
        
    # method to use the YOLO model to find monosaccharides
    # expects an image and a confidence threshold below which monosaccharides are thrown out and not returned; defaults to threshold of 0.5
    #returns a dictionary with monosaccharide bounding boxes, a mask dictionary, and the original image
    def find_monos(self,image=None, threshold = 0.5, logger_name='', **kw):
        logger = logging.getLogger(logger_name+'.monosaccharideid')
        monos = {}
        
        image = self.crop_largest(image)
        image = self.resize_image(image)
        #save original image, and then format it for masking
        monos["original"] = image
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_array, mask_array_name, mask_dict = self.get_masks(hsv)
        monos["mask_dict"] = mask_dict
        
        origin_image = image.copy()
        
        #use the YOLO model to get bounding boxes

        image, white_space = self.prep_YOLO_input(image)

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
                
                scores[class_id] = 0.
                class_options = {str(class_id): confidence}
                
                while np.any(scores):
                    #print(scores)
                    classopt = np.argmax(scores)
                    confopt = scores[classopt]
                    scores[classopt] = 0.
                    class_options[str(classopt)] = confopt
                
                if len(class_options) > 1:
                    message = "WARNING: More than one class possible: "+str(class_options)
                    print(message)
                    logger.warning(message)
                #print(class_options)
                box = boundingboxes.Detected(origin_image, confidence, class_=class_id,white_space=white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3], class_options = class_options)

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
        try:
            indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, threshold, 0.4)
        except TypeError:
            boxesfornms = [bbox.to_new_list() for bbox in monos_list]
            indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, threshold, 0.4)
        try:
            indexes = [index[0] for index in indexes]
        except IndexError:
            pass

        #print(f"\nGlycan detected: {len(boxes)}")
        #cv2.imshow("Image", detected_glycan)
        #cv2.waitKey(0)
        monos_list = [monos_list[i] for i in indexes]
        confidences = [confidences[i] for i in indexes]
        class_ids = [class_ids[i] for i in indexes]
        
        monos["contours"] = monos_list
        return monos
    
    #method to format the monosaccharide dictionary and annotate the final image, to be used after find_monos
    #requires an image, the dictionary returned by find_monos, a confidence threshold for class identification (separate from the find_monos confidence threshold)
    #requires the parent logger name to be passed
    #returns the formatted dictionary with annotated image
    def format_monos(self, monos_dict = None,conf_threshold = 0.0, logger_name='', **kw):
        logger = logging.getLogger(logger_name+'.monosaccharideid')
        monos_list = monos_dict["contours"]
        
        for mono in monos_list:
            if mono.confidence < conf_threshold:
                monos_list.remove(mono)
                
        monos_dict["contours"] = monos_list

        # read color range in config folder
        #origin = img.copy()
        image = monos_dict["original"]
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
    
    #prepare the input image for YOLO - add whitespace to borders.
    #takes the image, returns image with whitespace borders
    def prep_YOLO_input(self, image):
        
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
        return image, white_space
