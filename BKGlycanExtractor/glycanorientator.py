import os
import cv2
import numpy as np
import BKGlycanExtractor.boundingboxes as boundingboxes

import logging

#### base class, all orientation methods should be subclasses of the GlycanOrientator
#### all orientation methods should have method get_orientation which returns an orientation 0 (left-right), 1 (right-left), 2 (top-bottom), 3 (bottom-top)
#### potential to add other orientations (diagonal?) with numbers >3
class GlycanOrientator:
    def __init__(self,**kw):
        pass
    def get_orientation(self, **kw):
        raise NotImplementedError

#Default class for orientation determination. takes counts of horizontal and vertical connecting lines and determines right-left(horizontal) or bottom-top(vertical)
class DefaultOrientator(GlycanOrientator):
    def __init__(self, **kw):
        super().__init__()
    #expects keywords horizontal_count and vertical_count, returns number representing orientation
    def get_orientation(self, **kw):
        horiz_count = kw.get("horizontal_count",0)
        vert_count = kw.get("vertical_count",0)
        #assumes right-left
        if horiz_count > vert_count:
            return 1, None
        #assumes bottom-top
        elif vert_count > horiz_count:
            return 3, None
        #assumes right-left?
        else:
            return 1, None
        
#### the YOLO Orientator expects to be initiated with the weights and cfg file for the YOLO orientation model
class YOLOOrientator(GlycanOrientator):
    def __init__(self,configs):
        weights = configs.get("weights",None)
        net = configs.get("config",None)
        
        if not os.path.isfile(weights):
            raise FileNotFoundError()
        if not os.path.isfile(net):
            raise FileNotFoundError()
        
        self.net = cv2.dnn.readNet(weights,net)
        
        layer_names = self.net.getLayerNames()
        #compatibility with new opencv versions
        try:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
            #print(self.output_layers)
        except IndexError:
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
    #expects kw glycanimage for the image to be passd through the trained YOLO model
    #image should contain a single glycan
    #returns the class (orientation) with highest confidence of the best bounding box extracted from the image
    def get_orientation(self,**kw):
        glycanimage=kw.get("glycanimage",None)
        origin_image = glycanimage.copy()
        height, width, channels = glycanimage.shape
        ############################################################################################
        #fix issue with
        ############################################################################################
        white_space = 200
        bigwhite = np.zeros([glycanimage.shape[0] +white_space, glycanimage.shape[1] +white_space, 3], dtype=np.uint8)
        bigwhite.fill(255)
        half_white_space = int(white_space/2)
        bigwhite[half_white_space:(half_white_space + glycanimage.shape[0]), half_white_space:(half_white_space+glycanimage.shape[1])] = glycanimage
        glycanimage = bigwhite.copy()
        #detected_glycan = image.copy()
        # cv2.imshow("bigwhite", bigwhite)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ############################################################################################
        blob = cv2.dnn.blobFromImage(glycanimage, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        #print(outs)
        # loop through results and print them on images
        oriented_glycan_list = []
        confidences = []
        for out in outs:
            #print(out)
            for detection in out:
                #assign best-match class to the bounding box
                #print(detection)
                scores = detection[5:]
                #print(scores)
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                
                oriented_box = boundingboxes.Detected(origin_image, confidence, class_=class_id,white_space=white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])

                oriented_box.rel_to_abs()
                
                oriented_box.fix_image()
                
                oriented_box.center_to_corner()
                
                oriented_box.fix_borders()
                
                oriented_box.to_four_corners()

    #BOXES FORMAT IS A PROBLEM
                oriented_glycan_list.append(oriented_box)
                confidences.append(confidence)
        if oriented_glycan_list == []:
            return None, None
        else:
            #cv.dnn.NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]])
            #print(boxes)
            boxesfornms = [bbox.to_list() for bbox in oriented_glycan_list]
            #print(unpaddedboxesfornms)
            try:
                indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, 0.0, 0.4)
            except TypeError:
                boxesfornms = [bbox.to_new_list() for bbox in oriented_glycan_list]
                indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, 0.0, 0.4)
            try:
                indexes = [index[0] for index in indexes]
            except IndexError:
                pass

            #print(f"\nGlycan detected: {len(boxes)}")
            #cv2.imshow("Image", detected_glycan)
            #cv2.waitKey(0)
            oriented_glycans = [oriented_glycan_list[i] for i in indexes]
            confidences = [confidences[i] for i in indexes]
            try:
                best_index = np.argmax(confidences)
            except ValueError:
                return None, None
            #get the bounding box with highest confidence, return its class
            oriented_glycan = oriented_glycans[best_index]
            orientation = oriented_glycan.class_
            #print(orientation)
        return orientation, oriented_glycan

class RootMonoOrientator(YOLOOrientator):
    def get_orientation(self,**kw):
        glycanimage=kw.get("glycanimage",None)
        logger_name = kw.get("logger_name", '')
        logger = logging.getLogger(logger_name+'.glycanorientator')
        origin_image = glycanimage.copy()
        height, width, channels = glycanimage.shape
        ############################################################################################
        #fix issue with
        ############################################################################################
        white_space = 200
        bigwhite = np.zeros([glycanimage.shape[0] +white_space, glycanimage.shape[1] +white_space, 3], dtype=np.uint8)
        bigwhite.fill(255)
        half_white_space = int(white_space/2)
        bigwhite[half_white_space:(half_white_space + glycanimage.shape[0]), half_white_space:(half_white_space+glycanimage.shape[1])] = glycanimage
        glycanimage = bigwhite.copy()
        #detected_glycan = image.copy()
        # cv2.imshow("bigwhite", bigwhite)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        ############################################################################################
        blob = cv2.dnn.blobFromImage(glycanimage, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        #print(outs)
        # loop through results and print them on images
        mono_list = []
        confidences = []
        for out in outs:
            #print(out)
            for detection in out:
                #assign best-match class to the bounding box
                #print(detection)
                scores = detection[5:]
                #print(scores)
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                
                box = boundingboxes.Detected(origin_image, confidence, class_=class_id,white_space=white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])

                box.rel_to_abs()
                
                box.fix_image()
                
                box.center_to_corner()
                
                box.fix_borders()
                
                box.to_four_corners()

    #BOXES FORMAT IS A PROBLEM
                mono_list.append(box)
                confidences.append(confidence)
        if mono_list == []:
            return None, None
        else:
            #cv.dnn.NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]])
            #print(boxes)
            boxesfornms = [bbox.to_list() for bbox in mono_list]
            #print(unpaddedboxesfornms)
            try:
                indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, 0.0, 0.4)
            except TypeError:
                boxesfornms = [bbox.to_new_list() for bbox in mono_list]
                indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, 0.0, 0.4)
            try:
                indexes = [index[0] for index in indexes]
            except IndexError:
                pass

            #print(f"\nGlycan detected: {len(boxes)}")
            #cv2.imshow("Image", detected_glycan)
            #cv2.waitKey(0)
            all_monos = [mono_list[i] for i in indexes]
            confidences = [confidences[i] for i in indexes]
            
            root_monos = [ x for x in all_monos if x.class_ == 0 ]
            if len(root_monos) == 0:
                return None, None
            elif len(root_monos) == 1:
                root_mono = root_monos[0]
            else:
                confidences = [ mono.confidence for mono in root_monos ]
                best_index = np.argmax(confidences)
                root_mono = root_monos[best_index]
            
            orientation = None
            ###get the orientation from root mono and its position in image
            #YOLO x is left-right, y is top-bottom
            left = root_mono.rel_cen_x
            top = root_mono.rel_cen_y
            right = 1 - root_mono.rel_cen_x
            bottom = 1 - root_mono.rel_cen_y
            
            # orientation: 0: lr, 1: rl, 2: tb, 3: bt
            # here we get orientation based on index of the lowest number
            location = [left,right,top,bottom]
            logger.info(f"Root mono coordinates:\n{location}")
            
            #leave this edge case - since the root is the end goal this is not that important
            # center = (location.count(min(location)) > 1)
            
            # if center:
                
                    

            orientation = np.argmin(location)
            logger.info(f"Orientation: {orientation}")
            #print(orientation)
        return orientation, root_mono