import cv2
import numpy as np
import BKGLycanExtractor.boundingboxes as boundingboxes

class GlycanRectID:
    def __init__(self,**kw):
        pass
    def get_rects(self,**kw):
        raise NotImplementedError
    
class OriginalYOLO(GlycanRectID):
    def __init__(self,**kw):

        super().__init__()
        weights=kw.get("weights",)
        net=kw.get("net",)
        
        self.net = cv2.dnn.readNet(weights,net)
        
        layer_names = self.net.getLayerNames()
        #print(layer_names)
        #compatibility with new opencv versions
        try:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        except IndexError:
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        

    def get_rects(self,image=None, threshold = 0.0):
        #extract location of all glycan from image

        origin_image = image.copy()
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
        #cv2.imshow("bigwhite", bigwhite)
        #cv2.waitKey(0)

        ############################################################################################
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        # loop through results and print them on images
        class_ids = []
        confidences = []
        padded_boxes = []
        unpadded_boxes = []
        for out in outs:
            for detection in out:
                #print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                unpadded_box = boundingboxes.Detected(origin_image, confidence, white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])
                padded_box = boundingboxes.Detected(origin_image, confidence, white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])
                
                unpadded_box.rel_to_abs()
                padded_box.rel_to_abs()
                
                unpadded_box.fix_image()
                padded_box.fix_image()
                
                unpadded_box.center_to_corner()
                padded_box.center_to_corner()
                
                padded_box.pad_borders()
                
                unpadded_box.fix_borders()
                padded_box.fix_borders()
                
                unpadded_box.is_entire_image()
                padded_box.is_entire_image()

    #BOXES FORMAT IS A PROBLEM
                unpadded_boxes.append(unpadded_box)
                padded_boxes.append(padded_box)
                #boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        #cv.dnn.NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]])
        #print(boxes)
        unpaddedboxesfornms = [bbox.to_list() for bbox in unpadded_boxes]
        paddedboxesfornms = [bbox.to_list() for bbox in padded_boxes]
        
        unpadded_indexes = cv2.dnn.NMSBoxes(unpaddedboxesfornms, confidences, threshold, 0.4)
        padded_indexes = cv2.dnn.NMSBoxes(paddedboxesfornms, confidences, threshold, 0.4)
        
        unpadded_indexes = [index[0] for index in unpadded_indexes]
        padded_indexes = [index[0] for index in padded_indexes]
        #print(f"\nGlycan detected: {len(boxes)}")
        #cv2.imshow("Image", detected_glycan)
        #cv2.waitKey(0)
        unpadded_boxes = [unpadded_boxes[i] for i in unpadded_indexes]
        unpadded_confidences = [confidences[i] for i in unpadded_indexes]
        unpadded_class_ids = [class_ids[i] for i in unpadded_indexes]
        
        padded_boxes = [padded_boxes[i] for i in padded_indexes]
        padded_confidences = [confidences[i] for i in padded_indexes]
        padded_class_ids = [class_ids[i] for i in padded_indexes]
        return unpadded_boxes, padded_boxes

class OriginalYOLOPreFix(OriginalYOLO):
    def get_rects(self,image=None,threshold=0.5, pad = True):
        #extract location of all glycan from image

        origin_image = image.copy()
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
        bigwhite[0:image.shape[0], 0:image.shape[1]] = image
        image = bigwhite.copy()
        #detected_glycan = image.copy()
        #cv2.imshow("bigwhite", bigwhite)
        #cv2.waitKey(0)

        ############################################################################################
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        # loop through results and print them on images
        class_ids = []
        confidences = []
        padded_boxes = []
        unpadded_boxes = []
        for out in outs:
            for detection in out:
                #print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                unpadded_box = boundingboxes.Detected(origin_image, confidence, white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])
                padded_box = boundingboxes.Detected(origin_image, confidence, white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])
                
                unpadded_box.rel_to_abs()
                padded_box.rel_to_abs()
                
                unpadded_box.center_to_corner()
                padded_box.center_to_corner()
                
                padded_box.pad_borders()
                
                unpadded_box.fix_borders()
                padded_box.fix_borders()
                
                unpadded_box.is_entire_image()
                padded_box.is_entire_image()

    #BOXES FORMAT IS A PROBLEM
                unpadded_boxes.append(unpadded_box)
                padded_boxes.append(padded_box)
                #boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        #cv.dnn.NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]])
        #print(boxes)
        unpaddedboxesfornms = [bbox.to_list() for bbox in unpadded_boxes]
        paddedboxesfornms = [bbox.to_list() for bbox in padded_boxes]
        
        unpadded_indexes = cv2.dnn.NMSBoxes(unpaddedboxesfornms, confidences, threshold, 0.4)
        padded_indexes = cv2.dnn.NMSBoxes(paddedboxesfornms, confidences, threshold, 0.4)
        
        unpadded_indexes = [index[0] for index in unpadded_indexes]
        padded_indexes = [index[0] for index in padded_indexes]
        #print(f"\nGlycan detected: {len(boxes)}")
        #cv2.imshow("Image", detected_glycan)
        #cv2.waitKey(0)
        unpadded_boxes = [unpadded_boxes[i] for i in unpadded_indexes]
        unpadded_confidences = [confidences[i] for i in unpadded_indexes]
        unpadded_class_ids = [class_ids[i] for i in unpadded_indexes]
        
        padded_boxes = [padded_boxes[i] for i in padded_indexes]
        padded_confidences = [confidences[i] for i in padded_indexes]
        padded_class_ids = [class_ids[i] for i in padded_indexes]
        return unpadded_boxes, padded_boxes
    
class TrainingData(GlycanRectID):
    def get_rects(self,image=None,coord_file=None):
        boxes = []
        doc = open(coord_file)
        for line in doc:
            if line.replace(' ','') == '\n':
                continue
            split_line = line.split(' ')
            
            box = boundingboxes.Training(image, rel_cen_x = float(split_line[1]), rel_cen_y = float(split_line[2]), rel_w = float(split_line[3]), rel_h = float(split_line[4]))
            
            box.rel_to_abs()
            
            box.center_to_corner()
            
            box.to_four_corners()
            
            boxes.append(box)
        doc.close()
            
        return boxes