import cv2
import numpy as np
import BKGLycanExtractor.boundingboxes as bb

class glycanRectID:
    def __init__(self, image):
        self.image = image
    
class originalYOLO(glycanRectID):
    def __init__(self,image,**kwargs):
        #print(image)
        #print(kwargs)

        super().__init__(image)
        self.weights=kwargs.get("weight",)
        self.net=kwargs.get("coreyolo",)

    def getRects(self,threshold=0.5, pad = True):
        #extract location of all glycan from image file path
        #array = []
        #base = os.getcwd()
        #coreyolo = r"/home/nduong/demo/BKGLycanExtractor/configs/coreyolo.cfg"
        #weight = r"/home/nduong/demo/BKGLycanExtractor/configs/Glycan_300img_5000iterations.weights"
        weight = self.weights
        coreyolo = self.net
        image = self.image
        #print(f"weight1: type: {type(weight)} and: {weight}")
        #print(f"weight2: type: {type(weight2)} and: {weight2}")
        net = cv2.dnn.readNet(weight,coreyolo)
        #print(net)

        layer_names = net.getLayerNames()
        #print(layer_names)
        #compatibility with new opencv versions
        try:
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        except IndexError:
            output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        #colors = np.random.uniform(0, 255, size=(1, 3))

        #blue = (255, 0, 0)
        #green = (0, 255, 0)
        #red = (0, 0, 255)
        #count=0

        self.origin_image = image.copy()
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
        net.setInput(blob)
        outs = net.forward(output_layers)
        # loop through results and print them on images
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                #print(detection)
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                box = bb.Detected(self.origin_image, confidence, white_space, rel_cen_x = detection[0],rel_cen_y = detection[1], rel_w = detection[2],rel_h = detection[3])
                #print(box.whitespace)
                box.RelToAbs()
                
                box.CenterToCorner()
                #print(box.x)
                if pad:
                    box.padBorders()
                
                box.fixBorders()

                # Rectangle coordinates # where do these constants come from?
                # Looks like 0.2*w padding on left and right,
                # 0.2*h padding on top and bottom
                # x = int(center_x - w / 2)-int(0.2*w)
                # y = int(center_y - h / 2)-int(0.2*h)
                # w = int(1.4*w)
                # h = int(1.4 * h)

                # fix bug that prevent the image to be crop outside the figure object
                # if x<=0:
                #     x=0
                # if y <=0:
                #     y=0
                # if x+w >= (width):
                #     w=int((width)-x)
                # if y+h >= (height):
                #     h=int((height)-y)

                # If we are almost the entire image anyway, avoid cropping errors...

                # if w*h > 0.8*0.8*width*height:
                #     logging.info(f"\nEntire image? {x,w,width,float(w)/width,y,h,height,float(h)/height,w*h,width*height,float(w*h)/(width*height)}")
                #     logging.info("Reset to entire image...")
                #     x = 0; y = 0; w = width; h = height
                
                box.isEntireImage()

                # p1 = (x,y)
                # p2 = (x+w,y+h)
                #print(p1,p2)
                #cv2.rectangle(detected_glycan,p1,p2,(0,255,0),3)

    #BOXES FORMAT IS A PROBLEM
                boxes.append(box)
                #boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        #cv.dnn.NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]])
        #print(boxes)
        boxesfornms = [bbox.toList() for bbox in boxes]
        indexes = cv2.dnn.NMSBoxes(boxesfornms, confidences, threshold, 0.4)
        indexes = [index[0] for index in indexes]
        #print(f"\nGlycan detected: {len(boxes)}")
        #cv2.imshow("Image", detected_glycan)
        #cv2.waitKey(0)
        self.boxes = [boxes[i] for i in indexes]
        self.confidences = [confidences[i] for i in indexes]
        self.class_ids = [class_ids[i] for i in indexes]

class TrainingData(glycanRectID):
    def __init__(self,image,coord_file):
        super().__init__(image)
        self.coord_file = coord_file
    def getRects(self):
        boxes = []
        doc = open(self.coord_file)
        for line in doc:
            if line.replace(' ','') == '\n':
                continue
            split_line = line.split(' ')
            
            box = bb.Training(self.image, rel_cen_x = float(split_line[1]), rel_cen_y = float(split_line[2]), rel_w = float(split_line[3]), rel_h = float(split_line[4]))
            
            box.RelToAbs()
            
            box.CenterToCorner()
            
            box.toFourCorners()
            
            boxes.append(box)
        doc.close()
            
        self.boxes = boxes