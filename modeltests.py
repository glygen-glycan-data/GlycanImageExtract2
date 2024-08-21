
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import cv2
# import BKGlycanExtractor.compareBoxes as cb
import logging

logger = logging.getLogger("test")

class TestModel:
    def __init__(self):
        pass
    def is_glycan(self,boxes,con_threshold = 0.5):
        for i in boxes:
            confidence = boxes[4]
            if confidence >= con_threshold:
                return True
        return False
    def check_composition(self, known_comp_dict, model_comp_dict):
        for mono in known_comp_dict.keys():
            if int(known_comp_dict[mono]) != int(model_comp_dict[mono]):
                return False
        return True
    def compare(self, boxes_in, training, comparison_alg, image, conf_threshold = 0.0):
        boxes = [ box.clone() for box in boxes_in if box.confidence >= conf_threshold ] 
        [box.to_four_corners() for box in boxes]
        compare_dict = {}
        annotate_image = image.copy()
        for idx,dbox in enumerate(boxes):

            # p1 = (dbox.x,dbox.y)
            # p2 = (dbox.x2,dbox.y2)
            # cv2.rectangle(annotate_image,p1,p2,(0,255,0),3)
            dbox.set_name(str(idx))
            compare_dict[dbox.name] = (dbox, None)
            max_int = 0
            for tbox in training:
                if comparison_alg.have_intersection(tbox,dbox):
                    iou = comparison_alg.iou(tbox,dbox)
                    #print(intersect)
                    if iou > max_int:
                        max_int = iou
                        compare_dict[dbox.name] = (dbox,tbox)
                else:
                    continue
        results = []
        logger.info("Training box checks:")
        for tbox in training:
            #print(str(tbox))
            logger.info(str(tbox))

            found = False
            for dbox in boxes:
                if comparison_alg.have_intersection(tbox,dbox):
                    found = True
                    #print("intersection")
                    break
            if found:
                #print("hello")
                logger.info("Training box intersects with detected box(es).")
                continue
            else:                                       
                results.append('FN')
                #print("tbox not found")
                logger.info("FN, training box not detected.")
        # if self.pad_flag:
        #     cv2.imshow('image',annotate_image)
        # else:
        #     cv2.imshow('image2', annotate_image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        logger.info("Detected box checks:")
        for key,boxpair in compare_dict.items():
            dbox = boxpair[0]
            tbox = boxpair[1]

            assert dbox.name == key
            logger.info(str(dbox))
            if tbox is None:
                results.append("FP")
                #print("no tbox")
                logger.info("FP, detection does not intersect with training box.")
            else:
                if not comparison_alg.compare_class(tbox,dbox):
                    results.append("FP")
                    results.append("FN")
                    logger.info("FP/FN, incorrect class")
                else:
                    t_area = tbox.area()
                    d_area = dbox.area()
                    inter = comparison_alg.intersection_area(tbox,dbox)
                    if inter == 0:
                        results.append("FP")
                        #print("no tbox")
                        logger.info("FP, detection does not intersect with training box.")
                    elif inter == t_area:
                        #print(comparison_alg.training_contained(tbox,dbox))
                        if comparison_alg.training_contained(tbox,dbox):
                            #print("hello")
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN")
                            #print("dbox too big")
                            logger.info("FP/FN, detection area too large.")
                    elif inter == d_area:
                        if comparison_alg.detection_sufficient(tbox,dbox):
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FN")
                            results.append("FP")
                            logger.info("FP/FN, detection area too small.")
                    else:
                        if comparison_alg.is_overlapping(tbox,dbox):
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN")
                            logger.info("FP/FN, not enough overlap.")
                
                print("---------")
                print(results[-1])
                t1 = (tbox.x,tbox.y)
                t2 = (tbox.x2,tbox.y2)
                d1 = (dbox.x,dbox.y)
                d2 = (dbox.x2,dbox.y2)

                print(t1,t2)
                print(d1,d2)

                print("iou:",comparison_alg.iou(tbox,dbox))

                cv2.rectangle(annotate_image,t1,t2,(255,0,0),2)
                cv2.rectangle(annotate_image,d1,d2,(0,255,0),2)
                cv2.imshow('image', annotate_image)
                annotate_image = image.copy()
                cv2.waitKey(0)
                #cv2.destroyAllWindows()
        return results
    def compare_one(self, box, trainingbox, comparison_alg, conf_threshold = 0.0):
        if box.confidence < conf_threshold:
            logger.info(f'FN, training box not detected at confidence {conf_threshold}.')
            return False
        box.to_four_corners()
        if not comparison_alg.have_intersection(trainingbox,box):
            logger.info('FN, training box not detected.')
            return False
        else:                                       
            logger.info("Training box intersects with detected box(es).")
        if not comparison_alg.compare_class(trainingbox,box):
            logger.info("FP/FN, incorrect class")
            return False
        else:
            t_area = trainingbox.area()
            d_area = box.area()
            inter = comparison_alg.intersection_area(trainingbox,box)
            if inter == 0:
                #print("no tbox")
                logger.info("FP, detection does not intersect with training box.")
                return False
            elif inter == t_area:
                #print(comparison_alg.training_contained(tbox,dbox))
                if comparison_alg.training_contained(trainingbox,box):
                    logger.info("TP")
                    return True
                else:
                    logger.info("FP/FN, detection area too large.")
                    return False
            elif inter == d_area:
                if comparison_alg.detection_sufficient(trainingbox,box):
                    logger.info("TP")
                    return True
                else:
                    logger.info("FP/FN, detection area too small.")
                    return False
            else:
                if comparison_alg.is_overlapping(trainingbox,box):
                    logger.info("TP")
                    return True
                else:
                    logger.info("FP/FN, not enough overlap.")
                    return False

class MonosImprovement(TestModel):
    def compare(self, boxes, training, comparison_alg, conf_threshold = 0.0):
        fp = False
        for box in boxes:
            if box.confidence < conf_threshold:
                boxes.remove(box)
        [box.to_four_corners() for box in boxes]
        compare_dict = {}
        #annotate_image = self.image.copy()
        for idx,dbox in enumerate(boxes):
                
            # p1 = (dbox.x,dbox.y)
            # p2 = (dbox.x2,dbox.y2)
            # cv2.rectangle(annotate_image,p1,p2,(0,255,0),3)
            dbox.set_name(str(idx))
            compare_dict[dbox.name] = (dbox, None)
            max_int = 0
            for tbox in training:
                if comparison_alg.have_intersection(tbox,dbox):
                    intersect = comparison_alg.intersection_area(tbox,dbox)
                    #print(intersect)
                    if intersect > max_int:
                        max_int = intersect
                        compare_dict[dbox.name] = (dbox,tbox)
                else:
                    continue
        results = []
        logger.info("Training box checks:")
        for tbox in training:
            #print(str(tbox))
            logger.info(str(tbox))
            # p1 = (tbox.x,tbox.y)
            # p2 = (tbox.x2,tbox.y2)
            # cv2.rectangle(annotate_image,p1,p2,(255,0,0),3)
            found = False
            for dbox in boxes:
                if comparison_alg.have_intersection(tbox,dbox):
                    found = True
                    #print("intersection")
                    break
            if found:
                #print("hello")
                logger.info("Training box intersects with detected box(es).")
                continue
            else:                                       
                results.append('FN')
                #print("tbox not found")
                logger.info("FN, training box not detected.")
        # if self.pad_flag:
        #     cv2.imshow('image',annotate_image)
        # else:
        #     cv2.imshow('image2', annotate_image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        logger.info("Detected box checks:")
        for key,boxpair in compare_dict.items():
            dbox = boxpair[0]
            tbox = boxpair[1]
            assert dbox.name == key
            logger.info(str(dbox))
            if tbox is None:
                #print("no tbox")
                #don't count this fp, add to folder of fp monos
                logger.info("FP, detection does not intersect with training box.")
                fp = True
            else:
                if not comparison_alg.compare_class(tbox,dbox):
                    results.append("FP")
                    results.append("FN")
                    logger.info("FP/FN, incorrect class")
                else:
                    t_area = tbox.area()
                    d_area = dbox.area()
                    inter = comparison_alg.intersection_area(tbox,dbox)
                    if inter == 0:
                        results.append("FP")
                        #print("no tbox")
                        logger.info("FP, detection does not intersect with training box.")
                    elif inter == t_area:
                        #print(comparison_alg.training_contained(tbox,dbox))
                        if comparison_alg.training_contained(tbox,dbox):
                            #print("hello")
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN")
                            #print("dbox too big")
                            logger.info("FP/FN, detection area too large.")
                    elif inter == d_area:
                        if comparison_alg.detection_sufficient(tbox,dbox):
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FN")
                            results.append("FP")
                            logger.info("FP/FN, detection area too small.")
                    else:
                        if comparison_alg.is_overlapping(tbox,dbox):
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN")
                            logger.info("FP/FN, not enough overlap.")
        #print(results)
        return results, fp
    def getfps(self,boxes, training, comparison_alg, conf_threshold = 0.5):
        for box in boxes:
            if box.confidence < conf_threshold:
                boxes.remove(box)
        [box.to_four_corners() for box in boxes]
        compare_dict = {}
        #annotate_image = self.image.copy()
        for idx,dbox in enumerate(boxes):
                
            # p1 = (dbox.x,dbox.y)
            # p2 = (dbox.x2,dbox.y2)
            # cv2.rectangle(annotate_image,p1,p2,(0,255,0),3)
            dbox.set_name(str(idx))
            compare_dict[dbox.name] = (dbox, None)
            max_int = 0
            for tbox in training:
                if comparison_alg.have_intersection(tbox,dbox):
                    intersect = comparison_alg.intersection_area(tbox,dbox)
                    #print(intersect)
                    if intersect > max_int:
                        max_int = intersect
                        compare_dict[dbox.name] = (dbox,tbox)
                else:
                    continue
        newmonos = []
        logger.info("Training box checks:")
        for tbox in training:
            #print(str(tbox))
            logger.info(str(tbox))
            # p1 = (tbox.x,tbox.y)
            # p2 = (tbox.x2,tbox.y2)
            # cv2.rectangle(annotate_image,p1,p2,(255,0,0),3)
            found = False
            for dbox in boxes:
                if comparison_alg.have_intersection(tbox,dbox):
                    found = True
                    #print("intersection")
                    break
            if found:
                #print("hello")
                logger.info("Training box intersects with detected box(es).")
                continue

        
        logger.info("Detected box checks:")
        for key,boxpair in compare_dict.items():
            dbox = boxpair[0]
            tbox = boxpair[1]
            assert dbox.name == key
            logger.info(str(dbox))
            if tbox is None:
                #print("no tbox")
                #don't count this fp, add to folder of fp monos
                logger.info("FP, detection does not intersect with training box.")
                newmonos.append(dbox)

        #print(results)
        return newmonos
    
    
