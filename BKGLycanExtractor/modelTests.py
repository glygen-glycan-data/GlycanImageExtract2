import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import BKGLycanExtractor.compareBoxes as cb
import logging

logger = logging.getLogger("test")

class TestModel:
    def __init__(self,image,model):
        self.image = image
        self.model = model
    def drawRects(self,con_threshold=0.5, pad = True):
        glycans = self.model
        glycans.getRects(con_threshold, pad)
        self.model = glycans

class GlycanFound(TestModel):
    def isGlycan(self,con_threshold = 0.5):
        self.drawRects(con_threshold)
        for i in range(len(self.model.boxes)):
            confidence = self.model.confidences[i]
            if confidence >= con_threshold:
                return True
        return False

class bbAccuracy(TestModel):
    def __init__(self,image,model,training_boxes, pad = True):
       super().__init__(image, model)
       self.training = training_boxes
       self.pad_flag = pad
    def compare(self, con_threshold = 0.5):
        self.drawRects(con_threshold, pad = self.pad_flag)
        pad = self.pad_flag
        [box.toFourCorners() for box in self.model.boxes]
        compare_dict = {}
        #annotate_image = self.image.copy()
        for idx,dbox in enumerate(self.model.boxes):
            # p1 = (dbox.x,dbox.y)
            # p2 = (dbox.x2,dbox.y2)
            # cv2.rectangle(annotate_image,p1,p2,(0,255,0),3)
            dbox.setName(str(idx))
            compare_dict[dbox.name] = dbox
            max_int = 0
            for tbox in self.training.boxes:
                if pad:
                    pair = cb.ComparePaddedBox(dbox,tbox)
                else:
                    pair = cb.CompareRawBox(dbox,tbox)
                if pair.have_intersection():
                    intersect = pair.intersection_area()
                    if intersect > max_int:
                        max_int = intersect
                        compare_dict[dbox.name] = (pair)
                else:
                    continue
        results = []
        logger.info("Training box checks:")
        for tbox in self.training.boxes:
            logger.info(str(tbox))
            # p1 = (tbox.x,tbox.y)
            # p2 = (tbox.x2,tbox.y2)
            # cv2.rectangle(annotate_image,p1,p2,(255,0,0),3)
            found = False
            for dbox in self.model.boxes:
                if pad:
                    pair = cb.ComparePaddedBox(dbox,tbox)
                else:
                    pair = cb.CompareRawBox(dbox,tbox)
                if pair.have_intersection():
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
            try:
                n = boxpair.name
                assert n == key
                logger.info(str(boxpair))
                results.append("FP")
                #print("no tbox")
                logger.info("FP, detection does not intersect with training box.")
                continue
            except AttributeError:
                pass
            dbox = boxpair.dbox
            tbox = boxpair.tbox
            assert dbox.name == key
            logger.info(str(dbox))
            t_area = tbox.area()
            d_area = dbox.area()
            inter = boxpair.intersection_area()
            if inter == 0:
                results.append("FP")
                #print("no tbox")
                logger.info("FP, detection does not intersect with training box.")
            elif inter == t_area:
                if boxpair.training_contained():
                    results.append("TP")
                    logger.info("TP")
                else:
                    results.append("FP")
                    results.append("FN")
                    #print("dbox too big")
                    logger.info("FP/FN, detection area too large.")
            elif inter == d_area:
                if boxpair.detection_sufficient():
                    results.append("TP")
                    logger.info("TP")
                else:
                    results.append("FN")
                    results.append("FP")
                    logger.info("FP/FN, detection area too small.")
            else:
                if boxpair.is_overlapping():
                    results.append("TP")
                    logger.info("TP")
                else:
                    results.append("FP")
                    results.append("FN")
                    logger.info("FP/FN, not enough overlap.")
        return results
    def plotPrecisionRecall(self):
        precision = []
        recall = []
        for j in [x*0.1 for x in range(0,10,1)]:
            logger.info(f'Confidence: {j}')
            results = self.compare(j)
            fp = results.count("FP")
            tp = results.count("TP")
            pos = fp + tp
            fn = results.count("FN")
            tpfn = tp+fn
            try: 
                prec = tp/pos
            except ZeroDivisionError:
                prec = 0
            rec = tp/tpfn
            precision.append(prec)
            recall.append(rec)
        
        # min_recall = min(recall)
        # min_index = []
        # for idx,num in enumerate(recall):
        #     if num == min_recall:
        #         min_index.append(idx)
        
        # min_recall_prec = [precision[i] for i in min_index]
        # beginning_prec = max(min_recall_prec)
        
        # precision.append(beginning_prec)
        # recall.append(0)
            
        
        recall, precision = zip(*sorted(zip(recall, precision)))
        
        plt.plot(recall,precision)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        pr = plt.gcf()
        return pr