import matplotlib
# matplotlib.use('tkagg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
import logging
import configparser
import os
import numpy as np
import json
import logging

from .compareboxes import CompareBoxes
from . import Config_Manager, Image_Manager
from .semantics import Figure_Semantics

logger = logging.getLogger("test")


class BoxEvaluator:
    # cm = Config_Manager()
    # initialize CompareBoxes here with 3 different IOU values and pass the reference instead of creatig
    # the object agian and again later

    # make a plot which shows a distrubution of confidence values

    def runall(self,image_folder,base_pipeline,predictors,known):
        self.observations = {}
        # pipeline = self.cm.get_pipeline(self.base_pipeline)

        images = Image_Manager(image_folder,pattern="*.png,*.jpg")

        images = [image for image in images if os.path.basename(image).endswith('.png')]
        images = np.random.choice(images, 2) # select 100 of the files randomly

        for pred in predictors:
            self.observations[pred.__class__.__name__] = {}  
            # pred = self.cm.get_finder(pred_finder)

            for image in images:
                # run the base_pipeline
                # this doesnt need to be the full pipeline --> instantiate only what you need in the main_file
                figure_semantics = base_pipeline.run(image)

                glycan = figure_semantics.glycans()[0] #since its a single glycan image

                known_boxes = known.find_boxes(glycan.image_path())
                pred_boxes = pred.find_boxes(glycan.image())

                # compare the boxes
                for confidence in [c/100000 for c in range(99999,100000,10)]:
                    results = self.compare(pred_boxes,known_boxes,CompareBoxes(),image,confidence)
                    self.observations[pred.__class__.__name__].setdefault(confidence,[]).extend(results)

                # print("self.observations",self.observations)
    

    def compare(self, pred_boxes, known_boxes, comparison_alg, image, conf_threshold = 0.5):
        boxes = [box for box in pred_boxes if box.get('confidence') >= conf_threshold ]

        compare_dict = {}
        # should we take dbox and then tboxes or do the opposite? b/c rn a single dbox can have many tboxes ---> but that shouldnt happen --> one dbox box should have
        # only one tbox and all should be unique pairs ---> rn because of multiples ---> we are getting too many TP's
        for idx,dbox in enumerate(boxes):
            dbox.set('id',idx)
            compare_dict[dbox.get('id')] = (dbox, None)
            max_int = 0
            for tbox in known_boxes:
                if comparison_alg.have_intersection(tbox,dbox):
                    iou = comparison_alg.iou(tbox,dbox)
                    if iou > max_int:
                        max_int = iou
                        compare_dict[dbox.get('id')] = (dbox,tbox)
                else:
                    continue
        results = []
        logger.info("Training box checks:")
        for tbox in known_boxes:
            logger.info(str(tbox))

            found = False
            for dbox in boxes:
                if comparison_alg.have_intersection(tbox,dbox):
                    found = True
                    break
            if found:
                logger.info("Training box intersects with detected box(es).")
                continue
            else:                                       
                results.append('FN')
                logger.info("FN, training box not detected.")
        
        logger.info("Detected box checks:")
        for key,boxpair in compare_dict.items():
            dbox = boxpair[0]
            tbox = boxpair[1]

            assert dbox.get('id') == key
            logger.info(str(dbox))
            if tbox is None:
                results.append("FP")
                logger.info("FP, detection does not intersect with training box.")
            else:
                if not comparison_alg.compare_class(tbox,dbox):
                    results.append("FP")
                    results.append("FN") # detected culd be associated with multiple training, so might not bre Fn
                    logger.info("FP/FN, incorrect class")
                else:
                    t_area = tbox.area()
                    d_area = dbox.area()
                    inter = comparison_alg.intersection_area(tbox,dbox)
                    if inter == 0:
                        results.append("FP")
                        logger.info("FP, detection does not intersect with training box.")
                    elif inter == t_area:
                        if comparison_alg.training_contained(tbox,dbox):
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN") # detected culd be associated with multiple training, so might not bre Fn
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
                
        return results



    def plotprecisionrecall(self):

        directory = os.getcwd() + '/output'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.figure(1) 
        plt.figure(2)

        for finder_name, result_data in self.observations.items():
            precision = []
            recall = []
        
            for conf, results_list in result_data.items():
                fp = results_list.count("FP")
                tp = results_list.count("TP")
                pos = fp + tp 
                fn = results_list.count("FN")
                tpfn = tp + fn
                try: 
                    prec = tp / pos
                except ZeroDivisionError:
                    prec = 0
                rec = tp / tpfn
                precision.append(prec)
                recall.append(rec)
            
            # Sort the recall and precision for plotting
            recall, precision = zip(*sorted(zip(recall, precision)))

            # Plot on figure 1 
            plt.figure(1)
            if len(set(recall)) == 1 and len(set(precision)) == 1:
                # plt.plot(recall, precision, ".", label=f"{annotator_name}")
                plt.plot(recall, precision, ".", label=f"{finder_name}")
            else:
                plt.plot(recall, precision, ".-", label=f"{finder_name}")


            plt.figure(2)
            if len(set(recall)) == 1 and len(set(precision)) == 1:
                plt.plot(recall, precision, ".", label=f"{finder_name}")
            else:
                plt.plot(recall, precision, ".-", label=f"{finder_name}")  

        plt.figure(1)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.axhline(y=1, color='k', linestyle='--')
        plt.axvline(x=1, color='k', linestyle='--')
        plt.legend(loc="best")

        # figure 2 (zoomed-in graph)
        plt.figure(2)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.xlim([0.5, 1.1])
        plt.ylim([0.5, 1.1])
        plt.axhline(y=1, color='k', linestyle='--')
        plt.axvline(x=1, color='k', linestyle='--')
        plt.legend(loc="best")
        
        pr = plt.figure(1)
        pr_zoom = plt.figure(2) 

        pr.savefig(directory + '/plot1.png') 
        pr_zoom.savefig(directory + '/plot2.png')

        return pr,pr_zoom



