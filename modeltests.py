
import matplotlib
# matplotlib.use('tkagg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2
# import BKGlycanExtractor.compareBoxes as cb
import logging
import configparser
import os
import numpy as np
import json

from BKGlycanExtractor import monosaccharideid
from BKGlycanExtractor.compareboxes import CompareBoxes
import BKGlycanExtractor.monosaccharideid as mono_finder
from BKGlycanExtractor.glycanannotator import Config_Manager


logger = logging.getLogger("test")
        

class Finder_Evaluator:
    def __init__(self,pipeline_methods,km_pipeline_methods):
        self.configs = pipeline_methods
        self.km_configs = km_pipeline_methods
        self.obs = {}

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

    # changed conf_threshold = 0.5, originally it was conf_threshold = 0.0
    def compare(self, boxes_in, training, comparison_alg, image, conf_threshold = 0.5):
        # boxes = [ box.clone() for box in boxes_in if box.confidence >= conf_threshold ] 
        boxes = [box for box in boxes_in if box.confidence >= conf_threshold ]
        # print("--->>box class id",[(b.class_, b.confidence) for b in boxes])

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
                    # print("intersection")
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
                # print("\ntbox,dbox",tbox.class_,dbox.class_)
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
                
                t1 = (tbox.x,tbox.y)
                t2 = (tbox.x2,tbox.y2)
                d1 = (dbox.x,dbox.y)
                d2 = (dbox.x2,dbox.y2)

                # print(t1,t2)
                # print(d1,d2)

                # print("iou:",comparison_alg.iou(tbox,dbox))

                # cv2.rectangle(annotate_image,t1,t2,(255,0,0),2)
                # cv2.rectangle(annotate_image,d1,d2,(0,255,0),2)
                # cv2.imshow('image', annotate_image)
                # annotate_image = image.copy()
                # cv2.waitKey(0)
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
    

    def run(self,data_folder):
        # known_semantic_data = {}
        # predicted_semantic_data = {}

        compare_algo = CompareBoxes()
        glycan_files = [file for file in os.scandir(data_folder) if os.path.isfile(file) and file.name.endswith("png")]
        random_files = np.random.choice(glycan_files, 3) # select 100 of the files randomly 

        km = self.km_configs.get('mono_id')

        for name, pipeline_config in self.configs.items():
            self.obs.update({name: {}})
            # print("pipeline_config",pipeline_config)
            mf = pipeline_config.get('mono_id')

            for glycan_file in random_files:        
                file_name = glycan_file.name 
                print("\nfile name:",file_name)       
                if file_name.endswith("png"):
                    image = cv2.imread(glycan_file)
                    text_file = self.training_file(file=file_name, direc=data_folder)

                    bx1 = km.find_boxes(text_file)
                    bx2 = mf.find_boxes(glycan_file)

                    # can add bx3 (padded data) for comparision with ground truth data
                    # bx3 = mf.find_boxes(glycan_file,request_padding=True)

                    # there is some mistake here
                    for idx, gly_obj in enumerate(bx2.glycans()):
                        bx_2 = [monos['box'] for monos in gly_obj['monos']]
                        # predicted_semantic_data[file_name] = self.format_data(bx2.semantics)

                    for idx, gly_obj in enumerate(bx1.glycans()):
                        bx1_km = [monos['box'] for monos in gly_obj['monos']]
                        # known_semantic_data[file_name] = self.format_data(bx1.semantics)

                    for confidence in [c/100 for c in range(0,100)]:
                        results = self.compare(bx_2,bx1_km,compare_algo,image,confidence)
                        self.obs[name].setdefault(confidence, []).extend(results)


        # # * Semantics JSON string --> no need to include image and objects and save it in a file
        # create a method for json dumps, which also removes the ndarray and class obj from the data strcuture
        # predicted_monos_data = json.dumps(predicted_semantic_data)
        # known_monos_data = json.dumps(known_semantic_data)

        # # Save the JSON string to a file
        # with open('predicted_monos_data.json', 'w') as file:
        #     json.dump(predicted_monos_data, file, indent=4)
        
        # with open('known_monos_data.json', 'w') as file:
        #     json.dump(known_monos_data, file, indent=4)
        
        return self.obs

    def format_data(self,semantic_obj):
        if 'image' in semantic_obj:
            del semantic_obj['image']

        for gly_obj in semantic_obj['glycans']:
            if 'image' in gly_obj: 
                del gly_obj['image']

            for mono in gly_obj['monos']:
                del mono['box']

        # print("\nsemantic_obj",semantic_obj)
        return semantic_obj


    def training_file(self,file = None, direc = '.'):
        filename = file.split(".")[0]
        boxes_doc = os.path.join(direc,filename + '_map' + ".txt")
        if os.path.exists(boxes_doc):
            return boxes_doc
        else:
            return None

    def plotprecisionrecall(self,data_folder):
        self.run(data_folder)
        
        plt.figure(1) 
        plt.figure(2)

        for annotator_name, result_data in self.obs.items():
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
                plt.plot(recall, precision, ".", label=f"{annotator_name}")
            else:
                plt.plot(recall, precision, ".-", label=f"{annotator_name}")


            plt.figure(2)
            if len(set(recall)) == 1 and len(set(precision)) == 1:
                plt.plot(recall, precision, ".", label=f"{annotator_name}")
            else:
                plt.plot(recall, precision, ".-", label=f"{annotator_name}")  

        plt.figure(1)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.axhline(y=1, color='k', linestyle='--')
        plt.axvline(x=1, color='k', linestyle='--')
        plt.legend(loc="best")

        # Customize and finalize figure 2 (zoomed-in graph)
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

        pr.savefig('plot1.png') 
        pr_zoom.savefig('plot2.png')

        return pr,pr_zoom


class MonosImprovement(Finder_Evaluator):
    def compare(self, boxes, training, comparison_alg, image,conf_threshold = 0.0):
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
    
    
