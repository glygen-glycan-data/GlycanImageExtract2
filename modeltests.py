
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

from BKGlycanExtractor import monosaccharideid
from BKGlycanExtractor.compareboxes import CompareBoxes
import BKGlycanExtractor.monosaccharideid as mono_finder
from BKGlycanExtractor.glycanannotator import Config_Manager


logger = logging.getLogger("test")
        

class Finder_Evaluator:
    def __init__(self,pipeline_methods,reference='KnownSemantics',evaluation_type=None):
        self.configs = pipeline_methods
        self.km_configs = reference
        self.evaluation_type = evaluation_type
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

    def compare(self, boxes_in, training, comparison_alg, image, conf_threshold = 0.5):
        boxes = [box for box in boxes_in if box.confidence >= conf_threshold ]

        [box.to_four_corners() for box in boxes]
        compare_dict = {}
        annotate_image = image.copy()
        for idx,dbox in enumerate(boxes):
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

            assert dbox.name == key
            logger.info(str(dbox))
            if tbox is None:
                results.append("FP")
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
                        logger.info("FP, detection does not intersect with training box.")
                    elif inter == t_area:
                        if comparison_alg.training_contained(tbox,dbox):
                            results.append("TP")
                            logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN")
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
                logger.info("FP, detection does not intersect with training box.")
                return False
            elif inter == t_area:
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
    

    def run(self,images):
        # To Do [Discuss integrating link boxes,etc]- making this method more flexible so that it can accept - link_finder, etc

        # evaluation type is 'mono_finder' - it is straight forward, just implement mono_finder
        # if evaluation  type is 'link_finder' - you need to fill in Figure Semantics for 
        # everything in glycan_steps order before you can get details for 'link_finder' and then compare
        
        known_semantic_data = {}
        predicted_semantic_data = {}

        known_pipeline_name = None

        compare_algo = CompareBoxes()
        glycan_files = [image for image in images if os.path.isfile(image) and os.path.basename(image).endswith('.png')]
        random_files = np.random.choice(glycan_files, 3) # select 100 of the files randomly 

        # there will only be one pipeline for Known Semantics
        for km_pipeline in self.km_configs:
            known_pipeline_name = km_pipeline
            mono_idx = self.km_configs[km_pipeline]['steps'].index('mono_finder')
            km = self.km_configs[km_pipeline]['instantiated_finders'][mono_idx]

        # should handle multiple prediction pipelines
        for name, pipeline_config in self.configs.items():
            predicted_semantic_data[name] = []
            known_semantic_data[known_pipeline_name] = []
            bx1_km = None
            bx_2 = None

            print("\nPipeline name:",name)
            self.obs.update({name: {}})
            finder_idx = pipeline_config['steps'].index(self.evaluation_type)
            mf = pipeline_config['instantiated_finders'][finder_idx]

            for glycan_file in random_files:        
                file_name = os.path.basename(glycan_file)

                print("file name:",file_name)       
                if file_name.endswith("png"):
                    image = cv2.imread(glycan_file)
                    text_file = self.training_file(file=file_name, direc=os.path.dirname(glycan_file))

                    bx1 = km.find_boxes(text_file)
                    bx2 = mf.find_boxes(glycan_file)

                    for idx, gly_obj in enumerate(bx2.glycans()):
                        bx_2 = [monos['box'] for monos in gly_obj['monos']]
                        predicted_semantic_data[name].append({file_name: self.format_data(bx2.semantics)})

                    for idx, gly_obj in enumerate(bx1.glycans()):
                        bx1_km = [monos['box'] for monos in gly_obj['monos']]
                        known_semantic_data[known_pipeline_name].append({file_name: self.format_data(bx1.semantics)})

                    for confidence in [c/100 for c in range(0,100)]:
                        results = self.compare(bx_2,bx1_km,compare_algo,image,confidence)
                        self.obs[name].setdefault(confidence, []).extend(results)


        predicted_monos_data = json.dumps(predicted_semantic_data)
        known_monos_data = json.dumps(known_semantic_data)

        directory = os.getcwd() + '/output'

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the JSON string to a file
        with open(directory + '/predicted_monos_data.json', 'w') as file:
            json.dump(predicted_monos_data, file, indent=4)
            # file.write(predicted_monos_data)
        
        with open(directory + '/known_monos_data.json', 'w') as file:
            json.dump(known_monos_data, file, indent=4)
            # file.write(known_monos_data)
        
        return self.obs

    def format_data(self,semantic_obj):
        if 'image' in semantic_obj:
            del semantic_obj['image']

        for gly_obj in semantic_obj['glycans']:
            if 'image' in gly_obj: 
                del gly_obj['image']

            for mono in gly_obj['monos']:
                del mono['box']

        return semantic_obj


    def training_file(self,file = None, direc = '.'):
        filename = file.split(".")[0]
        boxes_doc = os.path.join(direc,filename + '_map' + ".txt")
        if os.path.exists(boxes_doc):
            return boxes_doc
        else:
            return None

    def plotprecisionrecall(self,images):
        self.run(images)

        directory = os.getcwd() + '/output'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
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


    
