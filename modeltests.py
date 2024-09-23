
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

from BKGlycanExtractor import monosaccharideid
from BKGlycanExtractor.compareboxes import CompareBoxes
import BKGlycanExtractor.monosaccharideid as mono_finder


logger = logging.getLogger("test")

class TestModel:
    def __init__(self,configs_dir, configs_file, pipeline_name,data_folder):
        self.data_folder = data_folder

        self.obs = {
            'unpadded': {},
            'padded': {}
        } 

        self.configs = self.read_pipeline_configs(configs_dir, configs_file, pipeline_name)

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
    
    def read_pipeline_configs(self, config_dir, config_file, pipeline):
        methods = {}
        config = configparser.ConfigParser()
        config.read(config_file)
        pipelines = []
        for key, value in config.items():
            if value.get("sectiontype") == "modeltest":
                pipelines.append(key)
        try:
            annotator_methods = config[pipeline]
        except KeyError:
            print(pipeline,"is not a valid pipeline.")
            print("Valid pipelines:", pipelines)
            sys.exit(1)
        
        method_descriptions = {
            "mono_id": {"prefix": "monosaccharideid.", "multiple": False},
            }

        for method, desc in method_descriptions.items():
            # print(method, desc)
            if desc.get("multiple"):
                method_names = annotator_methods.get(method).split(",")
                methods[method] = []
                for method_name in method_names:
                    # print(method_name)
                    methods[method].append(self.setup_method(
                        config, desc.get("prefix"), config_dir, method_name
                        ))
            else:
                method_name = annotator_methods.get(method)
                # print(method_name)
                methods[method] = self.setup_method(
                    config, desc.get("prefix"), config_dir, method_name
                    )
        return methods

    def setup_method(
            self, configparserobject, prefix, directory, method_name
            ):
        gdrive_dict = {
            "coreyolo.cfg":
                "1M2yMBkIB_VctyH01tyDe1koCHT0U8cwV",
            "Glycan_300img_5000iterations.weights":
                "1xEeMF-aJnVDwbrlpTHkd-_kI0_P1XmVi",
            "largerboxes_plusindividualglycans.weights":
                "16-AIvwNd-ZERcyXOf5G50qRt1ZPlku5H",
            "monos2.cfg":
                "15_XxS7scXuvS_zl1QXd7OosntkyuMQuP",
            "orientation_redo.weights":
                "1KipiLdlUmGSDsr0WRUdM0ocsQPEmNQXo",
            "orientation.cfg":
                "1AYren1VnmB67QLDxvDNbqduU8oAnv72x",
            "orientation_flipped.cfg":
                "1YXkSWjqjbx5_GkCrOdkIHrSocTAqu9WX",
            "orientation_flipped.weights":
                "1PQH6_JPpE_1o9WdhKAIGJdmOF5fI39Ew",
            "yolov3_monos_new_v2.weights":
                "1h-QiO2FP7fU7IbvZjoF7fPf55N0DkTPp",
            "yolov3_monos_random.weights": 
                "1m4nJqxrJLl1MamIugdyzRh6td4Br7OMg",
            "yolov3_monos_largerboxes.weights":
                "1WQI9UiJkqGx68wy8sfh_Hl5LX6q1xH4-",
            "rootmono.cfg":
                "1RSgCYxkNvrPYann5MG7WybyBZS2UA5v0",
            "yolov3_rootmono.weights":
                "1dUTFbPA7XV-HztWeM5uto2mF_xo5F-3Z"
        }
        
        method_values = configparserobject[method_name]
        method_class = method_values.pop("class")
        method_configs = {}
        for key, value in method_values.items():
            filename = os.path.join(directory,value)
            if os.path.isfile(filename):
                method_configs[key] = filename
            else:
                gdrive_id = gdrive_dict.get(value)
                if gdrive_id is None:
                    raise FileNotFoundError(
                        value + 
                        "was not found in configs directory or Google Drive"
                        )
                getfromgdrive.download_file_from_google_drive(
                    gdrive_id, filename
                    )
                method_configs[key] = filename
        if not method_configs:
            return eval(prefix+method_class+"()")
        return eval(prefix+method_class+"(method_configs)")

    def run(self):
        km = mono_finder.KnownMono()
        mf = self.configs.get('mono_id')
        compare_algo = CompareBoxes()

        glycan_files = [file for file in os.scandir(self.data_folder) if os.path.isfile(file) and file.name.endswith("png")]
        # select 100 of the files randomly 
        random_files = np.random.choice(glycan_files, 100) 

        for glycan_file in random_files:        
            name = glycan_file.name        
            if name.endswith("png"):
                image = cv2.imread(glycan_file)
                text_file = self.training_file(file=name, direc=self.data_folder)

                bx1 = km.find_boxes(text_file)
                bx2 = mf.find_boxes(glycan_file)

                for gly_obj in bx2.glycans():
                    bx2_unpadded = [monos['box'] for monos in gly_obj['monos_unpadded']]
                    bx2_padded = [monos['box'] for monos in gly_obj['monos_padded']]
                    

                for confidence in [c/100 for c in range(0,100)]:
                    unpadded_results = self.compare(bx2_unpadded,bx1,compare_algo,image,confidence)
                    padded_results = self.compare(bx2_padded,bx1,compare_algo,image,confidence)

                    self.obs['unpadded'].setdefault(confidence, []).extend(unpadded_results)
                    self.obs['padded'].setdefault(confidence, []).extend(padded_results)
        
        return self.obs

    def training_file(self,file = None, direc = '.'):
        filename = file.split(".")[0]
        boxes_doc = os.path.join(direc,filename + '_map' + ".txt")
        if os.path.exists(boxes_doc):
            return boxes_doc
        else:
            return None

    def plotprecisionrecall(self):
        plt.figure(1) 
        plt.figure(2)

        for padding_type, result_data in self.obs.items():
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
                plt.plot(recall, precision, ".", label=f"{padding_type + ' borders'}")
            else:
                plt.plot(recall, precision, ".-", label=f"{padding_type + ' borders'}")


            plt.figure(2)
            if len(set(recall)) == 1 and len(set(precision)) == 1:
                plt.plot(recall, precision, ".", label=f"{padding_type + ' borders'}")
            else:
                plt.plot(recall, precision, ".-", label=f"{padding_type + ' borders'}")  

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


class MonosImprovement(TestModel):
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
    
    
