# -*- coding: utf-8 -*-
"""
Create q-value curve for composition accuracy
"""

# set matplotlib backend to avoid cv2 conflicts
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from decimal import Decimal
import logging
import os
import sys

import cv2
import numpy as np

import BKGlycanExtractor.compareboxes as compareboxes
import BKGlycanExtractor.glycanannotator as glycanannotator
from BKGlycanExtractor.yolomodels import YOLOTrainingData

def check_composition(known_comp_dict, model_comp_dict):
    known_total = 0
    model_total = 0
    for mono, amount in known_comp_dict.items():
        known_total += amount
        if int(amount) != int(amount):
            return False
    for mono, amount in model_comp_dict.items():
        model_total += amount
    if int(known_total) != int(model_total):
        return False
    return True
    
# get composition count dictionary from log file
def get_count_dict(logfile=None):
    log_dict = {}
    log_file = open(logfile)
    for line in log_file.readlines():
        key = line.split(":")[0].strip()
        value = line.split(":")[1].strip()
        log_dict[key] = value
    log_file.close()
    
    true_count_dict = {}
    countstring = log_dict["composition"]
    countlist = countstring.split(" ")
    countlist = countlist[2:]
    for i in range(0, len(countlist), 2):
        true_count_dict[countlist[i]] = int(countlist[i+1])
    # "GlcNAc": 0, "NeuAc": 0, "Fuc": 0, "Man": 0, "GalNAc": 0, "Gal": 0
    if "GlcNAc" not in true_count_dict:
        true_count_dict["GlcNAc"] = 0
    if "NeuAc" not in true_count_dict:
        true_count_dict["NeuAc"] = 0
    if "Fuc" not in true_count_dict:
        true_count_dict["Fuc"] = 0
    if "Man" not in true_count_dict:
        true_count_dict["Man"] = 0
    if "GalNAc" not in true_count_dict:
        true_count_dict["GalNAc"] = 0
    if "Gal" not in true_count_dict:
        true_count_dict["Gal"] = 0
    if "Glc" not in true_count_dict:
        true_count_dict["Glc"] = 0
    if "NeuGc" not in true_count_dict:
        true_count_dict["NeuGc"] = 0
    return true_count_dict

# get log file
def get_log_doc(file=None, direc='.'):
    filename = file.split(".")[0]
    log_doc = os.path.join(direc, filename + ".log")
    if os.path.exists(log_doc):
        return log_doc
    else:
        return None
    
def get_training_box_doc(file=None, direc='.'):
    filename = file.split(".")[0]
    boxes_doc = os.path.join(direc, filename + ".txt")
    if os.path.exists(boxes_doc):
        return boxes_doc
    else:
        return None
        
def plot_qvalue(label, confidences, detections):
    confidences = list(confidences)
    confidences.sort()
    
    errorrate = []
    preds = []
    for conf in confidences:
        results = []
        # low to high threshold
        for glycanname, true_comp, detection_info in detections:
            logger.info(glycanname)
            detected_monos = detection_info.get_monos()
            break_flag = False
            for mono in detected_monos:
                # treat this as a failure to predict
                if mono.get_confidence() < conf:
                    break_flag = True

                    itext = f"Mono {mono.get_class()} at [{mono.to_list()}] has confidence less than {conf}.\nDropping prediction."
                    logger.info(itext)
                    break
            # continue - we get no prediction from this result at this conf
            if break_flag:
                continue
                    
            comp_dict = detection_info.get_composition()
            result = check_composition(true_comp, comp_dict)
            results.append(result)
            
        try:
            dec_value = sum(results) / len(results) 
        except ZeroDivisionError:
            dec_value = 0
        logger.info(f"Confidence {conf}: {sum(results)} correct predictions / {len(results)} predictions")
        errorrate.append(1 - dec_value)
        preds.append(len(results))

    minerror = min(errorrate)
    errorrate.append(minerror)
    preds.append(0)
    sorteder, sortedpreds = zip(*sorted(zip(errorrate, preds)))
    sorteder, sortedpreds = list(sorteder), list(sortedpreds)
    steper, steppreds = sorteder.copy(), sortedpreds.copy()
    if (len(set(sorteder)) == 1 and len(set(sortedpreds)) == 1):
        logger.info(f"predictions: {sortedpreds}")
        logger.info(f"error rate: {sorteder}")
        plt.plot(sorteder, sortedpreds, ".", label=label)
    else:
        to_add_x = []
        to_add_y = []
        for idx, yval in enumerate(steppreds):
            try:
                if steppreds[idx+1] < yval:
                    steppreds[idx+1] = yval
            except IndexError:
                continue
            try:
                if (steper[idx+1] > steper[idx]) and (steppreds[idx+1] > yval):
                    to_add_x.append(steper[idx+1])
                    to_add_y.append(yval)
            except IndexError:
                continue
        steper.extend(to_add_x)
        steppreds.extend(to_add_y)
        ploter, plotpreds = zip(*sorted(zip(steper, steppreds)))
        logger.info(f"predictions: {plotpreds}")
        logger.info(f"error rate: {ploter}")
        plt.plot(ploter, plotpreds, "-", label=label)

glycan_folder = sys.argv[1]

# detection_threshold = 0.80
# overlap_threshold = 0.75
# containment_threshold = 0.60

annotator = glycanannotator.Annotator()
training_box_interpreter = YOLOTrainingData()
# box_comparer = compareboxes.CompareDetectedClass(
#     detection_threshold = detection_threshold, 
#     overlap_threshold = overlap_threshold, 
#     containment_threshold = containment_threshold
#     )

try:
    configs_dir = sys.argv[3]
except IndexError:
    configs_dir = "./BKGlycanExtractor/config/"
configs_file = os.path.join(configs_dir, "configs.ini")

# provide models as space-separated list in quotes 
# i.e. "YOLOMonos YOLOMonos2"
try:
    models = sys.argv[2]
except IndexError:
    models = "YOLOMonosLargerBoxes YOLOMonos HeuristicMonos"
if os.path.isdir(models):
    models = "YOLOMonosLargerBoxes YOLOMonos HeuristicMonos"
    
# set type of model to be tested
# you should change this in the program
# it may affect the comparison methods you want to use
prefix = "monosaccharideid."

print("Testing models", models, "of type", prefix, "for composition.")
print("If type seems wrong, look at the script. You need to edit comparison finding for the appropriate type.")

logger = logging.getLogger("test")
logger.setLevel(logging.INFO)

annotatelogfile = os.path.abspath("./testing_log.txt")

handler = logging.FileHandler(annotatelogfile)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.info("Start.\n")
# logger.info("IOU thresholds:")
# logger.info("Monosaccharide fully contained in box: "+str(containment_threshold))
# logger.info("Detection smaller than monosaccharide: "+str(detection_threshold))
# logger.info("Overlapping: "+str(overlap_threshold))

models = models.split()
method_dict = {}    
for model in models:
    method = annotator.read_one_config(
        configs_dir, configs_file, prefix, model
        )
    method_dict[model] = method

max_predictions = 0

# # list all files in dir
# glycan_files = [file for file in os.scandir(glycan_folder) if os.path.isfile(file) and file.name.endswith("png")]

# # select 100 of the files randomly 
# random_files = np.random.choice(glycan_files, 100)

for desc, method in method_dict.items():
    print("Method:", desc)
    logger.info(f"Method: {desc}")
    results = []
    confidences = set()
    method.set_logger("test")

    for glycan_file in os.scandir(glycan_folder):
        name = glycan_file.name
        if not name.endswith("png"):
            continue
        print(name)
        logger.info(name)
        max_predictions += 1
        longname = os.path.abspath(glycan_file)
        image = cv2.imread(longname)
        training_box_doc = get_training_box_doc(
            file = name, direc = glycan_folder
            )
        
        log = get_log_doc(name, glycan_folder)
        if log is None:
            issue = f"No log file for image {name}."
            logger.warning(issue)
            print(issue)
            logger.info(f"Finished: {name}")
            continue
        
        true_comp_dict = get_count_dict(log)
        
        trained_monos = training_box_interpreter.read_boxes(
            image, training_box_doc
            )
        
        detection_info = method.find_monos(image, threshold=0.0, test=True)
        detections = detection_info.get_monos()
        # print(detections)
        
        confidences.update(
            [Decimal(str(round(detection.get_confidence(), 4))) for detection in detections]
            )
        
        
        # showim = image.copy()
        
        # for training in trained_monos:
        #     training.annotate(showim, colour=(255,0,0))
            
        # for dec in detections:
        #     dec.annotate(showim, colour=(0,255,0))
            
        # cv2.imshow('boxes', showim)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows

        results.append((name, true_comp_dict, detection_info))
    
    plt.figure(1)
    plot_qvalue(desc, confidences, results)

max_predictions = max_predictions / len(models)
plt.figure(1)
plt.ylabel('Total predictions')
plt.xlabel('Error rate')
plt.ylim(0.0, max_predictions+0.1*max_predictions)
plt.xlim([0.0,1.1])
plt.axhline(y=max_predictions, color = 'k', linestyle ='--')
plt.axvline(x=1, color='k', linestyle='--')
plt.legend(loc="best")

impath = os.path.abspath("./qvalue.png")
plt.savefig(impath)
plt.close(1)

#close handlers
handlers = logger.handlers[:]
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()