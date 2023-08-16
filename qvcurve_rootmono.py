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
        for image, true_root, detection_info in detections:
            detected_monos = detection_info.get_monos()
            result = False
            break_flag = False
            for mono in detected_monos:
                if mono.is_root():
                    if Decimal(str(round(mono.get_confidence(), 4))) >= conf:
                        if box_comparer.have_intersection(true_root, mono):
                            t_area = true_root.area()
                            d_area = mono.area()
                            inter = box_comparer.intersection_area(true_root, mono)
                            if inter == t_area:
                                if box_comparer.training_contained(true_root, mono):
                                    result = True
                            elif inter == d_area:
                                if box_comparer.detection_sufficient(true_root, mono):
                                    result = True
                            else:
                                if box_comparer.is_overlapping(true_root, mono):
                                    result = True
                    else:
                        break_flag = True
                    break
            if break_flag:
                continue
                    
            results.append(result)
            
        try:
            dec_value = sum(results) / len(results) 
        except ZeroDivisionError:
            dec_value = 0
        logger.info(f"Confidence {conf}: {sum(results)} correct predictions / {len(results)} predictions")
        errorrate.append(1 - dec_value)
        preds.append((results))

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

detection_threshold = 0.5
overlap_threshold = 0.5
containment_threshold = 0.5

annotator = glycanannotator.Annotator()
training_box_interpreter = YOLOTrainingData()

box_comparer = compareboxes.CompareDetectedClass(
    detection_threshold = detection_threshold, 
    overlap_threshold = overlap_threshold, 
    containment_threshold = containment_threshold
    )

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
    models = "YOLORootFinder YOLOOrientationRootFinder DefaultOrientationRootFinder"
if os.path.isdir(models):
    models = "YOLORootFinder YOLOOrientationRootFinder DefaultOrientationRootFinder"
    
# set type of model to be tested
# you should change this in the program
# it may affect the comparison methods you want to use
prefix = "rootmonofinding."

# I'm testing these all with YOLOMonos finding
mono_finder = annotator.read_one_config(configs_dir, configs_file, "monosaccharideid.", "YOLOMonos")
connector = annotator.read_one_config(configs_dir, configs_file, "glycanconnections.", "OriginalConnector")

print("Testing models", models, "of type", prefix)
print("If type seems wrong, look at the script. You need to edit comparison finding for the appropriate type.")

logger = logging.getLogger("test")
logger.setLevel(logging.INFO)

annotatelogfile = os.path.abspath("./testing_log.txt")

handler = logging.FileHandler(annotatelogfile)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.info("Start.\n")
logger.info("IOU thresholds:")
logger.info("Monosaccharide fully contained in box: "+str(containment_threshold))
logger.info("Detection smaller than monosaccharide: "+str(detection_threshold))
logger.info("Overlapping: "+str(overlap_threshold))

mono_finder.set_logger("test")
connector.set_logger("test")

models = models.split()
method_dict = {}    
for model in models:
    method = annotator.read_one_config(
        configs_dir, configs_file, prefix, model
        )
    method_dict[model] = method

max_predictions = 0

# # # list all files in dir
# glycan_files = [file for file in os.scandir(glycan_folder) if os.path.isfile(file) and file.name.endswith("png")]

# # # select 100 of the files randomly 
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
        longname = os.path.abspath(glycan_file)
        image = cv2.imread(longname)
        training_box_doc = get_training_box_doc(
            file = name, direc = glycan_folder
            )
        
        if training_box_doc is None:
            continue
        max_predictions += 1
        
        training = training_box_interpreter.read_boxes(
            image, training_box_doc
            )
        training_root = None
        for x in training:
            if x.class_ == 0:
                training_root = x
                break
        
        monos_info = mono_finder.find_monos(image, threshold=0.0, test=True)
        connector.connect(monos_info)
        rootconf = method.find_root_mono(monos_info)
        
        # print(rootconf)
        
        if rootconf is not None:
            confidences.add(Decimal(str(round(rootconf, 4))))
        
        
        # showim = image.copy()
        
        # for training in trained_monos:
        #     training.annotate(showim, colour=(255,0,0))
            
        # for dec in detections:
        #     dec.annotate(showim, colour=(0,255,0))
            
        # cv2.imshow('boxes', showim)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows

        results.append((image, training_root, monos_info))
    
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