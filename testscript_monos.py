# -*- coding: utf-8 -*-
"""
Create a precision recall curve 
and a q-value curve to assess detection success
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

import BKGlycanExtractor.compareboxes as compareboxes
import BKGlycanExtractor.glycanannotator as glycanannotator
from BKGlycanExtractor.yolomodels import YOLOTrainingData

def find_best_match(
        training_name, match_dictionary, pairs, conf, disallowed_box=None
        ):
    tbox, matches = match_dictionary[training_name]
    if disallowed_box is None:
        for dbox in matches:
            dbox_name = dbox.get_name()
            if dbox.get_confidence() >= conf:
                if dbox_name not in pairs:
                    pairs[dbox_name] = tbox
                    return
                else:
                    prev_t = pairs[dbox_name]
                    prev_iou = box_comparer.iou(prev_t, dbox)
                    new_iou = box_comparer.iou(tbox, dbox)
                    if new_iou > prev_iou:
                        pairs[dbox_name] = tbox
                        find_best_match(
                            prev_t.get_name(), match_dictionary,
                            pairs, conf, dbox
                            )
                    else:
                        continue
        return
    else:
        found = False
        for dbox in matches:
            if dbox == disallowed_box:
                found = True
                continue
            if found:
                dbox_name = dbox.get_name()
                if dbox.get_confidence() >= conf:
                    if dbox_name not in pairs:
                        pairs[dbox_name] = tbox
                        return
                    else:
                        prev_t = pairs[dbox_name]
                        prev_iou = box_comparer.iou(prev_t, dbox)
                        new_iou = box_comparer.iou(tbox, dbox)
                        if new_iou > prev_iou:
                            pairs[dbox_name] = tbox
                            find_best_match(
                                prev_t.get_name(), match_dictionary,
                                pairs, conf, dbox
                                )
                        else:
                            continue
        return
    
def get_training_box_doc(file = None, direc = '.'):
    filename = file.split(".")[0]
    boxes_doc = os.path.join(direc,filename+".txt")
    if os.path.exists(boxes_doc):
        return boxes_doc
    else:
        return None

def plot_prc(label, confidences, matched_boxes):
    # print("confidences", confidences)
    # print("matched boxes", matched_boxes)
    confidences = list(confidences)
    confidences.sort()
    
    precision = []
    recall = []
    for conf in confidences:
        results = []
        # low to high threshold
        for image, training_matches, detections in matched_boxes:
            pairs = {}
            for name, value in training_matches.items():
                find_best_match(name, training_matches, pairs, conf)
            # print(pairs)
            # pairs is now full, dboxname: tbox
            for name, value in training_matches.items():
                tbox = value[0]
                if tbox in pairs.values():
                    results.append("TP")
                    # tbox.annotate(image, colour=(255,0,0))
                else:
                    results.append("FN")
                    # print("training not in pairs")
            for detection in detections:
                if detection.get_confidence() >= conf:
                    try:
                        if detection.get_name() not in pairs:
                            results.append("FP")
                            # print("detection not matched")
                    # else:
                        # detection.annotate(image, colour=(0,255,0))
                    except AttributeError:
                        results.append("FP")
                        # print("detection not matched")
            # cv2.imshow("match", image)
        
            # cv2.waitKey(0)
            # cv2.destroyAllWindows
        fp = results.count("FP")
        tp = results.count("TP")
        pos = fp + tp
        fn = results.count("FN")
        tpfn = tp + fn
        try: 
            prec = tp/pos
        except ZeroDivisionError:
            prec = 0
        rec = tp/tpfn
        
        if prec <= 0 and rec <= 0:
            continue
        precision.append(prec)
        recall.append(rec)
        logger.info(f"Confidence {conf}: Precision {prec}, recall {rec}")
        # print(f"Confidence {conf}: Precision {prec}, recall {rec}")    
        
    recall, precision = zip(*sorted(zip(recall, precision)))
    logger.info(f"recall: {recall}")
    logger.info(f"precision: {precision}")
    if len(set(recall)) == 1 and len(set(precision)) == 1:
        plt.plot(recall, precision, ".", label=label)
    else:
        plt.plot(recall, precision, "-",label=label)
        
def plot_qvalue(label, confidences, matched_boxes):
    confidences = list(confidences)
    confidences.sort()
    
    errorrate = []
    preds = []
    for conf in confidences:
        results = []
        # low to high threshold
        for image, training_matches, detections in matched_boxes:
            pairs = {}
            for name, value in training_matches.items():
                find_best_match(name, training_matches, pairs, conf)
            # pairs is now full, dboxname: tbox
            for name, value in training_matches.items():
                tbox = value[0]
                if tbox in pairs.values():
                    results.append("TP")
                    # tbox.annotate(image, colour=(255,0,0))
                else:
                    results.append("FN")
                    # print("training not in pairs")
            for detection in detections:
                if detection.get_confidence() >= conf:
                    try:
                        if detection.get_name() not in pairs:
                            results.append("FP")
                            # print("detection not matched")
                    # else:
                        # detection.annotate(image, colour=(0,255,0))
                    except AttributeError:
                        results.append("FP")
                        # print("detection not matched")
            # cv2.imshow("match", image)
        
            # cv2.waitKey(0)
            # cv2.destroyAllWindows
        fp = results.count("FP")
        tp = results.count("TP")
        pos = fp + tp
        try:
            dec_value = tp / pos
        except ZeroDivisionError:
            dec_value = 0
        logger.info(
            f"Confidence {conf}: {tp} correct predictions / {pos} predictions"
            )
        errorrate.append(1 - dec_value)
        preds.append(pos)
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

detection_threshold = 0.80
overlap_threshold = 0.75
containment_threshold = 0.60

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
    models = "YOLOMonosLargerBoxes YOLOMonos HeuristicMonos"
if os.path.isdir(models):
    models = "YOLOMonosLargerBoxes YOLOMonos HeuristicMonos"
    
# set type of model to be tested
# you should change this in the program
# it may affect the comparison methods you want to use
prefix = "monosaccharideid."

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

models = models.split()
method_dict = {}    
for model in models:
    method = annotator.read_one_config(
        configs_dir, configs_file, prefix, model
        )
    method_dict[model] = method

counted = False
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
        longname = os.path.abspath(glycan_file)
        image = cv2.imread(longname)
        training_box_doc = get_training_box_doc(
            file = name, direc = glycan_folder
            )
        
        trained_monos = training_box_interpreter.read_boxes(
            image, training_box_doc
            )
        if not counted:
            for mono in trained_monos:
                max_predictions += 1
        
        detection_info = method.find_monos(image, threshold=0.0, test=True)
        detections = detection_info.get_monos()
        # print(detections)
        
        training_matches = box_comparer.match_to_training(
            trained_monos, detections
            )
        
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

        results.append((image, training_matches, detections))
    
    counted = True
    plt.figure(1)
    plot_prc(desc, confidences, results)
    
    plt.figure(2)
    plot_qvalue(desc, confidences, results)
    
plt.figure(1)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.xlim([0.0, 1.1])
plt.ylim([0.0, 1.1])
plt.axhline(y=1, color='k', linestyle='--')
plt.axvline(x=1, color='k', linestyle='--')
plt.legend(loc="best")

impath = os.path.abspath("./prc.png")
plt.savefig(impath)
plt.close(1)

plt.figure(2)
plt.ylabel('Total predictions')
plt.xlabel('Error rate')
plt.ylim(0.0, max_predictions+0.1*max_predictions)
plt.xlim([0.0,1.1])
plt.axhline(y=max_predictions, color = 'k', linestyle ='--')
plt.axvline(x=1, color='k', linestyle='--')
plt.legend(loc="best")

impath = os.path.abspath("./qvalue.png")
plt.savefig(impath)
plt.close(2)

#close handlers
handlers = logger.handlers[:]
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()