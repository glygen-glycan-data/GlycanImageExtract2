#set matplotlib backend to avoid cv2 conflicts
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

#import framework for getting boxes, comparing them
from BKGLycanExtractor import glycanrectangleid
from BKGLycanExtractor import compareboxes
from BKGLycanExtractor import modeltests

import sys, logging, cv2, os, random

import numpy as np


#get training box file
def get_training_box_doc(file = None, direc = '.'):
    filename = file.split(".")[0]
    boxes_doc = os.path.join(direc,filename+".txt")
    if os.path.exists(boxes_doc):
        return boxes_doc
    else:
        return None

#make plot
def plotprecisionrecall(*args):
    #args is dictionaries of precision/recall values at different confidences; dictionaries differ by some alg
    for dictionary in args:
        precision = []
        recall = []
        name = dictionary.get("name",'')
        dictionary.pop("name")
        
        for conf,results_list in dictionary.items():
            #print(results_list)
            fp = results_list.count("FP")
            tp = results_list.count("TP")
            pos = fp + tp
            fn = results_list.count("FN")
            tpfn = tp+fn
            try: 
                prec = tp/pos
            except ZeroDivisionError:
                prec = 0
            rec = tp/tpfn
            precision.append(prec)
            recall.append(rec)
            logger.info(f"Confidence 0.{conf}: Precision {prec}, recall {rec}")
            
        recall, precision = zip(*sorted(zip(recall, precision)))
        plt.figure(1)
        if len(set(recall)) == 1 and len(set(precision)) == 1:
            plt.plot(recall,precision, ".", label = name)
        else:
            plt.plot(recall,precision, ".-",label = name)
        plt.figure(2)
        if len(set(recall)) == 1 and len(set(precision)) == 1:
            plt.plot(recall,precision, ".", label = name)
        else:
            plt.plot(recall,precision, ".-",label = name)
    plt.figure(1)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([0.0,1.1])
    plt.ylim([0.0,1.1])
    plt.axhline(y=1, color='k', linestyle='--')
    plt.axvline(x=1, color='k', linestyle='--')
    plt.legend(loc="best")
    plt.figure(2)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([0.5,1.1])
    plt.ylim([0.5,1.1])
    plt.axhline(y=1, color='k', linestyle='--')
    plt.axvline(x=1, color='k', linestyle='--')
    plt.legend(loc="best")
    
    pr = plt.figure(1)
    pr_zoom = plt.figure(2)
    return pr, pr_zoom


#location for yolo configuration files
base_configs = "./BKGLycanExtractor/configs/"
weight1=base_configs+"yolov3_training_final.weights"
weight2=base_configs+"Glycan_300img_5000iterations.weights"
weight3 = base_configs +"retrain_v2.weights"
weight4 = base_configs + "largerboxes_plusindividualglycans.weights"
coreyolo=base_configs+"coreyolo.cfg"


#provide directory to search and directory to save plot
search_direc = sys.argv[1]
output_direc = sys.argv[2]

training_box_interpreter = glycanrectangleid.TrainingData()
padded_box_comparer = compareboxes.ComparePaddedBox()
unpadded_box_comparer = compareboxes.CompareRawBox()
glycan_checker = modeltests.TestModel()


#methods for comparison    
annotator = glycanrectangleid.OriginalYOLO(weights = weight1, net = coreyolo)
annotator2 = glycanrectangleid.OriginalYOLO(weights = weight2, net = coreyolo)
annotator3 = glycanrectangleid.OriginalYOLO(weights = weight3, net = coreyolo)
annotator4 = glycanrectangleid.OriginalYOLO(weights = weight4, net = coreyolo)

annotator_dict = {
    "original weights" : annotator2,
    "new weights" : annotator,
    "retrain v2" : annotator3,
    "larger training boxes" : annotator4
    }


if os.path.isdir(os.path.join(output_direc)):
    pass
else:
    os.makedirs(os.path.join(output_direc))

#set up logging
logger = logging.getLogger("test")
logger.setLevel(logging.INFO)

annotatelogfile = f"{output_direc}/prc_log.txt"
if os.path.isfile(annotatelogfile):
    tag = str(random.randint(1,100))
    annotatelogfile = f"{output_direc}/prc_log"+tag+".txt"
    print("Directory already contains prc curve log - check")

handler = logging.FileHandler(annotatelogfile)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.info("Start.\n")

#set up dictionaries
results_dicts = []
for desc, annotator in annotator_dict.items():
    unpadded_dict = {
                "0": [],
                "1": [],
                "2": [],
                "3": [],
                "4": [],
                "5": [],
                "6": [],
                "7": [],
                "8": [],
                "9": [],
                "name": "padded borders "+desc
                }
    padded_dict = {
                "0": [],
                "1": [],
                "2": [],
                "3": [],
                "4": [],
                "5": [],
                "6": [],
                "7": [],
                "8": [],
                "9": [],
                "name": "unpadded borders "+desc
                }
    results_dicts.append(unpadded_dict)
    results_dicts.append(padded_dict)

# list all files in dir
glycan_files = [file for file in os.scandir(search_direc) if os.path.isfile(file) and file.name.endswith("png")]

# select 100 of the files randomly 
random_files = np.random.choice(glycan_files, 100)


for glycan_file in random_files:
    break_flag = False
    
    name = glycan_file.name        
    if name.endswith("png"):
        print(name, "Start")
        logger.info(f"{name}: Start")
        longname = os.path.join(search_direc,name)
        image = cv2.imread(longname)
        training_box_doc = get_training_box_doc(file = name, direc = search_direc)
        if not training_box_doc:
            issue = f"No training data for image {name}."
            logger.warning(issue)
            print(issue)
            logger.info(f"Finished: {name}")
            continue
        training = training_box_interpreter.get_rects(image=image,coord_file = training_box_doc)
        
        #run each annotation method once, get results for each tested confidence level
        for desc,annotation_method in annotator_dict.items():
            logger.info(desc)
            
            unpadded_boxes, padded_boxes = annotation_method.get_rects(image = image)
            for j, confidence in enumerate([x*0.1 for x in range(0,10,1)]):
                logger.info(f'Confidence: {confidence}')
                logger.info("Padded boxes")
                padded_results = glycan_checker.compare(padded_boxes, training, padded_box_comparer, confidence)
                if j == 0:
                    if training == []:
                        if len(padded_boxes) > 0:
                            issue = f"False positive: No training glycans for image {name}."
                            logger.warning(issue)
                        else:
                            issue = f"No training glycans for image {name}."
                            logger.warning(issue)
                        break_flag = True
                        break
                    else:
                        if padded_results[0] == "FN" and len(set(padded_results)) == 1:
                            issue = "False negative: No detected glycans for image."
                            logger.warning(issue)
                logger.info("Unpadded boxes")
                unpadded_results = glycan_checker.compare(unpadded_boxes,training,unpadded_box_comparer, confidence)
                
                for dictionary in results_dicts:
                    if dictionary.get("name") == "unpadded borders "+desc:
                        [dictionary[str(j)].append(result) for result in unpadded_results]
                    elif dictionary.get("name") == "padded borders "+desc:
                        [dictionary[str(j)].append(result) for result in padded_results]

            if break_flag:
                continue
            
    elif name.endswith(".txt"):
        continue
    else:
        extn = name.split(".")[1]
        print('File %s had an Unsupported file extension: %s.'%(name,extn))
        logger.warning('File %s had an Unsupported file extension: %s.'%(name,extn))
        continue
    logger.info("%s Finished", name)


#make plot    
prc, zoomprc = plotprecisionrecall(*results_dicts)    


#save plots

path = f"{output_direc}/precisionrecallplot.png"
if os.path.isfile(path):
    path = f"{output_direc}/precisionrecallplot"+tag+".png"
    print("Directory already contains prc curve file - check")
plt.figure(1)
plt.savefig(path)
plt.close()

zoompath = f"{output_direc}/precisionrecallplot_zoomed.png"
if os.path.isfile(zoompath):
    path = f"{output_direc}/precisionrecallplot_zoomed"+tag+".png"
    print("Directory already contains zoomed prc curve file - check")
plt.figure(2)
plt.savefig(zoompath)
plt.close()

#close handlers
handlers = logger.handlers[:]
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()