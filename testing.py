import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import BKGLycanExtractor.SystemInteraction as si
import BKGLycanExtractor.glycanRectID as rectID
# import BKGLycanExtractor.MonoID as ms
# import BKGLycanExtractor.glycanConnections as c
# import BKGLycanExtractor.buildStructure as b
# import BKGLycanExtractor.glycanSearch as gs
import BKGLycanExtractor.compareBoxes as cb
import BKGLycanExtractor.modelTests as mt

#from BKGLycanExtractor.pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError

import sys, logging, cv2, os

def get_results(image,training,confidence,method,padded_comparison_method,raw_comparison_method):
    #compare padded boxes
    logger.info("Padded boxes (default):\n")
    padded_boxes = method.getRects(image = image, threshold = confidence)
    padded_results = glycan_checker.compare(padded_boxes, training, padded_comparison_method)
    if j == 0:
        if training == []:
            if len(padded_boxes) > 0:
                issue = f"False positive: No training glycans for image {str(item)}."
                logger.warning(issue)
                framework.find_problems("False positive: No training glycans.")
            else:
                issue = f"No training glycans for image {str(item)}."
                logger.warning(issue)
            logger.info("%s Finished", str(item))
            framework.close_log()
            return False
        else:
            if padded_results[0] == "FN" and len(set(padded_results)) == 1:
                issue = f"False negative: No detected glycans for image {str(item)}."
                logger.warning(issue)
                framework.find_problems("False negative: No detected glycans.")
    #compare raw boxes
    logger.info("Raw boxes:\n")
    raw_boxes = annotator.getRects(image = image, threshold = confidence, pad = False)
    raw_results = glycan_checker.compare(raw_boxes,training,raw_comparison_method)
    return padded_results,raw_results

def plotprecisionrecall(*args):
    #args is dictionaries of precision/recall values at different confidences; dictionaries differ by some alg
    for dictionary in args:
        precision = []
        recall = []
        name = dictionary.get("name",'')
        dictionary.pop("name")
        
        for results_list in dictionary.values():
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
            
        recall, precision = zip(*sorted(zip(recall, precision)))
        
        if len(set(recall)) == 1 and len(set(precision)) == 1:
            plt.plot(recall,precision, ".", label = name)
        else:
            plt.plot(recall,precision, ".-",label = name)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.axhline(y=1, color='k', linestyle='--')
    plt.axvline(x=1, color='k', linestyle='--')
    plt.legend(loc="best")
    pr = plt.gcf()
    return pr

logger = logging.getLogger("test")
logger.setLevel(logging.INFO)

framework = si.TestModel()
base_configs = "./BKGLycanExtractor/configs/"
weight1=base_configs+"yolov3_training_final.weights"
weight2=base_configs+"Glycan_300img_5000iterations.weights"
coreyolo=base_configs+"coreyolo.cfg"
#colors_range=base_configs+"colors_range.txt"

#color_range_dict = framework.get_color_range(colors_range)

# configs = {
#     "weights" : weight,
#     "net" : coreyolo,
#     "colors_range": color_range_dict}
search_direc = sys.argv[1]

training_box_interpreter = rectID.TrainingData()
padded_box_comparer = cb.ComparePaddedBox()
raw_box_comparer = cb.CompareRawBox()
glycan_checker = mt.TestModel()

    
annotator = rectID.originalYOLO(weights = weight1, net = coreyolo)
annotator2 = rectID.originalYOLO(weights = weight2, net = coreyolo)

for file in os.scandir(search_direc):
    break_flag = False
    
    item = file.name
    if item.endswith("txt") or item.endswith("zip"):
        continue

    print(item, "Start")
    longname = os.path.join(search_direc,item)
    
    framework.initialize_directory(name = item)
    
    workdir = framework.get_directory(name = item)
    
    framework.log(name = item, logger = logger)
    
    padded_results_dict_new = {
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
        "name": "padded borders new weights"
        }
    
    raw_results_dict_new = {
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
        "name": "raw borders new weights"
        }
    
    padded_results_dict_old = {
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
        "name": "padded borders old weights"
        }
    
    raw_results_dict_old = {
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
        "name": "raw borders old weights"
        }
    
    
    if item.endswith("png"):
        image = cv2.imread(longname)
        training_box_doc = framework.get_training_box_doc(file = item, direc = search_direc)
        if not training_box_doc:
            issue = f"No training data for image {str(item)}."
            logger.warning(issue)
            print(issue)
            logger.info("%s Finished", str(item))
            framework.close_log()
            continue
        training = training_box_interpreter.getRects(image=image,coord_file = training_box_doc)
        for j,confidence in enumerate([x*0.1 for x in range(0,10,1)]):
            logger.info(f'Confidence: {confidence}')
            logger.info("New weights")
            results = get_results(image, training, confidence, annotator, padded_box_comparer, raw_box_comparer)
            if not results:
                break_flag = True
                break
            padded_results, raw_results = results
            [padded_results_dict_new[str(j)].append(result) for result in padded_results]
            [raw_results_dict_new[str(j)].append(result) for result in raw_results]
            logger.info("Old weights")
            results = get_results(image, training, confidence, annotator2, padded_box_comparer, raw_box_comparer)
            if not results:
                break_flag = True
                break
            padded_results, raw_results = results
            [padded_results_dict_old[str(j)].append(result) for result in padded_results]
            [raw_results_dict_old[str(j)].append(result) for result in raw_results]
        if break_flag:
            continue
            
    else:
        extn = item.split(".")[1]
        print('File %s had an Unsupported file extension: %s.'%(item,extn))
        logger.warning('File %s had an Unsupported file extension: %s.'%(item,extn))
        continue
    
    prc = plotprecisionrecall(padded_results_dict_new,raw_results_dict_new,padded_results_dict_old,raw_results_dict_old)    
    framework.save_output(file = item, image = prc)
    
    logger.info("%s Finished", item)
    framework.close_log()