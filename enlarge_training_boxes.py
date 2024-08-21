import BKGLycanExtractor.glycanrectangleid as glycanrectangleid

import sys, cv2, os

import numpy as np


#get training box file
def get_training_box_doc(file = None, direc = '.'):
    filename = file.split(".")[0]
    boxes_doc = os.path.join(direc,filename+".txt")
    if os.path.exists(boxes_doc):
        return boxes_doc
    else:
        return None


training_direc = sys.argv[1]
training_box_interpreter = glycanrectangleid.TrainingData()

    
for glycan_file in os.scandir(training_direc):
    
    name = glycan_file.name
    if name.endswith("txt") or name.endswith("zip"):
        continue

    print(name, "Start")
    longname = os.path.join(training_direc,name)
    
    if name.endswith("png"):
        image = cv2.imread(longname)
        height, width, channels = image.shape
        ############################################################################################
        #fix issue with
        ############################################################################################
        white_space = 200
        bigwhite = np.zeros([image.shape[0] +white_space, image.shape[1] +white_space, 3], dtype=np.uint8)
        bigwhite.fill(255)
        half_white_space = white_space//2
        bigwhite[half_white_space:(half_white_space + image.shape[0]), half_white_space:(half_white_space+image.shape[1])] = image
        newimage = bigwhite.copy()
        training_box_doc = get_training_box_doc(file = name, direc = training_direc)
        if not training_box_doc:
            issue = f"No training data for image {name}."
            #logger.warning(issue)
            print(issue)
            continue
        training = training_box_interpreter.getRects(image=image,coord_file = training_box_doc)
        for box in training:
            box.resetImage(white_space)
            box.padBorders()
            box.AbsToRel()
            
    else:
        extn = name.split(".")[1]
        print('File %s had an Unsupported file extension: %s.'%(name,extn))
        #logger.warning('File %s had an Unsupported file extension: %s.'%(item,extn))
        continue
    
    im_path = os.path.join(training_direc,name)
    f = open(training_box_doc,"w")
    for box in training:
        toprint = box.toList()
        for x in toprint:
            f.write(str(x) + " ")
        f.write("\n")
    f.close()
    cv2.imwrite(im_path,newimage)