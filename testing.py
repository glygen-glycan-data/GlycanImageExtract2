
import BKGLycanExtractor.SystemInteraction as si
from BKGLycanExtractor.inputhandler import InputHandler
import BKGLycanExtractor.modelTests as mt
import BKGLycanExtractor.glycanRectID as rectid


import cv2, fitz, os, logging
import numpy as np

logger = logging.getLogger("test")
logger.setLevel(logging.INFO)


training_glycans = "./training_glycans/"

base_configs = "./BKGLycanExtractor/configs/"
weight=base_configs+"Glycan_300img_5000iterations.weights"
coreyolo=base_configs+"coreyolo.cfg"

for file in os.scandir(training_glycans):
    
    filename = file.name
    if filename.endswith("txt"):
        continue

    item = InputHandler(filename)
    
    #hasglycan = sys.argv[2]

    
    print(item, "Start")
    

    # colors_range=base_configs+"colors_range.txt"
    
    # color_range_file = open(colors_range)
    # color_range_dict = {}
    # for line in color_range_file.readlines():
    #     line = line.strip()
    #     name = line.split("=")[0].strip()
    #     color_range = line.split("=")[1].strip()
    #     color_range_dict[name] = np.array(list(map(int, color_range.split(","))))
    
    framework = si.TestModel(str(item), "original_yolo")
    
    framework.initialize_directory()
    
    workdir = framework.get_directory()
    
    
    framework.log()
    
    if item.is_file:
        framework.make_copy()
        extn = item.ext
        
        if extn in ('png','jpg','jpeg'):
            image = cv2.imread(item.item)
            model = rectid.originalYOLO(image, weight = weight, coreyolo = coreyolo)
            test = mt.GlycanFound(image,model)             
            isglycan = test.isGlycan()
            logger.info('File %s has at least one glycan: %s.'%(item,isglycan))
            if not isglycan:
                print("No glycan in image.")
                logger.info("%s Finished", str(item))
                framework.close_log()
                framework.delete_copy()
                continue
            
            training_box_doc = framework.get_training_box_doc()
            if not training_box_doc:
                logging.warning(f"No training data for image {str(item)}.")
                print(f"No training data for image {str(item)}.")
                logger.info("%s Finished", str(item))
                framework.close_log()
                framework.delete_copy()
                continue
            training = rectid.TrainingData(image,training_box_doc)
            training.getRects()
            if training.boxes == []:
                logger.warning(f"False positive: no training glycans for image {str(item)}.")
                logger.info("%s Finished", str(item))
                framework.close_log()
                framework.delete_copy()
                continue
            logger.info("Padded boxes (default):\n")
            padded_test = mt.bbAccuracy(image, model, training)
            pr = padded_test.plotPrecisionRecall()
            framework.get_output(pr)
            
            logger.info("Raw boxes:\n")
            raw_test = mt.bbAccuracy(image, model, training, pad = False)
            raw_pr = raw_test.plotPrecisionRecall()
            framework.get_output(raw_pr, pad = False)
        
        elif extn in ('pdf'):
            pdf = fitz.open(item.item)
            img_array = InputHandler.extract_img_obj(item.item)
            break_flag = False
            for p,page in enumerate(pdf.pages()):
                img_list = [image for image in img_array if image[0]==(p+1)]
                #print(img_list)
                #print(f"##### {page} found figures: {len(img_list)}")
                logger.info(f"\n##### {page} found figures: {len(img_list)}")
                for imgindex,img in enumerate(img_list):
                    #print(img)
                    xref = img[2]
                    #print(xref)
                    img_name = f"{p}-{img[1]}"
                    #print(img_name)
                    x0, y0, x1, y1 = img[3]
                    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                    h = y1 - y0
                    w = x1 - x0
                    # # rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
                    # #print(f"@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
                    # logging.info(f"\n@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
                    # print(f"\n@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
                    pixel = fitz.Pixmap(pdf, xref)
                    
                    
                    #page.drawRect(rectangle, color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=False)
                    if h > 60 and w > 60:
                        #pixel.save(rf"{workdir}/test/p{p}-{xref}.png")  # xref is the xref of the image
                        #print(f" save image to {workdir}test/p{p}-{xref}.png")
                        #annotate_log.write(f"\n save image to {workdir}test/p{p}-{xref}.png")
    
                        #time.sleep(0.1)
                        #print(rf"{workdir}/test/p{p}-{xref}.png")
                        imagedata = pixel.tobytes("png")
                        nparray = np.frombuffer(imagedata, np.uint8)
                        image = cv2.imdecode(nparray,cv2.IMREAD_COLOR)
                        model = rectid.originalYOLO(image, weight = weight, coreyolo = coreyolo)
                        test = mt.GlycanFound(image,model)   
                        #testroc = mt.GlycanFound(glycans,True)
                        isglycan = test.isGlycan()
                        if isglycan:
                            break_flag = True
                            print('File %s has at least one glycan: %s.'%(item,isglycan))
                            logger.info('File %s has at least one glycan: %s.'%(item,isglycan))
                            break
                if break_flag:
                    break
                else:
                    continue
                logger.info('No glycans in file %s.'%item)
                print('No glycans in file %s.'%item)
                        
            
        else:
            logger.warning('File %s had an Unsupported file extension: %s.'%(item,extn))
    
    else:
        model = rectid.originalYOLO(item, weight = weight, coreyolo = coreyolo)
        test = mt.GlycanFound(item,model)             
        isglycan = test.isGlycan()
        logger.info('Image %s has at least one glycan: %s.'%(item,isglycan))  
    
    logger.info("%s Finished", str(item))
    framework.close_log()
    framework.delete_copy()