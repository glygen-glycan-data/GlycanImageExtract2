import sys, os, shutil, time, json, cv2
import logging

from BKGLycanExtractor import glycanrectangleid
from BKGLycanExtractor import monosaccharideid
from BKGLycanExtractor import glycanconnections
from BKGLycanExtractor import buildstructure
from BKGLycanExtractor import glycansearch
from BKGLycanExtractor import glycanannotator


glycan_file = sys.argv[1]
try:
    provided_configs = sys.argv[2]
except IndexError:
    provided_configs = {}


#get all configuration files
base_configs = provided_configs.get("base_configs","./BKGLycanExtractor/configs/")

glycanidweights = provided_configs.get("glycanfinder_weights","largerboxes_plusindividualglycans.weights")
glycanidcfg = provided_configs.get("glycanfinder_cfg","coreyolo.cfg")
colors_range = provided_configs.get("colors_range","colors_range.txt")
monoidweights = provided_configs.get("monoid_weights","yolov3_monos3.weights")
monoidcfg = provided_configs.get("monoid_cfg","monos2.cfg")
orientationweights = provided_configs.get("orientation_weights","edwards_orientation.weights")
orientationcfg = provided_configs.get("orientation_cfg","orientation.cfg")
configs = {
           "glycanfinder_weights": base_configs + glycanidweights,
           "glycanfinder_cfg"    : base_configs + glycanidcfg,
           "colors_range"        : base_configs + colors_range,
           "monoid_weights"      : base_configs + monoidweights,
           "monoid_cfg"          : base_configs + monoidcfg,
           "orientation_weights" : base_configs + orientationweights,
           "orientation_cfg"     : base_configs + orientationcfg,
           }

#initialize directory for output
workdir = os.path.join("./files", glycan_file)
if os.path.isdir(os.path.join(workdir)):
    shutil.rmtree(os.path.join(workdir))
    time.sleep(5)
os.makedirs(os.path.join(workdir))
os.makedirs(os.path.join(workdir, "input"))
os.makedirs(os.path.join(workdir, "output")) 

#start logging
logger = logging.getLogger("search")
logger.setLevel(logging.INFO)

annotatelogfile = f"{workdir}/output/annotated_{glycan_file}_log.txt"
handler = logging.FileHandler(annotatelogfile)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
logger.info(f"Start: {glycan_file}")


#set up methods to get boxes / id monos / extract topology / get accession
annotator = glycanannotator.Annotator()


#the following can be changed to any class in the appropriate class file; swap monosaccharide identification method to/from heuristic/YOLO, etc
#search methods are used sequentially and can be added / subtracted at the appropriate point in glycan finding
boxgetter = glycanrectangleid.OriginalYOLO(configs)
monos = monosaccharideid.YOLOMonos(configs)
connector = glycanconnections.ConnectYOLO(configs)
builder = buildstructure.CurrentBuilder()
glycoctsearch = glycansearch.SearchGlycoCT()
gnomesearch = glycansearch.SendToGNOme()


outdirec = os.path.join(workdir,"input")
try:
    shutil.copyfile(os.path.join('.', glycan_file), os.path.join(outdirec,glycan_file))    
except FileNotFoundError:
    time.sleep(5)
    shutil.copyfile(os.path.join('.',glycan_file), os.path.join(outdirec,glycan_file))

annotated_doc = None
annotated_img = None






### find glycans    

glycan_images = annotator.extract_images(glycan_file)
logger.info(f"\nFound {len(glycan_images)} Figures.")
if glycan_images is not None:
    results = []
    for image in glycan_images:
        orig_image = annotator.interpret_image(glycan_file, image)
        if orig_image is None:
            continue
        count = 0
        unpadded_glycan_boxes,padded_glycan_boxes = annotator.find_glycans(glycan_file, orig_image, boxgetter)
        
        ###### choose one set of boxes - this is without border padding #########
        glycan_boxes = unpadded_glycan_boxes
        for box in glycan_boxes:
            monos_dict, count = annotator.find_monos(orig_image, box, monos, count)
            connect_dict = annotator.connect_monos(orig_image, box, monos_dict, connector)
            glycoCT = annotator.build_glycan(connect_dict,builder)
            #print(glycoCT)
            
            #search glycoCT to get accession
            glycoCT, accession = annotator.search(monos_dict, glycoCT, glycoctsearch, logger)
            if not accession:
                glycoCT, accession = annotator.search(monos_dict, glycoCT, gnomesearch, logger)
            #end search    
                
            result = annotator.format_result(glycan_file, image, box, monos_dict, glycoCT, accession, monos, count, logger)
            results.append(result)
    annotated_doc, annotated_img = annotator.create_annotation(glycan_file,glycan_images,results)
        
### end glycan finding - remainder is saving and file handling
    






    #save results and annotated pdf/png
    name = glycan_file.split('.')[0]
    if annotated_doc is not None:
        annotated_pdf_path = f"{workdir}/output/annotated_{name}.pdf"
        annotated_doc.save(annotated_pdf_path)
    if annotated_img is not None:
        annotated_img_path = f"{workdir}/output/annotated_{name}.png"
        cv2.imwrite(annotated_img_path,annotated_img)
    for idx,result in enumerate(results):
        original = result["origimage"]
        final = result["annotatedimage"]
        try:
            os.makedirs(f"{workdir}/test/{idx}/originals")
        except FileExistsError:
            pass
        try:
            os.makedirs(f"{workdir}/test/{idx}/annotated")
        except FileExistsError:
            pass
        orig_path = f"{workdir}/test/{idx}/originals/save_original.png"
        final_path = f"{workdir}/test/{idx}/annotated/annotated_glycan.png"
        cv2.imwrite(orig_path, original)
        cv2.imwrite(final_path, final)
        result["origimage"] = orig_path
        result["annotatedimage"] = final_path
    res = {
        "glycans": results,
        "rename": "annotated_"+glycan_file,
        "output_file_abs_path": os.path.join(workdir, "output", "annotated_" + glycan_file),
        "inputtype": annotator.get_object_type(glycan_file)
    }
    res = dict(id=glycan_file,result=res,finished=True)
    wh = open(f"{workdir}/output/results.json",'w')
    wh.write(json.dumps(res))
    wh.close()  


#shut down logger

logger.info("%s Finished", glycan_file)

handlers = logger.handlers[:]
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()    