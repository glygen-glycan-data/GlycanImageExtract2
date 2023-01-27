import sys, os, shutil, time, json, cv2
import logging

from BKGLycanExtractor import glycanannotator


#set up annotator to read configs and run pipeline
#the annotator has mix-and-match methods to run individual sections, as well as a method to run the full annotation pipeline
annotator = glycanannotator.Annotator()

#read configs file
#provide the directory with configs file and weights, YOLO .cfg, etc as the second command line argument
configs_dir=sys.argv[2]
configs_file = os.path.join(configs_dir,"configs.ini")
#this contains all methods set for each step, with the given configuration files (weights, YOLO .cfg, colours, etc)
annotation_methods = annotator.read_configs(configs_dir,configs_file)

### from here on could be wrapped into a for loop for multiple glycans with the same annotator


#get glycan file from command line - provide as the first command line argument
glycan_file = os.path.abspath(sys.argv[1])
glycan_name = os.path.basename(glycan_file)

#initialize directory to save output
workdir = os.path.join("./files", glycan_name)
if os.path.isdir(os.path.join(workdir)):
    shutil.rmtree(os.path.join(workdir))
    time.sleep(5)
os.makedirs(os.path.join(workdir))
os.makedirs(os.path.join(workdir, "input"))
os.makedirs(os.path.join(workdir, "output")) 

#start logging to a per-glycan log.txt file
logger = logging.getLogger(glycan_name)
logger.setLevel(logging.INFO)

annotatelogfile = os.path.join(workdir, "output", "annotated_"+glycan_name+"_log.txt")
handler = logging.FileHandler(annotatelogfile)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
logger.addHandler(handler)
logger.info(f"Start: {glycan_name}")

#copy the original image to an input directory for storage/recovery purposes
outdirec = os.path.join(workdir,"input")
try:
    shutil.copyfile(glycan_file, os.path.join(outdirec,glycan_name))    
except FileNotFoundError:
    time.sleep(5)
    shutil.copyfile(glycan_file, os.path.join(outdirec,glycan_name))

### find glycans - this is the all in one annotator method to find glycans, extract their composition, and build topology
results, annotated_doc, annotated_img = annotator.annotate_file(glycan_file,annotation_methods)    

if results is not None:
    #save results and annotated pdf/png
    output_file_path = os.path.join(workdir, "output", "annotated_" + glycan_name)
    if annotated_doc is not None:
        annotated_doc.save(output_file_path)
    if annotated_img is not None:
        cv2.imwrite(output_file_path,annotated_img)
    #save individual glycans extracted, and versions with monosaccharides annotated
    for idx,result in enumerate(results):
        original = result["origimage"]
        final = result["annotatedimage"]
        try:
            os.makedirs(os.path.join(workdir,"test",str(idx),"originals"))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join(workdir, "test", str(idx), "annotated"))
        except FileExistsError:
            pass
        orig_path = os.path.join(workdir, "test", str(idx), "originals", "save_original.png")
        final_path = os.path.join(workdir, "test", str(idx), "annotated", "annotated_glycan.png")
        cv2.imwrite(orig_path, original)
        cv2.imwrite(final_path, final)
        result["origimage"] = os.path.abspath(orig_path)
        result["annotatedimage"] = os.path.abspath(final_path)
    #save json file with results, composition, topology, links for each extracted glycan
    res = {
        "glycans": results,
        "rename": "annotated_"+glycan_name,
        "output_file_abs_path": os.path.abspath(output_file_path),
        "inputtype": annotator.get_object_type(glycan_file)
    }
    res = dict(id=glycan_file,result=res,finished=True)
    wh = open(os.path.join(workdir, "output", "results.json"),'w')
    wh.write(json.dumps(res))
    wh.close()  


#shut down logger

logger.info("%s Finished", glycan_name)

handlers = logger.handlers[:]
for handler in handlers:
    logger.removeHandler(handler)
    handler.close()    