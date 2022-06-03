import cv2, os, shutil, time, json, logging

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

class SystemInteraction:
    def get_directory(self):
        return self.workdir
    def initialize_directory(self):
        if os.path.isdir(os.path.join(self.workdir)):
            shutil.rmtree(os.path.join(self.workdir))
            time.sleep(5)
        os.makedirs(os.path.join(self.workdir))
    def log(self, filepath):
        try:
            handler = logging.FileHandler(filepath)
        except FileNotFoundError:
            time.sleep(5)
            handler = logging.FileHandler(filepath)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(message)s'))
        return handler
class FindGlycans(SystemInteraction):
    def __init__(self,name):
        self.name = name
        self.workdir = os.path.join("./files", self.name)
        self.results = []
    def get_output(self,array,doc=None,image=None):
        if doc is not None:
            annotated_path = f"{self.workdir}/output/annotated_{self.name.split('.')[0]}.pdf"
            doc.save(annotated_path)
        if image is not None:
            annotated_path = f"{self.workdir}/output/annotated_{self.name.split('.')[0]}.png"
            cv2.imwrite(annotated_path,image)
        for idx,result in enumerate(array):
            original = result["origimage"]
            final = result["annotatedimage"]
            try:
                os.makedirs(f"{self.workdir}/test/{idx}/originals")
            except FileExistsError:
                pass
            try:
                os.makedirs(f"{self.workdir}/test/{idx}/annotated")
            except FileExistsError:
                pass
            orig_path = f"{self.workdir}/test/{idx}/originals/save_original.png"
            final_path = f"{self.workdir}/test/{idx}/annotated/annotated_glycan.png"
            cv2.imwrite(orig_path, original)
            cv2.imwrite(final_path, final)
            result["origimage"] = orig_path
            result["annotatedimage"] = final_path
        self.results = array    
    def initialize_directory(self):
        super().initialize_directory()
        os.makedirs(os.path.join(self.workdir, "input"))
        os.makedirs(os.path.join(self.workdir, "output"))        
    def log(self):
        annotatelogfile = f"{self.workdir}/output/annotated_{self.name.split('.')[0]}_log.txt"
        logger = logging.getLogger("search")
        handler = super().log(annotatelogfile)
        logger.addHandler(handler)
        logger.info(f"Start: {self.name}")
    def make_copy(self):
        try:
            shutil.copyfile(self.name, os.path.join(self.workdir, "input", self.name))    
        except FileNotFoundError:
            time.sleep(5)
            shutil.copyfile(self.name, os.path.join(self.workdir, "input", self.name))
    def save_results_json(self,inputtype):
        res = {
            "glycans": self.results,
            "rename": "annotated_"+self.name,
            "output_file_abs_path": os.path.join(self.workdir, "output", "annotated_" + self.name),
            "inputtype": inputtype
        }
        res = dict(id=self.name,result=res,finished=True)
        wh = open(f"{self.workdir}/output/results.json",'w')
        wh.write(json.dumps(res))
        wh.close()
        

class TestModel(SystemInteraction):
    def __init__(self,name,model):
        self.file_name = name + "_" + model
        self.workdir = os.path.join("./testing", self.file_name)
        self.item_name = name.rsplit('.',1)[0]
        self.name = name
    def close_log(self):
        logger = logging.getLogger("test")
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    def delete_copy(self):
        path = os.path.join(".",self.name)
        os.remove(path)
    def get_output(self,precrec, pad = True):
        if pad:
            path = f"{self.workdir}/precisionrecallplot_paddedborders.png"
        else:
            path = f"{self.workdir}/precisionrecallplot_rawboxes.png"
        plt.savefig(path)
        plt.close()
    def get_training_box_doc(self):
        boxes_path = os.path.join("./training_glycans")
        boxes_doc = os.path.join(boxes_path,self.item_name+".txt")
        #print(boxes_doc)
        if os.path.exists(boxes_doc):
            return boxes_doc
        else:
            return None
    def log(self):
        annotatelogfile = f"{self.workdir}/annotated_log.txt"
        logger = logging.getLogger("test")
        handler = super().log(annotatelogfile)
        logger.addHandler(handler)
        logger.info(f"Start: {self.name}")
    def make_copy(self):
        try:
            shutil.copyfile(os.path.join("./training_glycans", self.name), os.path.join(".",self.name))    
        except FileNotFoundError:
            time.sleep(5)
            shutil.copyfile(os.path.join("./training_glycans",self.name), os.path.join(".",self.name))