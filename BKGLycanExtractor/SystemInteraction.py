import cv2, os, shutil, time, json, logging, pdfplumber
import numpy as np

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

class SystemInteraction:
    def __init__(self,**kw):
        pass
    def check_file(self,file):
        if os.path.isfile(file):
            return True
        else:
            return False
    def close_log(self,logger=None):
        if logger is not None:
            handlers = logger.handlers[:]
            for handler in handlers:
                logger.removeHandler(handler)
                handler.close()
    def extract_img_from_pdf(self,pdf_path):
        pdf_file = pdfplumber.open(pdf_path)
        array=[]
        count=0
       # print("Loading pages:")
        for i, page in enumerate(pdf_file.pages):
            page_h = page.height
          #  print(f"-- page {i} --")
            for j, image in enumerate(page.images):
             #   print(image)
                box = (image['x0'], page_h - image['y1'], image['x1'], page_h - image['y0'])
                image_id=f"p{image['page_number']}-{image['name']}-{image['x0']}-{image['y0']}"
                image_id =f"iid_{count}"
                image_xref=image['stream'].objid
                image_page = image['page_number']
                array.append((image_page,image_id,image_xref,box))
                count+=1
        return array
    def find_problems(self,**kw):
        pass
    def get_color_range(self,file):
        color_range_file = open(file)
        color_range_dict = {}
        for line in color_range_file.readlines():
            line = line.strip()
            name = line.split("=")[0].strip()
            color_range = line.split("=")[1].strip()
            color_range_dict[name] = np.array(list(map(int, color_range.split(","))))
        return color_range_dict
    def get_directory(self,**kw):
        raise NotImplementedError
    def get_training_box_doc(self,file = None, direc = '.'):
        name = file.split(".")[0]
        boxes_doc = os.path.join(direc,name+".txt")
        if os.path.exists(boxes_doc):
            return boxes_doc
        else:
            return None
    def initialize_directory(self,directory = None,**kw):
        if os.path.isdir(os.path.join(directory)):
            shutil.rmtree(os.path.join(directory))
            time.sleep(5)
        os.makedirs(os.path.join(directory))
    def log(self, file = None, logger = None, **kw):
        try:
            handler = logging.FileHandler(file)
        except FileNotFoundError:
            time.sleep(5)
            handler = logging.FileHandler(file)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(message)s'))
        return handler
    def make_copy(self, **kw):
        pass
    def save_output(self, **kw):
        raise NotImplementedError
        
class FindGlycans(SystemInteraction):
    def close_log(self):
        logger = logging.getLogger("search")
        super().close_log(logger)
    def get_directory(self,name=None):
        workdir = os.path.join("./files", name)
        return workdir
    def get_training_box_doc(self,**kw):
        pass
    def initialize_directory(self,name = None, **kw):
        workdir = self.get_directory(name=name)
        super().initialize_directory(directory = workdir)
        os.makedirs(os.path.join(workdir, "input"))
        os.makedirs(os.path.join(workdir, "output"))      
    def log(self,name = None, logger = None, **kw):
        workdir = self.get_directory(name = name)
        annotatelogfile = f"{workdir}/output/annotated_{name}_log.txt"
        handler = super().log(file = annotatelogfile)
        logger.addHandler(handler)
        logger.info(f"Start: {name}")
    def make_copy(self,file = None ,in_direc = '.'):
        workdir = self.get_directory(file)
        outdirec = os.path.join(workdir,"input")
        try:
            shutil.copyfile(os.path.join(in_direc, file), os.path.join(outdirec,file))    
        except FileNotFoundError:
            time.sleep(5)
            shutil.copyfile(os.path.join(in_direc,file), os.path.join(outdirec,file))
    def save_output(self,file = None,array = None,doc=None,image=None, **kw):
        workdir = self.get_directory(name = file)
        name = file.split('.')[0]
        try:
            inputtype = file.split('.')[1]
        except IndexError:
            inputtype = "Not a file"
        if doc is not None:
            annotated_path = f"{workdir}/output/annotated_{name}.pdf"
            doc.save(annotated_path)
        if image is not None:
            annotated_path = f"{workdir}/output/annotated_{name}.png"
            cv2.imwrite(annotated_path,image)
        for idx,result in enumerate(array):
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
            "glycans": array,
            "rename": "annotated_"+file,
            "output_file_abs_path": os.path.join(workdir, "output", "annotated_" + file),
            "inputtype": inputtype
        }
        res = dict(id=file,result=res,finished=True)
        wh = open(f"{workdir}/output/results.json",'w')
        wh.write(json.dumps(res))
        wh.close()  
        

class TestModel(SystemInteraction):
    def __init__(self,**kw):
        super().__init__()
        self.problem_file_path = os.path.join("./testing","probleminputs.txt")  
    def close_log(self):
        logger = logging.getLogger("test")
        super().close_log(logger)
    def find_problems(self, file='', problem=''):
        printstr = file+": "+problem+"\n"
        f = open(self.problem_file_path,'a')
        f.write(printstr)
        f.close()
    def get_directory(self,name=None):
        workdir = os.path.join("./testing", name)
        return workdir
    def initialize_directory(self,name = None, **kw):
        workdir = self.get_directory(name = name)
        super().initialize_directory(directory = workdir)
    def log(self, name = None, logger = None, **kw):
        workdir = self.get_directory(name=name)
        annotatelogfile = f"{workdir}/annotated_log.txt"
        handler = super().log(annotatelogfile)
        logger.addHandler(handler)
        logger.info(f"Start: {name}")
    def save_output(self, file = None, image = plt.gcf(), pad = True, **kw):
        workdir = self.get_directory(name = file)
        if pad:
            path = f"{workdir}/precisionrecallplot_paddedborders.png"
        else:
            path = f"{workdir}/precisionrecallplot_rawboxes.png"
        plt.savefig(path)
        plt.close()
class BatchTesting(SystemInteraction):
    def __init__(self,**kw):
        super().__init__()
        self.workdir = os.path.join("./modelPRCs") 
    def close_log(self):
        logger = logging.getLogger("test")
        super().close_log(logger)
    def get_directory(self,**kw):
        return self.workdir
    def initialize_directory(self,**kw):
        super().initialize_directory(directory = self.workdir)
    def log(self, logger = None):
        annotatelogfile = f"{self.workdir}/annotated_log.txt"
        handler = super().log(annotatelogfile)
        logger.addHandler(handler)
        logger.info("Start.\n")
    def save_output(self,image = plt.gcf(), **kw):
        path = f"{self.workdir}/precisionrecallplot.png"
        plt.savefig(path)
        plt.close()

class EnlargeTraining(SystemInteraction):
    def __init__(self,directory = "."):
        super().__init__()
        self.workdir = directory
    def close_log(self):
        pass
    def get_directory(self,**kw):
        return self.workdir
    def get_training_box_doc(self,file = '.', **kw):
        super().get_training_box_doc(file = file, direc = self.workdir)
    def initialize_directory(self,**kw):
        pass
    def log(self):
        pass
    def save_output(self, file = None, image = None, boxes = None, **kw):
        box_doc = self.get_training_box_doc(file = file)
        im_path = os.path.join(self.workdir,file)
        f = open(box_doc,"w")
        for box in boxes:
            box = box.toList()
            for x in box:
                f.write(x + " ")
            f.write("\n")
        f.close()
        cv2.imwrite(im_path,image)