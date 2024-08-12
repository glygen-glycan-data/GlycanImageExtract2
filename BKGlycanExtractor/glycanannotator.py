# -*- coding: utf-8 -*-
"""
Class to work with the glycan annotation pipeline, 
read in configs, set up annotation methods, etc
"""

import configparser
import json
import logging
import os
import shutil
import sys
import time

import cv2
# import fitz
import numpy as np
# import pdfplumber

from .config import getfromgdrive
from .pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError
from . import glycanfinding
from . import monosaccharideid
from . import glycanconnections
from . import rootmonofinding
from . import glycanbuilding
from . import glycansearch

class Annotator:
    def annotate_file(self, glycanfile, methods):        
        objecttype = self.get_object_type(glycanfile)
        # print(objecttype)
        if objecttype == "pdf":
            annotation, results = self.annotate_pdf(glycanfile, methods)
        elif objecttype == "png":
            annotation, results = self.annotate_png(glycanfile, methods)
        elif objecttype == "image":
            annotation, results = self.annotate_image(glycanfile, methods)
        else:
            pass
        
        return annotation, results
    
    def annotate_pdf(self, glycanpdf, methods):
        pdf_file = pdfplumber.open(glycanpdf)
        array=[]
        count=0
        # print("Loading pages:")
        for i, page in enumerate(pdf_file.pages):
            page_h = page.height
            #  print(f"-- page {i} --")
            for j, image in enumerate(page.images):
                #   print(image)
                box = (
                    image['x0'], page_h - image['y1'], 
                    image['x1'], page_h - image['y0']
                    )
                image_id = f"iid_{count}"
                image_xref=image['stream'].objid
                image_page = image['page_number']
                array.append((image_page, image_id, image_xref, box))
        pdf_file.close()
        
        self.logger.info(f"Found {len(array)} Figures.")
        
        results = []
        for glycanimage in array:
            count = 0
            pdf = fitz.open(glycanpdf)
            xref = glycanimage[2]
            x0, y0, x1, y1 = glycanimage[3]
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
            h = y1 - y0
            w = x1 - x0

            pixel = fitz.Pixmap(pdf, xref)
        
            if h > 60 and w > 60:
                imagedata = pixel.tobytes("png")
                nparray = np.frombuffer(imagedata, np.uint8)
                image = cv2.imdecode(nparray,cv2.IMREAD_COLOR)
            else:
                image = None
                
            pdf.close()
            if image is not None:
                _, imgresults = self.annotate_image(image, methods)
                for result in imgresults:
                    count += 1
                    result["page"] = glycanimage[0] - 1
                    result["figure"] = glycanimage[1]
                    result["imgref"] = f"{glycanimage[0]-1}-{xref}-{count}"
                results.extend(imgresults)
                    
        pdf = fitz.open(glycanpdf)
        for p,page in enumerate(pdf.pages()):
            img_list = [image for image in array if image[0]==(p+1)]
            result_list = [
                result for result in results if result.get("page")==p
                ]
            for glycanimage in img_list:
                x0, y0, x1, y1 = glycanimage[3]
                x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
                img_result_list = [
                    result for result in result_list 
                    if result.get("figure") == glycanimage[1]
                    ]
                img_name = f"{p}-{glycanimage[1]}"
                for result in img_result_list:
                    coordinates = result["pdfcoordinates"]
                    # print(coordinates)
                    g_x0 = coordinates["x0"]
                    g_y0 = coordinates["y0"]
                    g_x1 = coordinates["x1"]
                    g_y1 = coordinates["y1"]
                    
                    imgcoordinate_page = (
                        x0 + g_x1*float(x1-x0),  y0 + g_y1*float(y1-y0)
                        )
        
                    glycan_uri = result.get("gnomeurl",'')
                    glycan_id = result.get("imgref",'')
                    confidence = result.get("confidence",'')
                    # print(confidence)
                    # print(f"{str(confidence)}")
                    accession = result.get("accession",'')
                    glycoCT = result.get('glycoct','')
                    composition = result.get('name','')
                    page.insert_link({
                        'kind': 2, 
                        'from': fitz.Rect(
                            x0+g_x0*float(x1-x0), y0+g_y0*float(y1-y0), 
                            x0+g_x1*float(x1-x0), y0+g_y1*float(y1-y0)
                            ), 
                        'uri': glycan_uri
                        })
                    comment = f"Glycan id: {glycan_id} found with {str(confidence)} confidence."
                    comment += f'\nPredicted accession:\n{accession}'
                    comment += f'\nPredicted glycoCT:\n{glycoCT}'
                    comment += f'\nPredicted composition:\n{composition}'
                    page.add_text_annot(
                        imgcoordinate_page, comment, icon="Note"
                        )
                    page.draw_rect(
                        (g_x0, g_y0, g_x1,g_y1), 
                        color=fitz.utils.getColor("red"), 
                        fill=fitz.utils.getColor("red"), 
                        overlay=True
                        )
                if img_result_list!=[]:
                    page.add_text_annot(
                        (x0,y0), 
                        f"Found {len(img_result_list)} glycans\nObj: {img_name} at coordinate: {x0, y0, x1, y1} ", 
                        icon="Note"
                        )
                    page.draw_rect(
                        rectangle, color=fitz.utils.getColor("red"), 
                        fill=fitz.utils.getColor("red"), overlay=False
                        )
        
        return pdf, results
    
    def annotate_png(self, glycanpng, methods):
        glycanimage = cv2.imread(glycanpng)
        glycanimage, resultslist = self.annotate_image(glycanimage, methods)
        return glycanimage, resultslist
    
    def annotate_image(self, glycanimage, methods):
        glycanfinder = methods.get("glycanfinder")
        monos = methods.get("mono_id")
        connector = methods.get("connector")
        root_monos = methods.get("rootfinder") 
        builder = methods.get("builder")
        searches = methods.get("search")
        
        final = glycanimage.copy()
        results = []
        count = 0
        glycan_boxes = glycanfinder.find_glycans(glycanimage)
        for glycan in glycan_boxes:
            (x, y), (x2, y2) = glycan.to_image_coords()
            single_glycanimage = glycanimage[y:y2, x:x2].copy()
            glycan_info = monos.find_monos(single_glycanimage)
            count += 1
            connector.connect(glycan_info)
            rootconf = root_monos.find_root_mono(glycan_info)
            glycoCT = builder(glycan_info)
            
            gctparser = GlycoCTFormat()
            count_dictionary = glycan_info.get_composition()
            total_count = sum(count_dictionary.values())
            for searchmethod in searches:
                accession = None
                if glycoCT is not None:
                    try:
                        g = gctparser.toGlycan(glycoCT)
                    except GlycoCTParseError:
                        g = None
                    if g is not None:
                        comp = g.iupac_composition()
                        # this needs to be changed to add new monos
                        # not ideal
                        comptotal = sum(map(comp.get, (
                            "Glc", "GlcNAc", "Gal", "GalNAc",
                            "NeuAc", "NeuGc", "Man", "Fuc"
                            )))
                        if comptotal == total_count:
                            self.logger.info(
                                f"\n{type(searchmethod).__name__} submitting:{glycoCT}"
                                )
                            accession = searchmethod(glycoCT)
                        else:
                            glycoCT = None
                    else:
                        glycoCT = None
                if accession is not None:
                    break
            
            [x0, y0, x1, y1] = glycan.to_pdf_coords()
            (x, y), (x2, y2) = glycan.to_image_coords()
            
            p1, p2 = (x, y), (x2, y2)
            cv2.rectangle(final, p1, p2, (0,255,0), 3)
                
            returninfo = {
                "name" : glycan_info.get_composition_string(),
                "accession": accession,
                "origimage": glycan_info.get_image(),
                "confidence": str(round(glycan.get_confidence(), 2)),
                "annotatedimage": glycan_info.get_annotated_image(),
                "pdfcoordinates": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                "glycoCT": glycoCT,
                }

            returninfo = self.get_uri(returninfo, count_dictionary)
            results.append(returninfo)
        return final, results
    
    def close_logger(self, glycanfile):
        glycan_name = os.path.basename(glycanfile)
        base_glycan_name = glycan_name.rsplit('.', 1)[0]
        
        self.logger.info("%s Finished", base_glycan_name)

        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()
        
    def create_output_direc(self, glycanfile, pipeline):
        glycan_name = os.path.basename(glycanfile)
        base_glycan_name = glycan_name.rsplit('.', 1)[0]
        
        workdir = os.path.join("./files", base_glycan_name+"_"+pipeline)
        if os.path.isdir(os.path.join(workdir)):
            shutil.rmtree(os.path.join(workdir))
            time.sleep(5)
        os.makedirs(os.path.join(workdir))
        os.makedirs(os.path.join(workdir, "input"))
        os.makedirs(os.path.join(workdir, "output")) 
        
        return workdir
    
    def get_object_type(self, glycanobject):
        if isinstance(glycanobject, str):
            if (glycanobject.endswith(".png") 
                or glycanobject.endswith(".jpg") 
                or glycanobject.endswith(".jpeg")):
                
                return "png"
            elif glycanobject.endswith(".pdf"):
                return "pdf"
        elif isinstance(glycanobject, np.ndarray):
            return "image"
        else:
            return None
        
    def get_uri(self, result, comp_dict):
        uri_base = "https://gnome.glyomics.org/StructureBrowser.html?"
        if result["accession"] is None:
            self.logger.info("found: None")
            glycan_uri = uri_base + \
                f"Glc={comp_dict['Glc']}&GlcNAc={comp_dict['GlcNAc']}&GalNAc={comp_dict['GalNAc']}&NeuAc={comp_dict['NeuAc']}&NeuGc={comp_dict.get('NeuGc',0)}&Man={comp_dict['Man']}&Gal={comp_dict['Gal']}&Fuc={comp_dict['Fuc']}"
            result['linktype'] = 'composition'
            if result["glycoCT"] is not None:
                result['linkexpl'] = \
                    'composition, extracted topology not found'
            else:
                result['linkexpl'] = \
                    'composition only, topology not extracted'
            result['gnomeurl'] = glycan_uri
        else:
            self.logger.info(f"found: {result['accession']}")
            if result["accession"].startswith('G'):
                glycan_uri = uri_base + "focus=" + result["accession"]
            else:
                glycan_uri = \
                    uri_base + "ondemandtaskid=" + result["accession"]
            result['linktype'] = 'topology'
            result['linkexpl'] = 'topology extracted'
            result['gnomeurl'] = glycan_uri
        return result
    
    # return an instance of a single class, with specified configs
    # i.e. one glycan finder, one monosaccharide finder
    # use to work with a single module without a whole pipeline
    def read_one_config(self, config_dir, config_file, module, class_name):
        config = configparser.ConfigParser()
        config.read(config_file)
        return self.setup_method(config, module, config_dir, class_name)
        
    # read in info from the configs.ini file, for a named pipeline
    # returns a dictionary of methods to use for each pipeline step
    def read_pipeline_configs(self, config_dir, config_file, pipeline):
        methods = {}
        config = configparser.ConfigParser()
        config.read(config_file)
        pipelines = []
        for key, value in config.items():
            if value.get("sectiontype") == "annotator":
                pipelines.append(key)
        try:
            annotator_methods = config[pipeline]
        except KeyError:
            print(pipeline,"is not a valid pipeline.")
            print("Valid pipelines:", pipelines)
            sys.exit(1)
        
        method_descriptions = {
            "glycanfinder": {"prefix": "glycanfinding.", "multiple": False},
            "mono_id": {"prefix": "monosaccharideid.", "multiple": False},
            "connector": {"prefix": "glycanconnections.", "multiple": False},
            "rootfinder": {"prefix": "rootmonofinding.", "multiple": False},
            "builder": {"prefix": "glycanbuilding.", "multiple": False},
            "search": {"prefix": "glycansearch.", "multiple": True},
            }
        for method, desc in method_descriptions.items():
            # print(method, desc)
            if desc.get("multiple"):
                method_names = annotator_methods.get(method).split(",")
                methods[method] = []
                for method_name in method_names:
                    # print(method_name)
                    methods[method].append(self.setup_method(
                        config, desc.get("prefix"), config_dir, method_name
                        ))
            else:
                method_name = annotator_methods.get(method)
                # print(method_name)
                methods[method] = self.setup_method(
                    config, desc.get("prefix"), config_dir, method_name
                    )
        return methods
    
    def save_results(self, directory, glycanfile, results, annotation):
        glycan_name = os.path.basename(glycanfile)
        output_file_path = os.path.join(
            directory, "output", "annotated_" + glycan_name
            )
        
        objecttype = self.get_object_type(glycanfile)
        if objecttype == "pdf":
            annotation.save(output_file_path)
        elif objecttype == "png" or objecttype == "image":
            cv2.imwrite(output_file_path, annotation)
        
        for idx, result in enumerate(results):
            original = result["origimage"]
            final = result["annotatedimage"]
            try:
                os.makedirs(os.path.join(
                    directory, "test", str(idx), "originals"
                    ))
            except FileExistsError:
                pass
            try:
                os.makedirs(os.path.join(
                    directory, "test", str(idx), "annotated"
                    ))
            except FileExistsError:
                pass
            
            orig_path = os.path.join(
                directory, "test", str(idx), "originals", "save_original.png"
                )
            final_path = os.path.join(
                directory, "test", str(idx), 
                "annotated", "annotated_glycan.png"
                )
            cv2.imwrite(orig_path, original)
            cv2.imwrite(final_path, final)
            result["origimage"] = os.path.abspath(orig_path)
            result["annotatedimage"] = os.path.abspath(final_path)
            
        # save json file with results, composition, 
        # topology, links for each extracted glycan
        res = {
            "glycans": results,
            "rename": "annotated_"+glycan_name,
            "output_file_abs_path": os.path.abspath(output_file_path),
            "inputtype": self.get_object_type(glycanfile)
            }
        res = dict(id=glycanfile, result=res, finished=True)
        wh = open(os.path.join(directory, "output", "results.json"), 'w')
        wh.write(json.dumps(res))
        wh.close()   
    
    def set_loggers(self, directory, glycanfile, methods):
        glycan_name = os.path.basename(glycanfile)
        glycan_name = glycan_name.rsplit('.', 1)[0]
        
        self.logger = logging.getLogger(glycan_name)
        self.logger.setLevel(logging.INFO)
        
        annotatelogfile = os.path.join(
            directory, "output", "annotated_"+glycan_name+"_log.txt"
            )
        handler = logging.FileHandler(annotatelogfile)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
        self.logger.addHandler(handler)
        print("Start:", glycan_name)
        self.logger.info(f"Start: {glycan_name}")
        
        for desc, method in methods.items():
            if isinstance(method, list):
                for mth in method:
                    mth.set_logger(glycan_name)
            else:
                method.set_logger(glycan_name)
    
    def setup_method(
            self, configparserobject, prefix, directory, method_name
            ):
        gdrive_dict = {
            "coreyolo.cfg":
                "1M2yMBkIB_VctyH01tyDe1koCHT0U8cwV",
            "Glycan_300img_5000iterations.weights":
                "1xEeMF-aJnVDwbrlpTHkd-_kI0_P1XmVi",
            "largerboxes_plusindividualglycans.weights":
                "16-AIvwNd-ZERcyXOf5G50qRt1ZPlku5H",
            "monos2.cfg":
                "15_XxS7scXuvS_zl1QXd7OosntkyuMQuP",
            "orientation_redo.weights":
                "1KipiLdlUmGSDsr0WRUdM0ocsQPEmNQXo",
            "orientation.cfg":
                "1AYren1VnmB67QLDxvDNbqduU8oAnv72x",
            "orientation_flipped.cfg":
                "1YXkSWjqjbx5_GkCrOdkIHrSocTAqu9WX",
            "orientation_flipped.weights":
                "1PQH6_JPpE_1o9WdhKAIGJdmOF5fI39Ew",
            "yolov3_monos_new_v2.weights":
                "1h-QiO2FP7fU7IbvZjoF7fPf55N0DkTPp",
            "yolov3_monos_random.weights": 
                "1m4nJqxrJLl1MamIugdyzRh6td4Br7OMg",
            "yolov3_monos_largerboxes.weights":
                "1WQI9UiJkqGx68wy8sfh_Hl5LX6q1xH4-",
            "rootmono.cfg":
                "1RSgCYxkNvrPYann5MG7WybyBZS2UA5v0",
            "yolov3_rootmono.weights":
                "1dUTFbPA7XV-HztWeM5uto2mF_xo5F-3Z"
        }
        
        method_values = configparserobject[method_name]
        method_class = method_values.pop("class")
        method_configs = {}
        for key, value in method_values.items():
            filename = os.path.join(directory,value)
            if os.path.isfile(filename):
                method_configs[key] = filename
            else:
                gdrive_id = gdrive_dict.get(value)
                if gdrive_id is None:
                    raise FileNotFoundError(
                        value + 
                        "was not found in configs directory or Google Drive"
                        )
                getfromgdrive.download_file_from_google_drive(
                    gdrive_id, filename
                    )
                method_configs[key] = filename
        if not method_configs:
            return eval(prefix+method_class+"()")
        return eval(prefix+method_class+"(method_configs)")
