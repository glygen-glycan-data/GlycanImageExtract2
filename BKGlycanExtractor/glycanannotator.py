import numpy as np
import cv2, fitz
import os
import pdfplumber
import configparser
import logging
import sys

from BKGlycanExtractor.config import getfromgdrive

from BKGlycanExtractor.pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError

from BKGlycanExtractor import glycanrectangleid
from BKGlycanExtractor import monosaccharideid
from BKGlycanExtractor import glycanorientator
from BKGlycanExtractor import glycanconnections
from BKGlycanExtractor import buildstructure
from BKGlycanExtractor import glycansearch

### the glycan annotator class uses the methods passed to it to extract glycans from an image/pdf and get their composition/topology
# it has an annotate_file method to go through the complete annotation pipeline, from extracting glycans to searching for accession based on composition + topology
# it has individual methods for each step in the pipeline which can be called directly, avoiding the annotate_file method
# this allows its use to run one or multiple steps divorced from the full pipeline
# logging in the glycan annotator class assumes instantiation in the main program of a per-glycan logger named after the glycanobject; it creates a child logger which logs to the same file
#
# the annotator class also has a read_configs method to read the configs.ini file and create the requested class instances for the steps outlined in that file


class Annotator():
    def __init__(self, **kw):
        pass
    
    #method which goes through the complete pipeline - extract glycans, extract composition, orient the glycan, connect monosacchairdes, build topology, search for accession, format results
    #requires a glycanobject - pdf, image file, or already extracted array representing an image
    #requires an instance for each customisable step in the pipeline in the form of a dictionary; keys are step names and values are instances of the chosen class ready for use
    #returns a list of results containing bounding boxes for found glycans, original and monosaccharide-annotated images for each found glycan, extracted composition/topology, coordinates of each glycan in the original file, accession if found
    #returns annotated pdf or png, depending on original glycanobject format
    def annotate_file(self,glycanobject, methods):
        #set up chosen methods for each customisable step in the pipeline
        boxgetter=methods.get("rectfinder")
        monos=methods.get("mono_id")
        orienter=methods.get("orienter")
        connector=methods.get("connector")
        builder=methods.get("builder")
        searches=methods.get("search")
        
        #name glycan, create logger which should be a child of a logger created in the main program - see logging in mainprogram_example.py
        glycan_name = os.path.basename(glycanobject)
        logger_name=glycan_name+'.annotator'
        logger = logging.getLogger(logger_name)
        
        annotated_doc = None
        annotated_img = None
        results = None
        
        #get images from the glycanobject - if a pdf, this gets a list of figures; if a png or bare image array this returns the single image, in a list
        glycan_images = self.extract_images(glycanobject)
        logger.info(f"Found {len(glycan_images)} Figures.")
        if glycan_images is not None:
            results = []
            for image in glycan_images:
                #pdf figures are stored with additional information (figure and page number) required for the results list; interpret_image gets the image from this figure
                orig_image = self.interpret_image(glycanobject, image)
                if orig_image is None:
                    continue
                count = 0
                #find glycans in the image
                unpadded_glycan_boxes,padded_glycan_boxes = self.find_glycans(orig_image, boxgetter)
                
                ###### choose one set of boxes - this is without border padding - depends on training and if YOLO tends to return too-cropped glycans #########
                glycan_boxes = unpadded_glycan_boxes
                for box in glycan_boxes:
                    #get composition by identifying monosaccharides
                    monos_dict, count = self.find_monos(glycanobject, orig_image, box, monos, count)
                    #connect monosaccharides
                    connect_dict = self.connect_monos(monos_dict, connector, orienter, logger_name = logger_name)
                    #use topology to build glycoCT
                    glycoCT = self.build_glycan(connect_dict,builder)
                    #print(glycoCT)
                    
                    #search glycoCT to get accession - goes through methods sequentially and breaks the loop when it finds an accession
                    # so the preferred method should be first, etc
                    for searchmethod in searches:
                        glycoCT, accession = self.search(glycanobject, monos_dict, glycoCT, searchmethod)
                        if accession:
                            break
                            #end search    
                    #create formatted result for the results list
                    result = self.format_result(glycanobject, image, box, monos_dict, glycoCT, accession, monos, count)
                    results.append(result)
            #box glycans on the annotated image, create notes with text annotation on the annotated pdf
            annotated_doc, annotated_img = self.create_annotation(glycanobject,glycan_images,results)
                
        ### end glycan finding - remainder is saving and file handling which should be in your main program - see mainprogram_example.py
        return results, annotated_doc, annotated_img

    #method which takes extracted topology and uses PyGly to build a GlycoCT description
    #requires the dictionary resulting from connect_monos (monosaccharides, indicating connections to other monosaccharides and root/not)
    #requires an instance of a build class - the only current class is CurrentBuilder
    #returns GlycoCT if it can be built
    def build_glycan(self, connectdict, buildmethod):
        if connectdict != {}:
            glycoCT = buildmethod(mono_dict = connectdict)
        else:
            glycoCT = None
        return glycoCT
    
    #method which takes an image, glycan bounding box, and extracted composition and connects monosaccharides to extract topology
    #requires an image with glycan(s) and a single glycan bounding box returned by find_glycans
    #requires the dictionary returned by find_monos (specifically the monosaccharides and IDs)
    #requires an instance of a connect class to use - should be ConnectYOLO if using the YOLO monosaccharide finder, or OriginalConnector if using the heuristic monosaccharide finder
    #requires an instance of an orientation class to pass to the connect method - used to determine where the root should be
    #returns a dictionary of monosaccharides, their connections to each other, and status as the root monosaccharide
    def connect_monos(self, monodict, connectmethod, orientmethod, **kw):
        logger_name = kw.get("logger_name", '')
        
        
        connect_dict = connectmethod.connect(monos = monodict, orientation_method = orientmethod, logger_name = logger_name)
        return connect_dict
    
    #method which takes the original glycanobject, list of images in it, and formatted results list
    #annotates pdf with information about each glycan at the appropriate location, as well as a box around the image
    #annotates png with a box around each glycan
    #returns either an annotated pdf or an annotated png
    def create_annotation(self, glycanobject, glycanimages, results):
        if self.get_object_type(glycanobject) == 'pdf':
            pdf = fitz.open(glycanobject)
            for p,page in enumerate(pdf.pages()):
                img_list = [image for image in glycanimages if image[0]==(p+1)]
                result_list = [result for result in results if result.get("page")==p]
                for glycanimage in img_list:
                    x0, y0, x1, y1 = glycanimage[3]
                    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
                    rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
                    img_result_list = [result for result in result_list if result.get("figure")==glycanimage[1]]
                    img_name = f"{p}-{glycanimage[1]}"
                    for result in img_result_list:
                        coordinates = result["pdfcoordinates"]
                        g_x0 = coordinates.get("x0",)
                        g_y0 = coordinates.get("y0",)
                        g_x1 = coordinates.get("x1",)
                        g_y1 = coordinates.get("y1",)
                        
                        imgcoordinate_page= (x0+g_x1*float(x1-x0),  y0+g_y1*float(y1-y0))
            
                        glycan_uri = result.get("gnomeurl",'')
                        glycan_id = result.get("imgref",'')
                        confidence = result.get("confidence",'')
                        accession = result.get("accession",'')
                        glycoCT = result.get('glycoct','')
                        composition = result.get('name','')
                        page.insert_link({'kind': 2, 'from': fitz.Rect(x0+g_x0*float(x1-x0),y0+g_y0*float(y1-y0),x0+g_x1*float(x1-x0),y0+g_y1*float(y1-y0)), 'uri': glycan_uri})
                        comment = f"Glycan id: {glycan_id} found with {str(confidence * 100)[:5]}% confidence."  # \nDebug:{count_dictionary}|"+str(imgcoordinate_page)+f"|{str(x1-x0)},{g_x1},{y1-y0},{g_y1}"
                        comment += f'\nPredicted accession:\n{accession}'
                        comment += f'\nPredicted glycoCT:\n{glycoCT}'
                        comment += f'\nPredicted composition:\n{composition}'
                        page.add_text_annot(imgcoordinate_page, comment, icon="Note")
                        page.draw_rect((g_x0, g_y0, g_x1,g_y1), color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=True)
                    if img_result_list!=[]:
                        #print("hello")
                        page.add_text_annot((x0,y0),f"Found {len(img_result_list)} glycans\nObj: {img_name} at coordinate: {x0, y0, x1, y1} ", icon="Note")
                        page.draw_rect(rectangle, color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=False)
            return pdf, None
        else:
            for glycanimage in glycanimages:
                for result in results:
                    coordinates = result["imagecoordinates"]
                    x = coordinates.get("x")
                    y = coordinates.get("y")
                    x2 = coordinates.get("x2")
                    y2 = coordinates.get("y2")
                    p1 = (x,y)
                    p2 = (x2,y2)
                    cv2.rectangle(glycanimage,p1,p2,(0,255,0),3)
            return None, glycanimage
    
    #method which gets images from the glycanobject
    #returns a list of images which is trivial for png or image array types; non-trivial for pdf objects
    def extract_images(self,glycanobject):
        extn = self.get_object_type(glycanobject)
        if extn is None:
            return extn
        if extn == "image":
            return [glycanobject]
        elif extn == "png":
            return [cv2.imread(glycanobject)]
        elif extn == "pdf":
            img_array = self.extract_img_from_pdf(glycanobject)
            return img_array
    
    #method which extracts figures from pdfs
    #takes the path to the pdf file so it can be opened
    #returns a list of figures with page, id number, and location
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
                #image_id=f"p{image['page_number']}-{image['name']}-{image['x0']}-{image['y0']}"
                image_id =f"iid_{count}"
                image_xref=image['stream'].objid
                image_page = image['page_number']
                array.append((image_page,image_id,image_xref,box))
                count+=1
        return array
    
    #method which takes an image and extracts location of glycans in the image
    #requires an image and an instance of a glycanrectangleid class to be used for glycan location finding
    #returns two lists of glycan bounding boxes - one as returned from the glycan finder and one with padded borders
    #this is in case model training led to weights which tend to crop glycan arms/monosaccharides - padding the borders can prevent information loss
    #if the model is well-trained the padding shouldn't be necessary - one list should be chosen in the next step
    def find_glycans(self,glycanimage,rectidmethod, threshold = 0.5):
        unpadded_boxes,padded_boxes = rectidmethod.get_rects(image = glycanimage, threshold = threshold)
        return unpadded_boxes,padded_boxes
    
    #method which takes the glycan image, bounding box for a single glycan, and a counter for the number of glycans processed and extracts composition
    #requires the glycanobject for getting the appropriate logger
    #requires an image and a bounding box for a single glycan found in the image
    #requires an instance of a monosaccharideid class to find composition
    #requires a counter which represents the number of glycans processed for this image
    #returns a dictionary of monosaccharides with ids, composition counts, and original and annotated versions of the image cropped to the borders of the bounding box
    #returns a counter representing the number of glycans processed (count+1); this is used later for saving cropped images
    def find_monos(self, glycanobject, glycanimage, glycanbox, monoidmethod, count, threshold = 0.5, logger_name = None):
        if not logger_name:
            glycan_name = os.path.basename(glycanobject)
            logger_name=glycan_name+'.annotator'
        aux_cropped = glycanimage[glycanbox.y:glycanbox.y2,glycanbox.x:glycanbox.x2].copy()
        count += 1
        mono_dict = monoidmethod.find_monos(image = aux_cropped, logger_name=logger_name)
        mono_id_dict = monoidmethod.format_monos(image = aux_cropped, monos_dict = mono_dict, conf_threshold = threshold, logger_name=logger_name)
        return mono_id_dict, count
    
    #method which takes information from multiple pipeline steps to format a complete result for an individual glycan
    #requires the glycanobject for getting the appropriate logger and for figure/page number extraction if glycanobject is a pdf
    #requires an image/figure and a bounding box for a single glycan found in the image
    #requires the dictionary returned by find_monos (monosaccharides with ids and composition count; original and annotated cropped images)
    #requires the glycoCT - if it couldn't be built, None is an acceptable input
    #requires the accession - if not found, None is appropriate
    #requires an instance of the method for monosaccharide id - this is to interpret the composition count into a text string
    #requires a counter for the number of glycans processed for this image
    #returns a single formated result dictionary for this glycan
    def format_result(self, glycanobject, glycanfigure, glycanbox, monosdict, glycoCT, accession, monoidmethod, count):
        glycan_name = os.path.basename(glycanobject)
        logger_name=glycan_name+'.annotator'
        logger = logging.getLogger(logger_name)
        
        if not isinstance(glycanfigure,np.ndarray):
            #run the pdf handling here - in this case the figure has [0] for page and [2] for image data etc
            p = glycanfigure[0]-1
            xref = glycanfigure[2]
            fig = glycanfigure[1]

        else:
            p = 1
            xref = 1
            fig = 1
        
        count_dictionary = monosdict.get("count_dict")
        glycanbox.to_pdf_coords()
        
        #format the result as a dictionary
        result = dict(name=monoidmethod.compstr(count_dictionary), 
                      accession = accession,
                      origimage=monosdict.get("original"),
                      confidence=str(round(glycanbox.confidence,2)),
                      page=p,
                      figure=fig,
                      imgref=f"{p}-{xref}-{count}", 
                      annotatedimage = monosdict.get("annotated"), 
                      pdfcoordinates = {"x0" : glycanbox.x0, "y0" : glycanbox.y0, "x1" : glycanbox.x1, "y1": glycanbox.y1},
                      imagecoordinates = {"x": glycanbox.x, "y" : glycanbox.y, "x2" : glycanbox.x2, "y2": glycanbox.y2})
        #add glycoCT if found
        if glycoCT:
            result['glycoct'] = glycoCT

        #get url to link to accession, if found
        uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
        if not accession:
            logger.info("found: None")
            glycan_uri=uri_base+f"Glc={count_dictionary['Glc']}&GlcNAc={count_dictionary['GlcNAc']}&GalNAc={count_dictionary['GalNAc']}&NeuAc={count_dictionary['NeuAc']}&Man={count_dictionary['Man']}&Gal={count_dictionary['Gal']}&Fuc={count_dictionary['Fuc']}"
            result['linktype'] = 'composition'
            if glycoCT:
                result['linkexpl'] = 'composition, extracted topology not found'
            else:
                result['linkexpl'] = 'composition only, topology not extracted'
            result['gnomeurl'] = glycan_uri
        else:
            logger.info(f"found: {accession}")
            if accession.startswith('G'):
                glycan_uri =uri_base+"focus="+accession
            else:
                glycan_uri =uri_base+"ondemandtaskid="+accession
            result['linktype'] = 'topology'
            result['linkexpl'] = 'topology extracted'
            result['gnomeurl'] = glycan_uri
        return result

    #method to get the type of the object being annotated
    #requires an object, returns its type (png file, pdf, or bare image)
    def get_object_type(self,glycanobject):
        if isinstance(glycanobject,str):
            if glycanobject.endswith(".png") or glycanobject.endswith(".jpg") or glycanobject.endswith(".jpeg"):
                return "png"
            if glycanobject.endswith(".pdf"):
                return "pdf"
        elif isinstance(glycanobject,np.ndarray):
            return "image"
        else:
            return None
        
    def get_orientation(self, orientationmethod, glycanimage):
        orientation, oriented_glycan = orientationmethod.get_orientation(glycanimage = glycanimage)
        return oriented_glycan
    
    #method to get an image from either an image array - trivial - or a pdf figure representation - nontrivial
    #requires the object and the image representation
    #returns the image array
    def interpret_image(self, glycanobject, glycanimage):
        if not isinstance(glycanimage,np.ndarray):
            #run the pdf handlign here - in this case the image has [0] for page and [2] for image data etc
            pdf = fitz.open(glycanobject)
            #p = glycanimage[0]-1
            xref = glycanimage[2]
            #img_name = f"{p}-{glycanimage[1]}"
            #print(img_name)
            x0, y0, x1, y1 = glycanimage[3]
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
            h = y1 - y0
            w = x1 - x0
            #rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
            #print(f"@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
            #logger.info(f"\n@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")

            pixel = fitz.Pixmap(pdf, xref)
        
            if h > 60 and w > 60:
                imagedata = pixel.tobytes("png")
                nparray = np.frombuffer(imagedata, np.uint8)
                image = cv2.imdecode(nparray,cv2.IMREAD_COLOR)
            else:
                image = None
        else:
            image = glycanimage
        return image
    
    #method to read the configs.ini file
    #requires the directory and the path to the file
    #returns a dictionary where keys are the names of the pipeline steps and values are instances of the requested class
    ###use to automatically read in the configurations for the complete pipeline
    ###no changes needed to this code when changing the annotator to use a different pre-set class or configuration
    ###no changes if you add a new configuration option for an existing class
    ###if adding a new class, be sure to give it a 'class' key-value pair in the configuration file
    def read_configs(self, config_dir, config_file, pipeline):      
        methods = {}
        config = configparser.ConfigParser()
        config.read(config_file)
        pipelines = []
        for key, value in config.items():
            if value.get("sectiontype") == "annotator":
                pipelines.append(key)
        try:
            annotator_methods=config[pipeline]
        except KeyError:
            print(pipeline,"is not a valid pipeline.")
            print("Valid pipelines:", pipelines)
            sys.exit(1)
        
        method_descriptions = {
            "rectfinder": {"prefix": "glycanrectangleid.",
                           "multiple": False},
            "mono_id"   : {"prefix": "monosaccharideid.",
                           "multiple": False},
            "orienter"  : {"prefix": "glycanorientator.",
                           "multiple": False},
            "connector" : {"prefix": "glycanconnections.",
                           "multiple": False},
            "builder"   : {"prefix": "buildstructure.",
                           "multiple": False},
            "search"    : {"prefix": "glycansearch.",
                           "multiple": True}
            }
        for method, desc in method_descriptions.items():
            if desc.get("multiple"):
                method_names=annotator_methods.get(method).split(",")
                methods[method] = []
                for method_name in method_names:
                    methods[method].append(self.setup_method(config, desc.get("prefix"), config_dir, method_name))
            else:
                method_name = annotator_methods.get(method)
                methods[method] = self.setup_method(config, desc.get("prefix"), config_dir, method_name)
        return methods
    
    #function to take the name of a modular method (glycan finder, monosaccharide finder, etc)
    #and return an instance of that method which can be used
    #requires the directory, the path to the file, and the method you're searching for
    ### use this to work with a single module without having to create a pipeline for it
    def read_method(self, config_dir, config_file, prefix, method_name):
        config = configparser.ConfigParser()
        config.read(config_file)
        return self.setup_method(config, prefix, config_dir, method_name)
    
    #method to search a glycoCT using a chosen search method, attempting to get an accession
    #requires a glycanobject to get the logger
    #requires the dictionary returned by find_monos - this contains a dictionary of monosaccharide counts
    #requires a glycoCT - if this couldn't be build, it takes None
    #requires an instance of the chosen search method class
    #returns the glycoCT again, unless parsing fails or composition counts don't match up - then returns None
    #returns the accession, if found
    def search(self,glycanobject, monosdict, glycoCT, searchmethod):
        glycan_name = os.path.basename(glycanobject)
        logger_name=glycan_name+'.annotator'
        logger = logging.getLogger(logger_name)
        gctparser = GlycoCTFormat()
        
        count_dictionary = monosdict.get("count_dict")
        total_count = count_dictionary['Glc']+count_dictionary['GlcNAc']+\
                      count_dictionary['GalNAc']+count_dictionary['NeuAc']+\
                      count_dictionary['Man']+count_dictionary['Gal']+count_dictionary['Fuc']
                      
        accession = None
        if glycoCT:
            try:
                g = gctparser.toGlycan(glycoCT)
            except GlycoCTParseError:
                g = None
            if g:
                comp = g.iupac_composition()
                comptotal = sum(map(comp.get,("Glc","GlcNAc","Gal","GalNAc","NeuAc","Man","Fuc")))
                if comptotal == total_count:
                    logger.info(f"\n{type(searchmethod).__name__} submitting:{glycoCT}")
                    accession = searchmethod(glycoCT)
                else:
                    glycoCT = None
            else:
                glycoCT = None
        return glycoCT, accession
    
    #method to instantiate the requested class for a pipeline step (glycan finding, monosaccharide id, etc) during reading of the configs.ini file
    #takes the config parser, a dictionary created from the configs.ini file containing class names and whether the class requires its own configs
    #takes the path to the directory with the configs file
    #takes the name of the method as defined in the [Annotator] section of the configs file
    #returns an instance of the class as defined in the configs.ini file, initialised with the specified configuration (if any)
    def setup_method(self, configparserobject, prefix, directory, method_name):
        #these are the files available for download from the Google Drive
        #if you have named these configs but haven't downloaded them already they will be downloaded and placed in your configs directory
        gdrive_dict = {
            "coreyolo.cfg": "1M2yMBkIB_VctyH01tyDe1koCHT0U8cwV",
            "Glycan_300img_5000iterations.weights": "1xEeMF-aJnVDwbrlpTHkd-_kI0_P1XmVi",
            "largerboxes_plusindividualglycans.weights": "16-AIvwNd-ZERcyXOf5G50qRt1ZPlku5H",
            "monos2.cfg": "15_XxS7scXuvS_zl1QXd7OosntkyuMQuP",
            "orientation_redo.weights": "1KipiLdlUmGSDsr0WRUdM0ocsQPEmNQXo",
            "orientation.cfg": "1AYren1VnmB67QLDxvDNbqduU8oAnv72x",
            "yolov3_monos_new_v2.weights": "1h-QiO2FP7fU7IbvZjoF7fPf55N0DkTPp",
            "yolov3_monos_random.weights": "1m4nJqxrJLl1MamIugdyzRh6td4Br7OMg",
            "yolov3_monos_largerboxes.weights": "1WQI9UiJkqGx68wy8sfh_Hl5LX6q1xH4-"
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
                    raise FileNotFoundError(value+"was not found in configs directory or Google Drive")
                getfromgdrive.download_file_from_google_drive(gdrive_id, filename)
                method_configs[key] = filename
        if not method_configs:
            return eval(prefix+method_class+"()")
        return eval(prefix+method_class+"(method_configs)")
