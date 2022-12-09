import numpy as np
import cv2, fitz
import sys
import pdfplumber

from BKGLycanExtractor.pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError

class Annotator():
    def __init__(self):
        pass
    
    def build_glycan(self, connectdict, buildmethod):
        if connectdict != {}:
            glycoCT = buildmethod(mono_dict = connectdict)
        else:
            glycoCT = None
        return glycoCT
    
    def connect_monos(self, glycanimage, glycanbox, monodict, connectmethod):
        aux_cropped = glycanimage[glycanbox.y:glycanbox.y2,glycanbox.x:glycanbox.x2].copy()
        connect_dict = connectmethod.connect(image = aux_cropped, monos = monodict)
        return connect_dict
    
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
    
    def find_glycans(self,glycanobject,glycanimage,rectidmethod):
        unpadded_boxes,padded_boxes = rectidmethod.get_rects(image = glycanimage, threshold = 0.5)
        return unpadded_boxes,padded_boxes
    
    def find_monos(self, glycanimage, glycanbox, monoidmethod, count):
        aux_cropped = glycanimage[glycanbox.y:glycanbox.y2,glycanbox.x:glycanbox.x2].copy()
        count += 1
        mono_dict = monoidmethod.find_monos(image = aux_cropped)
        mono_id_dict = monoidmethod.format_monos(image = aux_cropped, monos_dict = mono_dict)
        return mono_id_dict, count
    
    def format_result(self, glycanobject, glycanimage, glycanbox, monosdict, glycoCT, accession, monoidmethod, count, logger = None):
        
        if not isinstance(glycanimage,np.ndarray):
            #run the pdf handlign here - in this case the image has [0] for page and [2] for image data etc
            pdf = fitz.open(glycanobject)
            p = glycanimage[0]-1
            xref = glycanimage[2]
            fig = glycanimage[1]
            #rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
            #print(f"@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
            #logger.info(f"\n@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")

            pixel = fitz.Pixmap(pdf, xref)
        
            imagedata = pixel.tobytes("png")
            nparray = np.frombuffer(imagedata, np.uint8)
            image = cv2.imdecode(nparray,cv2.IMREAD_COLOR)
        else:
            p = 1
            xref = 1
            fig = 1
            image = glycanimage
        
        count_dictionary = monosdict.get("count_dict")
        glycanbox.to_pdf_coords()
        
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
        if glycoCT:
            result['glycoct'] = glycoCT

        #get url to link to accession, if found
        uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
        if not accession:
            if logger:
                logger.info("\nfound: None")
            else:
                print("\nfound: None")
            glycan_uri=uri_base+f"Glc={count_dictionary['Glc']}&GlcNAc={count_dictionary['GlcNAc']}&GalNAc={count_dictionary['GalNAc']}&NeuAc={count_dictionary['NeuAc']}&Man={count_dictionary['Man']}&Gal={count_dictionary['Gal']}&Fuc={count_dictionary['Fuc']}"
            result['linktype'] = 'composition'
            if glycoCT:
                result['linkexpl'] = 'composition, extracted topology not found'
            else:
                result['linkexpl'] = 'composition only, topology not extracted'
            result['gnomeurl'] = glycan_uri
        else:
            if logger:
                logger.info(f"\nfound: {accession}")
            else:
                print(f"\nfound: {accession}")
            if accession.startswith('G'):
                glycan_uri =uri_base+"focus="+accession
            else:
                glycan_uri =uri_base+"ondemandtaskid="+accession
            result['linktype'] = 'topology'
            result['linkexpl'] = 'topology extracted'
            result['gnomeurl'] = glycan_uri
        return result


    def get_object_type(self,glycanobject):
        if isinstance(glycanobject,str):
            if glycanobject.endswith(".png") or glycanobject.endswith(".jpg"):
                return "png"
            if glycanobject.endswith(".pdf"):
                return "pdf"
        elif isinstance(glycanobject,np.ndarray):
            return "image"
        else:
            return None
        
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
    
    def search(self, monosdict, glycoCT, searchmethod, logger = None):
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
                    if logger:
                        logger.info(f"\nsubmitting:{glycoCT}")
                    else:
                        print(f"\nsubmitting:{glycoCT}")
                    accession = searchmethod(glycoCT)
                else:
                    glycoCT = None
            else:
                glycoCT = None
        return glycoCT, accession