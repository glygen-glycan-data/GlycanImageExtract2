import os, random, logging, cv2, sys, fitz, pdfplumber
from .pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError
import BKGLycanExtractor.glycanSearch as gs

import numpy as np

import BKGLycanExtractor.glycanRectID as rectid
import BKGLycanExtractor.MonoID as ms
import BKGLycanExtractor.buildStructure as b
import BKGLycanExtractor.glycanConnections as c

class InputHandler:
    def __init__(self,item):
        self.item = item
        if os.path.exists(self.item) or os.path.exists(os.path.join("./training_glycans", self.item)):
            self.is_file = True
            self.ext = self.item.lower().rsplit('.',1)[1]
        else:
            self.is_file = False

    def annotate(self, configs, logger = sys.stdout):
        logger = logging.getLogger("search")
        if self.is_file:
            if self.ext == "pdf":
                pdf = fitz.open(self.item)
                img_array = InputHandler.extract_img_obj(self.item)
                
                results = []


                logger.info(f"\nFound {len(img_array)} Figures.")
                #rest
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
                        rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
                        #print(f"@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
                        logger.info(f"\n@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")

                        pixel = fitz.Pixmap(pdf, xref)
                        
                        

                        if h > 60 and w > 60:
             
                            imagedata = pixel.tobytes("png")
                            nparray = np.frombuffer(imagedata, np.uint8)
                            image = cv2.imdecode(nparray,cv2.IMREAD_COLOR)
                            
                            output, z = InputHandler.annotateimage(image, configs)

                            logger.info(f"\nGlycans Found")

                            for result in output:
                                coordinates = results["coordinates"]
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
                                page.insert_link({'kind': 2, 'from': fitz.Rect(x0+g_x0*float(x1-x0),y0+g_y0*float(y1-y0),x0+g_x1*float(x1-x0),y0+g_y1*float(y1-y0)), 'uri': glycan_uri})
                                comment = f"Glycan id: {glycan_id} found with {str(confidence * 100)[:5]}% confidence."  # \nDebug:{count_dictionary}|"+str(imgcoordinate_page)+f"|{str(x1-x0)},{g_x1},{y1-y0},{g_y1}"
                                comment += f'\nPredicted accession:\n{accession}'
                                comment += f'\nPredicted glycoCT:\n{glycoCT}'
                                page.add_text_annot(imgcoordinate_page, comment, icon="Note")
                                page.draw_rect((g_x0, g_y0, g_x1,g_y1), color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=True)
                                results.append(result)
                            if output!=[]:
                                #print("hello")
                                page.add_text_annot((x0,y0),f"Found {len(results)} glycans\nObj: {img_name} at coordinate: {x0, y0, x1, y1} ", icon="Note")
                                page.draw_rect(rectangle, color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=False)
                            

                self.results = results
                self.annotated_doc = pdf
            elif self.ext == "png":
                image = cv2.imread(self.item)
                self.results, self.annotated_image = InputHandler.annotateimage(image,configs)
            else:
                print('File %s had an Unsupported file extension: %s.'%(self,self.ext))
                logger.warning('File %s had an Unsupported file extension: %s.'%(self,self.ext))
                sys.exit(1)
        else:
            self.results, self.annotated_image  = InputHandler.annotateimage(self.item, configs)
        
    @staticmethod
    def annotateimage(image, configs):
        logger = logging.getLogger("search")
        
        gctparser = GlycoCTFormat()
    
        weights = configs.get("weights",'')
        net = configs.get("net",'')
        colors_range = configs.get("colors_range",'')
    
        results = []
        
        height, width, channels = image.shape
        glycans = rectid.originalYOLO(image, weight = weights, coreyolo = net)
        glycans.getRects(0.5)
        boxes = glycans.boxes
    
        origin_image = image.copy()
        detected_glycans = image.copy()
    
        count = 0
        for box in boxes:
            if box.y<0:
                box.y = 0
                
            box.toFourCorners()
    
    
            #print(y, y + h, x, x + w)
            p1 = (box.x,box.y)
            p2 = (box.x2,box.y2)
            cv2.rectangle(detected_glycans,p1,p2,(0,255,0),3)
            aux_cropped = image[box.y:box.y2,box.x:box.x2].copy()
           # print(aux_cropped)
            count += 1
            monos = ms.HeuristicMonos(aux_cropped, colors = colors_range)
            mono_id_dict = monos.id_monos()
            final = mono_id_dict.get("annotated",)
            count_dictionary = mono_id_dict.get("count_dict")
            connector = c.HeuristicConnector(aux_cropped,monos = mono_id_dict)
            connect_dict = connector.connect()
            if connect_dict!={}:
                builder = b.CurrentBuilder(monos_dict = connect_dict)
                glycoCT = builder()
    
            else:
                glycoCT = None
                
            box.toPDFCoords()
            glycan_id = str(count)
            
            logger.info(f"\nImageRef: {str(count)}")
    
        
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
                        logger.info(f"\nsubmitting:{glycoCT}")
                        #print(f"\nsubmitting:{glycoCT}")
                        newsearch = gs.searchGlycoCT(glycoCT)
                        accession = newsearch()
                        if not accession:
                            gnomesearch = gs.sendToGNOme(glycoCT)
                            accession = gnomesearch()
                    else:
                        glycoCT = None
                else:
                    glycoCT = None
            
    
            result = dict(name=ms.MonoID.compstr(count_dictionary), 
                          accession = accession,
                          origimage=origin_image,
                          confidence=str(round(box.confidence,2)),
                          page=1,
                          figure=1,
                          imgref=glycan_id, 
                          annotatedimage = final, 
                          coordinates = {"x0" : box.x0, "y0" : box.y0, "x1" : box.x1, "y1": box.y1})
            if glycoCT:
                result['glycoct'] = glycoCT
    
            uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
            if not accession:
                logger.info(f"\nfound: None")
                #print(f"\nfound: None")
                glycan_uri=uri_base+f"Glc={count_dictionary['Glc']}&GlcNAc={count_dictionary['GlcNAc']}&GalNAc={count_dictionary['GalNAc']}&NeuAc={count_dictionary['NeuAc']}&Man={count_dictionary['Man']}&Gal={count_dictionary['Gal']}&Fuc={count_dictionary['Fuc']}"
                result['linktype'] = 'composition'
                if glycoCT:
                    result['linkexpl'] = 'composition, extracted topology not found'
                else:
                    result['linkexpl'] = 'composition only, topology not extracted'
                result['gnomeurl'] = glycan_uri
            else:
                logger.info(f"\nfound: {accession}")
                #print(f"\nfound: {accession}")
                if accession.startswith('G'):
                    glycan_uri =uri_base+"focus="+accession
                else:
                    glycan_uri =uri_base+"ondemandtaskid="+accession
                result['linktype'] = 'topology'
                result['linkexpl'] = 'topology extracted'
                result['gnomeurl'] = glycan_uri
    
            if total_count > 0:
                results.append(result)
    
        return results, detected_glycans   
    
    @staticmethod
    def extract_img_obj(path):
        pdf_file = pdfplumber.open(path)
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
    
    def __str__(self):
        if self.is_file:
            asstr = self.item
        else:
            asstr = "potentialglycans" + str(random.randint(10000, 99999))
        return asstr