import BKGLycanExtractor.SystemInteraction as si
import BKGLycanExtractor.glycanRectID as rectID
import BKGLycanExtractor.MonoID as ms
import BKGLycanExtractor.glycanConnections as c
import BKGLycanExtractor.buildStructure as b
import BKGLycanExtractor.glycanSearch as gs

from BKGLycanExtractor.pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError

import sys, logging, fitz, cv2

import numpy as np

def annotate_image(image,annotator,monos,builder,connector):
    
    gctparser = GlycoCTFormat()
    
    boxes = annotator.getRects(image = image)

    logger.info("\nGlycans Found")
    
    origin_image = image.copy()
    detected_glycans = image.copy()
    
    count = 0
    results = []
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
        mono_id_dict = monos.id_monos(image = aux_cropped)
        final = mono_id_dict.get("annotated")
        count_dictionary = mono_id_dict.get("count_dict")
        connect_dict = connector.connect(image = aux_cropped,monos = mono_id_dict)
        #print(connect_dict)
        if connect_dict!={}:
            glycoCT = builder(mono_dict = connect_dict)

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
                    accession = glycoctsearch(glycoCT)
                    if not accession:
                        accession = gnomesearch(glycoCT)
                else:
                    glycoCT = None
            else:
                glycoCT = None
        

        result = dict(name=monos.compstr(count_dictionary), 
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
            logger.info("\nfound: None")
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


logger = logging.getLogger("search")
logger.setLevel(logging.INFO)

framework = si.FindGlycans()
base_configs = "./BKGLycanExtractor/configs/"

item = sys.argv[1]



print(item, "Start")

weight=base_configs+"yolov3_training_final.weights"
coreyolo=base_configs+"coreyolo.cfg"
colors_range=base_configs+"colors_range.txt"

color_range_dict = framework.get_color_range(colors_range)

# configs = {
#     "weights" : weight,
#     "net" : coreyolo,
#     "colors_range": color_range_dict}


framework.initialize_directory(name = item)

workdir = framework.get_directory(name = item)

framework.log(name = item)

annotator = rectID.originalYOLO(weights = weight, net = coreyolo)
monos = ms.HeuristicMonos(colors = color_range_dict)
connector = c.HeuristicConnector()
builder = b.CurrentBuilder()
glycoctsearch = gs.searchGlycoCT()
gnomesearch = gs.sendToGNOme()

annotated_doc = None
annotated_img = None

if framework.check_file(item):
    framework.make_copy(file = item)
    extn = item.lower().rsplit('.',1)[1]
    if extn == "pdf":
        pdf = fitz.open(item)
        img_array = framework.extract_img_from_pdf(item)
        
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
                    
                    output,z = annotate_image(image,annotator,monos,builder,connector)
   
                    for result in output:
                        coordinates = result["coordinates"]
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
                    
        annotated_doc = pdf
        #output: results,pdf
    elif extn == "png":
        image = cv2.imread(item)
        results,annotated_img = annotate_image(image, annotator, monos, builder, connector)
    else:
        print('File %s had an Unsupported file extension: %s.'%(item,extn))
        logger.warning('File %s had an Unsupported file extension: %s.'%(item,extn))
        sys.exit(1)
else:
    results, annotated_img  = annotate_image(item, annotator,monos,builder,connector)
    extn = "Not a file" 
    
framework.save_output(file = item, array = results, doc = annotated_doc, image = annotated_img)

logger.info("%s Finished", item)
framework.close_log()