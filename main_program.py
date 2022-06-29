from BKGLycanExtractor import glycanrectangleid
from BKGLycanExtractor import monosaccharideid
from BKGLycanExtractor import glycanconnections
from BKGLycanExtractor import buildstructure
from BKGLycanExtractor import glycansearch

from BKGLycanExtractor.pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError

import sys, logging, fitz, cv2, os, shutil, time, pdfplumber, json

import numpy as np

def annotate_image(image,rectgetter,monos,builder,connector,glycoctsearch,gnomesearch):
    
    gctparser = GlycoCTFormat()
    
    #gets initial yolo output boxes and boxes with added border padding, confidence threshold of 0.5
    unpadded_boxes,padded_boxes = rectgetter.get_rects(image = image, threshold = 0.5)
    
    #choose one set of output boxes - this is without border padding
    boxes = unpadded_boxes

    logger.info("\nGlycans Found")
    
    origin_image = image.copy()
    detected_glycans = image.copy()
    
    count = 0
    results = []
    for box in boxes:
        if box.y<0:
            box.y = 0
            
        box.to_four_corners()

        #visually annotate image
        p1 = (box.x,box.y)
        p2 = (box.x2,box.y2)
        cv2.rectangle(detected_glycans,p1,p2,(0,255,0),3)
        
        #identify monosaccharides in each glycan, get an image of glycan with annotated monos
        aux_cropped = image[box.y:box.y2,box.x:box.x2].copy()
        count += 1
        mono_id_dict = monos.id_monos(image = aux_cropped)
        final = mono_id_dict.get("annotated")
        count_dictionary = mono_id_dict.get("count_dict")
        
        #connect monos to extract structure
        connect_dict = connector.connect(image = aux_cropped,monos = mono_id_dict)
        
        #build structure
        if connect_dict!={}:
            glycoCT = builder(mono_dict = connect_dict)
        else:
            glycoCT = None
        
        #format for results
        box.to_pdf_coords()
        glycan_id = str(count)
        logger.info(f"\nImageRef: {str(count)}")

        total_count = count_dictionary['Glc']+count_dictionary['GlcNAc']+\
                      count_dictionary['GalNAc']+count_dictionary['NeuAc']+\
                      count_dictionary['Man']+count_dictionary['Gal']+count_dictionary['Fuc']

        #attempt to use structure to get accession by submitting to database
        accession = None
        if glycoCT:
            try:
                g = gctparser.to_glycan(glycoCT)
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
        
        #more formatting for result
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


        #get url to link to accession, if found
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

def extract_img_from_pdf(pdf_path):
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


#location of configuration files - includes yolo weights and colors for heuristic mono id
base_configs = "./BKGLycanExtractor/configs/"

glycan_file = sys.argv[1]

print(glycan_file, "Start")

weight=base_configs+"retrain_v2.weights"
coreyolo=base_configs+"coreyolo.cfg"
colors_range=base_configs+"colors_range.txt"

#read in color ranges for mono id
color_range_file = open(colors_range)
color_range_dict = {}
for line in color_range_file.readlines():
    line = line.strip()
    name = line.split("=")[0].strip()
    color_range = line.split("=")[1].strip()
    color_range_dict[name] = np.array(list(map(int, color_range.split(","))))

# configs = {
#     "weights" : weight,
#     "net" : coreyolo,
#     "colors_range": color_range_dict}


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
boxgetter = glycanrectangleid.OriginalYOLO(weights = weight, net = coreyolo)
monos = monosaccharideid.HeuristicMonos(colors = color_range_dict)
connector = glycanconnections.HeuristicConnector()
builder = buildstructure.CurrentBuilder()
glycoctsearch = glycansearch.SearchGlycoCT()
gnomesearch = glycansearch.SendToGNOme()

annotated_doc = None
annotated_img = None


#start annotation
if os.path.isfile(glycan_file):
    outdirec = os.path.join(workdir,"input")
    try:
        shutil.copyfile(os.path.join('.', glycan_file), os.path.join(outdirec,glycan_file))    
    except FileNotFoundError:
        time.sleep(5)
        shutil.copyfile(os.path.join('.',glycan_file), os.path.join(outdirec,glycan_file))
    extn = glycan_file.lower().rsplit('.',1)[1]
    if extn == "pdf":
        pdf = fitz.open(glycan_file)
        img_array = extract_img_from_pdf(glycan_file)
        
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
                    
                    output,z = annotate_image(image,boxgetter,monos,builder,connector, glycoctsearch, gnomesearch)
   
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
        image = cv2.imread(glycan_file)
        results,annotated_img = annotate_image(image, boxgetter, monos, builder, connector, glycoctsearch, gnomesearch)
    else:
        print('File %s had an Unsupported file extension: %s.'%(glycan_file,extn))
        logger.warning('File %s had an Unsupported file extension: %s.'%(glycan_file,extn))
        sys.exit(1)
else:
    results, annotated_img  = annotate_image(glycan_file, boxgetter,monos,builder,connector, glycoctsearch, gnomesearch)
    extn = "Not a file" 
    
    

    
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
    "inputtype": extn
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