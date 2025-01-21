import fitz, sys, os, cv2,shutil, pdfplumber, time, ntpath, json, base64
from .submit import searchGlycoCTnew,  sendToGNOme
from .glycanExtractor import compare2img, countcolors, extractGlycanTopology,buildglycan
from .pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError
from BKGlycanExtractor import Config_Manager, Glycan_Semantics

from .monosaccharideid import YOLOMonos

import numpy as np
from shutil import *
##########  this script take both pdfplumber to extract coordinate and fitz to manipulate pdf file
##########  "Example command: py abbitatePDF.py <pdf file name> <training set>"  #####################


def checkpath(workdir):
    # check to see if test and pages exist
    check_path = f"{workdir}/test/pages/"
    isdir = os.path.isdir(check_path)
    #print(f"Checking \"{check_path}\" exist? {isdir}")
    if isdir == True:
        #print("There are files in \"test\" folder proceed to delete them.")
        os.makedirs(check_path)
    else:
        #print("Created path: test/pages.")
        os.makedirs(check_path)


def extract_img_obj(path):
    pdf_file = pdfplumber.open(path)
    array=[]
    count=0
    #print("Loading pages:")
    for i, page in enumerate(pdf_file.pages):
        page_h = page.height
        #print(f"-- page {i} --")
        for j, image in enumerate(page.images):
            #print(image)
            box = (image['x0'], page_h - image['y1'], image['x1'], page_h - image['y0'])
            #image_id=f"p{image['page_number']}-{image['name']}-{image['x0']}-{image['y0']}"
            image_id =f"iid_{count}"
            image_xref=image['stream'].objid
            image_page = image['page_number']
            array.append((image_page,image_id,image_xref,box))
            count+=1

    return array


def findglycans(image_path, workdir, base_configs, pipeline_name, log=None):
    #extract location of all glycan from image file path
    collection_dict = {}
    base = os.getcwd()
    
    weight = base_configs + "Glycan_300img_5000iterations.weights"
    coreyolo = base_configs + "coreyolo.cfg"

    config = Config_Manager()
    glycan_pipeline = config.get_pipeline(pipeline_name)

    figure_semantics = glycan_pipeline.run(image_path)

    count = 1
    gly = figure_semantics.glycans()[0]

    x, y, w, h = gly.semantics['box'].bbox()
    

    # dont need to build glyco_CT - use IUPAC string instead
    # glyco_CT = buildglycan(figure_semantics.semantics['glycans'][0].semantics['monos'])

    # count_dictionary = Glycan_Semantics.composition(gly.monosaccharides())

    # for mono in Glycan_Semantics.mono_syms:
    #     if mono not in count_dictionary:
    #         count_dictionary[mono] = 0

    composition_str = Glycan_Semantics.compstr(gly.monosaccharides())
    iupac_string = Glycan_Semantics.IUPAC(gly.semantics['monos'], gly.semantics['root'])
    json_data = figure_semantics.tojson()

    # print("json",json_data)

    basename = ntpath.basename(image_path).split('.')[0]
    if log:
        print(f"\nImageRef: {basename}-{str(count)}",file=log)

    
    image = cv2.imread(image_path)
    aux_cropped = image[y:y+h,x:x+w].copy() 
    count_dictionary, final, origin, mask_dict, return_contours = countcolors(aux_cropped, base_configs, log)

    save_origin = origin.copy()

    dummy, save_origin_png = cv2.imencode(".png", save_origin)
    save_origin_url = f"test/{basename}-{str(count)}/save_origin.png"

    try:
        os.makedirs(f"{workdir}/test/{basename}-{str(count)}/")
    except FileExistsError:
        pass
    cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/save_origin.png", save_origin)
    
    collection_dict = dict(
        box_corners = gly.semantics['box'].corners_relative(),
        count_dictionary = count_dictionary,
        basename = basename+"-"+str(count),
        iupac = iupac_string,
        save_origin_url = save_origin_url,
        composition_str = composition_str,
        monos_data = json_data
        )

    # array = [gly.semantics['box'].corners_relative(), confidence, count_dictionary, basename+"-"+str(count), iupac_string,  save_origin_url, composition_str]

    return collection_dict

    

    # net = cv2.dnn.readNet(weight, coreyolo)
    # layer_names = net.getLayerNames()

    # try:
    #     output_layers = [layer_names[i[0] - 1] 
    #                             for i in net.getUnconnectedOutLayers()]
    # except IndexError:
    #     output_layers = [layer_names[i - 1] 
    #                             for i in net.getUnconnectedOutLayers()]

    # #colors = np.random.uniform(0, 255, size=(1, 3))

    # #blue = (255, 0, 0)
    # #green = (0, 255, 0)
    # #red = (0, 0, 255)
    # count=0

    # image = cv2.imread(image_path)
    # origin_image = image.copy()
    # height, width, channels = image.shape
    # ############################################################################################
    # #fix issue with
    # ############################################################################################
    # white_space = 200
    # bigwhite = np.zeros([image.shape[0] +white_space, image.shape[1] +white_space, 3], dtype=np.uint8)
    # bigwhite.fill(255)
    # bigwhite[0:image.shape[0], 0:image.shape[1]] = image
    # image = bigwhite.copy()
    # detected_glycan = image.copy()
    # #cv2.imshow("bigwhite", bigwhite)
    # #cv2.waitKey(0)

    # ############################################################################################
    # blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # net.setInput(blob)
    # outs = net.forward(output_layers)
    # # loop through results and print them on images
    # class_ids = []
    # confidences = []
    # boxes = []
    # for out in outs:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]
    #         # filter only confidence higher than 60%
    #         if confidence > 0.6:
    #             center_x = int(detection[0] * (width+white_space))
    #             center_y = int(detection[1] * (height+white_space))
    #             w = int(detection[2] * (width+white_space))
    #             h = int(detection[3] * (height+white_space))

    #             # Rectangle coordinates # where do these constants come from?
    #             # Looks like 0.2*w padding on left and right,
    #             # 0.2*h padding on top and bottom
    #             x = int(center_x - w / 2)-int(0.2*w)
    #             y = int(center_y - h / 2)-int(0.2*h)
    #             w = int(1.4*w)
    #             h = int(1.4 * h)

    #             # fix bug that prevent the image to be crop outside the figure object
    #             if x<=0:
    #                 x=0
    #             if y <=0:
    #                 y=0
    #             if x+w >= (width):
    #                 w=int((width)-x)
    #             if y+h >= (height):
    #                 h=int((height)-y)

    #             # If we are almost the entire image anyway, avoid cropping errors...
    #             if log:
    #                 print("\nEntire image?",x,w,width,float(w)/width,y,h,height,float(h)/height,w*h,width*height,float(w*h)/(width*height),file=log)
    #             if w*h > 0.8*0.8*width*height:
    #                 if log:
    #                      print("Reset to entire image...",file=log)
    #                 x = 0; y = 0; w = width; h = height

    #             p1 = (x,y)
    #             p2 = (x+w,y+h)
    #             #print(p1,p2)
    #             cv2.rectangle(detected_glycan,p1,p2,(0,255,0),3)


    #             boxes.append([x, y, w, h])
    #             confidences.append(float(confidence))
    #             class_ids.append(class_id)
    # # cv.dnn.NMSBoxesRotated(        bboxes, scores, score_threshold, nms_threshold[, eta[, top_k]]        )
    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # #print(f"\nGlycan detected: {len(boxes)}")
    # #print(f"Path: {image_path}")
    # #cv2.imshow("Image", detected_glycan)
    # #cv2.waitKey(0)
    # image=origin_image.copy()
    # for i in range(len(boxes)):
    #     if i in indexes:
    #         x, y, w, h = boxes[i]
    #         #fix some minorbugs when y<0
    #         if y<=0:
    #             y=0


    #         #print(y, y + h, x, x + w)
    #         aux_cropped = image[y:y+h,x:x+w].copy()
    #         #print(y,y+h,x,x+w)
    #         #aux_crop=cv2.resize(aux_crop,None,fx=1,fy=1)


    #         #cv2.imshow("Image", aux_cropped)
    #         #cv2.waitKey(0)

    #         basename = ntpath.basename(image_path).split('.')[0]
    #         if log:
    #             print(f"\nImageRef: {basename}-{str(count)}",file=log)

    #         count+=1
    #         count_dictionary,final,origin,mask_dict, return_contours = countcolors(aux_cropped,base_configs,log)
    #         save_origin=origin.copy()
    #         mono_dict, a, b = extractGlycanTopology(mask_dict, return_contours, origin)
    #         if mono_dict!={}:
    #             glycoCT = buildglycan(mono_dict)
    #         else:
    #             glycoCT = None

    #         #print("glycoCT is", buildglycan(mono_dict))
    #         all_masks = list(mask_dict.keys())
    #         all_masks.remove("black_mask")
    #         all_masks = sum([mask_dict[a] for a in all_masks])

    #         #print(image_path)
    #         #basename = ntpath.basename(image_path).split('.')[0]
    #         try:
    #             os.makedirs(f"{workdir}/test/{basename}-{str(count)}/")
    #         except FileExistsError:
    #             pass
    #         # dummy,save_origin_png = cv2.imencode(".png", save_origin)
    #         save_origin_url = f"test/{basename}-{str(count)}/save_origin.png"
    #         cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/save_origin.png", save_origin)
    #         cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/final.png", final)
    #         cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/black_mask.png", mask_dict["black_mask"])
    #         cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/all_mask.png", all_masks)
    #         cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/a.png", a)
    #         cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/b.png", b)
    #         array.append((x/width,y/height,(x+w)/width,(y+h)/height,confidence,count_dictionary,basename+"-"+str(count),glycoCT,save_origin_url))

    # #if image.shape[0] > 1500 or image.shape[1] > 1200:
    # #    image = cv2.resize(image, None, fx=0.5, fy=0.5)  # , None, fx=1, fy=1)
    # #cv2.imshow("Image", image)
    # #key = cv2.waitKey(0)
    # return array #%of xy coordinate and with/height percentage

def compstr(counts):
    s = ""
    for sym,count in sorted(counts.items()):
        if count > 0:
            s += "%s(%d)"%(sym,count)
    return s

def jobstate(work_dict,state=False,results=None):
    work_dict["job_finished"]=state
    work_dict["results"] = results
    job_log_file = open(work_dict['joblogfile'], "w+")
    json.dump(work_dict,job_log_file)
    job_log_file.close()
    return True

def annotatePNGGlycan(work_dict):
    token = work_dict["token"]
    path = work_dict["infilename"]
    workdir = work_dict["workdir"]
    outfilename = work_dict["outfilename"]
    outfilename2 = work_dict["outfilename2"]
    base_configs = work_dict["base_configs"]
    pipeline_name = work_dict["pipeline_name"]
    
    work_dict["joblogfile"] = (outfilename.rsplit('.',1)[0]+"_job.json")
    work_dict["annotatelogfile"] = (outfilename.rsplit('.',1)[0]+"_log.txt")

    annotate_log=open(work_dict["annotatelogfile"],"w+")
    print(work_dict,file=annotate_log)
    
    jobstate(work_dict,False)

    try:
        checkpath(workdir)
    except PermissionError:
        checkpath(workdir)
    
    image_names = []

    annotate_log.write(f"{token}\n{path}\n{outfilename}")

    # gctparser = GlycoCTFormat()

    results = []
    gly_dict = findglycans(path,workdir,base_configs,pipeline_name,annotate_log)

    count_dictionary = dict(gly_dict['count_dictionary'])
    iupac_seq = gly_dict.get('iupac')
    comp_str = gly_dict.get('composition_str')
    origimage = gly_dict.get('save_origin_url')
    glycanindex = 1

    total_count = count_dictionary['Glc']+count_dictionary['GlcNAc']+\
                count_dictionary['GalNAc']+count_dictionary['NeuAc']+\
                count_dictionary['Man']+count_dictionary['Gal']+count_dictionary['Fuc']
    
    annotate_log.write(f"\nsubmitting:{iupac_seq}")

    accession = searchGlycoCTnew(iupac_seq)
    

    result = dict(name = comp_str,
                imageurl = f"static/files/{token}/{origimage}",
                imgref = "%s-%d"%(os.path.split(path)[1].rsplit('.')[0],glycanindex))
    
    if iupac_seq:
        result['IUPAC'] = iupac_seq

    uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
    if not accession:
        annotate_log.write(f"\nfound: None")
        glycan_uri = uri_base+f"Glc={count_dictionary['Glc']}&GlcNAc={count_dictionary['GlcNAc']}&GalNAc={count_dictionary['GalNAc']}&NeuAc={count_dictionary['NeuAc']}&Man={count_dictionary['Man']}&Gal={count_dictionary['Gal']}&Fuc={count_dictionary['Fuc']}"
        result['linktype'] = 'composition'
        if iupac_seq:
            result['linkexpl'] = 'composition, extracted topology not found'
        else:
            result['linkexpl'] = 'composition only, topology not extracted'
        result['gnomeurl'] = glycan_uri
    else:
        annotate_log.write(f"\nfound: {accession}")
        if accession.startswith('G'):
            glycan_uri = uri_base+"focus="+accession
        else:
            glycan_uri = uri_base+"ondemandtaskid="+accession
        result['linktype'] = 'topology'
        result['linkexpl'] = 'topology extracted'
        result['gnomeurl'] = glycan_uri


    result['monosaccharides'] = json.loads(gly_dict.get('monos_data'))

    if total_count > 0:
        results.append(result)

    annotate_log.close()

    # data contains - monos, root and links
    
    
    jobstate(work_dict, True, results)

    return True 


    # print("figure_semantics",figure_semantics.semantics['glycans'][0].semantics)

    # for glycanindex,found_glycan in enumerate(findglycans(path,workdir,base_configs,annotate_log)):
    #     g_x0,g_y0,g_x1,g_y1,confidence,count_dictionary,glycan_id,glycoCT,origimage=found_glycan
    
    #     total_count = count_dictionary['Glc']+count_dictionary['GlcNAc']+\
    #                   count_dictionary['GalNAc']+count_dictionary['NeuAc']+\
    #                   count_dictionary['Man']+count_dictionary['Gal']+count_dictionary['Fuc']

    #     accession = None
    #     if glycoCT:
    #         try:
    #             g = gctparser.toGlycan(glycoCT)
    #         except GlycoCTParseError:
    #             g = None
    #         if g:
    #             comp = g.iupac_composition()
    #             comptotal = sum(map(comp.get,("Glc","GlcNAc","Gal","GalNAc","NeuAc","Man","Fuc")))
    #             if comptotal == total_count:
    #                 annotate_log.write(f"\nsubmitting:{glycoCT}")
    #                 accession = searchGlycoCTnew(glycoCT)
    #                 if not accession:
    #                     accession = sendToGNOme(glycoCT)
    #             else:
    #                 glycoCT = None
    #         else:
    #             glycoCT = None

    #     result = dict(name=compstr(count_dictionary),
    #                   imageurl=f"static/files/{token}/{origimage}",
    #                   confidence=str(round(confidence,2)),page=1,figure=1,
    #                   imgref="%s-%d"%(os.path.split(path)[1].rsplit('.')[0],glycanindex))
    #     if glycoCT:
    #         result['glycoct'] = glycoCT

    #     uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
    #     if not accession:
    #         annotate_log.write(f"\nfound: None")
    #         glycan_uri=uri_base+f"Glc={count_dictionary['Glc']}&GlcNAc={count_dictionary['GlcNAc']}&GalNAc={count_dictionary['GalNAc']}&NeuAc={count_dictionary['NeuAc']}&Man={count_dictionary['Man']}&Gal={count_dictionary['Gal']}&Fuc={count_dictionary['Fuc']}"
    #         result['linktype'] = 'composition'
    #         if glycoCT:
    #             result['linkexpl'] = 'composition, extracted topology not found'
    #         else:
    #             result['linkexpl'] = 'composition only, topology not extracted'
    #         result['gnomeurl'] = glycan_uri
    #     else:
    #         annotate_log.write(f"\nfound: {accession}")
    #         if accession.startswith('G'):
    #             glycan_uri =uri_base+"focus="+accession
    #         else:
    #             glycan_uri =uri_base+"ondemandtaskid="+accession
    #         result['linktype'] = 'topology'
    #         result['linkexpl'] = 'topology extracted'
    #         result['gnomeurl'] = glycan_uri

    #     if total_count > 0:
    #         results.append(result)

    # annotate_log.close()

    # jobstate(work_dict, True, results)

    # return True    

def annotatePDFGlycan(work_dict):
    token = work_dict["token"]
    path = work_dict["infilename"]
    workdir = work_dict["workdir"]
    outfilename = work_dict["outfilename"]
    outfilename2 = work_dict["outfilename2"]
    base_configs = work_dict["base_configs"]
    pipeline_name = work_dict["pipeline_name"]

    work_dict["joblogfile"] = outfilename.rsplit('.',1)[0]+"_job.json"
    work_dict["annotatelogfile"] = outfilename.rsplit('.',1)[0]+"_log.txt"
    
    jobstate(work_dict,False)

    try:
        checkpath(workdir)
    except PermissionError:
        checkpath(workdir)
    image_names = []
    annotate_log=open(work_dict["annotatelogfile"],"a")

    annotate_log.write(f"{token}\n{path}\n{outfilename}")
    # get image name list
    #stream = open(path,"rb").read()
    doc = fitz.open(path)

    results = []

    image_array = extract_img_obj(path)
    #print(f"Found {len(image_array)} Figures.")
    annotate_log.write(f"\nFound {len(image_array)} Figures.")
    #rest
    for p,page in enumerate(doc.pages()):
        img_list = [image for image in image_array if image[0]==(p+1)]
        #print(f"##### {page} found figures: {len(img_list)}")
        annotate_log.write(f"\n##### {page} found figures: {len(img_list)}")
        for imgindex,img in enumerate(img_list):
            xref = img[2]
            img_name = f"{p}-{img[1]}"
            x0, y0, x1, y1 = img[3]
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
            h = y1 - y0
            w = x1 - x0
            rectangle = (x0 - 1, y0 - 1, x1 + 1, y1 + 1)
            #print(f"@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}")
            annotate_log.write(f"\n@@,xref:{xref},img_name:{img_name}, coordinate:{img[3]}, w:{w}, h:{h}, w*h:{h*w}")

            pixel = fitz.Pixmap(doc, xref)
            #page.drawRect(rectangle, color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=False)
            if (h > 60 and w > 60) or (h*w > 360):
                pixel.writePNG(rf"{workdir}/test/p{p}-{xref}.png")  # xref is the xref of the image
                #print(f" save image to {workdir}test/p{p}-{xref}.png")
                annotate_log.write(f"\n save image to {workdir}test/p{p}-{xref}.png")

                #time.sleep(0.1)
                array = findglycans(rf"{workdir}/test/p{p}-{xref}.png",workdir,base_configs,pipeline_name,log=annotate_log) # find glycan using object classification algorihtm
                #print(f"Glycans Found")
                annotate_log.write(f"\nGlycans Found")

                for glycanindex,glycan_bbox in enumerate(array):
                    g_x0,g_y0,g_x1,g_y1,confidence,count_dictionary,glycan_id,glycoCT,origimage=glycan_bbox
                    #print(count_dictionary)
                    imgcoordinate_page= (x0+g_x1*float(x1-x0),  y0+g_y1*float(y1-y0))

                    total_count = count_dictionary['Glc']+count_dictionary['GlcNAc']+count_dictionary['GalNAc']+count_dictionary['NeuAc']+count_dictionary['Man']+count_dictionary['Gal']+count_dictionary['Fuc']

                    #predict accession
                    accession = None
                    if glycoCT and total_count <= int(glycoCT.count("\n")/2):
                        annotate_log.write(f"\nsubmitting:{[glycoCT]}")
                        accession = searchGlycoCTnew(glycoCT)

                    result = dict(name=compstr(count_dictionary),
                                  imageurl=f"static/files/{token}/{origimage}",
                                  confidence=str(round(confidence,2)),page=(p+1),figure=(imgindex+1),
                                  imgref=rf"p{p}-{xref}-{glycanindex}")
                    if glycoCT:
                        result['glycoct'] = glycoCT

                    ############annotation of glycan image here#############################
                    uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
                    if not accession:
                        annotate_log.write(f"\nfound: None")
                        glycan_uri=uri_base+f"Glc={count_dictionary['Glc']}&GlcNAc={count_dictionary['GlcNAc']}&GalNAc={count_dictionary['GalNAc']}&NeuAc={count_dictionary['NeuAc']}&Man={count_dictionary['Man']}&Gal={count_dictionary['Gal']}&Fuc={count_dictionary['Fuc']}"
                        result['linktype'] = 'composition'
                        if glycoCT:
                            result['linkexpl'] = 'composition, extracted topology not found'
                        else:
                            result['linkexpl'] = 'composition, topology not extracted'
                        result['gnomeurl'] = glycan_uri
                    else:
                        annotate_log.write(f"\nfound: {accession}")
                        glycan_uri =uri_base+"focus="+accession
                        result['linktype'] = 'topology'
                        result['linkexpl'] = 'topology extracted and found'
                        result['gnomeurl'] = glycan_uri

                    if total_count > 0:
                        results.append(result)

                    page.insertLink({'kind': 2, 'from': fitz.Rect(x0+g_x0*float(x1-x0),y0+g_y0*float(y1-y0),x0+g_x1*float(x1-x0),y0+g_y1*float(y1-y0)), 'uri': glycan_uri})
                    comment = f"Glycan id: {glycan_id} found with {str(confidence * 100)[:5]}% confidence."  # \nDebug:{count_dictionary}|"+str(imgcoordinate_page)+f"|{str(x1-x0)},{g_x1},{y1-y0},{g_y1}"
                    comment += f'\nPredicted accession:\n{accession}'
                    comment += f'\nPredicted glycoCT:\n{glycoCT}'
                    page.addTextAnnot(imgcoordinate_page, comment, icon="Note")
                    #page.addFileAnnot(imgcoordinate_page, comment, icon="Note")
                    page.drawRect((g_x0, g_y0, g_x1,g_y1), color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=True)
                if array!=[]:
                    page.addTextAnnot((x0,y0),f"Found {len(array)} glycans\nObj: {img_name} at coordinate: {x0, y0, x1, y1} ", icon="Note")
                    page.drawRect(rectangle, color=fitz.utils.getColor("red"), fill=fitz.utils.getColor("red"), overlay=False)
    doc.save(outfilename)
    doc.save(outfilename2)
    #doc.save(f"{workdir}/output/Result.pdf")
    annotate_log.close()

    jobstate(work_dict, True, results)

    return True

#main!
if __name__ =="__main__":
    '''
    work_dict = {
                "token": token,
                "workdir": workdir,
                "infilename": infilename,
                "outfilename": outfilename,
                "base_configs": base_configs,
                "origfilename": origfilename,
                "outfilename2": outfilename2
            }
    '''
    '''
    work_dict = {
        "token": sys.argv[1],
        "workdir": sys.argv[2],
        "infilename": sys.argv[3],
        "outfilename": sys.argv[4],
        "base_configs": sys.argv[5],
        "origfilename": sys.argv[6],
        "outfilename2": sys.argv[7]
    }
    '''
    # annotatePDFGlycan(work_dict)
    #example
    work_dict_example0 = {
        "token": "Example0",
        "workdir": "tmp",
        "infilename": "tmp/Example0.pdf",
        "outfilename": "tmp/Anotate_Example0.pdf",
        "base_configs": "config/",
        "origfilename": "Example0.pdf",
        "outfilename2": "Example02.pdf"
    }
    annotatePDFGlycan(work_dict_example0)
