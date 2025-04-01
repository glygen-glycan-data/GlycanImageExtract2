import fitz, sys, os, cv2,shutil, pdfplumber, time, ntpath, json, base64
from .submit import searchGlyLookup, sendToGNOme, searchGlyImage
from PIL import Image

from BKGlycanExtractor import Config_Manager, Glycan_Semantics, Figure_Semantics
from . import glycanExtractor

from .glycanannotator import GlycanExtractorPipeline

import numpy as np
from shutil import *



def annotatePNGGlycan(work_dict):
    token = work_dict["token"]
    path = work_dict["infilename"]
    workdir = work_dict["workdir"]
    outfilename = work_dict["outfilename"]
    outfilename2 = work_dict["outfilename2"]
    base_configs = work_dict["base_configs"]
    file_type = work_dict['file_type']
    
    work_dict["joblogfile"] = (outfilename.rsplit('.',1)[0]+"_job.json")
    work_dict["annotatelogfile"] = (outfilename.rsplit('.',1)[0]+"_log.txt")

    annotate_log=open(work_dict["annotatelogfile"],"w+")
    
    jobstate(work_dict,False)

    page_num = 0


    # try:
    #     checkpath(workdir)
    # except PermissionError:
    #     checkpath(workdir)
    
    image_names = []

    annotate_log.write(f"{token}\n{path}\n{outfilename}")

    results = []
    # glycan_data, image_data = findglycans(path, workdir, outfilename, base_configs, file_type, annotate_log)

    # print("\n--->>>glycan_data",glycan_data)
    
    # check - switching pipelines (multi_glycan to single_glycan)
    # check
    # if not glycan_data:
    #     accepted_file_types = ["multi_glycan", "single_glycan"]

    #     new_file_type = next((t for t in accepted_file_types if t != file_type), None)

    #     if new_file_type:
    #         glycan_data, image_data = findglycans(path, workdir, outfilename, base_configs, new_file_type, annotate_log)


    #     work_dict['file_format_error'] = (
    #         f"No glycans were present/detected in the image; switched to {new_file_type} detection mode!"
    #         if glycan_data
    #         else "Something might be wrong with the file you used, please enter another file"
    #     )

    # parse_glycan_data(work_dict, glycan_data, results)

    # work_dict['file_data'] = [{"page_num": page_num, "image_data": image_data, "glycan_count": len(image_data)}]


    annotate_log.close()

    jobstate(work_dict, True, results)

    return True 

  
def annotatePDFGlycan(work_dict):
    token = work_dict["token"]
    path = work_dict["infilename"]
    workdir = work_dict["workdir"]
    outfilename = work_dict["outfilename"]
    outfilename2 = work_dict["outfilename2"]
    base_configs = work_dict["base_configs"]
    file_type = work_dict['file_type']

    work_dict["joblogfile"] = outfilename.rsplit('.',1)[0]+"_job.json"
    work_dict["annotatelogfile"] = outfilename.rsplit('.',1)[0]+"_log.txt"

    work_dict["file_data"] = []
    
    jobstate(work_dict,False)

    try:
        checkpath(workdir)
    except PermissionError:
        checkpath(workdir)

    pages_path = f"{workdir}/test/pages/"

    image_names = []
    annotate_log=open(work_dict["annotatelogfile"],"a")

    annotate_log.write(f"{token}\n{path}\n{outfilename}")

    if not os.path.exists(path):
        print(f"Error: File not found at {path}")

    doc = fitz.open(path)

    results = []

    image_array, pages_array = extract_img_obj(path)
    annotate_log.write(f"\nFound {len(image_array)} Figures.")
    for p,page in enumerate(doc.pages()):

        page_pix = doc.load_page(p).get_pixmap()
        
        page_image_path = os.path.join(pages_path, f"page_{p}.png")
        page_pix.save(page_image_path)

        img_list = [image for image in image_array if image[0]==(p+1)]
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
                pixel.save(rf"{workdir}/test/p{p}-{xref}.png")  # xref is the xref of the image
                annotate_log.write(f"\n save image to {workdir}test/p{p}-{xref}.png")

                glycan_data, image_data = findglycans(page_image_path, workdir, outfilename, base_configs, file_type, annotate_log)

                if not glycan_data:
                    annotate_log.write(f"\nNo Glycans Found on page {p}")
                    print(f"\nNo Glycans Found on page {p}")
                    continue 
                
                # print(f"Glycans Found", glycan_data)
                annotate_log.write(f"\nGlycans Found on page {p}")

                parse_glycan_data(work_dict, glycan_data, results, page_num=p)
                work_dict['file_data'].append({"page_num": p,  "image_data": image_data})

    print("outfilename",outfilename)
    print("outfilename2",outfilename2)
    
    annotate_log.close()

    # results['file_data'] = work_dict['file_data']
    jobstate(work_dict, True, results)
    print("*******************JOB DONE")

    return True


def checkpath(workdir):
    # check to see if test and pages exist
    check_path_pages = f"{workdir}/test/pages/"
    check_path_annotated = f"{workdir}/test/annotated_pages/"
    isdir = os.path.isdir(check_path_pages)
    #print(f"Checking \"{check_path}\" exist? {isdir}")
    if os.path.isdir(check_path_pages) and os.path.isdir(check_path_annotated):
        #print("There are files in \"test\" folder proceed to delete them.")
        os.makedirs(check_path_pages,exist_ok=True)
        os.makedirs(check_path_annotated,exist_ok=True)
    else:
        #print("Created path: test/pages.")
        os.makedirs(check_path_pages,exist_ok=True)
        os.makedirs(check_path_annotated,exist_ok=True)

# extracts images from a given PDF file using pdfplumber and 
# returns an array containing metadata about each image
def extract_img_obj(path):
    pdf_file = pdfplumber.open(path)
    array=[]
    page_array = []
    count=0
    for i, page in enumerate(pdf_file.pages):
        page_h = page.height
        for j, image in enumerate(page.images):
            box = (image['x0'], page_h - image['y1'], image['x1'], page_h - image['y0'])

            image_id =f"iid_{count}"
            image_xref=image['stream'].objid
            image_page = image['page_number']
            array.append((image_page,image_id,image_xref,box))
            count+=1

        page_array.append(page)

    return array, page_array


def findglycans(image_path, workdir, outfilename, base_configs, pipeline_name, log=None):
    collection_dict = {}
    base = os.getcwd()

    config = Config_Manager()
    pipeline = config.get_pipeline(pipeline_name)
    figure_semantics = pipeline.run(image_path)
    
    # do you need this loop?
    # adding glycan bounding boxes to the entire image
    entire_image = figure_semantics.image().copy()
    for i, _ in enumerate(figure_semantics.glycans()):
        print(f"Annotating glycan {i}")
        # entire_image = figure_semantics.annotate_glycans(entire_image, name = f'entire_image_{i}', **dict(idx=i))
        figure_semantics.annotate_glycans()

    print("Writing annotated file")
    image_basename = os.path.basename(image_path)
    basename = ntpath.basename(image_path).split('.')[0]
    pages_directory = os.path.join(workdir, "test", "annotated_pages", image_basename)
    cv2.imwrite(pages_directory, figure_semantics.image())
    

    # Steps to run the pipeline in three stages - 
    # 1) glycan_steps, 
    # 2) crop_image (not required for single glycan image),
    # 3) figure_steps
    
    # check if the image needs to be cleaned and make sure the image is not empty
    # print("----->>>>clean",glycan_pipeline.steps.get('clean',None))
    # if glycan_pipeline.steps.get('clean',None):
    #     crop_image = glycan_pipeline.steps['clean'].get('crop_image',False)
    #     clean_image = glycan_pipeline.steps['clean'].get('clean_image',False)

    #     for i, gly_semantics in enumerate(figure_semantics.glycans()):
    #         # print("CLEANING THE IMAGE", gly_semantics.image_path(), crop_image, clean_image)
    #         glycan_image = gly_semantics.semantics['image'].copy()
    #         if crop_image and glycan_image is not None and glycan_image.size != 0:
    #             glycan_image = glycanExtractor.crop_largest_component(glycan_image)
    #         if clean_image and glycan_image is not None and glycan_image.size != 0:
    #             glycan_image = glycanExtractor.clean_largest_component(glycan_image)

    #         gly_semantics.semantics['image'] = glycan_image    


    count = 1
    image_details = {}
    for i, gly_semantics in enumerate(figure_semantics.glycans()):
        errors = []

        glycan_image = gly_semantics.semantics.get('image')
        # print("\n--->>LOCATION:", f"{workdir}/test/{basename}-{str(count)}/save_origin.png")
        image_details[i] = {'location': f"{workdir}/test/{basename}-{str(count)}/save_origin.png"}

        # print("\nglycan_image",glycan_image)
        # print("\nglycan_image.size",glycan_image.size)

        if glycan_image is None or glycan_image.size == 0:
            image_details[i]['error'] = f"Skipping glycan {i}: Detected object is missing or empty"
            print(f"Skipping glycan {i}: Detected object is missing or empty")
            continue

        # print("\nRunning Glycan Pipeline ")
        # for gly_step in pipeline.get_steps('glycan'): 
        #     gly_step.execute(gly_semantics)

        # monos = gly_semantics.monosaccharides()
        count_dictionary = gly_semantics.composition()

        for mono in gly_semantics.mono_syms:
            if mono not in count_dictionary:
                count_dictionary[mono] = 0

        composition_str = gly_semantics.compstr()


        # adj_list = Glycan_Semantics.build_adjacency_list(gly_semantics, cleaned=True)
        # print("adj_list",adj_list)
        # links, monoid_collection = Glycan_Semantics.link_count(gly_semantics)

        # print("--->>monoid_collection",monoid_collection)
        # print("--->>actual mono ids", tuple(sorted(gly_semantics.semantics['monos'].keys())))
        
        # mono_count = len(monos)

        links_count = len(gly_semantics.undirected_links())
        monos_count = len(gly_semantics.monosaccharides())

        IUPAC = ''
        orientation = 'RL'
        error_found = False
        root_id = gly_semantics.root()

        # check for errors - TO DO
        if not gly_semantics.root():
            errors.append("Unable to detect root in the structure.")
            error_found = True
        # if monoid_collection != tuple(sorted(gly_semantics.semantics['monos'].keys())):
        #     errors.append("Monosaccharides were detected incorrectly.")
        #     error_found = True
        # if monos_count - 1 != links_count:
        #     errors.append("Links were detected incorrectly.")
        #     error_found = True
        


        if not error_found:
            # work on IUPAC() method
            IUPAC = gly_semantics.IUPAC()
            print("IUPAC", IUPAC, root_id, composition_str)
            orientation = gly_semantics.glycan_orientation()


        # json_data = gly_semantics.tojson()

        if log:
            print(f"\nImageRef: {basename}-{str(count)}",file=log)

        image = gly_semantics.semantics['image']
        save_origin_url = f"test/{basename}-{str(count)}/save_origin.png"


        try:
            os.makedirs(f"{workdir}/test/{basename}-{str(count)}/")
        except FileExistsError:
            pass
        cv2.imwrite(f"{workdir}/test/{basename}-{str(count)}/save_origin.png", image)
        
        # add the below details to the dict
        # links_count = links_count,
        # monos_count = monos_count,
        # errors = errors,
        collection_dict[count] = dict(
            box_corners = gly_semantics.semantics['box'].corners_relative(),
            count_dictionary = dict(count_dictionary),
            basename = basename+"-"+str(count),
            iupac = IUPAC,
            save_origin_url = save_origin_url,
            composition_str = composition_str,
            glycan_data = gly_semantics.tojson(),
            identified_glycan = f"{workdir}/test/{basename}-{str(count)}/save_origin.png",
            entire_annotated_image = pages_directory,
            orientation = orientation,
            errors = [],
        )

        count += 1
    
    return collection_dict, image_details

    

# def compstr(counts):
#     s = ""
#     for sym,count in sorted(counts.items()):
#         if count > 0:
#             s += "%s(%d)"%(sym,count)
#     return s

def jobstate(work_dict,state=False,results=None):
    work_dict["job_finished"]=state
    work_dict["results"] = results
    job_log_file = open(work_dict['joblogfile'], "w+")
    json.dump(work_dict,job_log_file)
    job_log_file.close()
    print("-------->>>JOB COMPLETED", state)
    return True


def parse_glycan_data(work_dict, glycan_data, results, page_num=0):
    token = work_dict["token"]
    path = work_dict["infilename"]
    annotate_log=open(work_dict["annotatelogfile"],"a")


    for glycan_idx, gly_dict in glycan_data.items():

        count_dictionary = gly_dict['count_dictionary']
        iupac_seq = gly_dict.get('iupac')
        comp_str = gly_dict.get('composition_str')
        origimage = gly_dict.get('save_origin_url')
        mono_count = gly_dict.get('mono_count')
        link_count = gly_dict.get('link_count')
        errors = gly_dict.get('errors')
        composition_str = gly_dict.get('composition_str')
        entire_annotated_image = gly_dict.get('entire_annotated_image')
        orientation = gly_dict.get('orientation')

        total_count = count_dictionary['Glc']+count_dictionary['GlcNAc']+\
                    count_dictionary['GalNAc']+count_dictionary['NeuAc']+\
                    count_dictionary['Man']+count_dictionary['Gal']+count_dictionary['Fuc']
        
    
        submit_sequence =  iupac_seq if iupac_seq and iupac_seq.strip() is not None else comp_str
        annotate_log.write(f"\nsubmitting:{submit_sequence}")

        try:
            accession = searchGlyLookup(submit_sequence)
        except:
            accession = searchGlyLookup(composition_str)
        
        
        # Instead of this - GlyImage will generate the required Image
        # if not accession:
        #     accession = sendToGNOme(submit_sequence)

        print("accession found",accession)

        # you can provide IUPAC and if that is not present mono composiiton will also work
        if submit_sequence:
            glyImage = searchGlyImage(submit_sequence, orientation=orientation)
        else:
            glyImage = searchGlyImage(composition_str, orientation=orientation)
        

        result = dict(name = comp_str,
                    imageurl = f"static/files/{token}/{origimage}",
                    imgref = "%s-%d"%(os.path.split(path)[1].rsplit('.')[0],glycan_idx),
                    mono_count = mono_count,
                    link_count = link_count,
                    accession = accession,
                    glyImage = glyImage,
                    entire_annotated_image = entire_annotated_image,
                    page_num = page_num,
                    orientation = orientation, 
                )
        
        if iupac_seq:
            result['IUPAC'] = iupac_seq
        else:
            result['IUPAC'] = None

        uri_base="https://gnome.glyomics.org/StructureBrowser.html?"
        # if an accession is not found - the image is not displayed
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


        mono_details = json.loads(gly_dict.get('glycan_data'))

        root = mono_details.get('root', None)

        if root is None:
            errors.append("Root was not found")
        else:
            result['root'] = root


        result['monosaccharides'] = mono_details['monos']
        result['errors'] = errors

        if total_count > 0:
            results.append(result)


    annotate_log.close()

