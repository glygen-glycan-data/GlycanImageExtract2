import fitz, sys, os, cv2,shutil, time, ntpath, json, base64, re, urllib.request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submit import searchGlyLookup, searchSubsumption, searchGlyImage, sendToGNOme
from PIL import Image
from hashlib import md5
from APIFramework import APIFramework
from BKGlycanExtractor import ImageSearch
from BKGlycanExtractor import Config_Manager, BoundingBox, PDFBoundingBox, CompareBoxes
from BKGlycanExtractor import STANDARD_DPI, PDFHandler
from BKGlycanExtractor import searchpmc

import numpy as np
from shutil import copyfile
import tarfile
from io import BytesIO
import xml.etree.ElementTree as ET
from pmc_xmlparser import XMLParser

class JobInstance:

    pipeline_mapping = {
        'Simple Glycan Image': 'SingleGlycanImage-YOLOFinders',
        # 'Single-Glycan Image': 'SingleGlycanImage-YOLOFinders',
        'Multi-Glycan Image': 'MultipleGlycanImage-YOLOFinders',
        'Manuscript': 'MultipleGlycanImage-YOLOFinders'
    }

    def __init__(self, task_detail, config = {}, msg_queue = None):
        self.task_detail = task_detail
        self.id = task_detail.get('id')
        self.msg_queue = msg_queue
        self.config = config
        self.original_file_name = task_detail.get('filename')
        self.submission_type = task_detail.get('submission_type')

        self.pmid = task_detail.get('pmid')

        # Base project directory (absolute)
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.workdir = os.path.join(self.base_dir, "static", "files", self.id)

        # Structured input/output directories
        self.input_dir = os.path.join(self.workdir, "input")
        self.output_dir = os.path.join(self.workdir, "output")

        # File paths
        self.input_filepath = os.path.join(self.input_dir, self.original_file_name)
        self.output_filename = f"annotated_{self.original_file_name}"
        self.output_filepath = os.path.join(self.output_dir, self.output_filename)

        self.log_file_path = os.path.splitext(self.output_filepath)[0] + "_log.txt"
        self.json_filepath = os.path.splitext(self.output_filepath)[0] + "_job.json"

        # Ensure necessary directories exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Open log file for writing
        self.log_file = open(self.log_file_path, 'w')

        self.job_finished = False
        self.results = []
        self.document_metadata = {}

    def update_status(self,status,state=None):
        msg = dict(id=self.id)
        if state is not None:
            msg['state'] = state
            msg['status'] = ""
        if status is not None:
            msg['status'] = status
        if state is not None or status is not None:
            self.msg_queue.put(msg)

    def update_state(self,state,status=None):
        self.update_status(status=None,state=state)

    @staticmethod
    def get_processor(task_detail,*args,**kwargs):
        submission_type = task_detail.get('submission_type')
        pmid = task_detail.get('pmid')

        if task_detail.get('curation_task', False) and submission_type == "Manuscript":
            return PDFJob(task_detail,*args,**kwargs) 
        elif submission_type == "Manuscript" and pmid is not None:
            return PMIDJob(task_detail,*args,**kwargs) 
        elif submission_type == "Manuscript":
            return PDFJob(task_detail,*args,**kwargs)
        elif submission_type in ("Simple Glycan Image","Multi-Glycan Image"):
            return ImageJob(task_detail,*args,**kwargs) 
        
        raise ValueError(f"Unsupported submission type: {submission_type}")


    def create_directories(self,*paths):
        for path in paths:
            os.makedirs(path, exist_ok=True)


    def save_image(self,image, path):
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"Error saving image at {path}: {e}")



    def abs_to_rel(self, abs_path):
        """
        Converts absolute path to relative path based on the base directory
        """
        # Debugging: Print the absolute path before conversion
        # print(f"Converting absolute path: {abs_path}")
        
        # Calculate the relative path using the base directory
        rel_path = os.path.relpath(abs_path, self.workdir)

        # Prefix the relative path with './' to match your required format
        final_path = rel_path
        
        # Debugging: Print the relative path after conversion
        # print(f"Converted relative path: {final_path}")
        
        return final_path

    def jobstate(self, state=False):
        """
        This function updates the paths in the JSON data to be relative.
        """
        self.job_finished = state

        if state:
            # Parse the results into JSON data
            data = [json.loads(s) for s in self.results]

            # Convert absolute paths to relative before saving to JSON
            for result in data:
                if 'annotated_image_path' in result:
                    result['annotated_image_path'] = self.abs_to_rel(result['annotated_image_path'])
                if 'image_path' in result:
                    result['image_path'] = self.abs_to_rel(result['image_path'])

                # Check if 'glycans' field exists and update paths inside it
                if 'glycans' in result:
                    for glycan in result['glycans']:
                        if 'extracted_image_path' in glycan:
                            glycan['extracted_image_path'] = self.abs_to_rel(glycan['extracted_image_path'])

                        if 'image_path' in glycan:
                            glycan['image_path'] = self.abs_to_rel(glycan['image_path'])

                        if 'glyImage' in glycan:
                            glycan['glyImage'] = self.abs_to_rel(glycan['glyImage'])

                        # Add more fields if necessary in the glycan object


            # Now propagate the same changes to self.results
            self.results = [json.dumps(result) for result in data] 

            # Write the updated data back to the JSON file
            with open(self.json_filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.log_file.close()
            # print("-------->>>JOB COMPLETED", state)

        return True

    def get_results(self):
        results = [json.loads(s) for s in self.results]
        return sorted(results, key=lambda x: x.get('image_count',0))

    def get_document_metadata(self):
        return self.document_metadata

    def check_and_create_paths(self,subdirs):
        """Ensure that all specified subdirectories exist under the work directory."""
        for subdir in subdirs:
            path = os.path.join(self.workdir, subdir)
            os.makedirs(path, exist_ok=True)
        return [os.path.join(self.workdir, subdir) for subdir in subdirs]

    def annotate_image(self,figure_semantics):
        """Annotate glycans and save the annotated image."""

        annotated_figures_path = os.path.join(self.workdir, "extracted_figures", "annotated_figures")
        self.create_directories(annotated_figures_path)

        for i, _ in enumerate(figure_semantics.glycans()):
            figure_semantics.annotate_glycans()

        # Save the annotated figure
        semanatic_fig_path = figure_semantics.image_path()
        fig_basename = os.path.basename(semanatic_fig_path)
        annotated_image_path = os.path.join(annotated_figures_path, fig_basename)
        self.save_image(figure_semantics.image(), annotated_image_path)
        figure_semantics.set('annotated_image_path',annotated_image_path)


    def get_IUPAC_metadata(self,gly_semantics):
        """
        Process IUPAC-related metadata for a single glycan.
        """

        IUPAC_data = {
            "composition_str": gly_semantics.compstr(),
            "orientation": "RL",
        }

        if len(gly_semantics.glycan_errors()) == 0:
            IUPAC_data["IUPAC"] = gly_semantics.IUPAC()
            # can get orienation only after the IUPAC is generated - because we access to directed links
            IUPAC_data["orientation"] = gly_semantics.glycan_orientation()

        iupac = IUPAC_data.get("IUPAC")
        compstr = IUPAC_data.get("composition_str")

        # GNOME URL
        uri_base = "https://gnome.glyomics.org/StructureBrowser.html?"

        if iupac:
            accession,wurcs = searchGlyLookup(iupac,
                                              baseurl=self.config.get('glylookup_url'),
                                              devemail=self.config.get('dev_email'))
        else:
            accession = None
            wurcs = None

        if accession:
            gnome_url = uri_base + 'focus=' + accession
        elif iupac:
            # subsumption = searchSubsumption(iupac,
            #                                 baseurl=self.config.get('subsumption_url'),
            #                                 devemail=self.config.get('dev_email'))
            # print(subsumption,file=sys.stderr)
            gnome_url = sendToGNOme(iupac,devemail=self.config.get('dev_email'))
        else:
            # converting composition format:
            # eg: "GlcNAc(5)NeuAc(2)" to "GlcNAc=5&NeuAc=2"
            matches = re.findall(r'([A-Za-z]+)\((\d+)\)', compstr)
            converted_composition = '&'.join(f"{name}={count}" for name, count in matches)
            gnome_url = 'https://gnome.glyomics.org/StructureBrowser.html?' + converted_composition

        
        if accession:
            IUPAC_data.update({
                "linkexpl": "Extracted successfully using accession",
                "gnomeurl": gnome_url,
                "accession": accession, 
                "glyImage": searchGlyImage(iupac, orientation=IUPAC_data["orientation"],
                                           baseurl=self.config.get('glymage_url'),
                                           devemail=self.config.get('dev_email'))
            })
            if wurcs:
                IUPAC_data['WURCS'] = wurcs
        elif iupac:
            IUPAC_data.update({
                "linkexpl": "Extracted structure using IUPAC.",
                "gnomeurl": gnome_url,
                "glyImage": searchGlyImage(iupac, orientation=IUPAC_data["orientation"],
                                           baseurl=self.config.get('glymage_url'),
                                           devemail=self.config.get('dev_email')),
            })
        elif compstr:
            # composition only
            IUPAC_data.update({
                "linkexpl": "Extracted structure using Composition.",
                "gnomeurl": gnome_url,
                "glyImage": searchGlyImage(compstr, orientation=IUPAC_data["orientation"],
                                           baseurl=self.config.get('glymage_url'),
                                           devemail=self.config.get('dev_email')),
            })

        return IUPAC_data



    def process_glycans(self,figure_semantics, image_folders):
        """
        Process all glycans in a figure, saving images and metadata.
        """

        basename = os.path.basename(figure_semantics.image_path()).split('.')[0]

        for i, gly_semantics in enumerate(figure_semantics.glycans()):
            glycan_image = gly_semantics.get('image')
            extracted_glycan_image = gly_semantics.get('extracted_image',glycan_image)

            if glycan_image is None or glycan_image.size == 0:
                print(f"Skipping glycan {i}: Detected object is missing or empty")
                continue

            image_name = f"{basename}-{i+1}.png"
            # save processed/cleaned extracted image
            image_url = os.path.join(image_folders['images_dir'], image_name)
            self.save_image(glycan_image,image_url)

            # save origial extracted imaged
            extracted_image_url = os.path.join(image_folders['extracted_images_dir'], image_name)
            self.save_image(extracted_glycan_image,extracted_image_url)

            # gly_semantics.set('image_path',save_origin_url)      
            gly_semantics.set('image_path', image_url)    
            gly_semantics.set('extracted_image_path',extracted_image_url)  
            gly_semantics.set('image_name', image_name)
            gly_semantics.set('fig_glycan_count', i+1)

            for key, val in self.get_IUPAC_metadata(gly_semantics).items():
                if key != "glyImage":
                    gly_semantics.set(key,val)
                else:
                   rest,imgfilename = val.rsplit('/',1)
                   glymage_image = os.path.join(image_folders['glymage_images_dir'], imgfilename)
                   wh = open(glymage_image,'wb')
                   with urllib.request.urlopen(val) as h:
                        wh.write(h.read())
                   wh.close()
                   gly_semantics.set(key,glymage_image)


    def progress_callback(self,**kwargs):
        if kwargs.get('stage') == "GLYCAN" and kwargs.get('checkpoint') == "DONE":
            nglycan = kwargs.get('nglycan')
            index = kwargs.get('index')
            image_number = kwargs.get("image_number", 0)
            page_num = kwargs.get("page_num",0)
            pmid_job = kwargs.get("pmid_job", False)
            figure_number = kwargs.get("figure_number","")
            caption = kwargs.get("caption","")
            
            if pmid_job:    # for PMID submissions
                if figure_number:
                    self.update_status("Processing figure %s, analyzed %d/%d glycan(s)"%(figure_number,index,nglycan))
                elif caption:
                    self.update_status("Processing %s, analyzed %d/%d glycan(s)"%(caption,index,nglycan))
                else:
                    self.update_status("Processing figure, analyzed %d/%d glycan(s)"%(index,nglycan))
            elif page_num == 0:     # for simple/multi glycans submissions
                self.update_status("Processing image, analyzed %d/%d glycan(s)"%(index,nglycan))
            else:   # for pdf submission
                if figure_number:
                    self.update_status("Processing figure %s, analyzed %d/%d glycan(s)"%(figure_number,index,nglycan))
                else:
                    self.update_status("Processing image %d from page %d, analyzed %d/%d glycan(s)"%(image_number,page_num,index,nglycan))


    def find_glycans(self, figure_path, image_folders, **kwargs):
        config = Config_Manager()
        self.pipeline_name = self.pipeline_mapping[self.submission_type]
        pipeline = config.get_pipeline(self.pipeline_name)

        figure_semantics = pipeline.run(figure_path, self.progress_callback, **kwargs)

        # Sort bbox L->R for UI
        sorted_glycans = sorted(
            figure_semantics.glycans(),
            key=lambda g: tuple(g.box().bbox())
        )
        figure_semantics.set_glycans(sorted_glycans)

        nglycan = len(figure_semantics.glycans())

        if nglycan > 0:
            if kwargs.get('pmid_job',False):
                if kwargs.get('figure_number'):
                    self.update_status("Processing figure %s, postprocessing %d glycan(s)"%(kwargs['figure_number'],nglycan))
                elif kwargs.get('caption'):
                    self.update_status("Processing %s, postprocessing %d glycan(s)"%(kwargs['caption'],nglycan))
                else:
                    self.update_status("Processing figure, postprocessing %d glycan(s)"%(nglycan))
            elif kwargs.get("page_num",0) == 0:
                self.update_status("Processing image, postprocessing %d glycan(s)" % (nglycan))
            else:
                if kwargs.get('figure_number'):
                    self.update_status("Processing figure %s, postprocessing %d glycan(s)" % (kwargs["figure_number"],nglycan))
                else:
                    self.update_status("Processing image %d from page %d, postprocessing %d glycan(s)" % (kwargs["image_number"], kwargs["page_num"], nglycan))

        self.annotate_image(figure_semantics)
        self.process_glycans(figure_semantics, image_folders)
        self.results.append(figure_semantics.tojson())

    def process_file(self):
        self.jobstate(False)
        self.update_state(APIFramework.RUNNING)

        base_path = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(base_path, "input", self.id, self.original_file_name)

        try:
            copyfile(input_file, self.input_filepath)
        except FileNotFoundError:
            time.sleep(5)
            copyfile(input_file, self.input_filepath)

        self.log_file.write(f"{self.id}\n{self.output_filepath}\n")

        figures_dir = os.path.join(self.workdir, "extracted_figures", "figures")
        extracted_images_dir = os.path.join(self.workdir, "extracted_figures", "extracted_images")
        images_dir = os.path.join(self.workdir, "extracted_figures", "images")
        glymage_dir = os.path.join(self.workdir, "glymage")

        image_folders = {'figures_dir': figures_dir, 'images_dir': images_dir, 'extracted_images_dir': extracted_images_dir, 'glymage_images_dir': glymage_dir}

        self.create_directories(*image_folders.values())

        # after required directories are ready - process the input file (figures)
        self.process_figures(image_folders)

        self.jobstate(True)


    def process_figures(self, image_folders):
        return NotImplementedError

class ImageJob(JobInstance):
    def process_figures(self, image_folders):
        self.update_status("Processing image")
        # self.task_detail['original_filepath'] = self.abs_to_rel(self.input_filepath)
        # self.task_detail['abs_original_filepath'] = self.input_filepath
        self.find_glycans(self.input_filepath,image_folders)

class PMIDJob(JobInstance):

    def process_figures(self, image_folders):
        base_path = os.path.dirname(os.path.abspath(__file__))

        self.update_status("Processing PMID manuscript")
        # self.task_detail['original_filepath'] = self.abs_to_rel(self.input_filepath)
        # self.task_detail['abs_original_filepath'] = self.input_filepath

        pmc_publication_info = self.task_detail.get('pmc_publication')

        figures_src = os.path.join(base_path, "input", self.id, f"PMID-{self.pmid}.tar.gz")
        figures_dest_dir = image_folders['figures_dir']

        fig_to_label_map = {}
        image_files = []

        self.figure_info_by_basename = {}
        self.figure_info_by_renamed = {}

        try:
            with tarfile.open(figures_src, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.lower().endswith('.nxml'):
                        file_obj = tar.extractfile(member)
                        if not file_obj:
                            continue

                        nxml_content = file_obj.read().decode('utf-8', errors='ignore')
                        xml_obj = XMLParser(nxml_content,pmc_publication=pmc_publication_info)

                        try:
                            xml_data = xml_obj.parse()

                            # document level information
                            self.document_metadata = {
                                k: v for k, v in xml_data.items() if k != "figure_info"
                            }
                            self.document_metadata['citation'] = xml_data.get('citation', None)

                            # per figure: basename --> fig_info
                            self.figure_info_by_basename = xml_data.get("figure_info") or {}
                            # print(self.figure_info_by_basename.items())

                            fig_to_label_map = {}
                            for basename, info in self.figure_info_by_basename.items():
                                # PMID 39988192 has no figure number for its title page graphic
                                # Do we need to support it? On PubMed, it is called "Graphical Abstract"
                                # and has no figure number. 
                                if 'figure_number' in info:
                                    fig_to_label_map[basename] = info['figure_number']
                                else:
                                    fig_to_label_map[basename] = ""

                        except Exception as e:
                            self.log_file.write(f"Warning: Could not parse nxml: {e}\n")
                            print("Parsing error", e)

                        break  

                # Second pass: extract figures and rename to their labels
                seen_basenames = set()

                for member in tar.getmembers():
                    filename = os.path.basename(member.name)
                    base_name, ext = os.path.splitext(filename)
                    ext = ext.lower()
                    if ext not in ('.jpg', '.jpeg', '.png'):
                        continue
                    if base_name not in fig_to_label_map:
                        continue
                    if base_name in seen_basenames:
                        continue  # already chose one format for this figure

                    file_obj = tar.extractfile(member)
                    if not file_obj:
                        continue

                    figure_number = fig_to_label_map[base_name]
                    renamed_file = f"{figure_number}{ext}"
                    renamed_file_path = os.path.join(figures_dest_dir, renamed_file)

                    with open(renamed_file_path, 'wb') as f:
                        f.write(file_obj.read())

                    image_files.append(renamed_file)
                    seen_basenames.add(base_name)

                    fig_info = self.figure_info_by_basename.get(base_name, {}).copy()
                    fig_info["figure_number"] = figure_number
                    self.figure_info_by_renamed[renamed_file] = fig_info

        except Exception as e:
            self.log_file.write(f"Error: opening tar {figures_src}: {e}\n")
            print("Error opening tar file", e)
            return

        # Sort to ensure correct order
        image_files.sort()

        image_count = 1
        for _, fig_name in enumerate(image_files, 1):
            fig_path = os.path.join(figures_dest_dir, fig_name)

            with Image.open(fig_path) as img:
                width, height = img.size

                base_fig_name = fig_name.rsplit('.', 1)[0]

                # look up XML metadata for this renamed figure (if any)
                fig_info = self.figure_info_by_renamed.get(fig_name, {})
                # if the figure_number is empty, can we assume Graphical Abstract?
                figure_metadata = {
                    "fig_bbox": [0, 0, width, height],
                    "image_count": image_count,
                    # XML-derived metadata (keys match XMLParser output)
                    "caption": fig_info.get("caption",""),
                    "figure_number": fig_info["figure_number"]
                }
                if not figure_metadata["figure_number"] and not figure_metadata["caption"]:
                    figure_metadata["caption"] = "Graphical Abstract"
                
                if figure_metadata["figure_number"]:
                    self.update_status("Processing figure %s" % figure_metadata["figure_number"])
                elif figure_metadata["caption"]:
                    self.update_status("Processing %s" % figure_metadata["caption"])
                else:
                    self.update_status("Processing figure")
                self.find_glycans(
                    fig_path,
                    image_folders,
                    pmid_job = True,
                    **figure_metadata,
                )

                image_count += 1

class PDFJob(JobInstance):
    """
    Main processing logic for PDF files.
    Handles file verification, page/image extraction, and glycan annotation.
    """
    
    def process_figures(self, image_folders):
        """
        Extract figure metadata from PDF using the two methods:
        a) xref based figure extraction (returns figures metadata) 
        b) Using Heuristics (PDFigCapX)- Text block and Image block positions (returns figures metadata) 

        Step 1
        - Both the above methods return figures metadata for the pdf. 
        Merge the info obtained --> to get the overall best figures metadata in the pdf.

        Step 2
        - Using the figures metadata, find all glycans using the object detection pipeline and generate semantics.
        """

        # update task_detail - with the original input_filepath
        # self.task_detail['original_filepath'] = self.abs_to_rel(self.input_filepath)
        # self.task_detail['abs_original_filepath'] = self.input_filepath
        # self.task_detail['pipeline_name'] = self.pipeline_name

        pdf = PDFHandler(self.input_filepath)
        doc = pdf.doc
        cite = pdf.get_citation()
        if cite:
            self.document_metadata['citation'] = cite['citation']
            self.document_metadata['pmid'] = cite['pmid']
            
        # Factory method
        image_search_instance = ImageSearch.search_method(self.task_detail['image_search_strategy'])
        pdf_images_metadata = image_search_instance.get_metadata(self.input_filepath)

        for page_num, fig_data in pdf_images_metadata.items():
            page = doc[page_num-1]
            for image_number, figure_info in fig_data.items():
                image_path = os.path.join(image_folders['figures_dir'], f"{figure_info['image_count']}.png")

                pix = PDFHandler.save_image(doc, page, figure_info['pdf_fig_bbox'], image_path, xref=figure_info.get('xref'), dpi=STANDARD_DPI, annots=True)

                figure_info['width'] = pix.width
                figure_info['height'] = pix.height

                self.log_file.write(f"\nSaved image to {image_path}")

                if figure_info.get('figure_number'):
                    self.update_status("Processing figure %s" % (figure_info['figure_number'],))
                else:
                    self.update_status("Processing image %d from page %d" % (image_number, page_num))

                self.find_glycans(image_path, image_folders, page_num=page_num, **figure_info)
