import fitz, sys, os, cv2,shutil, time, ntpath, json, base64, re, urllib.request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submit import searchGlyLookup, searchGlyImage, sendToGNOme
from PIL import Image
from hashlib import md5
from APIFramework import APIFramework
from BKGlycanExtractor import ImageSearch, FitzImageSearch, FigCapImageSearch, HybridImageSearch
from BKGlycanExtractor import Config_Manager, BoundingBox, PDFBoundingBox, CompareBoxes
from BKGlycanExtractor import PDFHandler, CompoundPDFImageFilter, PDFXRefImageFilter, PDFImageSizeFilter

import numpy as np
from shutil import copyfile
import tarfile
from io import BytesIO
import xml.etree.ElementTree as ET


class JobInstance:

    pipeline_mapping = {
        'Simple Glycan Image': 'SingleGlycanImage-YOLOFinders',
        # 'Single-Glycan Image': 'SingleGlycanImage-YOLOFinders',
        'Multi-Glycan Image': 'MultipleGlycanImage-YOLOFinders',
        'Manuscript': 'MultipleGlycanImage-YOLOFinders'
    }

    def __init__(self, task_detail, msg_queue = None):
        self.task_detail = task_detail
        self.id = task_detail.get('id')
        self.msg_queue = msg_queue
        self.original_file_name = task_detail.get('original_file_name')
        self.submission_type = task_detail.get('submission_type')

        self.pmid = task_detail.get('pmid')
        self.pmcid = task_detail.get('pmcid')

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

        self.pipeline_name = None
        self.job_finished = False
        self.results = []

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
        pmid = task_detail.get('pmid', None)

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
            accession,wurcs = searchGlyLookup(iupac)
        else:
            accession = None
            wurcs = None

        if accession:
            gnome_url = uri_base + 'focus=' + accession
        elif iupac:
            gnome_url = sendToGNOme(iupac)
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
                "glyImage": searchGlyImage(iupac, orientation=IUPAC_data["orientation"])
            })
            if wurcs:
                IUPAC_data['WURCS'] = wurcs
        elif iupac:
            IUPAC_data.update({
                "linkexpl": "Extracted structure using IUPAC.",
                "gnomeurl": gnome_url,
                "glyImage": searchGlyImage(iupac, orientation=IUPAC_data["orientation"]),
            })
        elif compstr:
            # composition only
            IUPAC_data.update({
                "linkexpl": "Extracted structure using Composition.",
                "gnomeurl": gnome_url,
                "glyImage": searchGlyImage(compstr, orientation=IUPAC_data["orientation"]),
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
            figure_num = kwargs.get("figure_num",0)
            page_num = kwargs.get("page_num",0)
            pmid_job = kwargs.get("pmid_job", False)
            
            if pmid_job:    # for PMID submissions
                self.update_status("Processing %s, analyzed %d/%d glycan(s)"%(kwargs['figure_name'],index,nglycan))
            elif page_num == 0:     # for simple/multi glycans submissions
                self.update_status("Processing image, analyzed %d/%d glycan(s)"%(index,nglycan))
            else:   # for pdf submission
                # self.update_status("Processing image %d from page %d, analyzed %d/%d glycan(s)"%(self.imageno,self.pageno,index,nglycan))
                self.update_status("Processing image %d from page %d, analyzed %d/%d glycan(s)"%(figure_num,page_num,index,nglycan))


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

        if kwargs.get("pmid_job", False):
            self.update_status("Processing %s, postprocessing %d glycan(s)"%(kwargs['figure_name'],nglycan))
        elif kwargs.get("page_num",0) == 0:
            self.update_status("Processing image, postprocessing %d glycan(s)" % (nglycan))
        else:
            self.update_status("Processing image %d from page %d, postprocessing %d glycan(s)" % (kwargs["figure_num"], kwargs["page_num"], nglycan))

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
        self.task_detail['original_filepath'] = self.abs_to_rel(self.input_filepath)
        self.task_detail['abs_original_filepath'] = self.input_filepath
        self.task_detail['pipeline_name'] = self.pipeline_name
        self.find_glycans(self.input_filepath,image_folders)

class PMIDJob(JobInstance):

    # Helper to get tag name without namespace from xml doc
    def get_tag_name(self, elem):
        return elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

    def find_figures(self, root):
        for elem in root.iter():
            if self.get_tag_name(elem) == 'fig':
                yield elem

    def extract_figure_info(self, fig):
        XLINK_NS = "http://www.w3.org/1999/xlink"  # W3C standard - consistent for xml files
        XLINK_HREF = f"{{{XLINK_NS}}}href"

        xml_fig_label = None
        fig_filename = None

        for child in fig.iter():
            tag = self.get_tag_name(child)
            if tag == 'label' and child.text:
                xml_fig_label = child.text.strip()
                # Should match "Fig. 1" and "Figure 1" and other variations...
                m = re.search(r'^\s*\w+\.?\s*(\d+)\.?\s*$',xml_fig_label)
                if m:
                    # normalize figure label
                    xml_fig_label = "figure %s"%(m.group(1))

            if tag in ('graphic', 'inline-graphic'):
                # Check both regular href and xlink:href
                href = child.get('href') or child.get(XLINK_HREF)
                if href:
                    fig_filename = os.path.basename(href)

        return fig_filename, xml_fig_label


    def process_figures(self, image_folders):

        base_path = os.path.dirname(os.path.abspath(__file__))

        self.update_status("Processing image")
        self.task_detail['original_filepath'] = self.abs_to_rel(self.input_filepath)
        self.task_detail['abs_original_filepath'] = self.input_filepath
        self.task_detail['pipeline_name'] = self.pipeline_name

        # zipped file location - which contains all info related to the PMID
        # copy all the figures from the zipped file to the extracted_figures folder (the figures are renamed to match the figure labels that appear in the manuscript)
        figures_src = os.path.join(base_path, "input", self.id, f"PMID-{self.pmid}.tar.gz")
        figures_dest_dir = image_folders['figures_dir']

        fig_to_label_map = {}
        image_files = []

        try:
            with tarfile.open(figures_src, "r:gz") as tar:

                # First pass: extract *nxml file and build label mapping (i.e create a dict which will help map the figure_no/name in the pdf with the extracted figure)
                # key - figure_name, value - figure label name from xml (that is present in the pdf)
                for member in tar.getmembers():
                    if member.name.lower().endswith('.nxml'):
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            nxml_content = file_obj.read().decode('utf-8', errors='ignore')

                            # Parse XML to find figure labels
                            try:
                                root = ET.fromstring(nxml_content)

                                for fig in self.find_figures(root):
                                    fig_filename, xml_fig_label = self.extract_figure_info(fig)

                                    # if both exist - then this is a True figure in the pdf (so the fig_name can be renamed to reflect what appears in the pdf)
                                    if fig_filename and xml_fig_label:
                                        fig_to_label_map[fig_filename] = xml_fig_label

                            except ET.ParseError as e:
                                self.log_file.write(f"Warning: Could not parse nxml: {e}")


                # Second pass: extract figure from the zipped location and rename them to match the figure labels in the pdf
                for member in tar.getmembers():
                    filename = os.path.basename(member.name)
                    base_name, ext = os.path.splitext(filename)
                    if ext.lower() == '.jpg':
                        # Extract file content directly using the tar module - extraction from a tar file requires thiese steps inorder to extract files to the correct directory
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            # check if the figure label is officially present in pdf - only then should the figure be displayed on the website
                            if base_name in fig_to_label_map:
                                label_id = fig_to_label_map[base_name]
                                renamed_file = f"{label_id}{ext}"
                                renamed_file_path = os.path.join(figures_dest_dir, renamed_file)

                                target_path = os.path.join(figures_dest_dir, filename)
                                with open(renamed_file_path, 'wb') as f:
                                    f.write(file_obj.read())

                                image_files.append(renamed_file)
        except Exception as e:
            self.log_file.write(f"Error: opening tar {figures_src}: {e}")
            return

        # Sort to ensure correct order
        image_files.sort()

        for figure_num, fig_name in enumerate(image_files, 1):
            fig_path = os.path.join(figures_dest_dir, fig_name)

            with Image.open(fig_path) as img:
                width, height = img.size
                
                base_fig_name = fig_name.rsplit('.',1)[0]
                figure_metadata = {"fig_bbox": [0, 0, width, height], "pmid_job": True, "figure_name": base_fig_name}
                self.update_status("Processing %s" % base_fig_name)
                self.find_glycans(fig_path, image_folders, **figure_metadata)


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
        self.task_detail['original_filepath'] = self.abs_to_rel(self.input_filepath)
        self.task_detail['abs_original_filepath'] = self.input_filepath
        self.task_detail['pipeline_name'] = self.pipeline_name

        doc = fitz.open(self.input_filepath)

        # Factory method
        image_search_instance = ImageSearch.search_method(self.task_detail['image_search_strategy'])
        pdf_images_metadata = image_search_instance.get_metadata(self.input_filepath)

        for page_num, fig_data in pdf_images_metadata.items():
            page = doc[page_num-1]
            for figure_num, figure_info in fig_data.items():
                image_path = os.path.join(image_folders['figures_dir'], f"{figure_info['image_count']}.png")

                pix = page.get_pixmap(clip=figure_info['pdf_fig_bbox'], dpi=figure_info['dpi'])        # x1,y1,x2,y2
                pix.save(image_path)
                figure_info['width'] = pix.width
                figure_info['height'] = pix.height

                self.log_file.write(f"\nSaved image to {image_path}")

                self.update_status("Processing image %d from page %d" % (figure_num, page_num))

                self.find_glycans(image_path, image_folders, **figure_info)
