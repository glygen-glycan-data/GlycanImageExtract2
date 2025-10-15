import fitz, sys, os, cv2,shutil, time, ntpath, json, base64, re, urllib.request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submit import searchGlyLookup, searchGlyImage, sendToGNOme
from PIL import Image
from hashlib import md5
from APIFramework import APIFramework

from BKGlycanExtractor import Config_Manager, BoundingBox

import numpy as np
from shutil import copyfile


class JobInstance:

    pipeline_mapping = {
        'Simple Glycan Image': 'SingleGlycanImage-YOLOFinders',
        # 'Single-Glycan Image': 'SingleGlycanImage-YOLOFinders',
        'Multi-Glycan Image': 'MultipleGlycanImage-YOLOFinders',
        'Manuscript': 'MultipleGlycanImage-YOLOFinders',
        'PMID': 'MultipleGlycanImage-YOLOFinders'
    }

    def __init__(self, task_detail, msg_queue = None):
        self.id = task_detail.get('id')
        self.msg_queue = msg_queue
        self.original_file_name = task_detail.get('original_file_name')
        self.submission_type = task_detail.get('submission_type')
        self.pmid = task_detail.get("pmid")
        self.pmcid = task_detail.get("pmcid")

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

        if submission_type == "Manuscript":
            return PDFJob(task_detail,*args,**kwargs)
        elif submission_type in ("Simple Glycan Image","Multi-Glycan Image"):
            return ImageJob(task_detail,*args,**kwargs)  
        elif submission_type == "PMID":
            return PMIDJob(task_detail,*args,**kwargs)          
        
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
            print("-------->>>JOB COMPLETED", state)

        return True


    def get_results(self):
        return [json.loads(s) for s in self.results]

    def check_and_create_paths(self,subdirs):
        """Ensure that all specified subdirectories exist under the work directory."""
        for subdir in subdirs:
            path = os.path.join(self.workdir, subdir)
            os.makedirs(path, exist_ok=True)
        return [os.path.join(self.workdir, subdir) for subdir in subdirs]


    def process_file(self):
        raise NotImplementedError


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
                "glyImage": searchGlyImage(accession, orientation=IUPAC_data["orientation"],accession=True),
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
            # Note: in some cases compstr is None (reason: maybe a False case of Glycan identification)
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
            # default for page_num becomes 0 - 0 is only for single images and not pdf
            page_num = kwargs.get("page_num",0)
            if page_num == 0:
                self.update_status("Processing image, analyzed %d/%d glycan(s)"%(index,nglycan))
            else:
                # self.update_status("Processing image %d from page %d, analyzed %d/%d glycan(s)"%(self.imageno,self.pageno,index,nglycan))
                self.update_status("Processing image %d from page %d, analyzed %d/%d glycan(s)"%(figure_num,page_num,index,nglycan))


    def find_glycans(self, figure_path, image_folders, **kwargs):
        config = Config_Manager()
        self.pipeline_name = self.pipeline_mapping[self.submission_type]
        pipeline = config.get_pipeline(self.pipeline_name)

        # if self.submission_type == 'PMID':
        #     # if PMID is the submission type -
        #     # then move figures from input submission to self.workdir - extracted figures
        #     # need to run pipeline on those extracted figures and generate figure semnatics
        #     # note: how to merge the semantics of different figures in one result json - treat it like
        #     # a pdf which contains figures - but for each dict within figure_result....along with - annotated_image_path, image_path, need to also mention the figure_path of the main figure this image was ecxtracted from
        #     print("RUN pipeline on the extracted figures stored in the extracted figures directory")
        #     print("generate figure semantics similar to a PDF")
        # else:
        figure_semantics = pipeline.run(figure_path, self.progress_callback, **kwargs)

        # Sort bbox from left->right for UI
        sorted_glycans = sorted(
            figure_semantics.glycans(),
            key=lambda g: tuple(g.box().bbox())
        )
        figure_semantics.set_glycans(sorted_glycans)

        nglycan = len(figure_semantics.glycans())
        if kwargs.get("page_num",0) == 0:
            self.update_status("Processing image, postprocessing %d glycan(s)" % (nglycan))
        else:
            # self.update_status("Processing image %d from page %d, postprocessing %d glycan(s)" % (self.imageno, kwargs["page_num"], nglycan))
            self.update_status("Processing image %d from page %d, postprocessing %d glycan(s)" % (kwargs["figure_num"], kwargs["page_num"], nglycan))

        self.annotate_image(figure_semantics)
        self.process_glycans(figure_semantics, image_folders)
        self.results.append(figure_semantics.tojson())

class BaseJob(JobInstance):
    """Base class for all job types with common functionality"""
    
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

        # Delegate to specific processing logic
        self._prepare_images(image_folders)
        self._process_images(image_folders)

        self.jobstate(True)

    def _prepare_images(self, image_folders):
        """Override in subclasses for specific image preparation"""
        pass  # Default: do nothing (for ImageJob)

    def _process_images(self, image_folders):
        """Override in subclasses for specific image processing"""
        raise NotImplementedError("Subclasses must implement _process_images")


class ImageJob(BaseJob):
    """Process direct image files"""
    
    def _process_images(self, image_folders):
        """Process single image directly"""
        self.update_status("Processing image")
        self.find_glycans(self.input_filepath, image_folders)


class PMIDJob(BaseJob):
    """Process figures from PMCID folder for Manuscripts"""
    
    def _prepare_images(self, image_folders):
        """Copy figures from PMCID folder"""
        base_path = os.path.dirname(os.path.abspath(__file__))
        figures_src = os.path.join(base_path, "input", self.pmcid)
        figures_dir = image_folders['figures_dir']

        # Note: Each figure has different formats (jpg, gif) -> so I selected jpg as the standard
        for name in os.listdir(figures_src):
            p = os.path.join(figures_src, name)
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() == ".jpg":
                shutil.copy(p, figures_dir)

        # after figures have been copied to their dedicated extracted_figures dir -> delete the zipped file obtained from pubmed central 
        # Clean up after copying - using absolute paths
        base_path = os.path.dirname(os.path.abspath(__file__))
        pmcid_folder_abs = os.path.abspath(os.path.join(base_path, "input", self.pmcid))
        tar_gz_file_abs = os.path.abspath(os.path.join(base_path, "input", f"{self.pmcid}.tar.gz"))


        # Delete the PMCID folder
        if os.path.exists(pmcid_folder_abs):
            shutil.rmtree(pmcid_folder_abs)
            # print(f"Deleted folder: {pmcid_folder_abs}")

        # Delete the zip file
        if os.path.exists(tar_gz_file_abs):
            os.remove(tar_gz_file_abs)
            # print(f"Deleted zip file: {tar_gz_file_abs}")
        


    def _process_images(self, image_folders):
        """Process all figures in figures directory"""
        figures_src = image_folders['figures_dir']

        for fig_name in os.listdir(figures_src):
            fig_path = os.path.join(figures_src, fig_name)

            with Image.open(fig_path) as img:
                width, height = img.size

            figure_metadata = {
                "fig_bbox": [0, 0, width, height]   # x,y,w,h
            }
    
            self.find_glycans(fig_path, image_folders, **figure_metadata)


class PDFJob(BaseJob):
    """Process figures extracted from PDF pages"""
    
    def _prepare_images(self, image_folders):
        """Extract figures from PDF"""
        doc = fitz.open(self.input_filepath)
        figure_num = 1

        for page_num, page in enumerate(doc.pages(), 1):
            info = page.get_image_info(xrefs=True)
            for img in info:
                xref = img["xref"]
                pdf_fig_box = fitz.Rect(img["bbox"])
                pdf_fig_height = pdf_fig_box.height
                pdf_fig_width = pdf_fig_box.width
                area = pdf_fig_height * pdf_fig_width

                self.log_file.write(
                    f"\nFigure number: {figure_num}, Page number: {page_num},  BBox: {img["bbox"]}, Width: {pdf_fig_width}, Height: {pdf_fig_height}, Area: {area}\n"
                )

                if (pdf_fig_height > 60 and pdf_fig_width > 60) or area > 360:
                    images_path = os.path.join(image_folders['figures_dir'], f"{xref}.png")

                    pix = fitz.Pixmap(doc, xref)
                    try:
                        pix.save(images_path)
                    except Exception as e:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(images_path)

                    self.log_file.write(f"\nSaved image to {images_path}")
                    self.update_status("Processing image %d from page %d" % (figure_num, page_num))

                    figure_num += 1

    def _process_images(self, image_folders):
        """Process all figures in figures directory"""
        figures_src = image_folders['figures_dir']

        for fig_name in os.listdir(figures_src):
            fig_path = os.path.join(figures_src, fig_name)

            with Image.open(fig_path) as img:
                width, height = img.size

            figure_metadata = {
                "fig_bbox": [0, 0, width, height]   # x,y,w,h
            }
    
            self.find_glycans(fig_path, image_folders, **figure_metadata)