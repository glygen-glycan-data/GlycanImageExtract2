'''
This 
'''
import fitz, sys, os, cv2,shutil, time, ntpath, json, base64, re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submit import searchGlyLookup, searchGlyImage, sendToGNOme
from PIL import Image
from hashlib import md5
from APIFramework import APIFramework




from BKGlycanExtractor import Config_Manager, BoundingBox
# from . import glycanExtractor

# from .glycanannotator import GlycanExtractorPipeline
from collections import Counter
import numpy as np
from shutil import copyfile


class JobInstance:

    pipeline_mapping = {
        'Simple Glycan Image': 'SingleGlycanImage-YOLOFinders',
        # 'Single-Glycan Image': 'SingleGlycanImage-YOLOFinders',
        'Multi-Glycan Image': 'MultipleGlycanImage-YOLOFinders',
        'Manuscript': 'MultipleGlycanImage-YOLOFinders'
    }

    def __init__(self, task_detail, msg_queue = None):
        self.id = task_detail.get('id')
        self.msg_queue = msg_queue
        self.original_file_name = task_detail.get('original_file_name')
        self.submission_type = task_detail.get('submission_type')

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
    def get_processor(task_detail):
        original_file_name = task_detail.get('original_file_name')
        file_extension = original_file_name.rsplit('.', 1)[-1]

        if file_extension in ('png', 'jpg', 'jpeg'):
            return ImageJob
        elif file_extension == 'pdf':
            return PDFJob
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


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
                "glyImage": searchGlyImage(accession, orientation=IUPAC_data["orientation"],accession=True)
            })
            if wurcs:
                IUPAC_data['WURCS'] = wurcs
        elif iupac:
            IUPAC_data.update({
                "linkexpl": "Extracted structure using IUPAC.",
                "gnomeurl": gnome_url,
                "glyImage": searchGlyImage(iupac, orientation=IUPAC_data["orientation"]),
            })
        else:
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
                gly_semantics.set(key,val)


    def progress_callback(self,**kwargs):
        if kwargs.get('stage') == "GLYCAN" and kwargs.get('checkpoint') == "DONE":
            nglycan = kwargs.get('nglycan')
            index = kwargs.get('index')
            figure_num = kwargs.get("figure_num",0)
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

        figure_semantics = pipeline.run(figure_path, self.progress_callback, **kwargs)

        # Sort bbox L->R for UI
        sorted_glycans = sorted(
            figure_semantics.glycans(),
            key=lambda g: tuple(g.box().bbox())
        )
        figure_semantics.set_glycans(sorted_glycans)

        fig_box_t = kwargs.get("pdf_fig_box")            # tuple (x0,y0,x1,y1)
        x_scale = kwargs.get("x_scale")
        y_scale = kwargs.get("y_scale")

        # if case - special for PDF - so maybe this might need some re-org?
        # Like a hook method for PDFJob class - leaving it for TO DO as of now
        if fig_box_t is not None and x_scale is not None and y_scale is not None:
            fig_box = fitz.Rect(fig_box_t)

            for gly_semantics in figure_semantics.glycans():
                x0, y0, x1, y1 = gly_semantics.box().corners()
                x0, x1 = sorted((x0, x1))
                y0, y1 = sorted((y0, y1))

                pdf_x0 = fig_box.x0 + x0 * x_scale
                pdf_x1 = fig_box.x0 + x1 * x_scale
                pdf_y0 = fig_box.y0 + y0 * y_scale
                pdf_y1 = fig_box.y0 + y1 * y_scale

                gly_semantics.set("pdf_glycan_box", (pdf_x0, pdf_y0, pdf_x1, pdf_y1))


        nglycan = len(figure_semantics.glycans())
        if kwargs.get("page_num",0) == 0:
            self.update_status("Processing image, postprocessing %d glycan(s)" % (nglycan))
        else:
            # self.update_status("Processing image %d from page %d, postprocessing %d glycan(s)" % (self.imageno, kwargs["page_num"], nglycan))
            self.update_status("Processing image %d from page %d, postprocessing %d glycan(s)" % (kwargs["figure_num"], kwargs["page_num"], nglycan))

        self.annotate_image(figure_semantics)
        self.process_glycans(figure_semantics, image_folders)
        self.results.append(figure_semantics.tojson())


class ImageJob(JobInstance):

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

        image_folders = {'figures_dir': figures_dir, 'images_dir': images_dir, 'extracted_images_dir': extracted_images_dir}

        self.create_directories(*image_folders.values())

        self.update_status("Processing image")
        # self.imageno = 1
        # self.pageno = 0

        # metadata = {}

        
        self.find_glycans(self.input_filepath,image_folders)
        # self.glycan_obj['file_format_error'] = (
        #     f"No glycans were present/detected in the image; switched to {new_file_type} detection mode!"
        #     if glycan_data
        #     else "Something might be wrong with the file you used, please enter another file"
        # )

        # self.log_file.close()
        self.jobstate(True)




class PDFJob(JobInstance):

    def process_file(self):
        """
        Main processing logic for PDF files.
        Handles file verification, page/image extraction, and glycan annotation.
        """
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

        image_folders = {'figures_dir': figures_dir, 'images_dir': images_dir, 'extracted_images_dir': extracted_images_dir}

        self.create_directories(*image_folders.values())

        # figures_metadata = self.extract_pdf_figure_metadata()
        # self.log_file.write(f"\nFound {len(figures_metadata)} figures in the PDF.")

        self.process_pdf_pages(image_folders)

        self.jobstate(True) 

    
    def process_pdf_pages(self, image_folders):
        """
        Extract figure metadata from PDF using fitz.
        On each figure - find all glycans using the object detection pipeline and generate semantics
        """

        doc = fitz.open(self.input_filepath)
        # figure_metadata = []
        figure_num = 0

        for page_num, page in enumerate(doc.pages()):
            # page = doc[page_num]
            info = page.get_image_info(xrefs=True)
            for img in info:
                xref = img["xref"]

                # Note: pdf_fig_box is not in pixels
                # this is in page cooridniates which is called points (1 point = 1/72 inch)
                pdf_fig_box = fitz.Rect(img["bbox"])
                pdf_fig_height = pdf_fig_box.height
                pdf_fig_width = pdf_fig_box.width
                area = pdf_fig_height * pdf_fig_width

            
                self.log_file.write(
                    f"\nFigure number: {figure_num}, Page number: {page_num},  BBox: {img["bbox"]}, Width: {pdf_fig_width}, Height: {pdf_fig_height}, Area: {area}\n"
                )

                if (pdf_fig_height > 60 and pdf_fig_width > 60) or area > 360:
                    # figure_data['figure_count'] = self.figure_count

                    images_path = os.path.join(image_folders['figures_dir'], f"{xref}.png")

                    pix = fitz.Pixmap(doc, xref)
                    try:
                        pix = fitz.Pixmap(doc, xref)

                        # Note - this is in pixels (which means it is the original dimensions of the image
                        # irrespective of the scaling (smaller/bigger) used to display it on the PDF)
                        fig_px_w, fig_px_h = pix.width, pix.height

                        pix.save(images_path)
                    except Exception as e:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        pix.save(images_path)


                    # Since the original image dimensions might be scaled on the PDF page - we
                    # need to make adjustments to map these image pixels wrt the page
                    x_scale = pdf_fig_box.width / float(fig_px_w) if fig_px_w else 0.0
                    y_scale = pdf_fig_box.height / float(fig_px_h) if fig_px_h else 0.0


                    figure_metadata = {
                        "page_num": page_num,
                        "figure_num": figure_num,
                        "xref": xref,
                        "pdf_fig_box": (pdf_fig_box.x0, pdf_fig_box.y0, pdf_fig_box.x1, pdf_fig_box.y1),
                        # "fig_px_w": fig_px_w,
                        # "fig_px_h": fig_px_h,
                        "x_scale": x_scale,
                        "y_scale": y_scale,
                    }
                    

                    self.log_file.write(f"\nSaved image to {images_path}")

                    self.update_status("Processing image %d from page %d" % (figure_num, page_num))

                    self.find_glycans(images_path, image_folders, **figure_metadata)

                    figure_num += 1

                    



    # def extract_pdf_figure_metadata(self):
    #     """
    #     Extract figure metadata using fitz.
    #     Returns list with fig bbox in page coords and pixel dims + scales.
    #     """
    #     doc = fitz.open(self.input_filepath)
    #     figure_metadata = []
    #     figure_num = 0

    #     for page_num in range(len(doc)):
    #         page = doc[page_num]
    #         info = page.get_image_info(xrefs=True)
    #         # info entries include 'xref' and 'bbox' (page coordinate rect)
    #         for img in info:
    #             xref = img["xref"]

    #             # Note: pdf_fig_bbox is not in pixels
    #             # this is in page cooridniates which is called points (1 point = 1/72 inch)
    #             pdf_fig_bbox = fitz.Rect(img["bbox"])

    #             # Use Pixmap on xref to get pixel size
    #             try:
    #                 pix = fitz.Pixmap(doc, xref)

    #                 # Note - this is in pixels (which means it is the original dimensions of the image
    #                 # irrespective of the scaling (smaller/bigger) used to display it on the PDF)
    #                 fig_px_w, fig_px_h = pix.width, pix.height
    #             except Exception:
    #                 # Fallback to displayed size if pixmap fails
    #                 # fig_px_w, fig_px_h = int(fig_bbox.width), int(fig_bbox.height)
    #                 self.log_file.write(f"Warning: Could not extract pixel dimensions for xref {xref}: {e}\n")
    #                 continue  # Skip this image entirely
                    

    #             # Since the original image might be scaled on the PDF page - we
    #             # need to make adjustments to map these image pixels wrt the page
    #             x_scale = pdf_fig_bbox.width / float(fig_px_w) if fig_px_w else 0.0
    #             y_scale = pdf_fig_bbox.height / float(fig_px_h) if fig_px_h else 0.0

    #             # reason for scaling and storing pixel info - because the pipeline runs
    #             # on the extracted figures and generates semnatics wrt figure pixels and then to create
    #             # annotations back on the pdf - we need to scale it according to PDF dimesnions
    #             metadata = {
    #                 "page_num": page_num,
    #                 "figure_num": figure_num,
    #                 "xref": xref,
    #                 "pdf_fig_bbox": (pdf_fig_bbox.x0, pdf_fig_bbox.y0, pdf_fig_bbox.x1, pdf_fig_bbox.y1),
    #                 "fig_px_w": fig_px_w,
    #                 "fig_px_h": fig_px_h,
    #                 "x_scale": x_scale,
    #                 "y_scale": y_scale,
    #             }
    #             figure_metadata.append(metadata)
    #             figure_num += 1

    #     return figure_metadata

    
    # def process_pdf_page(self, figures_metadata, image_folders):
    #     doc = fitz.open(self.input_filepath)
    #     for page_index, page in enumerate(doc.pages()):
    #         page_metadata = [d for d in figures_metadata if d['page_num'] == page_index]
    #         # self.process_pdf_page(doc, page_index, page_metadata, image_folders)

    #         for image_index, figure_data in enumerate(page_metadata):
    #             xref = figure_data["xref"]

    #             # Use fitz provided bbox in page coordinates
    #             pdf_fig_bbox = fitz.Rect(figure_data["pdf_fig_bbox"])   # since its a tuple - reconstruct it as a fitz Rect
    #             pdf_fig_height = pdf_fig_bbox.height
    #             pdf_fig_width = pdf_fig_bbox.width
    #             area = pdf_fig_height * pdf_fig_width

    #             self.log_file.write(
    #                 f"\nFigure number: {figure_data["figure_num"]}, Page number: {figure_data["page_num"]},  BBox: {tuple(figure_data["pdf_fig_bbox"])}, Width: {pdf_fig_width}, Height: {pdf_fig_height}, Area: {area}\n"
    #             )

    #             if (pdf_fig_height > 60 and pdf_fig_width > 60) or area > 360:
    #                 # figure_data['figure_count'] = self.figure_count

    #                 images_path = os.path.join(image_folders['figures_dir'], f"{xref}.png")

    #                 pix = fitz.Pixmap(doc, xref)
    #                 try:
    #                     pix.save(images_path)
    #                 except ValueError:
    #                     pix = fitz.Pixmap(fitz.csRGB, pix)
    #                     pix.save(images_path)

    #                 self.log_file.write(f"\nSaved image to {images_path}")

    #                 self.update_status("Processing image %d from page %d" % (figure_data["figure_num"], figure_data["page_num"]))

    #                 self.find_glycans(images_path, image_folders, **figure_data)





    # def process_pdf_pages(self, doc, page_index, page_metadata, image_folders):
    #     # page_index is used for status

    #     for image_index, figure_data in enumerate(page_metadata):
    #         xref = figure_data["xref"]

    #         # Use fitz provided bbox in page coordinates
    #         pdf_fig_bbox = fitz.Rect(figure_data["pdf_fig_bbox"])   # since its a tuple - reconstruct it as a fitz Rect
    #         pdf_fig_height = pdf_fig_bbox.height
    #         pdf_fig_width = pdf_fig_bbox.width
    #         area = pdf_fig_height * pdf_fig_width

    #         self.log_file.write(
    #             f"\nFigure number: {figure_data["figure_num"]}, Page number: {figure_data["page_num"]},  BBox: {tuple(figure_data["pdf_fig_bbox"])}, Width: {pdf_fig_width}, Height: {pdf_fig_height}, Area: {area}\n"
    #         )

    #         if (pdf_fig_height > 60 and pdf_fig_width > 60) or area > 360:
    #             # figure_data['figure_count'] = self.figure_count

    #             images_path = os.path.join(image_folders['figures_dir'], f"{xref}.png")

    #             # Use existing doc; do not reopen for each xref
    #             pix = fitz.Pixmap(doc, xref)
    #             try:
    #                 pix.save(images_path)
    #             except ValueError:
    #                 pix = fitz.Pixmap(fitz.csRGB, pix)
    #                 pix.save(images_path)

    #             self.log_file.write(f"\nSaved image to {images_path}")

    #             self.update_status("Processing image %d from page %d" % (figure_data["figure_num"], figure_data["page_num"]))

    #             self.find_glycans(images_path, image_folders, **figure_data)



