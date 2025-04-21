import fitz, sys, os, cv2,shutil, pdfplumber, time, ntpath, json, base64
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from submit import searchGlyLookup, searchGlyImage
from PIL import Image



from BKGlycanExtractor import Config_Manager, Glycan_Semantics, Figure_Semantics, GlycanExtractorPipeline
# from . import glycanExtractor

# from .glycanannotator import GlycanExtractorPipeline
from collections import Counter
import numpy as np
from shutil import *


class JobInstance:

    pipeline_mapping = {
        'single_figure_img': 'SingleGlycanImage-YOLOFinders',
        'multi_figure_img': 'MultipleGlycanImage-YOLOFinders',
        'multi_figure_pdf': 'MultipleGlycanImage-YOLOFinders'
    }

    def __init__(self, task_detail):
        self.id = task_detail.get('id')
        self.original_file_name = task_detail.get('original_file_name')
        self.file_type = task_detail.get('file_type')

        # other instance variables
        self.workdir = os.path.join("./static/files", self.id)
        self.input_filepath = os.path.join(self.workdir, "input", self.original_file_name)
        self.output_filepath = os.path.join(self.workdir, "output", "annotated_" + self.original_file_name)

        self.pipeline_name = None
        self.job_finished = False
        self.results = []

        self.log_file = open(self.output_filepath.rsplit('.', 1)[0] + "_log.txt",'w')
        self.json_filepath = self.output_filepath.rsplit('.', 1)[0] + "_job.json"

        # input_file = os.path.join("input", task_detail["id"])
        # copyfile(input_file, self.original_file_name)

        self.create_directories(f"./static/files/{self.id}/input", f"./static/files/{self.id}/output")
    
    
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


    def jobstate(self,state=False):
        self.job_finished = state
        
        if state:
            data = [json.loads(s) for s in self.results]

            with open(self.json_filepath, 'w') as f:
                json.dump(data,f,indent=2)
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
            print(f"Annotating glycan {i}")
            figure_semantics.annotate_glycans()

        # Save the annotated figure
        semanatic_fig_path = figure_semantics.image_path()
        fig_basename = os.path.basename(semanatic_fig_path)
        annotated_image_path = os.path.join(annotated_figures_path, fig_basename)
        self.save_image(figure_semantics.image(), annotated_image_path)
        # print(f"Annotated image saved at {annotated_image_path}")
        figure_semantics.set('annotated_image_path',annotated_image_path)


    def get_IUPAC_metadata(self,gly_semantics):
        """
        Process IUPAC-related metadata for a single glycan.
        """

        errors = []
        structure_errors = []
        error_found = False

        IUPAC_data = {
            "composition_str": gly_semantics.compstr(),
            "links_count": len(gly_semantics.undirected_links()),
            "monos_count": len(gly_semantics.monosaccharides()),
            "errors": errors,
            "structure_errors": structure_errors,
            "IUPAC": "",
            "orientation": "RL",
        }

        # if conditions to log errors 
        if not gly_semantics.root():
            error_message = f"Unable to detect reducing-end in the structure."
            self.log_file.write(error_message + '\n')
            structure_errors.append(error_message)
            errors.append(error_message)
            error_found = True

        is_tree, error_message = gly_semantics.traverse_tree()
        if not is_tree:
            error_found = True
            errors.append("Error in detection of links.")

        if error_message != '':
            self.log_file.write(error_message + '\n')
            structure_errors.append(error_message)

        if not error_found:
            IUPAC_data["IUPAC"] = gly_semantics.IUPAC()
            # can get orienation only after the IUPAC is generated - because we access to directed links
            IUPAC_data["orientation"] = gly_semantics.glycan_orientation()
            lookup_key = IUPAC_data["IUPAC"]
        else:
            lookup_key = IUPAC_data["composition_str"]

        accession = searchGlyLookup(lookup_key)

        # Build GNOME URL - work on this
        uri_base = "https://gnome.glyomics.org/StructureBrowser.html?"
        if not IUPAC_data["IUPAC"]:
            # glycan_uri = uri_base + "&".join(
            #     [f"{k}={v}" for k, v in gly_semantics.composition().items()]
            # )

            gly_image = searchGlyImage(lookup_key, orientation=IUPAC_data["orientation"])
            IUPAC_data.update({
                "linkexpl": "Couldn't generate structure because no accession was found.",
                "glyImage": gly_image
                # "gnomeurl": glycan_uri,
            })
        elif accession: 
            glycan_uri = (
                    uri_base + f"focus={accession}"
                    if accession.startswith('G') else uri_base + f"ondemandtaskid={accession}"
            )

            gly_image = searchGlyImage(lookup_key, orientation=IUPAC_data["orientation"])

            IUPAC_data.update({
                "linkexpl": "Extracted successfully using the accession",
                "gnomeurl": glycan_uri,
                "accession": accession, 
                "glyImage": gly_image
            })
            
        return IUPAC_data



    def process_glycans(self,figure_semantics, image_folders):
        """
        Process all glycans in a figure, saving images and metadata.
        """

        basename = os.path.basename(figure_semantics.image_path()).split('.')[0]

        for i, gly_semantics in enumerate(figure_semantics.glycans()):
            unprocessed_glycan_image = gly_semantics.semantics.get('unprocessed_image')
            glycan_image = gly_semantics.semantics.get('image')

            if glycan_image is None or glycan_image.size == 0:
                print(f"Skipping glycan {i}: Detected object is missing or empty")
                continue

            # image_folders = {'figures_path': figures_path, 'images_path': images_path, 'processed_images_path': processed_images_path}

            # Create directories and save image
            # glycan_dir = os.path.join(self.workdir, "test", f"{basename}-{i+1}")
            # self.create_directories(glycan_dir)

            image_name = f"{basename}-{i+1}.png"
            # save processed/cleaned extracted image
            image_url = os.path.join(image_folders['images_dir'], image_name)
            self.save_image(glycan_image,image_url)

            # save origial extracted imaged
            unprocessed_image_url = os.path.join(image_folders['unprocessed_images_dir'], image_name)
            self.save_image(unprocessed_glycan_image,unprocessed_image_url)

            # gly_semantics.set('image_path',save_origin_url)      
            gly_semantics.set('image_path', image_url)    
            gly_semantics.set('unprocessed_image_path',unprocessed_image_url)  
            gly_semantics.set('image_name', image_name)

            for key, val in self.get_IUPAC_metadata(gly_semantics).items():
                gly_semantics.set(key,val)

            # gly_semantics.set("page_num",page_num)


    def find_glycans(self, figure_path, image_folders):

        # Run the pipeline
        config = Config_Manager()
        self.pipeline_name = self.pipeline_mapping[self.file_type]
        pipeline = config.get_pipeline(self.pipeline_name)
        print("figure_path",figure_path)
        figure_semantics = pipeline.run(figure_path)

        # Annotate and save images
        self.annotate_image(figure_semantics)

        # Other glycan info
        self.process_glycans(figure_semantics,image_folders)
       
        self.results.append(figure_semantics.tojson())


class ImageJob(JobInstance):
    def __init__(self, task_detail):
        super().__init__(task_detail)


    def process_file(self):
        self.jobstate(False)

        input_file = os.path.join("input", self.id)

        try:
            copyfile(input_file, self.input_filepath)
        except FileNotFoundError:
            time.sleep(5)
            copyfile(input_file, self.input_filepath)

        self.log_file.write(f"{self.id}\n{self.output_filepath}\n")

        figures_dir = os.path.join(self.workdir, "extracted_figures", "figures")
        unprocessed_images_dir = os.path.join(self.workdir, "extracted_figures", "unprocessed_images")
        images_dir = os.path.join(self.workdir, "extracted_figures", "images")

        image_folders = {'figures_dir': figures_dir, 'images_dir': images_dir, 'unprocessed_images_dir': unprocessed_images_dir}

        self.create_directories(*image_folders.values())

        self.find_glycans(self.input_filepath,image_folders)

        # self.glycan_obj['file_format_error'] = (
        #     f"No glycans were present/detected in the image; switched to {new_file_type} detection mode!"
        #     if glycan_data
        #     else "Something might be wrong with the file you used, please enter another file"
        # )

        # self.log_file.close()
        self.jobstate(True)




class PDFJob(JobInstance):
    def __init__(self, task_detail):
        super().__init__(task_detail)

    def process_file(self):
        """
        Main processing logic for PDF files.
        Handles file verification, page/image extraction, and glycan annotation.
        """
        self.jobstate(False)  

        input_file = os.path.join("input", self.id)

        try:
            copyfile(input_file, self.input_filepath)
        except FileNotFoundError:
            time.sleep(5)
            copyfile(input_file, self.input_filepath)

        self.log_file.write(f"{self.id}\n{self.output_filepath}\n")

        figures_dir = os.path.join(self.workdir, "extracted_figures", "figures")
        unprocessed_images_dir = os.path.join(self.workdir, "extracted_figures", "unprocessed_images")
        images_dir = os.path.join(self.workdir, "extracted_figures", "images")

        image_folders = {'figures_dir': figures_dir, 'images_dir': images_dir, 'unprocessed_images_dir': unprocessed_images_dir}

        self.create_directories(*image_folders.values())

        images_metadata = self.extract_images_from_pdf()
        self.log_file.write(f"\nFound {len(images_metadata)} figures in the PDF.")

        doc = fitz.open(self.input_filepath)
        for page_index, page in enumerate(doc.pages()):
            page_pix = doc.load_page(page_index).get_pixmap()
            figure_path = os.path.join(image_folders['figures_dir'], f"{page_index}.png")
            # page_pix.save(figure_path)

            # maybe we dont need the below two lines (and also dont need images_metadata) 
            # and just need find_glycans() here with figure path?
            # self.find_glycans(figure_path,image_folders)
            page_metadata = [data for data in images_metadata if data['image_page'] == page_index+1]
            self.process_pdf_page(figure_path, page_metadata, image_folders)

        # self.log_file.close()
        self.jobstate(True) 



    def extract_images_from_pdf(self):
        """
        Extract image metadata from the PDF file.
        Returns an array of image metadata and page array.
        """
        pdf_file = pdfplumber.open(self.input_filepath)
        image_metadata = []
        # page_array = []
        image_counter = 0

        # xref - identifies the specific image object in the pdf
        for page_index, page in enumerate(pdf_file.pages):
            page_height = page.height
            for image in page.images:
                box = (image['x0'], page_height - image['y1'], image['x1'], page_height - image['y0'])
                metadata = {
                    "image_page": image['page_number'],
                    "image_id": f"iid_{image_counter}",
                    "xref": image['stream'].objid,
                    "box": box,
                }
                image_metadata.append(metadata)
                image_counter += 1
            # page_array.append(page)

        return image_metadata
    
    # add figure based number incrementally
    def process_pdf_page(self, figure_path, page_metadata, image_folders):

        # Glycan detection
        # self.find_glycans(figure_path,image_folders)
        
        # do we need the below? since find_glycans() is already identifying each image
        for image_data in page_metadata:
            xref = image_data["xref"]
            img_name = f"{image_data['image_id']}"
            box = image_data["box"]
            x0, y0, x1, y1 = map(float, box)
            height = y1 - y0
            width = x1 - x0
            area = height * width

            self.log_file.write(
                f"\nImage ID: {img_name}, Coordinates: {box}, Width: {width}, Height: {height}, Area: {area}\n"
            )

            if height > 60 and width > 60 or area > 360:
                # output_path = os.path.join(self.workdir, "test", f"{xref}.png")
                images_path = os.path.join(image_folders['figures_dir'], f"{xref}.png")
                # fitx.Pixmap - uses this xref (as ID) to extract the pixel data for that image and save it.
                pix = fitz.Pixmap(fitz.open(self.input_filepath), xref)
                pix.save(images_path)

                self.log_file.write(f"\nSaved image to {images_path}")

                self.find_glycans(images_path,image_folders)

        
        
            

    # def process_pdf_page(self, page, image_metadata):
    #     """
    #     Process a single page of the PDF, extracting images and annotating glycans.
    #     """
    #     page_image_path = os.path.join(figures_path, f"page_{page_index}.png")
    #     page_pixmap = page.get_pixmap()
    #     page_pixmap.save(page_image_path)

    #     self.log_file.write(f"\n##### Page {page_index + 1} contains {len(page_images)} images.")

    #     for image_data in image_metadata:
    #         self.process_figure_from_page(page_index, image_data)


    # def process_figure_from_page(self, page_index, image_data):
    #     """
    #     Process a single image extracted from a page.
    #     Save the image and annotate glycans if criteria are met.
    #     """
    #     xref = image_data["xref"]
    #     img_name = f"{page_index}-{image_data['image_id']}"
    #     box = image_data["box"]
    #     x0, y0, x1, y1 = map(float, box)
    #     height = y1 - y0
    #     width = x1 - x0
    #     area = height * width

    #     self.log_file.write(
    #         f"\nImage ID: {img_name}, Coordinates: {box}, Width: {width}, Height: {height}, Area: {area}"
    #     )

    #     if height > 60 and width > 60 or area > 360:
    #         output_path = os.path.join(self.workdir, "test", f"{xref}.png")
    #         # fitx.Pixmap - uses this xref (as ID) to extract the pixel data for that image and save it.
    #         pix = fitz.Pixmap(fitz.open(self.input_filepath), xref)
    #         pix.save(output_path)

    #         self.log_file.write(f"\nSaved image to {output_path}")

    #         # Glycan detection
    #         self.find_glycans(output_path,page_num=page_index)


        


