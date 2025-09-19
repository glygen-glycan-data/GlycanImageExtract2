'''
This 
'''
import fitz, sys, os, cv2,shutil, pdfplumber, time, ntpath, json, base64, re
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
        'Single-Glycan Image': 'SingleGlycanImage-YOLOFinders',
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
        print(f"Converting absolute path: {abs_path}")
        
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


    # def jobstate(self,state=False):
    #     self.job_finished = state
        
    #     if state:
    #         data = [json.loads(s) for s in self.results]

    #         with open(self.json_filepath, 'w') as f:
    #             json.dump(data,f,indent=2)
    #         self.log_file.close()
    #         print("-------->>>JOB COMPLETED", state)
    #     return True

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

            # image_folders = {'figures_path': figures_path, 'images_path': images_path, 'processed_images_path': processed_images_path}

            # Create directories and save image
            # glycan_dir = os.path.join(self.workdir, "test", f"{basename}-{i+1}")
            # self.create_directories(glycan_dir)

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

            # gly_semantics.set("page_num",page_num)

    def progress_callback(self,**kwargs):
        if kwargs.get('stage') == "GLYCAN" and kwargs.get('checkpoint') == "DONE":
            nglycan = kwargs.get('nglycan')
            index = kwargs.get('index')
            if self.pageno == 0:
                self.update_status("Processing image, analyzed %d/%d glycan(s)"%(index,nglycan))
            else:
                self.update_status("Processing image %d from page %d, analyzed %d/%d glycan(s)"%(self.imageno,self.pageno,index,nglycan))

    def find_glycans(self, figure_path, image_folders, **kwargs):

        # Run the pipeline
        config = Config_Manager()
        # print("submission_type",self.submission_type)
        self.pipeline_name = self.pipeline_mapping[self.submission_type]
        pipeline = config.get_pipeline(self.pipeline_name)
        # *****
        # print("BEFORE",kwargs)
        figure_semantics = pipeline.run(figure_path,self.progress_callback, **kwargs)


        # by defauly glycans are sorted by confidence - but we need to sort them
        # based on bbox so that they can be numbered sequentially from left to right
        # purely for webpage present purpose
        # sort glycans by bbox (left-to-right)
        sorted_glycans = sorted(
            figure_semantics.glycans(),
            key=lambda g: tuple(g.box().bbox())
        )
        figure_semantics.set_glycans(sorted_glycans)

        nglycan = len(figure_semantics.glycans())
        if self.pageno == 0:
            self.update_status("Processing image, postprocessing %d glycan(s)"%(nglycan))
        else:
            self.update_status("Processing image %d from page %d, postprocessing %d glycan(s)"%(self.imageno,self.pageno,nglycan))

        # Annotate and save images
        self.annotate_image(figure_semantics)

        # Other glycan info
        self.process_glycans(figure_semantics,image_folders)
       
        self.results.append(figure_semantics.tojson())


class ImageJob(JobInstance):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

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
        self.imageno = 1
        self.pageno = 0

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
    # def __init__(self, task_detail):
    #     super().__init__(task_detail)

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

        images_metadata = self.extract_images_from_pdf()
        self.log_file.write(f"\nFound {len(images_metadata)} figures in the PDF.")

        self.figure_count = 0
        doc = fitz.open(self.input_filepath)
        for page_index, page in enumerate(doc.pages()):
            page_pix = doc.load_page(page_index).get_pixmap()
            figure_path = os.path.join(image_folders['figures_dir'], f"{page_index}.png")
            # page_pix.save(figure_path)

            # maybe we dont need the below two lines (and also dont need images_metadata) 
            # and just need find_glycans() here with figure path?
            # self.find_glycans(figure_path,image_folders)
            page_metadata = [data for data in images_metadata if data['image_page'] == page_index]
            self.pageno = (page_index)
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
                # box = (image['x0'], page_height - image['y1'], image['x1'], page_height - image['y0'])
                box = BoundingBox(x1=image['x0'], y1=page_height - image['y1'], x2=image['x1'], y2=page_height - image['y0'])
                metadata = {
                    "image_page": image['page_number']-1,
                    "image_id": f"id_{image_counter}",
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

        # continue with figure number 
        
        # do we need the below? since find_glycans() is already identifying each image
        for image_index,figure_data in enumerate(page_metadata):
            xref = figure_data["xref"]
            img_name = f"{figure_data['image_id']}"
            box = figure_data["box"]
            x0, y0, x1, y1 = box.corners()
            height = y1 - y0
            width = x1 - x0
            area = height * width

            # figure_data['height'] = height
            # figure_data['width'] = width

            self.log_file.write(
                f"\nImage ID: {img_name}, Coordinates: {box}, Width: {width}, Height: {height}, Area: {area}\n"
            )

            if height > 60 and width > 60 or area > 360:
                figure_data['figure_count'] = self.figure_count
                # output_path = os.path.join(self.workdir, "test", f"{xref}.png")
                images_path = os.path.join(image_folders['figures_dir'], f"{xref}.png")
                # fitx.Pixmap - uses this xref (as ID) to extract the pixel data for that image and save it.
                pix = fitz.Pixmap(fitz.open(self.input_filepath), xref)
                goodsave = False
                try:
                    pix.save(images_path)
                    goodsave = True
                except ValueError:
                    pass
                if not goodsave:
                    pix = fitz.Pixmap(fitz.csRGB,pix)
                    pix.save(images_path)

                self.log_file.write(f"\nSaved image to {images_path}")
                
                self.imageno = (image_index+1)
                self.update_status("Processing image %d from page %d"%(self.imageno,self.pageno))

                self.find_glycans(images_path,image_folders, **figure_data)

            self.figure_count += 1



