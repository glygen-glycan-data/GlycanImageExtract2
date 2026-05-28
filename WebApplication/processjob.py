import fitz, sys, os, cv2,shutil, time, ntpath, json, base64, re, urllib.request
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from hashlib import md5
from APIFramework import APIFramework
from BKGlycanExtractor import ImageSearch
from BKGlycanExtractor import Config_Manager, BoundingBox, PDFBoundingBox, CompareBoxes
from BKGlycanExtractor import STANDARD_DPI, PDFHandler, PDFXRefImageFilter, PDFImageSizeFilter, PDFLargeImageSizeFilter
from BKGlycanExtractor import PDFCreator
from BKGlycanExtractor.glyomicsclient import GlyLookupClient, GlymageClient, SubsumptionClient
from BKGlycanExtractor import PMCData, PMCTarFile

import numpy as np
from shutil import copyfile
import copy
import tarfile
from io import BytesIO
import xml.etree.ElementTree as ET
from urllib.parse import urljoin

class MultiImageJob:
    '''
    Generic pipeline class for all types of jobs.
    '''

    pipelines_allowed = [
        'SingleGlycanImage-YOLOFinders',
        'MultipleGlycanImage-YOLOFinders',
    ]

    image_search_strategy_allowed = [
        'fitz',
        'figcap',
        'hybrid'
    ]
    
    def __init__(self, task_detail, config = {}, msg_queue = None):
        self.task_detail = task_detail
        self.id = task_detail.get('id')
        self.msg_queue = msg_queue
        self.config = config
        self.original_file_name = task_detail.get('filename')
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

        # other directories and path to stores downloaded files
        self.figures_dir = os.path.join(self.workdir, "extracted_figures", "figures")
        self.extracted_images_dir = os.path.join(self.workdir, "extracted_figures", "extracted_images")
        self.images_dir = os.path.join(self.workdir, "extracted_figures", "images")
        self.glymage_dir = os.path.join(self.workdir, "glymage")
        self.create_directories(*{self.figures_dir, self.images_dir, self.extracted_images_dir, self.glymage_dir})

        # Ensure necessary directories exist
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Open log file for writing
        self.log_file = open(self.log_file_path, 'w')

        self.job_finished = False
        self.results = []
        self.document_metadata = {}    

        # webservice clients
        self.glylookup_client = GlyLookupClient(apiurl=self.config.get('glylookup_url'),
                                    developer_email=self.config.get('dev_email'))

        self.glymage_client = GlymageClient(apiurl=self.config.get('glymage_url'),
                                    developer_email=self.config.get('dev_email'))

        self.gnome_client = SubsumptionClient(apiurl=self.config.get('subsumption_url'),
                                    developer_email=self.config.get('dev_email'))

    # generic/basic methods
    def create_directories(self,*paths):
        for path in paths:
            os.makedirs(path, exist_ok=True)

    def save_image(self,image, path):
        try:
            cv2.imwrite(path, image)
        except Exception as e:
            print(f"Error saving image at {path}: {e}")

    def abs_to_rel(self, abs_path=None):
        """
        Converts absolute path to relative path based on the base directory
        """
        # Debugging: Print the absolute path before conversion
        # print(f"Converting absolute path: {abs_path}")
        if abs_path is None:
            return None

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

    def set_document_metadata(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self.document_metadata[k] = v

    def check_and_create_paths(self,subdirs):
        """Ensure that all specified subdirectories exist under the work directory."""
        for subdir in subdirs:
            path = os.path.join(self.workdir, subdir)
            os.makedirs(path, exist_ok=True)
        return [os.path.join(self.workdir, subdir) for subdir in subdirs]

    def tar_filepath(self, pmid=None):
        if pmid is None:
            raise ValueError("PMID not provided")
        return os.path.join(self.base_dir, "input", self.id, f"PMID-{self.pmid}.tar.gz")

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

    def set_glycan_info(self, figure_semantics):
        
        glycans = list(figure_semantics.glycans())

        glylookup_seqs = []
        glymage_jobs = []
        
        for idx, glycan in enumerate(glycans):
            if not glycan.get('composition_str'):
                continue
            elif not glycan.has('IUPAC'):
                glycan.set('linkexpl', 'Extracted structure using Composition.')
                glymage_jobs.append((idx, 
                    self.glymage_client.submit_glymage(
                        seq=glycan.get('composition_str'), 
                        orientation=glycan.glycan_orientation()
                    )
                ))

                gnomeurl = self.gnome_client.get_gnome_url(
                    compositionstr=glycan.get('composition_str'))
                glycan.set('gnomeurl', gnomeurl)

            else:
                # iupac exists
                # build GlyLookup collection for batch - get accesson and wurcs from batch retrieve later
                glylookup_seqs.append((idx, glycan.get('IUPAC')))

        if glylookup_seqs:
            for i, result in self.glylookup_client.getmany([t[1] for t in glylookup_seqs]):
                glycan_idx = glylookup_seqs[i][0]
                glycan = glycans[glycan_idx]

                if result.get("accession"):
                    glycan.set('accession', result['accession'])

                    for sequence_type in result.get("sequences", []):
                        if sequence_type['format'] == 'WURCS':
                            glycan.set('WURCS', sequence_type['seq'])

                    # build gnome_url using accession
                    gnomeurl = self.gnome_client.get_gnome_url(acc=glycan.get('accession'))
                    glycan.set('gnomeurl', gnomeurl)

                    glycan.set('linkexpl', 'Extracted successfully using accession')
                    glymage_jobs.append((glycan_idx, 
                        self.glymage_client.submit_glymage(
                            acc=glycan.get('accession'), 
                            orientation=glycan.glycan_orientation(),
                        )
                    ))  
                else:
                    # submit iupac - for glymage and gnome
                    glycan.set('linkexpl', 'Extracted structure using IUPAC.')
                    glymage_jobs.append((glycan_idx, 
                        self.glymage_client.submit_glymage(
                            seq=glycan.get('IUPAC'), 
                            orientation=glycan.glycan_orientation()
                        )
                    ))

                    # if no accession - gnome_url should be created using iupac
                    try:
                        gnomeurl = self.gnome_client.get_gnome_url(seq=glycan.get('IUPAC'))
                        glycan.set('gnomeurl', gnomeurl)
                    except Exception as e:
                        sys.stderr.write(f"Warning: gnome subsumption failed for glycan {glycan_idx}: {e}\n")
                        self.log_file.write(f"Warning: gnome subsumption failed for glycan {glycan_idx}: {e}\n")
                        glycan.set('gnomeurl', '')

        # Retrieve Glymage, download and save the images to the correct folder
        for j, result in self.glymage_client.retrieve_many(*[t[1] for t in glymage_jobs]):
            glycan = glycans[glymage_jobs[j][0]]
            try:
                glyImage_path = result['result']
                rest, imgfilename = os.path.split(glyImage_path)

                glymage_image = os.path.join(self.glymage_dir, imgfilename)

                with open(glymage_image, 'wb') as wh:
                    # build http url glymage_path - so that it can be downloaded from the webservice
                    glyImage_url = urljoin(self.glymage_client.url(), glyImage_path.lstrip('/'))
                    with urllib.request.urlopen(glyImage_url, timeout=30) as h:
                        wh.write(h.read())
                    glycan.set('glyImage', glymage_image)

                    if not glycan.get('linkexpl'):
                        glycan.set('linkexpl', 'Extracted structure.')

            except (urllib.error.URLError, ValueError, OSError) as e:
                # Set to None or empty string, or skip setting it
                print(f"Failed glymage download for job {j}: {e}")
                glycan.set('glyImage', '')

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
        processor = task_detail.get('processor')
        if not processor:
            raise ValueError("processor is required")

        job_cls = globals()[task_detail["processor"]]
        return job_cls(task_detail, *args, **kwargs)

    def process_file(self):
        self.jobstate(False)
        self.update_state(APIFramework.RUNNING)

        input_file = os.path.join(self.base_dir, "input", self.id, self.original_file_name)

        try:
            copyfile(input_file, self.input_filepath)
        except FileNotFoundError:
            time.sleep(5)
            copyfile(input_file, self.input_filepath)

        self.log_file.write(f"{self.id}\n{self.output_filepath}\n")
        self.process_figures()
        self.jobstate(True)

    # pipeline methods
    def extract_figures(self) -> list[dict]:
        raise NotImplementedError

    def process_figures(self):
        # STEPS:
        # loop over figures
        # update status message - based on the metadata you have 
        # find glycans
        for figure_data in self.extract_figures():
            image_path = figure_data.pop("image_path", None) 
            if not image_path or not os.path.isfile(image_path):
                self.log_file.write("Warning: skipping figure with no image_path: %s\n" % (figure_data,))
                continue
            status_message = self._status_message(**figure_data)
            self.update_status(status_message)
            self.find_glycans(image_path, **figure_data)

    def _status_message(self, **kwargs):
        if kwargs.get("status_message"):
            return kwargs["status_message"]
        pmid_job = bool(kwargs.get("pmid_job"))
        page_num = kwargs.get("page_number") or 0
        figure_number = kwargs.get("figure_number")
        caption = (kwargs.get("caption") or "").strip()
        image_number = kwargs.get("image_number")
        has_fig_num = (
            figure_number is not None and str(figure_number).strip() != ""
        )
        
        if pmid_job:
            if has_fig_num:
                return f"Processing figure {figure_number}"
            if caption:
                return f"Processing {caption}"
            return "Processing figure"
        if page_num == 0:
            return "Processing image"
        if has_fig_num:
            return f"Processing figure {figure_number}"
        if image_number is not None and page_num != 0:
            return f"Processing image {image_number} from page {page_num}"
        return "Processing figure"

    def glycan_postprocessing(self,figure_semantics):
        """
        Process all glycans in a figure, saving images and metadata.
        """

        # Batch request - GlyImage and GlyLookup and gnome for each figure
        self.set_glycan_info(figure_semantics)

        basename = os.path.basename(figure_semantics.image_path()).split('.')[0]

        for i, gly_semantics in enumerate(figure_semantics.glycans()):
            glycan_image = gly_semantics.get('image')
            extracted_glycan_image = gly_semantics.get('extracted_image',glycan_image)

            if glycan_image is None or glycan_image.size == 0:
                print(f"Skipping glycan {i}: Detected object is missing or empty")
                continue

            image_name = f"{basename}-{i+1}.png"
            # save processed/cleaned extracted image
            image_url = os.path.join(self.images_dir, image_name)
            self.save_image(glycan_image,image_url)

            # save origial extracted imaged
            extracted_image_url = os.path.join(self.extracted_images_dir, image_name)
            self.save_image(extracted_glycan_image,extracted_image_url)

            gly_semantics.set('image_path', image_url)    
            gly_semantics.set('extracted_image_path',extracted_image_url)  
            gly_semantics.set('image_name', image_name)
            gly_semantics.set('fig_glycan_count', i+1)

    def progress_callback(self,**kwargs):
        if kwargs.get('stage') == "GLYCAN" and kwargs.get('checkpoint') == "DONE":
            nglycan = kwargs.get('nglycan')
            index = kwargs.get('index')
            if index is None or nglycan is None:
                return

            status_message = self._status_message(**kwargs)
            self.update_status(f"{status_message}, analyzed {index}/{nglycan} glycan(s)")

    accepted_pipeline_args = {
        "caption",
        "figure_number",
        "image_count",
        "page_number",
        "image_number",
        "fig_bbox",
        "pdf_fig_bbox",
        "pdf_fig_width",
        "pdf_fig_height",
        "page_width",
        "page_height",
        "pmid_job",
        "figure_name",
        "status_message",
    }

    def find_glycans(self, image_path, **kwargs):
        
        # pipeline_name is a static variable in each derived class
        if self.pipeline_name not in self.pipelines_allowed:
            allowed = ", ".join(sorted(self.pipelines_allowed))
            raise ValueError(f"The pipeline name {self.pipeline_name} is not valid. Allowed {allowed}")

        config = Config_Manager()
        
        pipeline_kwargs = {
            k: v for k, v in kwargs.items()
            if k in self.accepted_pipeline_args
        }

        pipeline = config.get_pipeline(self.pipeline_name)
        figure_semantics = pipeline.run(image_path, self.progress_callback, **pipeline_kwargs)

        # Sort bbox L->R for UI
        sorted_glycans = sorted(
            figure_semantics.glycans(),
            key=lambda g: tuple(g.box().bbox())
        )
        figure_semantics.set_glycans(sorted_glycans)

        nglycan = len(figure_semantics.glycans())

        if nglycan > 0:
            status_message = self._status_message(**kwargs)
            self.update_status(f"{status_message}, postprocessing {nglycan} glycan(s)")

        self.glycan_postprocessing(figure_semantics)
        self.annotate_image(figure_semantics)
        self.results.append(figure_semantics.tojson())

class SingleImageJob(MultiImageJob):
    '''
    Single Image - which has multiple glycans
    '''    
    pipeline_name = 'MultipleGlycanImage-YOLOFinders'

    def extract_figures(self) -> list[dict]:
        return [{'image_path': self.input_filepath, 'status_message': 'Processing image'}]

class SimpleImageJob(SingleImageJob):
    '''
    Single Image - which has a single/simple glycan
    '''
    pipeline_name = 'SingleGlycanImage-YOLOFinders'

class PMIDImageJob(MultiImageJob):

    pipeline_name = 'MultipleGlycanImage-YOLOFinders'

    def __init__(self, task_detail, config = {}, msg_queue = None):
        super().__init__(task_detail, config=config, msg_queue=msg_queue)
        self.pmid = task_detail.get('pmid')

    def extract_figures(self) -> list[dict]:
        '''
        gets PMC figures and metadata (captions, figure_number, citations)
        '''
        tar_filepath = self.tar_filepath(self.pmid)
        pmc_api = PMCTarFile(tar_filepath=tar_filepath)

        # set citation
        citation = PMCData.citation_details(self.pmid)
        citation_txt = ''
        if citation:
            citation_txt = citation.get('ascii_citation') or citation.get('citation', '')
        self.set_document_metadata(citation=citation_txt)

        return pmc_api.figures_metadata(self.figures_dir, input_dir=self.input_dir)

class PDFJob(MultiImageJob):
    pipeline_name = 'MultipleGlycanImage-YOLOFinders'

    def __init__(self, task_detail, config = {}, msg_queue = None):
        super().__init__(task_detail, config=config, msg_queue=msg_queue)
        self.image_search_strategy = 'hybrid'   # default, but should be able to update this

        # reanalyze on a synthetic pdf will use 'fitz' (set by form_task) - so the above image_search_startegy will have to be updated
        strategy = task_detail.get('image_search_strategy')
        if strategy:
            if strategy in self.image_search_strategy_allowed:
                self.image_search_strategy = strategy
            else:
                allowed = ", ".join(sorted(self.image_search_strategy_allowed))
                raise ValueError(f"Image search strategy {strategy!r} is not valid. Allowed {allowed}")
       
    def extract_figures(self) -> list[dict]:
        strategy = ImageSearch.search_method(self.image_search_strategy)

        metadata = strategy.get_metadata(self.input_filepath, self.figures_dir)
        
        if self.task_detail.get('pmid'):
            citation = PMCData.citation_details(self.task_detail['pmid'])
            if citation:
                self.set_document_metadata(citation=citation['citation'],
                                           pmid=self.task_detail['pmid']) 
        else:
            handler = PDFHandler(self.input_filepath)
            citation = handler.get_citation()
            if citation:
                self.set_document_metadata(citation=citation['citation'],
                                           pmid=citation['pmid'])
        return metadata

class PMIDSyntheticPDFJob(PDFJob):
    pipeline_name = 'MultipleGlycanImage-YOLOFinders'

    def __init__(self, task_detail, config = {}, msg_queue = None):
        super().__init__(task_detail, config=config, msg_queue=msg_queue)
        self.pmid = task_detail.get('pmid')
        self.image_search_strategy = 'fitz'     # always fitz by default - no one should be able to change it

    def extract_figures(self) -> list[dict]:
        tar_filepath = self.tar_filepath(self.pmid)
        pmc_api = PMCTarFile(tar_filepath=tar_filepath)

        pmc_figures = list(pmc_api.figures_metadata(self.figures_dir, input_dir=self.input_dir))

        if not pmc_figures:
            return

        # create PDF - write images and captions, citations to the pdf
        pdfwriter = PDFCreator(self.pmid)
        for i, fig_data in enumerate(pmc_figures, 1):
            pdfwriter.add_image(fig_data['image_path'], f"Figure {i}. {fig_data.get('caption', '')}")
        pdfwriter.write(self.input_filepath)

        citation = pdfwriter.citation.get('ascii_citation') or pdfwriter.citation.get('citation')
        self.set_document_metadata(citation=citation)

        # image_paths are required for glycan pipleine analysis on image
        # synthetic pdfs have images, but we also have the individual images (used during pdf_creator step)
        # so provide the image_paths
        image_path_dict = {f["image_count"]: f["image_path"] for f in pmc_figures}

        strategy = ImageSearch.search_method(self.image_search_strategy)
        metadata = strategy.get_metadata(self.input_filepath, self.figures_dir, image_path_dict=image_path_dict)

        return metadata

