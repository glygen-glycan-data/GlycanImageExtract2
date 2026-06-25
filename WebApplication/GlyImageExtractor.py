#!/bin/env python3.12
import sys
import os
import argparse
import base64
import shutil
import copy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path to import BKGlycanExtractor
sys.path.append(parent_dir)


from APIFramework import APIFramework, APIErrorBase, APIParameterError, APIDataError
from processjob import MultiImageJob
from BKGlycanExtractor import PMCData, PMCTarFile, PMCFiles
from BKGlycanExtractor import annotate_from_webapp

from shutil import copyfile

import os
import sys
import time
import random
import hashlib
import multiprocessing
import secrets
import flask
import json
import cv2
import traceback
import threading
import subprocess

class GlyImageExtractor(APIFramework):

    default_render_kwargs = dict(
        google_analytics_url_match = "extractor.glyomics.org",
        google_analytics_id = "G-47WSZ1WYRZ"
    )

    def __init__(self):
        super().__init__()

        self._resultid_locks = {}
        self._lock = threading.Lock()
        config = self._worker_config

        for k,v in self.default_render_kwargs.items():
            self.set_template_render_kwarg(**{k: config.get(k,v)})
        
        # default figure search type for PDF files...
        self._image_search_type = config.get('image_search_type')
        assert self._image_search_type in (None,"fitz", "hybrid", "figcap")

    def process(self):
        submission_type = flask.request.args.get("type")

        if submission_type == 'Manuscript':
            placeholder_url = 'https://example.com/document.pdf'
        else:
            placeholder_url = 'https://example.com/image.png'

        kwargs = dict(**self._template_render_kwargs)
        return flask.render_template(self._process_html, submission_type=submission_type, placeholder_url=placeholder_url, **kwargs)

    def examples(self):
        # Mcleod - https://www.neb.com/en-us/-/media/nebus/files/application-notes/appnote_characterization_of_glycans_from_erbitux_rituxan_and_enbrel_using_recombinant_pngase_f.pdf?rev=581a874aebbc4351bec05e10c07f96ea&hash=C8D5EB5AF1B7D5C331DFAB13BB87F649
        example_cards = [
            {"title": "Sassi et al., 2014", "desc": "", "url": f"{self._prefix}/result/mgp1", "icon": f"{self._prefix}/static/images/pdf.svg" },
            {"title": "Huang & Orlando, 2017", "desc": "", "url": f"{self._prefix}/result/mgp4", "icon": f"{self._prefix}/static/images/pdf.svg" },
            {"title": "Mcleod, 2024", "desc": "", "url": f"{self._prefix}/result/mgp3", "icon": f"{self._prefix}/static/images/pdf.svg"},
            {"title": "Figure 1, Kri\u0161ti\u0107 et al., 2018", "desc": "", "url": f"{self._prefix}/result/mgi4", "icon": f"{self._prefix}/static/images/multi-image.svg" },
            {"title": "Figure 2, Zhang et al., 2021", "desc": "", "url": f"{self._prefix}/result/mgi6", "icon": f"{self._prefix}/static/images/multi-image.svg" },
            {"title": "Mass Spectrometry of Glycans Webpage, Millipore Sigma", "desc": "", "url": f"{self._prefix}/result/mgi5", "icon": f"{self._prefix}/static/images/multi-image.svg" },
            {"title": "G16150CJ - Compact N-Glycan", "desc": "", "url": f"{self._prefix}/result/sgi4", "icon": f"{self._prefix}/static/images/single-image.svg"},
            {"title": "G83439SR - N-Glycan Toplogy ", "desc": "", "url": f"{self._prefix}/result/sgi6", "icon": f"{self._prefix}/static/images/single-image.svg" },
            {"title": "G69233PF - O-Glycan Fully-defined", "desc": "", "url": f"{self._prefix}/result/sgi5", "icon": f"{self._prefix}/static/images/single-image.svg" },
        ]
        kwargs = dict(**self._template_render_kwargs)
        return flask.render_template(self._examples_html, example_cards=example_cards, **kwargs)

    def get_result(self,resultid):
        
        if resultid in self.result_cache:
            return self.result_cache[resultid]

        path_files = f"static/files/{resultid}/results.json"
        if os.path.exists(path_files):
            with open(path_files, "r") as f:
                return json.load(f)

        path_examples = f"static/examples/{resultid}/results.json"
        if os.path.exists(path_examples):
            with open(path_examples, "r") as f:
                result = json.load(f)
            result["location"] = "examples"
            return result

        return {"Error": "resultid (%s) not found" % (resultid,)}

    task_params = [
        "filename",    
        "submission_type",   
        "submission_mode",           # derived   
        "pmid",
        "image_search_strategy",     # derived
        "processor",                 # derived
    ]

    def form_task(self, p: dict):
        '''
        User submitted data, derived data - should be defined here.

        Note - if submission_mode is already provided i.e
        it is 'Local' submission_mode, so based on other paramaters, decide
        the processor - because we already have the necessary input files downloaded - 
        so file download steps or pmid validation steps can be skipped - and you directly move on running the job.
        '''

        # additional/derieved keys have an underscore in front (so they can be stripped out before
        # the final JSON is written)

        if not p.get("submission_type"):
            raise APIParameterError("submission_type is required")
        submission_type = p['submission_type']

        pmid = p.get('pmid', None)
        if pmid:
            pmid = pmid.strip()

        # submission_mode
        submission_mode = None
        if p.get('filePath'):
            submission_mode = 'Local'
        elif p.get('file'):
            submission_mode = 'Upload'
        elif p.get('fileURL'):
            submission_mode = 'URL'
        elif submission_type == 'Manuscript' and pmid:
            if pmid.endswith('.pdf'):
                pmid = pmid.split('.', 1)[0]
                submission_mode = 'PMID.PDF'
            else:
                submission_mode = 'PMID'
        
        if not submission_mode:
            raise APIParameterError("No input file: Upload a file/image, paste a URL or submit PMID")


        # This block uses both submission_mode and submission_type to determine job processor
        processor = None
        if submission_type == "Simple Glycan Image":
            # processor = 'SimpleImageJob'
            processor = 'SimpleImageSyntheticPDFJob'
        elif submission_type == "Multi-Glycan Image":
            # processor = 'SingleImageJob'
            processor = 'SingleImageSyntheticPDFJob'
        elif submission_mode == 'PMID':
            processor = 'PMIDSyntheticPDFJob'
        elif submission_mode == 'PMID.PDF':
            processor = 'PDFJob'
        else:
            processor = 'PDFJob' 

        image_search_strategy = p.get('image_search_strategy',self._image_search_type)
                  
        params_dict = dict(p)
        params_dict['submission_mode'] = submission_mode
        params_dict['processor'] = processor
        if image_search_strategy:
            params_dict['image_search_strategy'] = image_search_strategy
        if pmid:
            params_dict['pmid'] =  pmid

        task = {}
        for k in self.task_params:
            v = params_dict.get(k)
            if v is None or v == '':
                continue
            task[k] = v

        # provide all parameter values in a predictable list (deterministic) 
        # incase we want to make the id's reproducible 
        task["id"] = self.makeid(*(task.get(k) for k in self.task_params),
                                 random=True,length=10)
        
        return task

    @staticmethod
    def worker(pid, task_queue, result_queue, params):

        while True:
            task_detail = task_queue.get(block=True)

            calculation_start_time = time.time()
            result = None
            document_metadata = None
            error = []

            token = task_detail["id"]

            PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
            workdir = os.path.join(PROJECT_ROOT, "static", "files", token)

            # Factory method - get_processor() 
            job_instance = None
            try:
                job_instance = MultiImageJob.get_processor(task_detail, config=params, msg_queue=result_queue)
                job_instance.process_file()
                result = job_instance.get_results()
                document_metadata = job_instance.get_document_metadata()
            except:
                traceback.print_exc()
                error.append(traceback.format_exc())

            calculation_end_time = time.time()
            calculation_time_cost = calculation_end_time - calculation_start_time

            status = ""
            if len(error) == 0:
                state = APIFramework.COMPLETE
            else:
                state = APIFramework.ERROR
                if "AttributeError: 'NoneType' object has no attribute 'shape'" in error[-1]:
                    status = "File could not be interpreted as an image."
                elif "pdfminer.pdfparser.PDFSyntaxError: No /Root object! - Is this really a PDF?" in error[-1]:
                    status = "File could not be interpreted as a PDF."

            # information for webservice (API Framework)
            res = {
                "id": token,
                "start_time": calculation_start_time,
                "end_time": calculation_end_time,
                "runtime": calculation_time_cost,
                "error": error,
                "figures": result,
                "finished": True,
                "state": state,
                "status": status,
            }

            if document_metadata:
                res.update(document_metadata)

            result_queue.put(res)

    def lock_result(self,resultid,timeout=2):
        if not self._lock.acquire(timeout=timeout):
            print("Status: ERROR:LOCK_TIMEOUT, ResultID: %s"%(resultid,),file=sys.stderr)
            return False
        if resultid not in self._resultid_locks:
            self._resultid_locks[resultid] = threading.Lock()
        self._lock.release()
        return self._resultid_locks[resultid].acquire(timeout=2)
    
    def release_result(self,resultid):
        if resultid in self._resultid_locks:
            self._resultid_locks[resultid].release()

    def save_result(self,resultid,result):
        # a generic method for save_result can exist in the base class,
        # but not sure if the folder sturcture to store files will be respected for all the different applications
        # so keeping this method here for now

        location = result.get("location") or "files"
        path = os.path.join("static", location, resultid, "results.json")
        tmp_path = path + ".tmp"

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as wh:
                json.dump(result, wh, indent=2)
            
            os.replace(tmp_path, path)
            self.result_cache[resultid] = result
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise APIErrorBase("save_result failed for %s: %s" % (resultid, e)) from e
        # finally:
        #     self.release_result(resultid) 
        
    def _input_filename(self, params: dict):
        '''
        Overrides the base class method - to support PMID based file naming
        '''
        filename = None

        if not params.get("submission_type"):
            raise APIParameterError("submission_type is required")
        submission_type = params['submission_type']

        pmid = params.get('pmid', None)
        if pmid:
            pmid = pmid.strip()

        # A manuscript submission - can be a file upload, fileURL, or PMID
        if submission_type == "Manuscript" and pmid:
            if pmid.endswith(".pdf"):
                pmid = pmid.rsplit(".", 1)[0]
            return "PMID-" + pmid + ".pdf"

        return super()._input_filename(params)

    # overrides base class method
    def prepare_job_input(self, params: dict, task_detail: dict, current_file_path: str):
        submission_mode = task_detail["submission_mode"]

        if submission_mode == "Local":
            src = params.get("filePath")
            if not src or not os.path.isdir(os.path.dirname(src)):
                raise APIErrorBase("Local resubmit: invalid filePath")

            src_dir = os.path.dirname(os.path.abspath(src))
            dst_dir = os.path.dirname(current_file_path)  # .../new_id/input
            os.makedirs(dst_dir, exist_ok=True)

            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            return
      
        elif submission_mode in ("PMID","PMID.PDF"):
            pmid = task_detail['pmid']
            filename = task_detail['filename']
            file_dir = os.path.dirname(current_file_path)

            # validates PMID and download the necessary files needed in the provided file_dir
            # response, status = PMCTarFile.download_and_prepare_pmid_data(pmid, file_dir, filename)
            response, status = PMCFiles.download_and_prepare_pmid_data(pmid, file_dir, filename)

            if status != 200:
                raise APIErrorBase(response.get("error") or "PMID download failed")

            return 

        return super().prepare_job_input(params, task_detail, current_file_path)

    def _get_base_url_from_request(self):
        base_url = None
        if flask.has_request_context():
            referer = flask.request.headers.get('Referer')
            # print("Referer:",referer)
            if referer:
                if '/result/' in referer:
                    base_url = referer.split('/result/')[0]
                elif referer.endswith('/jobs'):
                    base_url = referer.split('/jobs')[0]
        return base_url

    def _update_annotation_result_paths(self, resultid, result, annotated_pdf_file, annotated_tsv_file):
        # make sure you add the annoated file keys to the json result - so that this is added to the json file on disk
        result['annotated_pdf_file'] = annotated_pdf_file
        result['annotated_tsv_file'] = annotated_tsv_file
        # make sure to add annotated file keys to the in-memory cache (result_cache) - so
        # that the frontend has access to these keys via API payloads
        cache_entry = self.result_cache.get(resultid)
        if cache_entry is not None:
            if cache_entry.get('result') is None:
                cache_entry['result'] = {}
            cache_entry['result']['annotated_pdf_file'] = annotated_pdf_file
            cache_entry['result']['annotated_tsv_file'] = annotated_tsv_file
    
    def annotate_results(self, resultid=None):

        if not resultid:
            return flask.jsonify(dict(error="ERROR: resultid not provided", valid=False)), 400

        location = "files"

        if flask.request.is_json:
            location = (flask.request.get_json(silent=True) or {}).get('location') or "files"
        elif flask.request.args.get('location'):
            location = flask.request.args.get('location')

        json_file = self.abspath(f"static/{location}/{resultid}/results.json")

        if not os.path.exists(json_file):
            print(f"No results found for job {resultid}", file=sys.stderr)
            return flask.jsonify(dict(error=f"No results found for job {resultid}.", valid=False)), 404

        locked = False
        try:
            locked = self.lock_result(resultid, timeout=10)
            if not locked:
                print(f"timeout occured during result annotations: {resultid}, try again in a few seconds", file=sys.stderr)
                return flask.jsonify(dict(error=f"timeout occured during result annotations: {resultid}, try again in a few seconds"), valid=False), 400

            with open(json_file, 'r') as f:
                json_data = json.load(f)

            base_dir = self.abspath(f"static/{location}/{resultid}/")

            result = json_data.get('result', {})
            if result is None:
                result = {}
            # ensure changes to `result` always persist in json_data
            json_data['result'] = result

            subdetails = json_data.get('submission_detail', {})
            pdf_path = os.path.join(base_dir, "input", subdetails.get('filename', ""))
            # print(pdf_path)
            if not pdf_path or not os.path.isfile(pdf_path):
                print(f"Input file not found for job {resultid}.", file=sys.stderr)
                return flask.jsonify(dict(error=f"Input file not found for job {resultid}.", valid=False)), 404

            output_dir = os.path.join(base_dir, "annotated_files")
            os.makedirs(output_dir, exist_ok=True)

            pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
            annotated_pdf = os.path.join(output_dir, pdf_basename + ".annotated.pdf")
            annotated_pdf_file = pdf_basename + ".annotated.pdf"
            annotated_tsv_file = pdf_basename + ".annotated.tsv"

            # print("Expecting annotated PDF at: %s" % (annotated_pdf,), file=sys.stderr)

            # Check if the annotated results (pdf and tsv) are up to date with the json file?
            # i.e if the annotated results were modified at the time after the json was modified - then
            # no need to annotate files again -> serve the results directly to the user
            # getmtime --> helps with getting the last modified time of a file
            if os.path.exists(annotated_pdf):
                json_mtime = os.path.getmtime(json_file)
                pdf_mtime = os.path.getmtime(annotated_pdf)
                if pdf_mtime >= json_mtime:
                    # self._update_annotation_result_paths(
                    #     resultid=resultid,
                    #     result=result,
                    #     annotated_pdf_file=annotated_pdf_file,
                    #     annotated_tsv_file=annotated_tsv_file
                    # )
                    # with open(json_file, 'w') as f:
                    #     json.dump(json_data, f, indent=2)

                    print("Status: OK (annotated PDF up-to-date), ResultID: %s." % (resultid,), file=sys.stderr)
                    return flask.jsonify(dict(status="OK", resultid=resultid, valid=True)), 200

            base_url = self._get_base_url_from_request()

            if not base_url:
                base_url = f"http://{self.host()}:{self.port()}"

            # Regenerate annotated results
            isnewannotatedpdf = (annotated_pdf and not os.path.exists(annotated_pdf))

            print("Building annotated PDF/TSV for ResultID: %s." % (resultid,), file=sys.stderr)
            annotate_from_webapp(json_file, pdf_path, base_url, output_dir=output_dir)

            # after the results are ready and wrriten to the file system (via annotate_from_webapp), using a small delay to ensure
            # everything is set.
            time.sleep(0.2)

            # Verify that the annotated_pdf exists
            if annotated_pdf and os.path.exists(annotated_pdf):
                # make sure to add annotated file keys to the in-memory cache (result_cache) - so
                # that the front end has access to these keys via the API payload
                if isnewannotatedpdf: # if not, nothing to update...
                    self._update_annotation_result_paths(
                        resultid=resultid,
                        result=result,
                        annotated_pdf_file=annotated_pdf_file,
                        annotated_tsv_file=annotated_tsv_file
                    )
                    # Write back to same file
                    with open(json_file, 'w') as f:
                        json.dump(json_data, f, indent=2)
                
                    # Touch the annotated_pdf file so that we don't recreate it next time!
                    os.utime(annotated_pdf, None)

                print("Status: OK, ResultID: %s." % (resultid,), file=sys.stderr)
                return flask.jsonify(dict(status="OK", resultid=resultid, valid=True)), 200

            print("Status: Annotated PDF was not created", file=sys.stderr)
            return flask.jsonify(dict(error="Annotated PDF was not created.", valid=False)), 500

        except json.JSONDecodeError:
            print(f"Status: Results file for job {resultid} is corrupted.", file=sys.stderr)
            return flask.jsonify(dict(error=f"Status: Results file for job {resultid} is corrupted.", valid=False)), 500

        except Exception as e:
            traceback.print_exc()
            print(f"Status: Annotation failed due to an unexpected error: {e}", file=sys.stderr)
            return flask.jsonify(dict(errpr=f"Status: Annotation failed due to an unexpected error: {e}", valid=False)), 500

        finally:
            if locked:
                self.release_result(resultid)

    def on_task_finished(self, resid):
        result = self.result_cache[resid]
        submission_detail = result['submission_detail']
        # remove all the extra derived paramters from submission_detail before writing the
        # json file to disk
        for key in list(submission_detail.keys()):
            if key.startswith('_'):
                submission_detail.pop(key)

        submission_type = submission_detail.get("submission_type")
        submission_mode = submission_detail.get("submission_mode")
        processor = submission_detail.get("processor")

        # write the json file to disk
        abs_json_path = self.abspath(os.path.join("static/files/"+resid, "results.json"))
        with open(abs_json_path, 'w') as f:
            json.dump(self.result_cache[resid],f,indent=2)

        # make the method flexible to use result_cache is json is not avaibale??
        # Note: currently annotate_results uses the json dict written on disk for details
        # if you want to use this feature before the json files are written, the the result_cache will have to be accessed
        if processor in ("PDFJob","PMIDSyntheticPDFJob", "SimpleImageSyntheticPDFJob", "SingleImageSyntheticPDFJob"):
            self.annotate_results(resultid=resid)

    document_metadata_keys = [
        # '_citation'
    ]

    def _build_resubmit_task(self, submission_detail, result, tid, input_file):
        '''
        Override base method - if you need to add some extra parameters that will get passed
        to the Job Processor class.

        Since these paramters are extras/derived - they will have an underscore attached, 
        so that they can be identified and cleaned up from submission detail
        '''

        p = super()._build_resubmit_task(
            submission_detail, result, tid, input_file
        )

        if p.get("pmid"):
            if p.get("submission_mode") == "PMID.PDF":
                p["pmid"] += ".pdf"
            if 'filePath' in p:
                del p['filePath']

        for key in self.document_metadata_keys:
            cleaned_key = key.lstrip('_')
            if cleaned_key in result and result[cleaned_key] is not None:
                p[key] = result[cleaned_key]

        return p

    def validate_pmid(self, pmid):
        # wrapper used so that proper json responses are created after the helper returns a response.
        # TODO - Ticket for better design PMCData and PMCTarFile classes
        pmid = pmid.strip()
        if pmid.endswith('.pdf'):
            pmid = pmid.rsplit('.',1)[0]
        body, status = PMCData.validate_pmid(pmid)
        return flask.jsonify(body), status

    def mark(self):
        # when votes are updated - the annotated pdf and tsv file will also be updated accordingly
        resultid = flask.request.args['resultid']
        glycanid = flask.request.args['glycanid']
        note = flask.request.args['note']

        try:
            if self.lock_result(resultid,timeout=2):
    
                res = self.get_result(resultid)
                if res.get('location') == 'examples':
                    self.release_result(resultid)
                    print("Status: ERROR:EXAMPLE, ResultID: %s, GlycanID: %s, Note: %s."%(resultid,glycanid,note),file=sys.stderr)
                    return flask.jsonify(dict(status="ERROR"))
    
                figureindex,glycanindex=map(int,glycanid.split('.'))
                glycan = res['result']['figures'][figureindex]['glycans'][glycanindex]
                votes = glycan.get('upvotes',0) - glycan.get('downvotes',0)
                if note == "upvote":
                    votes += 1
                elif note == "downvote":
                    votes -= 1
                else:
                    glycan['note'] = note
                if votes >= 0:
                    glycan['upvotes'] = votes
                    glycan['downvotes'] = 0
                else:
                    glycan['upvotes'] = 0
                    glycan['downvotes'] = -votes
                self.save_result(resultid,res)

            else:
                print("Status: ERROR:RESULT_TIMEOUT, ResultID: %s, GlycanID: %s, Note: %s."%(resultid,glycanid,note),file=sys.stderr)
                return flask.jsonify(dict(status="ERROR"))

        except:
            self.release_result(resultid)
            traceback.print_exc()
            print("Status: ERROR, ResultID: %s, GlycanID: %s, Note: %s."%(resultid,glycanid,note),file=sys.stderr)
            return flask.jsonify(dict(status="ERROR"))
        
        self.release_result(resultid)
        print("Status: OK, ResultID: %s, GlycanID: %s, UpVotes: %s, DownVotes: %s, Note: %s."%(resultid,glycanid,glycan.get('upvotes',0),glycan.get('downvotes',0),glycan.get('note',"")),file=sys.stderr)
        return flask.jsonify(dict(status="OK",resultid=resultid,glycanid=glycanid,upvotes=glycan.get('upvotes',0),downvotes=glycan.get('downvotes',0),note=glycan.get('note',"")))

    def load_route(self):
        super().load_route()

        self._flask_app.add_url_rule("/mark", "mark", self.mark, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/annotate_results/<resultid>", "annotate_results", self.annotate_results, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/pmid/<pmid>", "validate_pmid", self.validate_pmid, methods=["GET", "POST"])

if __name__ == '__main__':
    multiprocessing.freeze_support()

    extractor = GlyImageExtractor()
    extractor.start()
 