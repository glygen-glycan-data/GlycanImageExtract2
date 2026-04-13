#!../.venv/bin/python

import sys
import os
import argparse
import base64


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path to import BKGlycanExtractor
sys.path.append(parent_dir)


from APIFramework import APIFramework
from processjob import JobInstance

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
        self._image_search_type = config.get('image_search_type','fitz')
        assert self._image_search_type in ("fitz","hybrid","figcap")

    task_params = ["filename",
                   "submission_type",
                   "submission_mode",
                   "fileURL",
                   "pmid",
                   "image_search_strategy",
                   ]

    def form_task(self, params):
        # set  default values, if appropriate
        task = {}

        submission_type = params['submission_type']
        submission_mode = params['submission_mode']
        has_pmid = bool(params.get('pmid'))

        # you need image_search_strategy only for PDF/PMID-PDF based jobs
        if submission_type not in ("Simple Glycan Image", "Multi-Glycan Image"):
            # if submission mode if local and for instance if analysis was done on a synethic pdf ealier
            # then its good to do re-analysis over the same pdf using fitz, so for reanalyze optinally pass the image_search_strategy as well
            if params.get('image_search_strategy'):
                task['image_search_strategy'] = params['image_search_strategy']
            if submission_mode == 'PMID-PDF':
                task['image_search_strategy'] = 'fitz'
            elif submission_mode != 'PMID' and not (submission_mode == 'Local' and has_pmid):   # use the image_search_strategy from either config file (if present) or defaults to 'fitz
                task['image_search_strategy'] = self._image_search_type

        # get these parameters from the form
        for k in self.task_params:
            if params.get(k):
                task[k] = params[k]

        # provide all parameter values in a predictable list to make a 
        # reproducible id, if desired
        task["id"]=self.makeid(*(task.get(k) for k in self.task_params),
                               random=True,length=10)
        return task

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

    @staticmethod
    def worker(pid, task_queue, result_queue, params):
        # print(pid, "Start")

        while True:
            task_detail = task_queue.get(block=True)

            calculation_start_time = time.time()
            result = None
            error = []

            token = task_detail["id"]

            # setting absolute paths - useful for docker
            PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
            workdir = os.path.join(PROJECT_ROOT, "static", "files", token)
            # workdir = os.path.join("./static/files", token)
            # os.makedirs(os.path.join(workdir, "input"), exist_ok=True)
            # os.makedirs(os.path.join(workdir, "output"), exist_ok=True)

            # Factory method - get_processor() 
            document_metadata = None
            try:
                job_instance = JobInstance.get_processor(task_detail, config=params, msg_queue=result_queue)
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
            updated_task_detail = job_instance.task_detail
            res = {
                "id": token,
                "start_time": calculation_start_time,
                "end_time": calculation_end_time,
                "runtime": calculation_time_cost,
                "error": error,
                # "filepath": updated_task_detail['filepath'],
                # "abs_original_filepath": updated_task_detail['abs_original_filepath'],
                "figures": result,
                "job_type": job_instance.__class__.__name__,
                "finished": True,
                "state": state,
                "status": status,
            }

            if document_metadata:
                res.update(**document_metadata)

            result_queue.put(res)


    # def home(self):
    #     return flask.render_template(self._home_html, urlprefix=self._prefix)

    # def examples(self):
    #     return flask.render_template(self._examples_html, basedir="static/examples")

    # def abstract(self):
    #     return flask.render_template(self._abstract_html)

    def result(self,id=None):
        if not id:
            id = flask.request.args['id']
        return flask.render_template(self._result_html, list_id=id, **self._template_render_kwargs)

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
    
if __name__ == '__main__':
    multiprocessing.freeze_support()

    extractor = GlyImageExtractor()
    extractor.start()










