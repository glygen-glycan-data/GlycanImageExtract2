#!../.venv/bin/python

import sys
import os
import argparse
import base64


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path to import BKGlycanExtractor
sys.path.append(parent_dir)


from APIFramework import APIFramework
# from BKGlycanExtractor import JobInstance
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

class ReferenceAPIParaBased(APIFramework):
    pass

import subprocess
class ReferenceAPIFileBased(APIFramework):

    def __init__(self):
        super().__init__()

        # self.pipeline_name = pipeline_name

    def form_task(self, p):
        res = {}

        # Prevent name collision
        res["original_file_name"] = p["original_file_name"]
        res['submission_type'] = p['submission_type']
        res["id"] = self.makeid(p["original_file_name"],p["submission_type"],random=True,length=10)

        return res


    @staticmethod
    def worker(pid, task_queue, result_queue, params):
        # print(pid, "Start")

        while True:
            task_detail = task_queue.get(block=True)

            calculation_start_time = time.time()
            error = []

            token = task_detail["id"]

            # setting absolute paths - useful for docker
            PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
            workdir = os.path.join(PROJECT_ROOT, "static", "files", token)
            # workdir = os.path.join("./static/files", token)
            # os.makedirs(os.path.join(workdir, "input"), exist_ok=True)
            # os.makedirs(os.path.join(workdir, "output"), exist_ok=True)

            # Factory method - get_processor() 
            try:
                job_instance = JobInstance.get_processor(task_detail, msg_queue=result_queue)
                job_instance.process_file()
            except:
                traceback.print_exc()
                error.append(traceback.format_exc())

            result = job_instance.get_results()

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
                "start time": calculation_start_time,
                "end time": calculation_end_time,
                "runtime": calculation_time_cost,
                "error": error,
                "figure_result": result,
                "finished": True,
                "state": state,
                "status": status,
            }
            result_queue.put(res)

            # res1 = dict(id=token,result=res,finished=res['finished'],state=res['state'],status=res['status'],submission_detail=task_detail)
            # file_path = os.path.join(workdir, "results.json")
            # with open(file_path, 'w') as f:
            #    json.dump(res1,f,indent=2)


    # def home(self):
    #     return flask.render_template(self._home_html, urlprefix=self._prefix)

    # def examples(self):
    #     return flask.render_template(self._examples_html, basedir="static/examples")

    def abstract(self):
        return flask.render_template(self._abstract_html)

    def result(self,id=None):
        if not id:
            id = flask.request.args['id']
        return flask.render_template(self._result_html, urlprefix=self._prefix, list_id=id)

    def mark(self):
        resultid = flask.request.args['resultid']
        glycanid = flask.request.args['glycanid']
        note = flask.request.args['note']

        if self._lock.acquire(timeout=2):
            if resultid not in self._resultid_locks:
                self._resultid_locks[resultid] = threading.Lock()
            self._lock.release()
        else:
            print("Status: ERROR:LOCK_TIMEOUT, ResultID: %s, GlycanID: %s, Note: %s."%(resultid,glycanid,note),file=sys.stderr)
            return flask.jsonify(dict(status="ERROR"))

        try:

            if self._resultid_locks[resultid].acquire(timeout=2):
    
                res = self.get_result(resultid)
                if res.get('location') == 'examples':
                    self._resultid_locks[resultid].release()
                    print("Status: ERROR:EXAMPLE, ResultID: %s, GlycanID: %s, Note: %s."%(resultid,glycanid,note),file=sys.stderr)
                    return flask.jsonify(dict(status="ERROR"))
    
                figureindex,glycanindex=map(int,glycanid.split('.'))
                glycan = res['result']['figure_result'][figureindex]['glycans'][glycanindex]
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
            self._resultid_locks[resultid].release()
            traceback.print_exc()
            print("Status: ERROR, ResultID: %s, GlycanID: %s, Note: %s."%(resultid,glycanid,note),file=sys.stderr)
            return flask.jsonify(dict(status="ERROR"))
        
        self._resultid_locks[resultid].release()
        print("Status: OK, ResultID: %s, GlycanID: %s, UpVotes: %s, DownVotes: %s, Note: %s."%(resultid,glycanid,glycan.get('upvotes',0),glycan.get('downvotes',0),glycan.get('note',"")),file=sys.stderr)
        return flask.jsonify(dict(status="OK",resultid=resultid,glycanid=glycanid,upvotes=glycan.get('upvotes',0),downvotes=glycan.get('downvotes',0),note=glycan.get('note',"")))
    
if __name__ == '__main__':
    multiprocessing.freeze_support()

    fb_api = ReferenceAPIFileBased()
    fb_api.parse_config("GlyImageExtractor.ini")

    fb_api._resultid_locks = {}
    fb_api._lock = threading.Lock()
    fb_api.start()










