
import sys
import os
import argparse


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
        task_str = p["original_file_name"] + str(random.randint(10000, 99999))
        task_str = task_str.encode("utf-8")
        list_id = hashlib.sha256(task_str).hexdigest()

        res["id"] = list_id
        res["original_file_name"] = p["original_file_name"]
        res['file_type'] = p['file_type']

        return res


    @staticmethod
    def worker(pid, task_queue, result_queue, params):
        print(pid, "Start")

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

            # Factory method - class reference is passed via get_prcessor() 
            # which is instantiated below
            job_class = JobInstance.get_processor(task_detail)
            job_instance = job_class(task_detail)
            job_instance.process_file()

            result = job_instance.get_results()

            calculation_end_time = time.time()
            calculation_time_cost = calculation_end_time - calculation_start_time

            # information for webservice (API Framework)
            res = {
                "id": token,
                "start time": calculation_start_time,
                "end time": calculation_end_time,
                "runtime": calculation_time_cost,
                "error": error,
                "figure_result": result
            }

            result_queue.put(res)

            res1 = dict(id=token,result=res,finished=True,submission_detail=task_detail)

            file_path = os.path.join(workdir, "results.json")
            with open(file_path, 'w') as f:
                json.dump(res1,f,indent=2)


    def home(self):
        return flask.render_template(self._home_html)

    # def examples(self):
    #     return flask.render_template(self._examples_html, basedir="static/examples")

    def abstract(self):
        return flask.render_template(self._abstract_html)

    def result(self):
        id = flask.request.args['id']
        print(f"{id}\n",file=sys.stderr)
        return flask.render_template(self._result_html, list_id=id)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    fb_api = ReferenceAPIFileBased()
    fb_api.parse_config("GlyImageExtractor.ini")

    fb_api.start()










