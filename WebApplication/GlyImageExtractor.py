
import sys
import os
import argparse


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path to import BKGlycanExtractor
sys.path.append(parent_dir)


from APIFramework import APIFrameWork
from BKGlycanExtractor.annotatePDF import annotatePDFGlycan, annotatePNGGlycan

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


class ReferenceAPIParaBased(APIFrameWork):
    pass


import subprocess
class ReferenceAPIFileBased(APIFrameWork):

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

            error = []
            calculation_start_time = time.time()

            original_file_name = task_detail["original_file_name"]


            list_id = task_detail["id"]

            # TODO finished by you from here##############################################################
            input_folder = os.path.join(os.getcwd(), "input")
            print("task_detail",task_detail)

            print(task_detail["original_file_name"])

            input_file = os.path.join("input", task_detail["id"])
            output_file_abs_path = os.path.abspath(os.path.join("output", list_id))

            origfilename = task_detail["original_file_name"]
            file_type = task_detail["file_type"]
            token = list_id

            os.makedirs(os.path.join("./static/files", token, "input"))
            os.makedirs(os.path.join("./static/files", token, "output"))
            workdir = os.path.join("./static/files", token)

            #infilename = os.path.join(workdir, "input", origfilename)
            infilename = os.path.join(workdir, "input", origfilename)
            outfilename = os.path.join(workdir, "output", "annotated_" + origfilename)


            base_configs = "../BKGlycanExtractor/config/"
            outfilename2 = output_file_abs_path
            work_dict = {
                "token": str(token),
                "workdir": str(workdir),
                "infilename": str(infilename),
                "outfilename": str(outfilename),
                "base_configs": str(base_configs),
                "origfilename": str(origfilename),
                "outfilename2": str(outfilename2),
                "input_file": str(input_file),
                "file_type": str(file_type)
            }

            try:
                copyfile(work_dict["input_file"], work_dict["infilename"])
            except FileNotFoundError:
                time.sleep(5)
                copyfile(work_dict["input_file"], work_dict["infilename"])

            extn = origfilename.lower().rsplit('.', 1)[-1]
            print("origfilename and extension",origfilename, extn)

            if extn in ('png','jpg','jpeg'):
                try:
                    annotatePNGGlycan(work_dict) 
                    

                    if "file_format_error" in work_dict:
                        error.append(work_dict["file_format_error"])

                except Exception as e:
                    error.append('File %s generated an exception in annotatePNGGlycan: %s'%(origfilename,repr(e)))
                    print(traceback.format_exc(),file=sys.stderr)
            elif extn in ('pdf'):
                try:
                    annotatePDFGlycan(work_dict)

                    if "file_format_error" in work_dict:
                        error.append(work_dict["file_format_error"])

                except Exception as e:
                    error.append('File %s generated an exception in annotatePDFGlycan: %s'%(origfilename,repr(e)))
                    print(traceback.format_exc(),file=sys.stderr)
            else:
                error.append('File %s had an Unsupported file extension: %s.'%(origfilename,extn))
                # self.output(1,'File %s had an Unsupported file extension: %s.'%(origfilename,extn))


            #print("#############end calling subprocess here################\n\n\n\n")


            # TODO END#####################################################################################



            calculation_end_time = time.time()
            calculation_time_cost = calculation_end_time - calculation_start_time

            # option = {"as_attachment": True}
            option = {"as_attachment": True, "mimetype": 'application/pdf'}

            total_glycan_count = sum(len(data['image_data']) for data in work_dict['file_data'])

            # print("--->>count",count)

            res = {
                "id": list_id,
                "start time": calculation_start_time,
                "end time": calculation_end_time,
                "runtime": calculation_time_cost,
                "error": error,
                "glycans": work_dict.get('results',[]),
                "output_file_abs_path": output_file_abs_path,
                "flask_download_option": option,
                "inputtype": extn,
                "original_file": infilename,
                "annotated_file_path": outfilename,
                "annotated_filename": "annotated_"+original_file_name,
                "file_data": work_dict['file_data'],
                "total_glycan_count": total_glycan_count
            }

            result_queue.put(res)
            res = dict(id=list_id,result=res,file_data=work_dict['file_data'],finished=True,submission_detail=task_detail)
            wh = open(f"static/files/{list_id}/results.json",'w')
            wh.write(json.dumps(res))
            wh.close()

    def home(self):
        return flask.render_template(self._home_html)

    def examples(self):
        return flask.render_template(self._examples_html, basedir="static/examples")

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










