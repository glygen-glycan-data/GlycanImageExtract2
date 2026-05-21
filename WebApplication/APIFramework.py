
from __future__ import print_function

import os
import sys
import copy
import time
import json
import flask
import requests
import werkzeug
import atexit
import hashlib
import multiprocessing
import json
from collections import defaultdict
import threading
import random as random_module
import base64
import glob

import os, ssl
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
import shutil
import tarfile
import traceback

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl,'_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

try:
    # Python3 import
    import queue
except ImportError:
    # Python2 import
    import Queue as queue

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

class APIErrorBase(RuntimeError):

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class APIParameterError(APIErrorBase):
    pass

class APIDataError(APIErrorBase):
    pass

class APIFramework:

    # job states
    UNKNOWN = 'Unknown'
    QUEUED = 'Queued'
    RUNNING = 'Running'
    ERROR = 'Error'
    COMPLETE = 'Complete'

    def __init__(self,name=None):

        if not name:
            name = self.__class__.__name__

        self._verbose_level = 100

        self._host = "localhost"
        self._port = 10980
        self._debug = True

        self._worker_num = 1
        self._clean_start = True
        self._file_based_job = False

        self._app_name = name
        self._prefix = ""

        self._input_file_folder  = self.abspath("input")
        # self._output_file_folder = self.abspath("output")

        self._allowed_file_ext = ["txt", "doc", "docx", "pdf", "jpg", "png"]

        self._worker_para = {}

        self.result_cache = {}
        self.task_queue   = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()

        self.task_index = 0
        self.task_index_lock = threading.Lock()

        self.task_list = set()
        self.task_list_lock = threading.Lock()

        self.session_task_list = defaultdict(list)
        self.session_task_list_lock = threading.Lock()

        self._template_folder = None
        self._home_html = None

        self._examples_html = "examples.html"
        # self._abstract_html = "abstract.html"
        self._result_html = "result.html"
        self._jobs_html = 'jobs.html'
        self._process_html = 'process.html'      # page which lets you submit your file/url
        self._file_upload_finished_html = None
        self._template_render_kwargs = {'urlprefix': self._prefix}

        self.parse_config(name + ".ini")
        self.set_app_name(name)

    # Proper APIs for changing config
    def host(self):
        return self._host

    def set_host(self, h):
        self._host = h

    def port(self):
        return self._port

    def set_port(self, p):
        if isinstance(p, int):
            self._port = p
        else:
            raise APIParameterError("Port number requires integer, %s is not acceptable")

    def debug(self):
        return self._debug

    def set_debug(self, debug):
        self._debug = (debug.lower() in ("true","t","1","yes","y"))

    def worker_num(self):
        return self._worker_num

    def set_worker_num(self, w):
        self._worker_num = w

    def verbose_level(self):
        return self._verbose_level

    def set_verbose_level(self, v):
        assert isinstance(v, int)
        assert 0 <= v <= 100
        self._verbose_level = v

    def set_app_name(self, an):
        self._app_name = an
        if self._template_folder is None:
            self._flask_app = flask.Flask(self._app_name)
        else:
            self._flask_app = flask.Flask(self._app_name, template_folder=self.abspath(self._template_folder))
        self._flask_app.secret_key = self.makeid(self._app_name,random=False,length=16)
        # self._flask_app.config['PERMANENT_SESSION_LIFETIME'] = 30*24*3600 # 30 days

    def set_prefix(self, prefix):
        self._prefix = "/" + prefix.strip().strip('/')
        self.set_template_render_kwarg(urlprefix=self._prefix)

    def input_file_folder(self):
        return self._input_file_folder

    def set_input_file_folder(self, fp):
        self._input_file_folder = self.abspath(fp)

    # def output_file_folder(self):
    #     return self._output_file_folder

    # def set_output_file_folder(self, fp):
    #     self._output_file_folder = self.abspath(fp)

    def allowed_file_ext(self):
        return self._allowed_file_ext

    def clear_allowed_file_ext(self):
        self._allowed_file_ext = []

    def add_allowed_file_ext(self, ext):
        ext = ext.lower()
        if ext not in self._allowed_file_ext:
            self._allowed_file_ext.append(ext)

    def rm_allowed_file_ext(self, ext):
        ext = ext.lower()
        if ext in self._allowed_file_ext:
            self._allowed_file_ext.remove(ext)

    def output(self, lvl, msg):
        if lvl <= self.verbose_level():
            print(msg, file=sys.stderr)


    def abspath(self, fp):
        base = os.path.dirname(os.path.abspath(sys.argv[0]))
        res = os.path.join(base, fp)
        return res

    def parse_config(self, config_file_name):
        config_path = self.abspath(config_file_name)

        config = configparser.ConfigParser()
        config.read_file(open(config_path))

        res = {}
        for each_section in config.sections():
            res[each_section] = {}
            for (each_key, each_val) in config.items(each_section):
                if each_val != "":
                    res[each_section][each_key] = each_val

        self._base_config = res.get("basic",{})
        self._worker_config = res.get(self._app_name,{})

        base = self._base_config

        #for k,v in base.items():
        #    print("%s: |%s|" % (k,v))

        if "host" in base:
            self.set_host(base["host"])

        if "port" in base:
            self.set_port(int(base["port"]))

        if "debug" in base:
            self.set_debug(base["debug"])

        if "cpu_core" in base:
            self.set_worker_num(int(base["cpu_core"]))

        # check these, string -> bool may not do what is intended
        if "clean_start" in base:
            self._clean_start = bool(base["clean_start"])

        if "file_based_job" in base:
            self._file_based_job = bool(base["file_based_job"])

        if "input_file_folder" in base:
            self.set_input_file_folder(base["input_file_folder"])

        if "template_folder" in base:
            self._template_folder = base["template_folder"]

        if "home_page" in base:
            self._home_html = base["home_page"]

        if "file_upload_finished_page" in base:
            self._file_upload_finished_html = base["file_upload_finished_page"]

        if "allowed_file_ext" in base:
            allowed_file_ext = base["allowed_file_ext"].split(",")
            self.clear_allowed_file_ext()
            for ext in allowed_file_ext:
                ext = ext.strip()
                self.add_allowed_file_ext(ext)

        if "prefix" in base:
            self.set_prefix(base["prefix"])

    def makeid(self,*params,random=False,length=16,sep=":"):
        msgparts = list(params)
        if random:
            msgparts.append("".join([ random_module.choice("0123456789") for i in range(16)]))
        msg = sep.join(map(lambda p: str(p) if p is not None else "",msgparts))
        return base64.b32encode(hashlib.sha256(msg.encode()).digest()).decode()[:length].lower()

    def get_session(self):
        sessionid = flask.session.get('sessionid')
        if not sessionid:
            sessionid = self.makeid(random=True,length=16)
            flask.session.permanent = True        
            flask.session['sessionid'] = sessionid
        print("Sessionid:",sessionid,file=sys.stderr)
        return sessionid

    # Worker function
    @staticmethod
    def worker(pid, task_queue, result_queue, params):
        # Params are key value pairs from configuration file, section app_name
        raise NotImplemented

    def set_template_render_kwarg(self,**kwargs):
        self._template_render_kwargs.update(dict(**kwargs)) 

    # FLASK related functions starts here

    # FLASK handlers, need to be overwrite for your own app
    def _render_page(self, template_attr, page_response=None,**page_args):
        template_name = getattr(self, template_attr, None)
        if not template_name:
            if page_response:
                return page_response
            return flask.jsonify({"error": "Not Implemented"}), 501

        args = dict(self._template_render_kwargs)
        args.update(page_args)
        return flask.render_template(template_name, **args)

    def home(self, **kwargs):
        sessionid = self.get_session()
        page_response = flask.jsonify("Hello from %s:%s" % (self.host(), self.port())), 200
        return self._render_page('_home_html', page_response=page_response, **kwargs)

    def jobs(self, **kwargs):
        return self._render_page('_jobs_html', **kwargs)

    def process(self, **kwargs):
        return self._render_page('_process_html', **kwargs)

    def examples(self, **kwargs):
        return self._render_page('_examples_html', **kwargs)

    def jobs(self, **kwargs):
        return self._render_page('_jobs_html', **kwargs)

    def result(self,id=None, **kwargs):
        return self._render_page('_result_html', list_id=id, **kwargs)

    def file_upload_finished_page(self, **kwargs):
        return self._render_page('_file_upload_finished_html', **kwargs)

    def get_jobs_ahead(self,tid):
        tind = self.result_cache[tid].get("task_index",1e+10)
        ahead = 0
        with self.task_list_lock:
            for tid1 in self.task_list:
                if self.result_cache[tid1].get('state') == self.QUEUED and \
                   self.result_cache[tid1].get("task_index",0) < tind:
                    ahead += 1
        return ahead

    def get_job_status(self,tid=None):
        params = self.api_para()
        if 'tid' in params:
            tid = params['tid']
        status = "Job status not available."
        state = self.UNKNOWN
        finished = False
        self.update_results(getall=True)
        result = self.get_result(tid)
        if 'Error' not in result:
            status = result.get("status","Job status not available.")
            state =  result.get("state",state)
            finished = result.get("finished",False)
            if state == self.QUEUED:
                status = "Position %d in the job queue"%(self.get_jobs_ahead(tid)+1,)
        return flask.jsonify(dict(status=status,state=state,finished=finished))

    def get_job_counts(self):
        self.update_results(getall=False)
        states = defaultdict(int)
        for x in self.result_cache.values():
            states[x["state"]] += 1
        return flask.jsonify(states)


    def get_recent_jobs_api(self):

        sid = self.get_session()
        recent_jobs = []

        with self.session_task_list_lock:
            # Refresh all results so get_result returns the latest data
            self.update_results(getall=True)

            for tid in map(lambda t: t[0], sorted(self.session_task_list[sid], key=lambda t: -t[1])[:10]):
                task1 = dict((k, v) for k, v in self.get_result(tid).items() if k != 'result')

                task1['job_status'] = f"{self._prefix}/job_status/{tid}"
                if task1['state'] == self.QUEUED:
                    task1['status'] = "Position %d in queue"%(self.get_jobs_ahead(tid)+1,)

                recent_jobs.append(task1)
        return flask.jsonify(recent_jobs)

    def get_next_task_index(self):
        with self.task_index_lock:
            self.task_index += 1
            task_index = self.task_index
        return task_index

    def add_to_task_lists(self,tid,sid,stime):
        with self.task_list_lock:
            self.task_list.add(tid)
        with self.session_task_list_lock:
            self.session_task_list[sid].append((tid,stime))

    def remove_from_task_list(self,tid):
        with self.task_list_lock:
            if tid in self.task_list:
                self.task_list.remove(tid)

    # TODO this method is similar to upload_file - probably should collapse these methods into one
    def submit(self):
        if flask.request.method in ['GET', 'POST']:
            p = self.api_para()
        else:
            return flask.jsonify("METHOD %s is not suppoted" % flask.request.method)

        if "tasks" not in p:
            return flask.jsonify("Please submit with actual tasks")

        sessionid = self.get_session()

        # Suppose to be a list
        raw_tasks = json.loads(p["tasks"])
        res = []
        for raw_task in raw_tasks:
            task_detail = self.form_task(raw_task)
            if "id" not in task_detail:
                raise APIParameterError(
                    "No id provided for your job(%s), probably check the form_task method"
                    % task_detail)

            res.append(copy.deepcopy(task_detail))
            list_id = task_detail["id"]
            status = {
                "id": list_id,
                "task_index": self.get_next_task_index(),
                "submission_detail": task_detail,
                "finished": False,
                "state": self.QUEUED,
                "status": "",
                "submit_time": time.time(),
                "sessionid": sessionid,
                "result": {}
            }

            if list_id in self.result_cache:
                pass
            else:
                self.task_queue.put(task_detail)
                self.result_cache[list_id] = status
                self.add_to_task_lists(list_id, sessionid, status['submit_time'])
            self.output(1, "Job received by API: %s" % (task_detail))

        return flask.jsonify(res)

    def get_result(self,list_id):
        if list_id in self.result_cache:
            result = self.result_cache[list_id]
        return {"Error": "list_id (%s) not found" % (list_id,)}

    def retrieve(self):
        if flask.request.method in ['GET', 'POST']:
            p = dict(self.api_para())
        else:
            return flask.jsonify("METHOD %s is not suppoted" % flask.request.method)

        if "task_ids" in p:
            p["list_ids"] = p["task_ids"]
        elif "task_id" in p:
            p["list_ids"] = json.dumps([ p["task_id"] ])
        if "list_ids" not in p:
            return flask.jsonify("Please provide with list_id(s)")

        self.update_results(getall=True)

        # Suppose to be a list
        list_ids = json.loads(p["list_ids"])
        res = []
        for list_id in list_ids:
            res.append(self.get_result(list_id))
        return flask.jsonify(res)

    def _input_filename(self, params: dict):
        '''
        Handles filename for: file (upload), fileURL, filePath (file stored on disk)
        
        Other applications can override this for submissions that dont fit the generic use case.
        '''
        filename = None

        if params.get('file'):
            filename = werkzeug.utils.secure_filename(params['file'].filename)
        elif params.get('filePath'):
            filename = werkzeug.utils.secure_filename(os.path.split(params['filePath'])[1])
        elif params.get('fileURL'):
            filename = werkzeug.utils.secure_filename(os.path.basename(params['fileURL'].split('?')[0]))
            if not os.path.splitext(filename)[1]:
                try:
                    head_resp = requests.head(params['fileURL'], timeout=10, allow_redirects=True)
                    head_resp.raise_for_status()   
                    content_disposition = head_resp.headers.get('content-disposition')
                    if content_disposition:
                        filename = content_disposition.split('filename=')[-1].strip('"')
                except requests.exceptions.RequestException as e:
                    raise APIParameterError(
                        "Could not create filename for the provided URL: %s" % (params["fileURL"],)
                    ) from e

        if filename is None:
            raise APIParameterError('Could not create a filename for the provided input')

        return filename

    def prepare_job_input(self, params: dict, task_detail: dict, current_file_path: str):
        '''
        params: contains user submitted details
        task_detail: may contain derived details which could be helpful
        '''

        if params.get('file') and self.allow_file_ext(params['file'].filename):
            params["file"].save(current_file_path)
            return 
        elif params.get('filePath'):
            src = params["filePath"]
            if not os.path.isfile(src):
                raise APIParameterError("filePath does not exist or is not a file")
            try:
                shutil.copyfile(src, current_file_path)
            except OSError as e:
                raise APIDataError("Could not copy filePath") from e
            return
        elif params.get('fileURL'):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
                'Accept': '*/*'
            }
            try:
                with requests.get(params["fileURL"], headers=headers, stream=True, timeout=10) as response:
                    if not response.ok:
                        raise APIDataError("Download failed (HTTP %s)" % response.status_code)
                    with open(current_file_path, "wb") as f:
                        for chunk in response.iter_content(1024):
                            if chunk:
                                f.write(chunk)
                return 
            except requests.exceptions.RequestException as e:
                raise APIDataError("Could not download file from provided URL") from e

        raise APIParameterError(
            "No supported file source (file, filePath, or fileURL), or file type not allowed"
        ) 
    
    def upload_file(self, task=None):

        if not (flask.request.method == 'POST' or task):
            return flask.jsonify({"error": "Invalid request method"}), 400

        try:
            # parse_request accesses the form fields 
            params = self.parse_request(task)  # task=None for FormData UI
        except APIParameterError as e:
            return flask.jsonify({"error": str(e), "valid": False}), 400

        try:
            filename = self._input_filename(params)
        except APIParameterError as e:
            return flask.jsonify({"error": str(e), "valid": False}), 400

        # form task
        params_dict = dict(params)
        params_dict['filename'] = filename

        try:
            task_detail = self.form_task(params_dict)
        except APIParameterError as e:
            return flask.jsonify({"error": str(e), "valid": False}), 400

        sessionid = self.get_session()
        list_id = task_detail["id"]
        file_dir = os.path.join(self.input_file_folder(), list_id)
        os.makedirs(file_dir, exist_ok=True)
        current_file_path = os.path.join(file_dir, filename)

        try:
            self.prepare_job_input(params_dict, task_detail, current_file_path)
        except APIParameterError as e:
            return flask.jsonify({"error": str(e), "valid": False}), 400
        except APIErrorBase as e:
            return flask.jsonify({"error": str(e), "valid": False}), 400
        except requests.exceptions.RequestException:
            return flask.jsonify({"error": "Can't download or fetch input.", "valid": False}), 400
        except Exception as e:
            traceback.print_exception(type(e), e, e.__cause__)
            return flask.jsonify({"error": f"Unexpected error: {e}", "valid": False}), 500

        status = {
            "id": list_id,
            "task_index": self.get_next_task_index(),
            "submission_detail": task_detail,
            "state": self.QUEUED,
            "status": "",
            "finished": False,
            "submit_time": time.time(),
            "sessionid": sessionid,
            "result": {},
        }

        if list_id in self.result_cache:
            pass
        else:
            self.task_queue.put(task_detail)
            self.result_cache[list_id] = status
            self.add_to_task_lists(list_id, sessionid, status['submit_time'])

        self.output(1, "Job received by API: %s" % (task_detail))
        return flask.jsonify([status]), 200

    def _build_resubmit_task(self, submission_detail, result, tid, input_file):
        """Can override in derived classes to add or change resubmit fields."""
        params_dict = {k: v for k, v in submission_detail.items() if k not in ('id')}
        params_dict["filePath"] = input_file
        return params_dict

    def resubmit_file(self, tid=None):
        '''
        Resubmit a prior job as a new job (reanalyze)
        Not currently a feature, but probably paramters could be tweaked by user for resubmission
        '''

        if tid is None:
            tid = flask.request.args.get("tid") or flask.request.form.get("tid")
        if not tid:
            return flask.jsonify({"error": "task id is required", "valid": False}), 400
                
        result = self.get_result(tid)
        if not result:
            return flask.jsonify({"error": "unknown task id %s" % tid, "valid": False}), 404

        if not result.get('finished'):
            return flask.jsonify({"error": "Previous task %s was not completed successfully" % tid, "valid": False}), 404

        submission_detail = result.get("submission_detail") or {}
        
        input_file = os.path.join(
            "static", result.get("location", "files"), tid, "input",
            submission_detail.get("filename") or submission_detail.get("original_file_name", ""),
        )   
        if not os.path.isfile(input_file):
          return flask.jsonify({"error": "input file not found for resubmit", "valid": False}), 404

        params_dict = self._build_resubmit_task(submission_detail, result['result'], tid, input_file)

        response, code = self.upload_file(task=json.dumps(params_dict))

        if code != 200:
            resp_body = response.get_json(silent=True) or {}
            err = resp_body.get("error") if isinstance(resp_body, dict) else str(resp_body)
            return flask.jsonify({"error": "Resubmission failed: %s" % err, "valid": False}), code

        return flask.redirect(self._prefix + '/jobs')
        
    def download_file(self):
        if flask.request.method in ['GET', 'POST']:
            p = self.api_para()
        else:
            return flask.jsonify("METHOD %s is not suppoted" % flask.request.method)

        if "list_id" not in p:
            return flask.jsonify("Please provide with list_id(s)")

        list_id = p["list_id"]

        if not self.result_cache[list_id]["finished"]:
            return flask.jsonify("The computation haven't finished yet")

        target_file_path = self.result_cache[list_id]["result"]["output_file_abs_path"]

        download_file_name = target_file_path
        if "rename" in self.result_cache[list_id]["result"]:
            download_file_name = self.result_cache[list_id]["result"]["rename"]

        download_option = {}
        if "flask_download_option" in self.result_cache[list_id]["result"]:
            download_option = self.result_cache[list_id]["result"]["flask_download_option"]

        try:
            return flask.send_file(target_file_path,
                                   attachment_filename=download_file_name,
                                   **download_option)
        except:
            flask.abort(404)

    def form_task(self, p: dict):
        #
        """
        task = {
            "id": list_id,
            "key": value...
        }
        return task
        """
        raise NotImplemented

    @staticmethod
    def api_para():
        if flask.request.method == "GET":
            return flask.request.args
        elif flask.request.method == "POST":
            return flask.request.form
        else:
            raise APIErrorBase

    def parse_request(self, task=None) -> dict:
        # generic method - which handles different types of data: JSON, arguments, post, etc
        # form-data/multipart - form + optional file
        # application/json - request.get_json(silent=true)
        # optionally application/www-form-urlencoded

        req = flask.request
        
        params = {}

        # 1) flask provides form fields (task dict) - for files and text fields.
        raw = req.form.get("task") if req.form else None
        if raw is not None and str(raw).strip():
            try:
                form_params = json.loads(raw)
            except (TypeError, ValueError) as e:
                raise APIParameterError(f"Invalid JSON in form form field 'task': {e}")
            if not isinstance(form_params, dict):
                raise APIParameterError("Form field 'task' must be a JSON object")
            params = form_params

        # 2) Explicit task argument:
        # useful when resubmit is used and task_dict is provided
        elif task is not None:
            if isinstance(task, dict):
                params = dict(task)
            else:
                if isinstance(task, bytes):
                    task = task.decode("utf-8")
                if isinstance(task, str):
                    s = task.strip()
                    if s:
                        try:
                            parsed = json.loads(s)
                        except (TypeError, ValueError) as e:
                            raise APIParameterError("task argument must be valid JSON: %s" % (e,))
                    if not isinstance(parsed, dict):
                        raise APIParameterError("task JSON must be an object")
                    params = parsed
                else:
                    raise APIParameterError("task must be dict, str, bytes, or None")
        
        # 3) JSON body (API style application/json)
        elif req.is_json:
            body = req.get_json(silent=True)
            if isinstance(body, dict):
                params = dict(body)

        # below code ensures that form details are added properly
        if req.form:
            for key in req.form:
                if key == "task":
                    continue
                val = req.form.get(key)
                if val is not None:
                    params[key] = val

        # if files are present - it ensures that they are added as well
        if req.files:
            for key in req.files:
                f = req.files.get(key)
                if f and getattr(f, 'filename', None):
                    params[key] = f

        return params

    def update_results(self, getall=False):

        i = 0
        while True:

            if not getall and i >= 5:
                break

            i += 1

            try:
                res = self.result_queue.get_nowait()
            except queue.Empty:
                break
            
            assert "id" in res
            resid = res["id"]

            if 'state' in res:
                self.result_cache[resid]['state'] = res["state"]
            if 'status' in res:
                self.result_cache[resid]['status'] = res["status"]
            if res.get("finished",False):
                for key in ("start_time","end_time","runtime","state","status","finished","error","id"):
                    if key in res:
                        self.result_cache[resid][key] = res[key]
                        del res[key]

                self.result_cache[resid]["result"] = res
                self.remove_from_task_list(resid)   

                # derived class method can override this to add functionality wrt result updates 
                self.on_task_finished(resid) 

    
    def on_task_finished(self, resid):
        '''
        Override in subclass - if the results need to be saved some where
        and if special cleanup steps is required
        '''
        pass


    def allow_file_ext(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_file_ext()

    def robots(self):
        response = flask.make_response(open("./static/robots.txt").read())
        response.mimetype = 'text/plain'
        return response

    # Load route and handler to flask app
    def load_route(self):
        # TODO custom route?
        self._flask_app.add_url_rule("/", "home", self.home, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/retrieve", "retrieve", self.retrieve, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/examples", "examples", self.examples, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/result/<id>", "result", self.result, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/job_counts", "get_job_counts", self.get_job_counts, methods=["GET"])
        self._flask_app.add_url_rule("/job_status/<tid>", "get_job_status", self.get_job_status, methods=["GET"])
        self._flask_app.add_url_rule("/recent_jobs", "get_recent_jobs_api", self.get_recent_jobs_api, methods=["GET"])
        self._flask_app.add_url_rule("/process", "process", self.process, methods=["GET"])  # keep here, but the method can live in the derived
        self._flask_app.add_url_rule("/jobs", "jobs", self.jobs, methods=["GET"])
        self._flask_app.add_url_rule("/resubmit_file/<tid>", "resubmit_file", self.resubmit_file, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/robots.txt", "robots.txt", self.robots, methods=["GET", "POST"])

        if self._file_based_job:
            self._flask_app.add_url_rule("/file_upload", "upload_file", self.upload_file, methods=["GET", "POST"])
            self._flask_app.add_url_rule("/file_download", "download_file", self.download_file, methods=["GET", "POST"])
        else:
            self._flask_app.add_url_rule("/submit", "submit", self.submit, methods=["GET", "POST"])

    def manipulate_dirs(self):
        if not os.path.exists(self.input_file_folder()):
            os.makedirs(self.input_file_folder())
        """
        if not os.path.exists(self.output_file_folder()):
            os.makedirs(self.output_file_folder())

        if self._clean_start:
            for folder in [self.input_file_folder(), self.output_file_folder()]:
                for fn in os.listdir(folder):
                    # Clean up the input and output folder
                    fp = os.path.join(folder, fn)
                    os.remove(fp)
        """
        return


    def populate_session_tasks(self):
        for f in glob.glob("static/files/*/results.json"):
            try:
                data = json.loads(open(f).read())
            except json.decoder.JSONDecodeError:
                continue
            if 'sessionid' in data and 'id' in data:
                tid = data['id']
                sid = data['sessionid']
                stime = data['submit_time']
                self.session_task_list[sid].append((tid,stime))
        # print(self.session_task_list)

    def start(self):
        self.load_route()
        self.manipulate_dirs()
        self.populate_session_tasks()

        self._deamon_process_pool = []
        for i in range(self._worker_num):
            p = multiprocessing.Process(target=self.worker, args=(i, self.task_queue, self.result_queue, self._worker_config ))
            self._deamon_process_pool.append(p)

        for p in self._deamon_process_pool:
            p.start()

        self.cleanup()

        self._flask_app.run(self.host(), self.port(), debug=self.debug())

    def cleanup(self):
        atexit.register(self.terminate_all)


    def terminate_all(self):
        for p in self._deamon_process_pool:
            p.terminate()


if __name__ == '__main__':
    multiprocessing.freeze_support()


