
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

class APIFramework:

    # job states
    UNKOWN = 'Unkown'
    QUEUED = 'Queued'
    RUNNING = 'Running'
    ERROR = 'Error'
    COMPLETE = 'Complete'

    def __init__(self):

        self._verbose_level = 100

        self._host = "localhost"
        self._port = 10980
        self._debug = True

        self._worker_num = 1
        self._clean_start = True
        self._file_based_job = False

        self._app_name = "testing"
        self._flask_app = flask.Flask(self._app_name)
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


        if "basic" in res:

            #for k,v in res["basic"].items():
            #    print("%s: |%s|" % (k,v))

            if "host" in res["basic"]:
                self.set_host(res["basic"]["host"])

            if "port" in res["basic"]:
                self.set_port(int(res["basic"]["port"]))

            if "debug" in res["basic"]:
                self.set_debug(res["basic"]["debug"])

            if "cpu_core" in res["basic"]:
                self.set_worker_num(int(res["basic"]["cpu_core"]))

            if "clean_start" in res["basic"]:
                self._clean_start = bool(res["basic"]["clean_start"])

            if "file_based_job" in res["basic"]:
                self._file_based_job = bool(res["basic"]["file_based_job"])

            if "input_file_folder" in res["basic"]:
                self.set_input_file_folder(res["basic"]["input_file_folder"])

            # if "output_file_folder" in res["basic"]:
            #     self.set_output_file_folder(res["basic"]["output_file_folder"])

            if "template_folder" in res["basic"]:
                self._template_folder = res["basic"]["template_folder"]

            if "home_page" in res["basic"]:
                self._home_html = res["basic"]["home_page"]

            if "file_upload_finished_page" in res["basic"]:
                self._file_upload_finished_html = res["basic"]["file_upload_finished_page"]

            if "allowed_file_ext" in res["basic"]:
                allowed_file_ext = res["basic"]["allowed_file_ext"].split(",")
                self.clear_allowed_file_ext()
                for ext in allowed_file_ext:
                    ext = ext.strip()
                    self.add_allowed_file_ext(ext)

            if "app_name" in res["basic"]:
                self.set_app_name(res["basic"]["app_name"])

                if res["basic"]["app_name"] in res:
                    self._worker_para = res[res["basic"]["app_name"]]

            if "prefix" in res["basic"]:
               self.set_prefix(res["basic"]["prefix"])

    def makeid(self,*params,random=False,length=16,sep=":"):
        msgparts = list(params)
        if random:
            msgparts.append("".join([ random_module.choice("0123456789") for i in range(16)]))
        msg = sep.join(map(str,msgparts))
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

    # FLASK related functions starts here

    # FLASK handlers, need to be overwrite for your own app
    def home(self, **kwargs):
        sessionid = self.get_session()
        if self._home_html is None:
            return flask.jsonify("Hello from %s:%s" % (self.host(), self.port()))
        else:
            return flask.render_template(self._home_html, urlprefix=self._prefix, **kwargs)

    def process(self):
        submission_type = flask.request.args.get("type")

        if submission_type == 'Manuscript':
            placeholder_url = 'https://example.com/document.pdf'
        else:
            placeholder_url = 'https://example.com/image.png'

        return flask.render_template(self._process_html, submission_type=submission_type, placeholder_url=placeholder_url, urlprefix=self._prefix)


    def examples(self):
        # Mcleod - https://www.neb.com/en-us/-/media/nebus/files/application-notes/appnote_characterization_of_glycans_from_erbitux_rituxan_and_enbrel_using_recombinant_pngase_f.pdf?rev=581a874aebbc4351bec05e10c07f96ea&hash=C8D5EB5AF1B7D5C331DFAB13BB87F649
        example_cards = [
            {"title": "Sassi et al., 2014", "desc": "", "url": f"{self._prefix}/result/mgp1", "icon": f"{self._prefix}/static/images/pdf.svg" },
            {"title": "Mcleod, 2024", "desc": "", "url": f"{self._prefix}/result/mgp3", "icon": f"{self._prefix}/static/images/pdf.svg"},
            {"title": "Simple PDF", "desc": "", "url": f"{self._prefix}/result/mgp2", "icon": f"{self._prefix}/static/images/pdf.svg" },
            {"title": "Multi Glycan Figure", "desc": "", "url": f"{self._prefix}/result/mgi1", "icon": f"{self._prefix}/static/images/multi-image.svg" },
            {"title": "Multi Glycan Example", "desc": "", "url": f"{self._prefix}/result/mgi2", "icon": f"{self._prefix}/static/images/multi-image.svg" },
            {"title": "Multiple Glycans", "desc": "", "url": f"{self._prefix}/result/mgi3", "icon": f"{self._prefix}/static/images/multi-image.svg" },
            {"title": "Simple Glycan Image", "desc": "", "url": f"{self._prefix}/result/sgi1", "icon": f"{self._prefix}/static/images/single-image.svg"},
            {"title": "Simple Glycan Page", "desc": "", "url": f"{self._prefix}/result/sgi2", "icon": f"{self._prefix}/static/images/single-image.svg" },
            {"title": "Simple Glycan Image", "desc": "", "url": f"{self._prefix}/result/sgi3", "icon": f"{self._prefix}/static/images/single-image.svg" },
        ]
        
        return flask.render_template(self._examples_html, example_cards=example_cards, urlprefix=self._prefix)


    def jobs(self):
        return flask.render_template(self._jobs_html, urlprefix=self._prefix)

    # def about(self):
    #     return flask.render_template("about.html", urlprefix=self._prefix)

    def file_upload_finished_page(self, **kwargs):
        if self._file_upload_finished_html is None:
            return flask.jsonify("Not Implemented")
        else:
            return flask.render_template(self._file_upload_finished_html, urlprefix=self._prefix, **kwargs)

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
        state = self.UNKOWN
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

                task1['job_status'] = f"{self._prefix}/get_job_status/{tid}"
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
        thing = {"Error": "list_id (%s) not found" % list_id}
        if list_id in self.result_cache:
            thing = self.result_cache[list_id]
        elif os.path.exists(f"static/files/{list_id}/results.json"):
            thing = json.loads(open(f"static/files/{list_id}/results.json").read())
        elif os.path.exists(f"static/examples/{list_id}/results.json"):
            thing = json.loads(open(f"static/examples/{list_id}/results.json").read())
            thing['location'] = 'examples'
        return thing

    def save_result(self,list_id,result):
        # We should lock so that we don't get two at once...
        self.result_cache[list_id] = result
        location = result.get('location','files')
        wh = open(f"static/{location}/{list_id}/results.json",'wt')
        json.dump(result,wh,indent=2)
        wh.close()

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

    def upload_file(self):
        # print("UPLOAD FILE")
        if flask.request.method == 'POST':

            file = flask.request.files.get('file')  
            if 'task' in flask.request.form:
                task = json.loads(flask.request.form.get('task'))
                file_url = task.get('fileURL')
                submission_type = task.get('submission_type')
            else:
                file_url = flask.request.form.get('fileURL')
                submission_type = flask.request.form.get('submission_type')
            # pipeline_name = flask.request.form.get('fileType')
            
            # print("FILE", file, file_type)
            # print("file_type",file_type)
            # print("file url", file_url)

            if not file and not file_url:
                return flask.abort(400, "No file or url provided")

            if file and file.filename:
                filename = werkzeug.utils.secure_filename(file.filename)

            elif file_url:
                filename = werkzeug.utils.secure_filename(os.path.basename(file_url.split('?')[0]))
                # print("URL FILE",filename)

                if not os.path.splitext(filename)[1]:  
                    content_disposition = requests.head(file_url).headers.get('content-disposition')
                    if content_disposition:
                        filename = content_disposition.split('filename=')[-1].strip('"')

            else:
                return flask.jsonify("Invalid file or URL"), 400

            sessionid = self.get_session()

            # Create task details
            task_detail = self.form_task({"original_file_name": filename, "submission_type": submission_type})
            list_id = task_detail["id"]
            file_dir = os.path.join(self.input_file_folder(), list_id)
            os.makedirs(file_dir, exist_ok=True)
            # file_path = os.path.join(self.input_file_folder(), list_id)
            file_path = os.path.join(file_dir, filename)

            try:
                if file and self.allow_file_ext(file.filename):
                    file.save(file_path)
                elif file_url:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
                        'Accept': '*/*'
                    }

                    with requests.get(file_url, headers=headers, stream=True, timeout=10) as response:
                        response.raise_for_status()
                        with open(file_path, "wb") as f:
                            for chunk in response.iter_content(1024):
                                f.write(chunk)

                else:
                    return flask.jsonify(f"Unsupported file type: {filename}"), 400

            except requests.exceptions.RequestException as e:
                return flask.jsonify(f"Failed to download file: {str(e)}"), 400
            except Exception as e:
                return flask.jsonify(f"Unexpected error: {str(e)}"), 500


            status = {
                "id": list_id,
                "task_index": self.get_next_task_index(),
                "submission_detail": task_detail,
                "state": self.QUEUED,
                "status": "",
                "finished": False,
                "submission_type": submission_type,
                "submit_time": time.time(),
                "sessionid": sessionid,
                "result": {}
            }

            if list_id in self.result_cache:
                pass
            else:
                self.task_queue.put(task_detail)
                self.result_cache[list_id] = status
                self.add_to_task_lists(list_id,sessionid,status['submit_time'])
            self.output(1, "Job received by API: %s" % (task_detail))

        return flask.jsonify([status])



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


    # FLASK helper functions
    def form_task(self, p):
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

    def update_results(self, getall=False):

        i = 0
        while True:

            if not getall and i >= 5:
                break

            i += 1

            try:
                res = self.result_queue.get_nowait()
                if 'state' in res:
                    self.result_cache[res["id"]]['state'] = res["state"]
                if 'status' in res:
                    self.result_cache[res["id"]]['status'] = res["status"]
                if res.get("finished",False):
                    self.result_cache[res["id"]]["result"] = res
                    self.result_cache[res["id"]]['finished'] = True
                    self.remove_from_task_list(res["id"])    

                    file_path = os.path.join("static/files/"+res["id"], "results.json")
                    with open(file_path, 'w') as f:
                        json.dump(self.result_cache[res["id"]],f,indent=2)

            except queue.Empty:
                break
            except KeyError:
                self.output(1, "Job ID %s is not present" % res["id"])


    def allow_file_ext(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_file_ext()

    # Load route and handler to flask app
    def load_route(self):
        # TODO custom route?
        self._flask_app.add_url_rule("/", "home", self.home, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/retrieve", "retrieve", self.retrieve, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/abstract", "abstract", self.abstract, methods=["GET", "POST"])
        # self._flask_app.add_url_rule("/examples", "examples", self.examples, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/result", "result", self.result, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/result/<id>", "result", self.result, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/mark", "mark", self.mark, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/get_job_counts", "get_job_counts", self.get_job_counts, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/get_job_status", "get_job_status", self.get_job_status, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/get_job_status/<tid>", "get_job_status", self.get_job_status, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/api/recent_jobs", "get_recent_jobs_api", self.get_recent_jobs_api, methods=["GET"])
        self._flask_app.add_url_rule("/process", "process", self.process, methods=["GET"])
        self._flask_app.add_url_rule("/examples", "examples", self.examples, methods=["GET"])
        self._flask_app.add_url_rule("/jobs", "jobs", self.jobs, methods=["GET"])

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
            if 'sessionid' in data:
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
            p = multiprocessing.Process(target=self.worker, args=(i, self.task_queue, self.result_queue, self._worker_para ))
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


