
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

from BKGlycanExtractor import annotate_from_webapp

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

    # flask.request.url_root seems to work poorly with respect to proxies etc.
    # and the proxy code (_prefix) didn't seem to work correctly either
    # 
    # The referer url (source of the request) is a better choice, since
    # we know what page should be calling this. We must make sure we do not fail
    # badly if called by another referer, but we only need this to work correctly
    # when clicked on from the appropriate page. 
    # 
    # def get_base_url(self):
    #     if flask.has_request_context():
    #         
    #         base = flask.request.url_root.rstrip('/')
    #         if self._prefix:
    #             prefix = self._prefix.strip('/')
    #             return f"{self._prefix}"
    #         return f"{base}"
    #

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
    def home(self, **kwargs):
        sessionid = self.get_session()
        if self._home_html is None:
            return flask.jsonify("Hello from %s:%s" % (self.host(), self.port()))
        kwargs.update(dict(**self._template_render_kwargs))
        return flask.render_template(self._home_html, **kwargs)

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


    def jobs(self):
        kwargs = dict(**self._template_render_kwargs)
        return flask.render_template(self._jobs_html, **kwargs)

    # def about(self):
    #     return flask.render_template("about.html", urlprefix=self._prefix)

    def file_upload_finished_page(self, **kwargs):
        if self._file_upload_finished_html is None:
            return flask.jsonify("Not Implemented")
        else:
            kwargs.update(dict(**self._template_render_kwargs))
            return flask.render_template(self._file_upload_finished_html, **kwargs)

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

    
    def validate_pmid(self, pmid=None):
        '''
        Validates is the given PMID has a PMCID and that the resources for the PMCID are Open Access (check if zip file can be retrieved)
        '''
        developer_email="nje5%2bextractor@georgetown.edu"

        if pmid is None:
            pmid = flask.request.json.get('pmid')

        pmid = pmid.strip()
        if pmid.endswith(".pdf"):
            pmid = pmid.rsplit('.',1)[0]
            
        pmid_to_pmc_converter_api = f'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={pmid}&tool=extract&email={developer_email}&idtype=pmid&format=json'

        try:
            resp = requests.get(pmid_to_pmc_converter_api, timeout=5)
            #
            # This API raises 400 errors for bad parameters, which need a 
            # nicer error message than the exception handler can give...
            # 
            # resp.raise_for_status()
            resp_json = resp.json()

            if not resp_json.get('records') or len(resp_json['records']) == 0:
                return flask.jsonify({'valid': False, 'error': f'PMID {pmid} is not in PubMed Central'}), 400
                
            pmcid = resp_json['records'][0].get('pmcid')
            if not pmcid:
                return flask.jsonify({'valid': False, 'error': f"PMID {pmid} is not in PubMed Central"}), 400

            # check if it is possible to retrieve the zipped file using PMCID
            pmc_resp, pmc_status = self.validate_pmcid_resources(pmid, pmcid)
            pmc_resp_json = pmc_resp.get_json()

            if pmc_status != 200 or not pmc_resp_json.get('valid'):
                return pmc_resp, pmc_status
                
            return flask.jsonify(pmc_resp_json), 200
        
        except (requests.exceptions.Timeout,requests.exceptions.ReadTimeout,requests.exceptions.RequestException) as e:
            return flask.jsonify({'valid': False, 'error': str(e)}), 500


    def validate_pmcid_resources(self, pmid, pmcid):
        pmc_api = f'https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}'
        r = requests.get(pmc_api, timeout=5)
        r.raise_for_status()

        root = ET.fromstring(r.text)
        link = root.find(".//link[@format='tgz']")
        record = root.find(".//record")

        if link is None:
            return flask.jsonify({
                'valid': False,
                'error': f"PMID {pmid} is not Open Access in PubMed Central"
            }), 400

        return flask.jsonify({
            'valid': True,
            'success': f'Given PMCID: {pmcid} is Open Access',
            'resource': {
                'href': link.get('href'),
                'format': link.get('format'),
                'pmcid': pmcid,
                # 'pmc_publication': record.get("citation")
            }
        }), 200

    def _parse_upload_inputs(self, task=None):
        task = json.loads(flask.request.form.get('task', task if task else '{}'))
        return {
            "task": task,
            "file": flask.request.files.get('file'),
            "file_url": flask.request.form.get('fileURL', task.get('fileURL')),
            "input_file_path": flask.request.form.get('filePath', task.get('filePath')),
            "pmid": flask.request.form.get('pmid', task.get('pmid')),
            # if search_stratgey given - use the mentioned search_stratgey (useful for re-analyze job, when submission_mode is Local and synthetic pdf will benifit from using the previsuly used fitz image_search_strategy)
            "image_search_strategy": flask.request.form.get('image_search_strategy', task.get('image_search_strategy')),
            "submission_type": flask.request.form.get('submission_type', task.get('submission_type')),
        }

    def _get_submission_mode(self, upload_params):
        submission_mode = None
        pmid = upload_params.get('pmid', None)
        
        if upload_params['input_file_path']:
            submission_mode = "Local"
        elif upload_params['file']:
            submission_mode = "Upload"
        elif upload_params['file_url']:
            submission_mode = "URL"
        elif upload_params['submission_type'] == "Manuscript" and pmid:
            pmid = pmid.strip()
            if pmid.endswith(".pdf"):
                submission_mode = "PMID-PDF"
                pmid = pmid.split('.', 1)[0]
            else:
                submission_mode = "PMID"

        upload_params['submission_mode'] = submission_mode
        upload_params['pmid'] = pmid
        
        return submission_mode

    def _get_filename(self, upload_params):
        submission_mode = upload_params.get('submission_mode')
        input_file_path = upload_params['input_file_path']
        file_url = upload_params['file_url']
        pmid = upload_params.get('pmid', None)
        file = upload_params['file']

        filename = None

        # Extract info using Pubmed API and get filename of the pdf based on PMID and at the same time extract figures as well - everything is present in the zipped file
        if submission_mode and pmid and submission_mode in ("PMID", "PMID-PDF"):
            # pdf file that goes in the input folder should be renamed as "PMID-<PMID>.pdf"
            filename = 'PMID-' + pmid + ".pdf"
        elif input_file_path:
            filename = werkzeug.utils.secure_filename(os.path.split(input_file_path)[1])
        elif file_url:
            filename = werkzeug.utils.secure_filename(os.path.basename(file_url.split('?')[0]))
            if not os.path.splitext(filename)[1]:
                try:
                    head_resp = requests.head(file_url, timeout=10, allow_redirects=True)
                    head_resp.raise_for_status()   
                    content_disposition = head_resp.headers.get('content-disposition')
                    if content_disposition:
                        filename = content_disposition.split('filename=')[-1].strip('"')
                except requests.exceptions.RequestException:
                    # Keep derived filename
                    pass
        elif file:
            filename = werkzeug.utils.secure_filename(file.filename)

        if filename:
            upload_params['filename'] = filename
            return filename

        return None

    def _handle_local_copy(self, input_file_path, file_dir):
        # Local means its a reanalyze step - copy over all files from the mentioned input folder to your current input folder
        # copy over files (pdf/tar.gz) from the old task id's folder i.e static/files/old_task_id/input/ --> input/curr_task_id
        old_input_directory = os.path.dirname(input_file_path)
        shutil.copytree(old_input_directory, file_dir, dirs_exist_ok=True)

    def _download_and_prepare_pmid_data(self, pmid, file_dir):
        # validate if PMCID resources are Open Access before proceeding
        pmc_resp, pmc_status = self.validate_pmid(pmid)
        pmc_resp_json = pmc_resp.get_json()

        if pmc_status != 200 or not pmc_resp_json.get('valid'):
            return pmc_resp, pmc_status, None
                    
        resource = pmc_resp_json.get("resource")
        pmcid = resource.get("pmcid")

        # 1) get citation from json response - if available
        # pmc_publication - is the PMC publication information obtained
        # from hittin the PMC API (useful when publication information is only partially present in the xml document provided by pmc)
        # pmc_publication = resource.get("pmc_publication")

        # 2) extract the href link, which is in ftp (NCBI supports both ftp and https protocols)
        href = resource.get("href")
        # Convert FTP to HTTPS
        download_url = href.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")

        # 3) Download the zipped file to the input folder
        zipped_path = os.path.join(file_dir, f"PMID-{pmid}.tar.gz")
        with open(zipped_path, "wb") as f:
            with requests.get(download_url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

        # 4) Extract data from tar file
        # Note: The zip file may contain multiple pdf's, so the main pdf filename is same
        # as the xml filename - the below code tracks and finds the correct pdf to use
        with tarfile.open(zipped_path, "r:gz") as tar:
            # Single pass: collect nxml files and their corresponding PDFs
            nxml_files = []
            pdf_files = {}
            for member in tar.getmembers():
                base = os.path.basename(member.name).lower()
                ext = os.path.splitext(base)[1]
                if ext == '.nxml':
                    nxml_basename = os.path.splitext(base)[0]
                    nxml_files.append((member, nxml_basename))
                elif ext == '.pdf':
                    pdf_basename = os.path.splitext(base)[0]
                    pdf_files[pdf_basename] = member
            # Now extract the main pdf and rename it as PMID-<PMID>.pdf
            for nxml_member, nxml_basename in nxml_files:
                if nxml_basename in pdf_files:
                    pdf_member = pdf_files[nxml_basename]
                    tar.extract(pdf_member, file_dir, filter="data")
                    # Rename the PDF
                    old_pdf_path = os.path.join(file_dir, pdf_member.name)
                    new_pdf_path = os.path.join(file_dir, f"PMID-{pmid}.pdf")
                    if os.path.exists(old_pdf_path):
                        os.rename(old_pdf_path, new_pdf_path)
                    else:
                        print(f"File not found: {old_pdf_path}")
        # renamed the zipped file to PMID-<PMID>, while using the tarfile modeule a residual empty folder was created with the original zipped file name --> so deleting this empty folder
        pmc_folder_path = os.path.join(file_dir, pmcid)
        try:
            shutil.rmtree(pmc_folder_path)
        except FileNotFoundError:
            pass
        return None, None

    def _save_file(self, upload_params):
        file = upload_params['file']
        file_url = upload_params['file_url']
        input_file_path = upload_params['input_file_path']
        current_file_path = upload_params['current_file_path']
        filename = upload_params['filename']

        if file and self.allow_file_ext(file.filename):
            file.save(current_file_path)
            return None
        if file_url:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
                'Accept': '*/*'
            }
            try:
                with requests.get(file_url, headers=headers, stream=True, timeout=10) as response:
                    response.raise_for_status()
                    with open(current_file_path, "wb") as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                return None
            except requests.exceptions.RequestException:
                return flask.jsonify({"error": f"Can't download from provided URL."}), 400
        if input_file_path:
            shutil.copyfile(input_file_path, current_file_path)
            return None
        return flask.jsonify({"error": f"File format not supported: {filename}"}), 400


    def upload_file(self, task=None):
        if not (flask.request.method == 'POST' or task):
            return flask.jsonify({"error": "Invalid request method"}), 400

        upload_params = self._parse_upload_inputs(task)
        file = upload_params["file"]
        file_url = upload_params["file_url"]
        input_file_path = upload_params["input_file_path"]
        # pmid = upload_params["pmid"]
        # image_search_strategy = upload_params["image_search_strategy"]
        submission_type = upload_params["submission_type"]
        
        # pmc_publication = None

        submission_mode = self._get_submission_mode(upload_params)
        if not submission_mode:
            return flask.jsonify({"error": "Invalid file or URL"}), 400

        filename = self._get_filename(upload_params)
        if not filename:
            return flask.jsonify({"error": "Invalid file or URL"}), 400

        sessionid = self.get_session()
        # default image_search_strategy in configs file is 'hybrid', but if nothing is mentioned in configs file, then 'fitz' is assumed as default
        # Note - for PMID-PDF - synethetic pdf's are created using figures, so the deafult image search stratgey for this is fitz
        # Create task details
        task_detail = self.form_task({
            "filename": filename,  # derived
            "fileURL": file_url,
            "submission_type": submission_type,
            "submission_mode": submission_mode,  # derived
            "pmid": upload_params.get('pmid'),
            "image_search_strategy": upload_params.get('image_search_strategy')  # if provided, else form_task method will assign an image_search_strategy derieved from configs file or default will be used
        })
        list_id = task_detail["id"]
        file_dir = os.path.join(self.input_file_folder(), list_id)
        os.makedirs(file_dir, exist_ok=True)
        current_file_path = os.path.join(file_dir, filename)
        # adding current_file_path to the params dict, so that other methods can easily access it
        upload_params['current_file_path'] = current_file_path
        try:
            if submission_mode == 'Local':
                self._handle_local_copy(input_file_path, file_dir)
            elif submission_mode in ("PMID", "PMID-PDF"):
                pmid = upload_params['pmid']
                pmc_resp, pmc_status = self._download_and_prepare_pmid_data(pmid, file_dir)
                if pmc_resp is not None:
                    return pmc_resp, pmc_status
            else:
                save_error = self._save_file(upload_params)
                if save_error is not None:
                    return save_error
        except requests.exceptions.RequestException:
            return flask.jsonify({"error": "Submitted input is invalid."}), 400
        except Exception as e:
            return flask.jsonify({"error": f"Unexpected error: {str(e)}"}), 400
        # TODO: remove pmc_publication and use e-utils method of getting the citation
        # if upload_params.get('pmid') and pmc_publication:
        #     task_detail.update({"pmc_publication": pmc_publication})
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

    def resubmit_file(self,tid=None):
        '''
        Resubmit pathways:
        1) if submission_mode is - PMID, Single Image, Multi Image --> analysis will be done 
        on the figures again after they are copied from the old tasks input folder.

        2) other submission_modes (PDF, PMID-PDF) - uses pdf or adds figures to a pdf, 
        so the re-analyze task should directly run on the existing pdf (maybe synthetically created pdf) 
        after obtaining the pdf from the old tasks input folder.
        '''

        params = self.api_para()
        if 'tid' in params:
            tid = params['tid']
        result = self.get_result(tid)
        oldtask = result['submission_detail']

        submission_mode = oldtask['submission_mode']
        submission_type = oldtask['submission_type']
        input_file = os.path.join('static', result.get('location','files'), tid, 'input', oldtask.get('filename',oldtask.get('original_file_name')))

        newtask = {
            'submission_type': submission_type,
            'fileUrl': oldtask.get('fileURL'),
            'filePath': input_file,
            'image_search_strategy': oldtask.get('image_search_strategy')
        }

        if submission_mode == 'PMID' or (submission_mode == 'Local' and oldtask.get('pmid')):
            newtask.update(**{'pmid': oldtask['pmid']})

        response = self.upload_file(task=json.dumps(newtask))
        # print(response.get_json())
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
        resultid = flask.request.args.get('resultid') or resultid

        if resultid is None:
            return flask.jsonify(dict(status="ERROR"))

        location = "files"

        if flask.request.is_json:
            location = (flask.request.get_json(silent=True) or {}).get('location') or "files"
        elif flask.request.args.get('location'):
            location = flask.request.args.get('location')

        json_file = self.abspath(f"static/{location}/{resultid}/results.json")

        if not os.path.exists(json_file):
            print("Status: ERROR:NO_JSON, ResultID: %s." % (resultid,), file=sys.stderr)
            return flask.jsonify(dict(status="ERROR"))

        locked = False
        try:
            locked = self.lock_result(resultid, timeout=10)
            if not locked:
                print("Status: ERROR:RESULT_TIMEOUT, ResultID: %s." % (resultid,), file=sys.stderr)
                return flask.jsonify(dict(status="ERROR"))

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
                print("Status: ERROR:NO_PDF, ResultID: %s." % (resultid,), file=sys.stderr)
                return flask.jsonify(dict(status="ERROR"))

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
                    self._update_annotation_result_paths(
                        resultid=resultid,
                        result=result,
                        annotated_pdf_file=annotated_pdf_file,
                        annotated_tsv_file=annotated_tsv_file
                    )
                    with open(json_file, 'w') as f:
                        json.dump(json_data, f, indent=2)

                    print("Status: OK (annotated PDF up-to-date), ResultID: %s." % (resultid,), file=sys.stderr)
                    return flask.jsonify(dict(status="OK", resultid=resultid))

            base_url = self._get_base_url_from_request()

            if not base_url:
                base_url = f"http://{self.host()}:{self.port()}"

            # Regenerate annotated results
            print("Building annotated PDF/TSV for ResultID: %s." % (resultid,), file=sys.stderr)
            annotate_from_webapp(json_file, pdf_path, base_url, output_dir=output_dir)

            # after the results are ready and wrriten to the file system (via annotate_from_webapp), using a small delay to ensure
            # everything is set.
            time.sleep(0.2)

            # Verify that the annotated_pdf exists
            if annotated_pdf and os.path.exists(annotated_pdf):
                # make sure to add annotated file keys to the in-memory cache (result_cache) - so
                # that the front end has access to these keys via the API payload
                self._update_annotation_result_paths(
                    resultid=resultid,
                    result=result,
                    annotated_pdf_file=annotated_pdf_file,
                    annotated_tsv_file=annotated_tsv_file
                )

                # Write back to same file
                with open(json_file, 'w') as f:
                    json.dump(json_data, f, indent=2)

                print("Status: OK, ResultID: %s." % (resultid,), file=sys.stderr)
                return flask.jsonify(dict(status="OK", resultid=resultid))

            print("Status: ERROR:PDF_NOT_CREATED, ResultID: %s." % (resultid,), file=sys.stderr)
            return flask.jsonify(dict(status="ERROR"))

        except Exception:
            traceback.print_exc()
            print("Status: ERROR:EXCEPTION, ResultID: %s." % (resultid,), file=sys.stderr)
            return flask.jsonify(dict(status="ERROR"))

        finally:
            if locked:
                self.release_result(resultid)

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

                abs_json_path = self.abspath(os.path.join("static/files/"+resid, "results.json"))
                with open(abs_json_path, 'w') as f:
                    json.dump(self.result_cache[resid],f,indent=2)

                submission_detail = self.result_cache[resid]['submission_detail']
                submission_type = submission_detail['submission_type']
                submission_mode = submission_detail['submission_mode']

                # DO NOT ANNOTATE - if image based job, PMID figures, local mode but it is PMID figures based job
                # for all other cases you can create annotated pdf and tsv file
                if (submission_type not in ('Simple Glycan Image', 'Multi-Glycan Image')
                    and submission_mode != 'PMID'
                    and not (submission_mode == 'Local' and submission_detail.get('pmid'))):
                    self.annotate_results(resultid=resid)

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
        # self._flask_app.add_url_rule("/abstract", "abstract", self.abstract, methods=["GET", "POST"])
        # self._flask_app.add_url_rule("/examples", "examples", self.examples, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/result", "result", self.result, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/result/<id>", "result", self.result, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/mark", "mark", self.mark, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/annotate_results", "annotate_results", self.annotate_results, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/annotate_results/<rid>", "annotate_results", self.annotate_results, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/get_job_counts", "get_job_counts", self.get_job_counts, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/get_job_status", "get_job_status", self.get_job_status, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/get_job_status/<tid>", "get_job_status", self.get_job_status, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/resubmit_file", "resubmit_file", self.resubmit_file, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/resubmit_file/<tid>", "resubmit_file", self.resubmit_file, methods=["GET", "POST"])
        self._flask_app.add_url_rule("/api/recent_jobs", "get_recent_jobs_api", self.get_recent_jobs_api, methods=["GET"])
        self._flask_app.add_url_rule("/process", "process", self.process, methods=["GET"])
        self._flask_app.add_url_rule("/examples", "examples", self.examples, methods=["GET"])
        self._flask_app.add_url_rule("/jobs", "jobs", self.jobs, methods=["GET"])
        self._flask_app.add_url_rule("/pmid", "validate_pmid", self.validate_pmid, methods=["POST"])
        self._flask_app.add_url_rule("/pmid/<pmid>", "validate_pmid", self.validate_pmid, methods=["GET"])
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


