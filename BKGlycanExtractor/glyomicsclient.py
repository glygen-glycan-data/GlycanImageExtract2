
__all__ = [ "ExtractorClient", "ExtractorDevClient", "BadTaskIDError", "GlyLookupClient" ]

import sys, os, glob, json
import requests, time

class APISubmitError(RuntimeError):
    pass

class APINoResponse(RuntimeError):
    pass

class APIUnfinishedError(RuntimeError):
    pass

class BadTaskIDError(RuntimeError):
    pass

class APIFrameworkClient:

    apiurl = 'http://localhost:10980'
    port = None
    developer_email="nje5+glyomicsclient_module@georgetown.edu"
    max_retrieve_wait = 300
    nocache = False
    status_callback = None

    def __init__(self,**kwargs):
        self._apiurl= kwargs.get('apiurl',self.apiurl)
        port = kwargs.get('port',self.port)
        if port is not None:
            self._apiurl += ":%s"%(port,)
        self._email = kwargs.get('developer_email',self.developer_email)
        self._nocache = kwargs.get('nocache',self.nocache)
        self._max_retry = kwargs.get('max_request_retry',3)
        self._interval = kwargs.get('request_interval',5)
        self._max_retry_for_unfinished_task = kwargs.get('max_retrieve_wait',self.max_retrieve_wait)
        self._statusfn = kwargs.get('status_callback',self.status_callback)

    def url(self):
        return self._apiurl

    def request(self, sub, params=None, files=None):
        for i in range(self._max_retry):
            if files is not None:
                files1 = dict((k,open(v,'rb')) for k,v in files.items())
            else:
                files1 = None
            params1 = params
            if params is None:
                params1 = {}
            response = None
            try:
                # print(self._apiurl + "/" + sub,params)
                if params1 or files1:
                    response = requests.post(self._apiurl + "/" + sub, params1, files=files1)
                else:
                    response = requests.get(self._apiurl + "/" + sub)
            except:
                pass
            finally:
                if files1 is not None:
                    dummy  = list(map(lambda fh: fh.close(),files1.values()))
            if response is not None:
                return response
            time.sleep(self._interval)
        raise APINoResponse

    def retrieve(self, task_id):
        for i in range(self._max_retry_for_unfinished_task):
            time.sleep(self._interval)
            try:
                res = self.status(task_id)
                return res
            except APIUnfinishedError as e:
                if self._statusfn is not None:
                    self._statusfn(*e.args) 
                continue
            except APINoResponse:
                continue
            except KeyError:
                raise BadTaskIDError(task_id) from None
            except ValueError:
                continue
                
        raise APIUnfinishedError("The task %s is not finished yet" % task_id)

    def get(self, **kwargs):
        task_id = self.submit(**kwargs)
        resjson = self.retrieve(task_id)
        return resjson

    def submit(self, task={}, request="submit", **kwargs):
        param = {"task": json.dumps(task), "developer_email": self._email}
        if self.nocache:
            param["nocache"] = 'true'
        res1 = self.request(request, param, **kwargs)
        submit_result = res1.json()
        try:
            task_id = submit_result[0][u"id"]
            return task_id
        except TypeError:
            pass
        raise APISubmitError(submit_result)

    def get_job_status(self, task_id):
        res = self.request("get_job_status/"+task_id).json()
        if not res[u"finished"]:
            raise APIUnfinishedError(task_id,res["state"],res["status"])
        return self.retrieve_once(task_id)

    def retrieve_once(self, task_id, asis=False):
        param = {"task_id": task_id }
        try:
            res2 = self.request("retrieve", param)
            if res2 is None:
                raise ValueError("No response")
            res2json = res2.json()[0]
        except:
            raise
        if not res2json[u"finished"] and not asis:
            raise APIUnfinishedError(task_id,"NotComplete","The task %s is not finished yet" % task_id)
        return res2json
    
    def status(task_id):
        return self.retreive_once(task_id)

class GlyLookupClient(APIFrameworkClient):
    apiurl="http://glylookup.glyomics.org"
    request_interval=1

    def get_accession_for_sequence(self,seq):
        data = self.get(task=dict(seq=seq))
        if len(data['result']) == 0:
            return None
        return data['result'][0]['accession']

class ExtractorClient(APIFrameworkClient):
    request_interval=5
    max_retrieve_wait = 1200
    apiurl="https://extractor.glyomics.org"

    def status(self,taskid):
        return self.get_job_status(taskid)

    def submit_url(self,mode,url):
        assert mode in ("Manuscript",
                        "Multi-Glycan Image",
                        "Single-Glycan Image")
        task = dict(submission_type=mode,fileURL=url)
        return self.submit(task=task,request="file_upload")
    
    def submit_file(self,mode,filename):
        assert mode in ("Manuscript",
                        "Multi-Glycan Image",
                        "Single-Glycan Image")
        task = dict(submission_type=mode)
        return self.submit(task=task,request="file_upload",files=dict(file=filename))

    def submit_manuscript_url(self,url):
        return self.submit_url("Manuscript",url)

    def analyze_manuscript_url(self,url):
        taskid = self.submit_manuscript_url(url)
        return self.retrieve(taskid)
    
    def submit_manuscript_file(self,filename):
        return self.submit_file("Manuscript",filename)
    
    def analyze_manuscript_file(self,filename):
        taskid = self.submit_manuscript_file(filename)
        return self.retrieve(taskid)
    
    def submit_multiglycanimg_url(self,url):
        return self.submit_url("Multi-Glycan Image",url)

    def analyze_multiglycanimg_url(self,url):
        taskid = self.submit_multiglycanimg_url(url)
        return self.retrieve(taskid)
    
    def submit_multiglycanimg_file(self,filename):
        return self.submit_file("Multi-Glycan Image",filename)
    
    def analyze_multiglycanimg_file(self,filename):
        taskid = self.submit_multiglycanimg_file(filename)
        return self.retrieve(taskid)
    
    def submit_singleglycanimg_url(self,url):
        return self.submit_url("Single-Glycan Image",url)

    def analyze_singleglycanimg_url(self,url):
        taskid = self.submit_singleglycanimg_url(url)
        return self.retrieve(taskid)
    
    def submit_singleglycanimg_file(self,filename):
        return self.submit_file("Single-Glycan Image",filename)
    
    def analyze_singleglycanimg_file(self,filename):
        taskid = self.submit_singleglycanimg_file(filename)
        return self.retrieve(taskid)
    
    @staticmethod
    def status_callback(*args):
        if args[2]:
            print("Task %s: %s - %s"%args)
        else:
            print("Task %s: %s"%args[:2])

class ExtractorDevClient(ExtractorClient):
    apiurl="http://localhost"
    port = 10982

