
__all__ = [ "ExtractorClient", "ExtractorDevClient", "BadTaskIDError", "GlyLookupClient" ]

import sys, os, glob, json
import requests, time
import traceback

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

    def request(self, sub, params=None, files=None, pmids=None):
        for i in range(self._max_retry):
            if files is not None:
                files1 = dict((k,open(v,'rb')) for k,v in files.items())
            else:
                files1 = None
            params1 = params
            if params is None:
                params1 = {}

            # Add pmids to form data if provided
            if pmids is not None:
                # pmids is a dict like {'pmid': '12345'}
                params1.update(pmids)       # add pmid to form data

            response = None
            try:
                if params1 or files1:
                    response = requests.post(
                        self._apiurl + "/" + sub, 
                        data=params1,   # form data (includes pmid from pmids dict)
                        files=files1    # files - pdf
                    )
                else:
                    response = requests.get(self._apiurl + "/" + sub)
            except Exception as e:
                print("Exception occuered: ", e)
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
        if isinstance(submit_result,dict) and submit_result.get('error'):
            raise APISubmitError(submit_result)
        try:
            task_id = submit_result[0][u"id"]
            return task_id
        except (TypeError,KeyError):
            pass
        except:
            traceback.print_exc()
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
        if not res2json.get("finished",False) and not asis:
            raise APIUnfinishedError(task_id,"NotComplete","The task %s is not finished yet" % task_id)
        return res2json
    
    def status(self,task_id):
        return self.retreive_once(task_id)

class BatchAPIFrameworkClient(APIFrameworkClient):
    """
    Extends APIFrameworkClient to support batch submit/retrieve operations.
    """

    def submit_batch(self, tasks, request="submit", **kwargs):
        param = {"tasks": json.dumps(tasks), "developer_email": self._email}
        if self._nocache:
            param["nocache"] = 'true'
        param.update(kwargs)
        
        res = self.request(request, param)

        try:
            submit_result = res.json()
        except ValueError as e:
            raise APISubmitError(
                f"Invalid JSON response from API: {res.text[:500]}"
            ) from e

        try:
            return [job["id"] for job in submit_result]
        except (TypeError, KeyError, IndexError) as e:
            raise APISubmitError(
                f"Unexpected submit response format: {submit_result}"
            ) from e


    def retrieve_batch(self, jobids, delay=None, maxretry=None):
        delay = delay or self._interval
        maxretry = maxretry or self._max_retry_for_unfinished_task
        
        nretries = 0
        while True:
            param = {"task_ids": json.dumps(jobids)}
            try:
                res = self.request("retrieve", param)
                if res is None:
                    raise ValueError("No response")
                if res.status_code != 200:
                    raise ValueError(
                        f"Retrieve failed with status {res.status_code}: {res.text[:500]}"
                    )
                try:
                    results = res.json()
                except ValueError as e:
                    raise ValueError(
                        f"Invalid JSON response. Status: {res.status_code}"
                    ) from e
            except Exception as e:
                if nretries >= maxretry:
                    raise APIUnfinishedError(
                        f"Failed to retrieve batch after {maxretry} retries: {e}"
                    ) from e
                time.sleep(delay)
                nretries += 1
                continue
            
            all_done = all(job.get('finished', False) for job in results)
            if all_done:
                return results
            
            if nretries >= maxretry:
                raise APIUnfinishedError(
                    f"Batch jobs not finished after {maxretry} retries"
                )
            
            if self._statusfn is not None:
                self._statusfn("batch", f"Waiting for {len(jobids)} jobs", True)
            
            time.sleep(delay)
            nretries += 1
        
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

    def makeurl(self,path):
        return self.url() + '/' + path.lstrip('/')
    
    def status(self,taskid):
        return self.get_job_status(taskid)

    def submit_pmid(self,mode,pmid,aspdf=False):
        assert mode in ("Manuscript",)
        if aspdf:
            pmid = str(pmid) + ".pdf"
        task = dict(submission_type=mode,pmid=pmid)
        return self.submit(task=task,request="file_upload")
    
    def submit_local(self,mode,filepath):
        assert mode in ("Manuscript",
                        "Multi-Glycan Image",
                        "Simple Glycan Image")
        task = dict(submission_type=mode,filePath=filepath)
        return self.submit(task=task,request="file_upload")
    
    def submit_url(self,mode,url):
        assert mode in ("Manuscript",
                        "Multi-Glycan Image",
                        "Simple Glycan Image")
        task = dict(submission_type=mode,fileURL=url)
        return self.submit(task=task,request="file_upload")
    
    def submit_file(self,mode,filename):
        assert mode in ("Manuscript",
                        "Multi-Glycan Image",
                        "Simple Glycan Image")
        task = dict(submission_type=mode)
        return self.submit(task=task,request="file_upload",files=dict(file=filename))

    def submit_manuscript_local(self,filepath):
        return self.submit_local("Manuscript",filepath)

    def analyze_manuscript_local(self,filepath):
        taskid = self.submit_manuscript_local(filepath)
        return self.retrieve(taskid)
    
    def submit_manuscript_pmid(self,pmid,aspdf=False):
        return self.submit_pmid("Manuscript",pmid,aspdf)

    def analyze_manuscript_pmid(self,pmid,aspdf=False):
        taskid = self.submit_manuscript_pmid(pmid,aspdf)
        return self.retrieve(taskid)
    
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

