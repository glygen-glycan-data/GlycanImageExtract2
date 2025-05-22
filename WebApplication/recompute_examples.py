#!../.venv/bin/python

import sys, os, glob, json
import time, shutil
import requests

class APIFrameworkClient:

    class APISubmitError(RuntimeError):
        pass

    class APIUnfinishedError(RuntimeError):
        pass

    apiurl = 'http://localhost:10980'
    developer_email = None
    max_retrieve_wait = 300
    nocache = False

    def __init__(self,**kwargs):
        self._apiurl= kwargs.get('apiurl',self.apiurl)
        self._email = kwargs.get('developer_email',self.developer_email)
        self._nocache = kwargs.get('nocache',self.nocache)
        self._max_retry = kwargs.get('max_request_retry',3)
        self._interval = kwargs.get('request_interval',1)
        self._max_retry_for_unfinished_task = kwargs.get('max_retrieve_wait',self.max_retrieve_wait)

    def request(self, sub, params, files={}):
        for i in range(self._max_retry):
            files1 = dict((k,open(v,'rb')) for k,v in files.items())
            response = None
            try:
                # print(self._apiurl + "/" + sub,params)
                response = requests.post(self._apiurl + "/" + sub, params, files=files1)
            except:
                pass
            finally:
                dummy  = list(map(lambda fh: fh.close(),files1.values()))
            if response is not None:
                return response
            time.sleep(self._interval)

    def retrieve(self, task_id):
        for i in range(self._max_retry_for_unfinished_task):
            time.sleep(self._interval)
            try:
                res = self.retrieve_once(task_id)
                return res
            except APIFrameworkClient.APIUnfinishedError:
                continue
        raise APIFrameworkClient.APIUnfinishedError("The task %s is not finished yet" % task_id)

    def get(self, **kwargs):
        task_id = self.submit(**kwargs)
        resjson = self.retrieve(task_id)
        return resjson

    def submit(self, task={}, request="submit"):
        param = {"task": json.dumps(task), "developer_email": self._email}
        if self.nocache:
            param["nocache"] = 'true'
        res1 = self.request(request, param)
        submit_result = res1.json()
        try:
            task_id = submit_result[0][u"id"]
            return task_id
        except TypeError:
            pass
        raise APIFrameworkClient.APISubmitError(submit_result)

    def retrieve_once(self, task_id):
        param = {"task_id": task_id }
        try:
            res2 = self.request("retrieve", param)
            res2json = res2.json()[0]
        except:
            raise
        if not res2json[u"finished"]:
            raise APIFrameworkClient.APIUnfinishedError("The task %s is not finished yet" % task_id)
        return res2json

# Should get the port number from the ini file...
port = 10982
apiurl = 'http://localhost:%s'%(port,)
extractor = APIFrameworkClient(apiurl=apiurl,
                               developer_email="nje5@georgetown.edu",
                               request_interval=5,
                               max_retrieve_wait = 1200)

tasks = []
for resultfile in glob.glob("static/examples/*/results.json"):
    basedir = os.path.split(resultfile)[0]
    result = json.loads(open(resultfile).read())
    inputfilename = result['submission_detail']['original_file_name']
    inputpath = basedir+"/input/"+inputfilename
    filetype = result['submission_detail']['file_type']
    task = dict(fileType=filetype,fileURL=(apiurl+"/"+inputpath))
    exampledir = os.path.split(basedir)[1]
    tasks.append((exampledir,extractor.submit(task=task,request="file_upload")))
    print("Example %s submitted. "%(exampledir,))
    time.sleep(1)

for exampledir,taskid in tasks:
    result = {}
    try:
        result = extractor.retrieve(taskid)
    except APIFrameworkClient.APIUnfinishedError:
        pass
    if result.get('finished',False):
        shutil.rmtree("static/examples/"+exampledir)
        shutil.copytree("static/files/"+taskid,
                        "static/examples/"+exampledir)
        print("Example %s done."%(exampledir,))
    else:
        print("Example %s not updated."%(exampledir,))
    

