#!../.venv/bin/python

import sys, os, glob, json
import time, shutil
import requests

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path to import BKGlycanExtractor
sys.path.append(parent_dir)

from BKGlycanExtractor.compareboxes import CompareBoxes
from BKGlycanExtractor.bbox import BoundingBox

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

patterns = ["*"]
if len(sys.argv) > 1:
    patterns = sys.argv[1:]

tasks = []
for pat in patterns:
  for resultfile in sorted(glob.glob("static/examples/%s/results.json"%(pat,))):
    basedir = os.path.split(resultfile)[0]
    result = json.loads(open(resultfile).read())
    inputfilename = result['submission_detail']['original_file_name']
    inputpath = basedir+"/input/"+inputfilename
    filetype = result['submission_detail']['file_type']
    task = dict(fileType=filetype,fileURL=(apiurl+"/"+inputpath))
    exampledir = os.path.split(basedir)[1]
    tasks.append((exampledir,extractor.submit(task=task,request="file_upload")))
    print("Example %s submitted (%s). "%(exampledir,tasks[-1][1]))
    time.sleep(1)

def update_votes(instance):
    result = json.loads(open("static/examples/"+instance+"/results.json").read())
    correct = json.loads(open("static/answers/"+instance+"/correct.json").read())
    for f1,f2 in zip(result["result"]["figure_result"],correct["result"]["figure_result"]):
        for g1 in f1["glycans"]:
            g1bb = BoundingBox(**dict(zip("xywh",g1['bbox'])))
            bestg2 = None
            bestiou = -1
            for g2 in f2["glycans"]:
                g2bb = BoundingBox(**dict(zip("xywh",g2['bbox'])))
                iou = CompareBoxes.iou(g1bb,g2bb)
                if iou > 0.7 and iou > bestiou:
                    bestiou = iou
                    bestg2 = g2
            if not bestg2:
                continue
            g2 = bestg2
            if g1.get("IUPAC"):
                if g1.get("IUPAC") == g2.get("IUPAC","__XXXXXX__"):
                    g1['upvotes'] = 1; g1['downvotes'] = 0;
                else:
                    g1['upvotes'] = 0; g1['downvotes'] = 1
            else:
                if g1.get("composition_str") == g2.get("composition_str","__XXXXXX__"):
                    g1['upvotes'] = 1; g1['downvotes'] = 0;
                else:
                    g1['upvotes'] = 0; g1['downvotes'] = 1
    with open("static/examples/"+instance+"/results.json",'wt') as wh:
        json.dump(result,wh,indent=2)

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
        update_votes(exampledir)
        print("Example %s done (%s)."%(exampledir,taskid))
    else:
        print("Example %s not updated (%s)."%(exampledir,taskid))

