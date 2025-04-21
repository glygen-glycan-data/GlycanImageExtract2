import sys
import time
import json
import requests

from urllib.request import urlopen
from urllib.parse import urlencode


devemail="nje5+converter@georgetown.edu"

def request(target,**kw):
    baseurl = "https://glylookup.glyomics.org/"
    return json.loads(urlopen(baseurl+target,urlencode(kw).encode('utf8')).read())

# def request1(target,**kw):
#     baseurl = "https://subsumption.glyomics.org/"
#     return json.loads(urlopen(baseurl+target,urlencode(kw).encode('utf8')).read())

# def request1(target, task, developer_email):
#     params = {"task": task, "developer_email": developer_email}
#     print("task",task)
#     baseurl = "https://subsumption.glyomics.org/"
#     try:
#         response = urlopen(baseurl + target, urlencode(params).encode('utf8')) 
#         return json.loads(response.read())

#     except Exception as e:
#         print(f"Error during request: {e}")
#         return None


def request2(target,**kw):

    baseurl = "https://glymage.glyomics.org/"

    response = urlopen(baseurl+target,urlencode(kw).encode('utf8')).read()

    return json.loads(response)


def searchGlyImage(*seqs, orientation='RL',display='normal', delay=1, maxretry=10):

    baseurl = "https://glymage.glyomics.org/"
    
    tasks = [{"seq": seq.strip() if seq else "", "orientation": orientation, "display": display} for seq in seqs]

    data = request2("submit",tasks=json.dumps(tasks), developer_email=devemail, baseurl=baseurl)
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request2("retrieve",task_ids=json.dumps(jobids))
        print("task ids",json.dumps(jobids))
        done = True
        for job in data:
            if not job.get('finished'):
                done = False
                break
        if done:
            break
        if nretries >= maxretry:
            break
        time.sleep(delay)
        nretries += 1

    retval = []
    for job in data:
        retval.append(f"{baseurl}/{job['result']}")

    if len(seqs) == 1:
        return retval[0]
    return retval
    

# def sendToGNOme(*seqs):
#     seqparams = dict()
#     for i,seq in enumerate(seqs):
#         seqparams['Query'] = seq.strip()
#     params = dict(seqs=seqparams)
#     data = request1("submit",task=json.dumps(params),developer_email=devemail)
#     jobids = []
#     for job in data:
#         jobids.append(job["id"])
#     return jobids[-1]

# if IUPAC sequence is None - pass name
# change the name of this function
def searchGlyLookup(*seqs, delay=1, maxretry=10):
    params = []
    for seq in seqs:
        # param = dict(seq=seq.strip())
        param = dict(seq=seq.strip() if seq is not None else "")
        params.append(param)

    print("PARAMS",params)
    data = request("submit",tasks=json.dumps(params),developer_email=devemail)
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request("retrieve",task_ids=json.dumps(jobids))
        print("task ids",json.dumps(jobids) )
        done = True
        for job in data:
            if not job.get('finished'):
                done = False
                break
        if done:
            break
        if nretries >= maxretry:
            break
        time.sleep(delay)
        nretries += 1

    retval = []
    for job in data:
        result = None
        for res in job.get("result",[]):
            result = res['accession']
            break
        retval.append(result)

    if len(seqs) == 1:
        return retval[0]
    return retval

