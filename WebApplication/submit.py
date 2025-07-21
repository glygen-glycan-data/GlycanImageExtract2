import sys
import time
import json
import requests

from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

devemail="nje5+converter@georgetown.edu"

def request_api(baseurl, target, **kwargs):
    """Generic function to call glyomics.org APIs"""
    url = baseurl + target
    data = urlencode(kwargs).encode('utf8')
    attempts = 0
    while True:
        try:
            attempts += 1
            response = urlopen(url, data).read()
            break
        except HTTPError:
            if attempts > 5:
                raise 
        time.sleep(5)
    return json.loads(response)


def sendToGNOme(*seqs):

    # get id (task_id) from subsumption
    baseurl = "https://subsumption.glyomics.org/"
    tasks = [{"seq": seq.strip() if seq else ""} for seq in seqs]
    data = request_api(baseurl, "submit", tasks=json.dumps(tasks), developer_email=devemail)

    task_id = data[0]['id']
    # use Gnome Structure Browser
    return f"https://gnome.glyomics.org/StructureBrowser.html?ondemandtaskid={task_id}"


def searchGlyImage(*seqs, orientation='RL',display='normal', image_format='svg', accession=False ,delay=1, maxretry=10):

    baseurl = "https://glymage.glyomics.org/"
    
    key = 'acc' if accession else 'seq'

    tasks = [
        {
            key: s.strip() if s else "", 
            "orientation": orientation, 
            "display": display, 
            "image_format": image_format
        }
        for s in seqs
    ]

    data = request_api(baseurl, "submit", tasks=json.dumps(tasks),developer_email=devemail)
    
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request_api(baseurl,"retrieve",task_ids=json.dumps(jobids))
        print("glyimage task ids",json.dumps(jobids))
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



def searchGlyLookup(*seqs, delay=1, maxretry=10):
    baseurl = "https://glylookup.glyomics.org/"

    params = []
    for seq in seqs:
        # param = dict(seq=seq.strip())
        param = dict(seq=seq.strip() if seq is not None else "")
        params.append(param)

    # print("PARAMS",params)
    data = request_api(baseurl, "submit",tasks=json.dumps(params),developer_email=devemail)
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request_api(baseurl, "retrieve",task_ids=json.dumps(jobids))
        # print("task ids",json.dumps(jobids) )
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
        result = None,None
        for res in job.get("result",[]):
            wurcs = None
            for seqrec in res.get("sequences",[]):
                if seqrec['format'] == 'WURCS' and seqrec['source'].startswith('GlyTouCan:'):
                    wurcs = seqrec['seq']
                    break
            result = res['accession'],wurcs
            break
        retval.append(result)

    if len(seqs) == 1:
        return retval[0]
    return retval

