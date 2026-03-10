import sys
import time
import json
import requests

from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

default_dev_email="nje5+extractor@georgetown.edu"

def request_api(baseurl, target, **kwargs):
    """Generic function to call glyomics.org APIs"""
    url = baseurl + target
    # print(url,kwargs,file=sys.stderr)
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


def sendToGNOme(*seqs, baseurl=None, devemail=None):
    if not baseurl:
        baseurl = "https://subsumption.glyomics.org/"
    if not devemail:
        devemail = default_dev_email

    # get id (task_id) from subsumption
    tasks = [{"seq": seq.strip() if seq else ""} for seq in seqs]
    data = request_api(baseurl, "submit", tasks=json.dumps(tasks), developer_email=devemail)

    task_id = data[0]['id']
    # use Gnome Structure Browser
    return f"https://gnome.glyomics.org/StructureBrowser.html?ondemandtaskid={task_id}"


def searchGlyImage(*seqs, orientation='RL', display='normal', image_format='svg', accession=False, baseurl=None, devemail=None, delay=1, maxretry=10):
    if not baseurl:
        baseurl = "https://glymage.glyomics.org/"
    if not devemail:
        devemail = default_dev_email

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

    data = request_api(baseurl, "submit", tasks=json.dumps(tasks), developer_email=devemail)
    
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request_api(baseurl,"retrieve",task_ids=json.dumps(jobids))
        # print("glyimage task ids",json.dumps(jobids))
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

def searchGlyLookup(*seqs, baseurl=None, devemail=None, delay=1, maxretry=10):
    if not baseurl:
        baseurl = "https://glylookup.glyomics.org/"
    if not devemail:
        devemail = default_dev_email

    params = []
    for seq in seqs:
        # param = dict(seq=seq.strip())
        param = dict(seq=seq.strip() if seq is not None else "")
        params.append(param)

    # print("PARAMS",params)
    data = request_api(baseurl,"submit",tasks=json.dumps(params),developer_email=devemail)
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request_api(baseurl,"retrieve",task_ids=json.dumps(jobids))
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

# Duplicated code from GlyLookup, should probably use the glyomics client...
def searchSubsumption(seq, baseurl=None, devemail=None, delay=1, maxretry=10):
    if not baseurl:
        baseurl = "https://subsumption.glyomics.org/"
    if not devemail:
        devemail = default_dev_email

    params = [dict(seq=seq.strip())]
    
    # print("PARAMS",params)
    data = request_api(baseurl,"submit",tasks=json.dumps(params),developer_email=devemail)
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request_api(baseurl,"retrieve",task_ids=json.dumps(jobids))
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

    retval = None
    for job in data:
        # print(job,file=sys.stderr)
        subsumedby = []
        for k,vs in job['result']['relationship'].items():
            if k == 'Query':
                subsumes = list(vs)
            elif "Query" in vs:
                subsumedby.append(k)
        equiv=job['result']['equivalent'].get('Query',"")
        retval = dict(equivalent=equiv,
                      subsumes=subsumes,
                      subsumedby=subsumedby)
    
    return retval

