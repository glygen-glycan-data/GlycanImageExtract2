import json, time

from urllib.request import urlopen
from urllib.parse import urlencode

class GlycanSearch:
    def __init__(self):
        pass
    def __call__(self,glycan):
        return self.search(glycan)
    def request(self, target, **kwargs):
        return json.loads(urlopen(self.baseurl+target,urlencode(kwargs).encode('utf8')).read())

class SearchGlycoCT(GlycanSearch):
    def __init__(self):
        self.delay = 1
        self.maxretry = 10
        self.baseurl = "https://glylookup.glyomics.org/"
    def search(self,glycan):
        params = []
        #print(params)
        #print(glycan)
        param = dict(seq=str(glycan).strip())
        # for seq in self.glycan:
        #     param = dict(seq=str(seq).strip())
        params.append(param)
        data = self.request("submit",tasks=json.dumps(params),developer_email="mmv71@georgetown.edu")
        #print(data)
        jobids = []
        for job in data:
            jobids.append(job["id"])

        nretries = 0
        while True:
            data = self.request("retrieve",list_ids=json.dumps(jobids))
            done = True
            for job in data:
                if not job.get('finished'):
                    done = False
                    break
            if done:
                break
            if nretries >= self.maxretry:
                break
            time.sleep(self.delay)
            nretries += 1

        retval = []
        for job in data:
            result = None
            for gtc in job.get("result",[]):
                result = gtc
                break
            retval.append(result)
        #print(retval)
        if len(glycan) == 1:
            return retval[0]
        if retval[0] is not None:
            return ''.join(retval)
        else:
            return None
    
class SendToGNOme(GlycanSearch):
    def __init__(self):
        self.baseurl = "https://subsumption.glyomics.org/"
    def search(self,glycan):
        seqparams = dict()
        for i,seq in enumerate(glycan):
            seqparams['Query'] = str(seq).strip()
        params = dict(seqs=seqparams)
        data = self.request("submit",tasks=json.dumps([params]),developer_email = "mmv71@georgetown.edu")
        jobids = []
        #print(data)
        for job in data:
            #print(job)
            jobids.append(job["id"])
        return jobids[-1]