import json, time

from urllib.request import urlopen
from urllib.parse import urlencode


## class to submit a glycan description and return an accession
## all subclasses need a search method
class GlycanSearch:
    def __init__(self):
        pass
    def __call__(self,glycan):
        return self.search(glycan)
    ## method to request and read from url
    def request(self, target, **kwargs):
        return json.loads(urlopen(self.baseurl+target,urlencode(kwargs).encode('utf8')).read())
    def search(self, glycan):
        raise NotImplementedError

#class to search glycoCT to get accession
class SearchGlycoCT(GlycanSearch):
    def __init__(self):
        self.delay = 1
        self.maxretry = 10
        self.baseurl = "https://glylookup.glyomics.org/"
    #requires a glycoCT, returns an accession or else returns None
    def search(self,glycan):
        params = []
        #print(params)
        #print(glycan)
        param = dict(seq=str(glycan).strip())
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
            for res in job.get("result",[]):
                result = res['accession']
                break
            retval.append(result)
                                                                                                                            
        return retval[0]

#classs to send glycoCT description to GNOme
#should be subsequent to more preferred search methods    
class SendToGNOme(GlycanSearch):
    def __init__(self):
        self.baseurl = "https://subsumption.glyomics.org/"
        self.delay = 1
        self.maxretry = 10
    #requires glycoCT description, returns accession if found
    def search(self,glycan):
        #print(glycan)
        seqparams = dict()
        seqparams['Query'] = glycan.strip()
        params = dict(seqs=seqparams)
        data = self.request("submit",tasks=json.dumps([params]),developer_email = "mmv71@georgetown.edu")
        jobids = []
        #print(data)
        for job in data:
            #print(job)
            jobids.append(job["id"])
            
        nretries = 0
        while True:
            data = self.request("retrieve",list_ids=json.dumps(jobids))
            #print(data)
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
            result_dict = job.get("result",None)
            #print(result_dict)
            if result_dict is None:
                return None
            result = result_dict.get("equivalent",None)
            if result == {}:
                return None
            
            
            retval.append(result)
        #print(retval)
        if retval[0] is not None:
            return retval[0]["Query"]
        else:
            return None
        #return jobids[-1]
