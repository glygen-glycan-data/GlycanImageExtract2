
__all__ = [ "ExtractorClient", "ExtractorDevClient", "GlyLookupClient" , "GlyLookupClient", "GlymageClient", "GnomeClient"]

import sys, os, glob, json, re
import requests, time
import traceback
from datetime import datetime

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
    request_interval = 5
    max_request_retry = 3
    nocache = False
    status_callback = None
    verbose = False

    def __init__(self,**kwargs):
        self._apiurl = kwargs.get('apiurl') or self.apiurl
        port = kwargs.get('port') or self.port
        if port is not None:
            self._apiurl += ":%s"%(port,)
        self._email = kwargs.get('developer_email') or self.developer_email
        
        self._nocache = kwargs.get('nocache',self.nocache) 
        self._max_retry = kwargs.get('max_request_retry',self.max_request_retry)
        self._interval = kwargs.get('request_interval',self.request_interval)
        self._max_retry_for_unfinished_task = kwargs.get('max_retrieve_wait',self.max_retrieve_wait)
        self._statusfn = kwargs.get('status_callback',self.status_callback)
        self._verbose = kwargs.get('verbose',self.verbose)
    
    def url(self):
        return self._apiurl

    def request(self, sub, params=None, files=None):
        if self._verbose:
            now = datetime.now()
            now.replace(microsecond=0)
            print(now,self.__class__.__name__,sub,params,file=sys.stderr)   
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
                if params1 or files1:
                    response = requests.post(
                        self._apiurl + "/" + sub, 
                        data=params1,   # form data (includes pmid from pmids dict)
                        files=files1    # files - pdf
                    )
                else:
                    response = requests.get(self._apiurl + "/" + sub)
            except Exception as e:
                # print("Exception occuered: ", e)
                pass
            finally:
                if files1 is not None:
                    dummy  = list(map(lambda fh: fh.close(),files1.values()))
            if response is not None:
                return response
            time.sleep(self._interval)
        raise APINoResponse

    def submit(self, *, task=None, tasks=[], request="submit", **kwargs):
    
        assert task or tasks, APISubmitError("No tasks submitted.")
        assert not task or not tasks, APISubmitError("Both single task and tasks submitted.")
    
        if task:
            param = {"task": json.dumps(task), "developer_email": self._email} 
            singletask = True
        else:
            param = {"tasks": json.dumps(tasks), "developer_email": self._email}    
            singletask = False
    
        if self._nocache:
            param["nocache"] = 'true'
            
        res = self.request(request, param, **kwargs)
        try:
            submit_result = res.json()
        except ValueError as e:
            raise APISubmitError(
                f"Invalid JSON response from API: {res.text[:500]}"
            ) from e
        try:
            if singletask:
                return submit_result[0]["id"]
            return [job["id"] for job in submit_result]
        except (IndexError,ValueError,TypeError,KeyError):
            pass
        except:
            traceback.print_exc()
        raise APISubmitError(submit_result)

    def retrieve(self, task_id):
        for i in range(self._max_retry_for_unfinished_task//self._interval + 1):
            time.sleep(self._interval)
            try:
                res = self.status(task_id)
                return res
            except APIUnfinishedError as e:
                if self._statusfn is not None:
                    self._statusfn(*e.args) 
                continue
            except KeyError:
                raise BadTaskIDError(task_id) from None
            except (APINoResponse,ValueError):
               continue
                
        raise APIUnfinishedError("The task %s is not finished yet" % task_id)

    def status(self,task_id):
        return self.retrieve_nowait(task_id)

    def retrieve_nowait(self, task_id, raise_unfinished=False):
        param = {"task_id": task_id }
        try:
            res2 = self.request("retrieve", param)
            if res2 is None:
                raise APINoResponse
            res2json = res2.json()[0]
        except:
            raise
        if not res2json.get("finished",False) and raise_unfinished:
            raise APIUnfinishedError(task_id,"NotComplete","The task %s is not finished yet" % task_id)
        return res2json

    def retrieve_many(self, *task_ids):
        taskid2index = {}
        for i,t in enumerate(task_ids):
            if t not in taskid2index:
                taskid2index[t] = []
            taskid2index[t].append(i)
        seen = set()
        while len(seen) < len(taskid2index):
            param = { "task_ids": json.dumps([ tid for tid in taskid2index if tid not in seen ]) }
            for i in range(self._max_retry_for_unfinished_task//self._interval + 1):
                time.sleep(self._interval)
                try:
                    res = self.request("retrieve", param)
                    resjson = res.json()
                except (ValueError,KeyError,APINoResponse):
                    # traceback.print_exc()
                    continue
                any = False
                for res in resjson:
                    if res.get('finished',False) and res['id'] not in seen:
                        for index in taskid2index[res['id']]:
                            yield index,res
                        any = True
                        seen.add(res['id'])
                if any:
                    break

        if len(seen) < len(taskid2index):
            taskid = [ tid for tid in taskid2index if tid not in seen ][0]
            raise APIUnfinishedError(task_id,"NotComplete","The task %s is not finished yet" % task_id)

    def getone(self, task, **kwargs):
        task_id = self.submit(task=task,**kwargs)
        resjson = self.retrieve(task_id)
        return resjson
    
    def getmany(self, tasks, **kwargs):
        task_ids = self.submit(tasks=tasks,**kwargs)
        for index,resjson in self.retrieve_many(*task_ids):
            yield index,resjson

    def getmany_inorder(self,tasks,**kwargs):
        task_ids = self.submit(tasks=tasks,**kwargs)
        nextindex = 0
        store = dict()
        for index,resjson in self.retrieve_many(*task_ids):
            if index == nextindex:
                yield index,resjson
                nextindex += 1
                while nextindex in store:
                    yield nextindex,store[nextindex]
                    del store[nextindex]
                    nextindex += 1
            else:
                store[index] = resjson

    def tolist(self,seqs):
        if len(seqs) == 1 and not isinstance(seqs[0],str):
            # detect iterable
            return list(seqs[0])
        return seqs
        
class GlyLookupClient(APIFrameworkClient):
    apiurl="https://glylookup.glyomics.org/"
    request_interval=1

    def getmany(self,seqs):
        tasks = [dict(seq=seq) for seq in seqs]
        for index,result in super().getmany(tasks):
            if len(result['result']) == 1:
                yield index,result['result'][0]
            else:
                yield index,{}

    def get_accessions(self,*seqs):
        seqs = self.tolist(seqs)
        for index,data in self.getmany(seqs):
            yield index,seqs[index],data.get('accession')
    
    def get_accession(self,seq):
        for index,seq,accession in self.get_accessions(seq):
            return accession

class GlymageClient(APIFrameworkClient):
    apiurl = 'https://glymage.glyomics.org/'
    # default_orientation = 'RL'
    display = 'normal'
    image_format = 'svg'
    use_accession = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._orientation = kwargs.get("orientation") or self.default_orientation
        self._display = kwargs.get("display") or self.display
        self._image_format = kwargs.get("image_format") or self.image_format
        self._use_accession = kwargs.get("use_accession") or self.use_accession

    def submit_glymage(self, *, acc=None, seq=None, **kwargs):
        # priority order - if accession (self._use_accession), iupac, composition

        if acc and seq:
            raise ValueError("Provide either acc or seq, not both.")

        task = {'orientation': kwargs.get('orientation') or 'RL', 
                    'display': self._display, 
                    'image_format': self._image_format,
                }
        if acc:
            task["acc"] = acc
        elif seq:
            task["seq"] = seq
        else:
            raise ValueError("Provide: acc or seq")

        return self.submit(task=task)
    
    
    # def getmany(self, seqs, image_orientations=[]):
    #     # every sequence can request a specific orientation for the image

    #     if len(seqs) != len(image_orientations):
    #         # use the default orientation
    #         image_orientations = [self._orientation] * len(seqs)
        
    #     # build tasks for submission
    #     key = 'acc' if self._use_accession else 'seq'
    #     tasks = [
    #         {key: seq, 'orientation': image_orientations[idx], 'display': self._display, 'image_format': self._image_format}
    #         for idx, seq in enumerate(seqs)
    #     ]

    #     # submits and retrieves many
    #     for index,result in super().getmany(tasks):
    #         if len(result['result']) == 1:
    #             yield index,result['result'][0]
    #         else:
    #             yield index,{}
    
    # def get_glymages(self, *seqs, image_orientations=[]):
    #     seqs = self.tolist(seqs)
    #     for index, data in self.getmany(seqs, image_orientations=image_orientations):
    #         yield index, seqs[index], data

    # def get_glymage(self, seq, orientation=None):
    #     orientation = [] if orientation is None else [orientation]
    #     for index, data in self.getmany([seq], image_orientations=orientation):
    #         return data

class SubsumptionClient(APIFrameworkClient):
    apiurl = 'https://subsumption.glyomics.org/'
    gnomeurl = 'https://gnome.glyomics.org/'

    def get_gnome_url(self,*,seq=None,acc=None,compositionstr=None,**composition):
        if acc:
            return self.gnomeurl + 'StructureBrowser.html?focus=' + acc
        if seq:
            taskid = self.submit(task=dict(seq=seq))
            return self.gnomeurl + 'StructureBrowser.html?ondemandtaskid=' + taskid
        if compositionstr:
            matches = re.findall(r'([A-Za-z]+)\((\d+)\)', compositionstr)
            converted_composition = '&'.join(f"{name}={count}" for name, count in matches)
        else:
            converted_composition = '&'.join(f"{name}={count}" for name, count in composition)
        return self.gnomeurl + 'StructureBrowser.html?' + converted_composition

class ExtractorClient(APIFrameworkClient):
    request_interval=5
    max_retrieve_wait = 1200
    apiurl="https://extractor.glyomics.org"

    def makeurl(self,path):
        return self.url() + '/' + path.lstrip('/')
    
    def status(self, task_id):
        res = self.request("job_status/"+task_id).json()
        if not res[u"finished"]:
            raise APIUnfinishedError(task_id,res["state"],res["status"])
        return self.retrieve_nowait(task_id)
    
    def submit(self, **kwargs):
        assert not kwargs.get('tasks'), "ExtractorClient requires single task per submission"
        return super().submit(**kwargs)
    
    def submit_pmid(self,submission_type,pmid,aspdf=False):
        assert submission_type in ("Manuscript",)
        if aspdf:
            pmid = str(pmid) + ".pdf"
        task = dict(submission_type=submission_type,pmid=pmid)
        return self.submit(task=task,request="file_upload")
    
    def submit_local(self,submission_type,filepath):
        assert submission_type in ("Manuscript",
                                   "Multi-Glycan Image",
                                   "Simple Glycan Image")
        task = dict(submission_type=submission_type,filePath=filepath)
        return self.submit(task=task,request="file_upload")
    
    def submit_url(self,submission_type,url):
        assert submission_type in ("Manuscript",
                        "Multi-Glycan Image",
                        "Simple Glycan Image")
        task = dict(submission_type=submission_type,fileURL=url)
        return self.submit(task=task,request="file_upload")
    
    def submit_file(self,submission_type,filename):
        assert submission_type in ("Manuscript",
                        "Multi-Glycan Image",
                        "Simple Glycan Image")
        task = dict(submission_type=submission_type)
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
    port = 10981

if __name__ == "__main__":

    import sys

    glylookup = GlyLookupClient()
    acc = glylookup.get_accession(sys.argv[1])
    print(sys.argv[1],acc)
    for ind,seq,acc in glylookup.get_accessions(sys.argv[1:]):
        print(ind,seq,acc)