#!/bin/env python3.12

import random
import time
import sys
import socket
import queue
import secrets
import os
import os.path
import hashlib
import multiprocessing
import traceback
import threading
import subprocess
from collections import defaultdict

from multiprocessing.managers import SyncManager

class DistributedProcessing(object):
    def __init__(self,target=None,workerargs=None,host=None,port=None,secret=None):
        self.hostname = socket.gethostname()
        self.target = target
        if workerargs is not None:
            self.workerargs = workerargs
        else:
            self.workerargs = [ "worker", "%(ncpus)s", "%(server)s" ]
        if host:
            self.host = host
        else:
            self.host = self.hostname
        if port:
            self.port = port
        else:
            self.port = self.random_port(open(sys.argv[0]).read())
        if secret:
            self.secret = secret
        else:
            self.secret = self.random_secret(open(sys.argv[0]).read())

    @staticmethod
    def random_port(seed_string=None):
        if seed_string:
            seed = int(hashlib.sha256(seed_string.encode()).hexdigest(),16)
            random.seed(seed)
        return random.randint(60900,60999)

    @staticmethod
    def random_secret(seed_string=None):
        if seed_string:
            seed = int(hashlib.sha256(seed_string.encode()).hexdigest(),16)
            random.seed(seed)
        return ("".join([random.choice('0123456789abcdef') for _ in range(16)])).encode()

    def make_server_manager(self):

        class JobQueueManager(SyncManager):
            pass

        self._tasks =  queue.Queue()
        self._results = queue.Queue()
        self._workers = queue.Queue()
        self._manager = queue.Queue()
        self._data = dict()

        JobQueueManager.register('get_task_queue', callable=lambda: self._tasks)
        JobQueueManager.register('get_result_queue', callable=lambda: self._results)
        JobQueueManager.register('get_worker_queue', callable=lambda: self._workers)
        JobQueueManager.register('get_manager_queue', callable=lambda: self._manager)
        JobQueueManager.register('get_shared_data', callable=lambda: self._data)
        
        self.manager = JobQueueManager(address=("", self.port), authkey=self.secret)
        self.manager.start()
        print('Server started at port %s' % self.port, file=sys.stderr)
        self.tasks = self.manager.get_task_queue()
        self.results = self.manager.get_result_queue()
        self.worker_messages = self.manager.get_worker_queue()
        self.manager_messages = self.manager.get_manager_queue()
        self.shared_data = self.manager.get_shared_data()
        self.njobs = 0

    def make_client_manager(self):
        class ServerQueueManager(SyncManager):
            pass

        ServerQueueManager.register('get_task_queue')
        ServerQueueManager.register('get_result_queue')
        ServerQueueManager.register('get_worker_queue')
        ServerQueueManager.register('get_manager_queue')
        ServerQueueManager.register('get_shared_data')

        self.manager = ServerQueueManager(address=(self.host, self.port), authkey=self.secret)
        ntries = 4
        for i in range(ntries):
            try:
                self.manager.connect()
                break
            except IOError:
                if i == (ntries - 1):
                    raise
                print('Client failed to connect, attempt %d'%(i+1,),file=sys.stderr)
                time.sleep(5)
        print('Client connected to %s:%s' % (self.host, self.port), file=sys.stderr)
        self.tasks = self.manager.get_task_queue()
        self.results = self.manager.get_result_queue()
        self.worker_messages = self.manager.get_worker_queue()
        self.manager_messages = self.manager.get_manager_queue()
        self.shared_data = self.manager.get_shared_data()

    def start_workers(self,ncpus):
        self.procs = []
        pid = os.getpid()
        for i in range(ncpus):
            workerid = "%s:%s"%(pid,i+1)
            proc = multiprocessing.Process(target=self.worker,args=(workerid,))
            self.procs.append(proc)
            proc.start()

    def put_task(self,task_index,task):
        self.tasks.put(dict(task_index=task_index,task=task))

    def get_task(self):
        return self.tasks.get()

    def put_result(self,worker_index,task_index,result):
        self.results.put(dict(status="RESULT",hostname=self.hostname,worker_index=worker_index,
                              task_index=task_index,result=result))

    def put_error(self,worker_index,task_index,excep):
        self.results.put(dict(status="ERROR",hostname=self.hostname,worker_index=worker_index,
                              task_index=task_index,traceback=traceback.format_exception(excep)))

    def get_result(self,timeout=-1):
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            return None

    def do_task(self,task,**kwargs):
        if not self.target:
            raise NotImplemented("Neither target nor derived class do_task method defined.")
        return self.target(task,**kwargs)

    def init(self):
        return

    def heartbeat(self,worker_index):
        while True:
            self.worker_messages.put(("HEARTBEAT",worker_index))
            time.sleep(60)

    def worker(self,worker_index):
        self.worker_messages.put(("WORKERID",worker_index))
        t=threading.Thread(target=self.heartbeat,args=(worker_index,))
        t.daemon = True
        t.start()
        init_called = False
        while True:
            task = self.get_task()
            task_index = task['task_index']
            task = task['task']
            if task is None:
                break
            self.worker_messages.put(("TASKID",task_index,worker_index))
            if not init_called:
                self.init()
                init_called = True
            try:
                result = self.do_task(task,hostname=self.hostname,worker_index=worker_index,task_index=task_index,shared_data=self.shared_data)
            except Exception as excep:
                self.put_error(worker_index,task_index,excep)
            else:
                self.put_result(worker_index,task_index,result)
        return

    def wait_workers(self):
        for p in self.procs:
            p.join()

    def shutdown(self):
        self.manager.shutdown()

    def client(self,ncpus):
        assert ncpus > 0
        self.make_client_manager()
        self.start_workers(ncpus)
        self.wait_workers()

    def server(self):
        self.make_server_manager()
        return self

    def procspec(self,arg):
        procspec = {None: 0}
        for ps in arg.split(','):
            sps = ps.split(':')
            if len(sps) == 1:
                procspec[None] = int(sps[0])
            else:
                procspec[sps[0].strip()] = int(sps[1])
        for k,v in procspec.items():
            if not k:
                continue
            self.start_remote_workers(k,v)
        return procspec[None]

    def start_remote_workers(self,worker,ncpus):
        dirname,progname = os.path.split(os.path.abspath(sys.argv[0]))
        cmd = 'ssh %s nohup %s %s'%(worker,sys.executable,os.path.abspath(sys.argv[0]))
        for arg in self.workerargs:
            cmd += " "+arg%dict(server=self.hostname,ncpus=ncpus)
        cmd += " &"
        subprocess.run(cmd,shell=True,check=True)

    def execute(self,tasks,workers="",**shared_data):
        self.shared_data.update(shared_data)
        self.alltasks = list(tasks)
        self.start_workers(self.procspec(workers))
        return self

    def __iter__(self):
        return self.iterresults()

    def iterresults(self):

        for i,task in enumerate(self.alltasks):
            self.put_task(i+1,task)

        self.workerids = set()
        self.donetasks = set()
        self.taskattempts = defaultdict(int)
        self.failedtasks = set()
        self.heartbeat = defaultdict(float)
        self.task2worker = dict()
        while not self.tasks.empty() or (len(self.donetasks) + len(self.failedtasks)) < len(self.alltasks):
 
            while not self.worker_messages.empty():
                msg = self.worker_messages.get()
                # print(msg,file=sys.stderr)
                if msg[0] == "WORKERID":
                    self.workerids.add(msg[1])
                elif msg[0] == "TASKID":
                    self.task2worker[msg[1]] = msg[2]
                elif msg[0] == "HEARTBEAT":
                    self.heartbeat[msg[1]] = time.time()

            if self.tasks.empty():
                for i,task in enumerate(self.alltasks):
                    taskid = (i+1)
                    if taskid not in self.donetasks and taskid not in self.failedtasks:
                        if taskid not in self.task2worker or (time.time() - self.heartbeat[self.task2worker[taskid]]) > 120:
                            print("Reqeueing missing task %d..."%(taskid,),file=sys.stderr)
                            self.put_task(taskid,task)

            result = self.get_result(15)
            if result is None:
                continue

            status = result['status']
            if status == "ERROR":
                print("Worker %s:%s: Task %s error...\n%s"%(result.get('hostname'),result.get('worker_index'),
                                                            result.get('task_index'),"".join(result.get('traceback',[]))),
                                                            file=sys.stderr)
                taskid = result.get('task_index')
                self.taskattempts[taskid] += 1
                if self.taskattempts[taskid] < 3:
                    print("Reqeueing on error task %d..."%(taskid,),file=sys.stderr)
                    self.put_task(taskid,self.alltasks[taskid-1])
                else:
                    print("Failed task %d..."%(taskid,),file=sys.stderr)
                    self.failedtasks.add(taskid)
            elif status == "RESULT":
                taskid = result.get('task_index')
                if taskid not in self.donetasks:
                    self.donetasks.add(taskid)
                    yield result
            else:
                raise RuntimeError("Bad result status")

        print("Task summary: %s tasks completed, %s tasks failed."%(len(self.donetasks),len(self.failedtasks)),file=sys.stderr)

        while not self.worker_messages.empty():
            msg = self.worker_messages.get()
            if msg[0] == "WORKERID":
                self.workerids.add(msg[1])

        for i in range(len(self.workerids)):
            self.put_task(-1,None)
            
        self.wait_workers()  

        time.sleep(5)
        self.shutdown()

    def serial(self,tasks,**shared_data):
        self.shared_data = shared_data
        self.alltasks = list(tasks)
        self.iterresults = self.serialiterresults
        return self

    def serialiterresults(self):
        pid = os.getpid()
        workerid = "%s"%(pid,)
        self.init()
        for i,task in enumerate(self.alltasks):
            result = self.do_task(task,hostname=self.hostname,task_index=(i+1),worker_index=workerid,shared_data=self.shared_data)
            yield dict(status='RESULT',hostname=self.hostname,worker_index=workerid,task_index=(i+1),result=result)

def do_task(task,**kwargs):
    # print("Worker %s:%s: Task %s delay %s starting..."%(kwargs.get('hostname'),kwargs.get('worker_index'),
    #                                                     kwargs.get('task_index'),task),file=sys.stderr)
    # print("Shared data:",kwargs.get('shared_data'))
    time.sleep(task)
    assert(random.random() < 0.95)
    # print("Worker %s:%s: Task %s delay %s done..."%(kwargs.get('hostname'),kwargs.get('worker_index'),
    #                                                 kwargs.get('task_index'),task),file=sys.stderr)
    return task

def process_result(result,**kwargs):
    print("Result:",result,kwargs,file=sys.stderr)

if __name__ == "__main__":
    tasks = [ random.randint(0,20) for i in range(100) ]

    if len(sys.argv) <= 1:

        for result in DistributedProcessing(target=do_task).serial(tasks):
            process_result(**result)

    elif sys.argv[1] == "manager":

        p = DistributedProcessing(target=do_task,workerargs=["worker","%(ncpus)s","%(server)s"]).server()
        for result in p.execute(tasks,workers=sys.argv[2]):
            process_result(**result)
                        
    elif sys.argv[1] == "worker":
                        
        ncpus = int(sys.argv[2])
        DistributedProcessing(target=do_task,host=sys.argv[3]).client(ncpus)
