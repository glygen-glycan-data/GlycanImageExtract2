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

def request1(target, task, developer_email):
    params = {"task": task, "developer_email": developer_email}
    print("task",task)
    baseurl = "https://subsumption.glyomics.org/"
    try:
        response = urlopen(baseurl + target, urlencode(params).encode('utf8')) 
        return json.loads(response.read())

    except Exception as e:
        print(f"Error during request: {e}")
        return None

def sendToGNOme(*seqs):
    seqparams = dict()
    for i,seq in enumerate(seqs):
        seqparams['Query'] = seq.strip()
    params = dict(seqs=seqparams)
    data = request1("submit",task=json.dumps(params),developer_email=devemail)
    jobids = []
    for job in data:
        jobids.append(job["id"])
    return jobids[-1]

def searchGlycoCTnew(*seqs, delay=1, maxretry=10):
    params = []
    for seq in seqs:
        param = dict(seq=seq.strip())
        params.append(param)

    print("PARAMS",params)
    data = request("submit",tasks=json.dumps(params),developer_email=devemail)
    jobids = []
    for job in data:
        jobids.append(job["id"])

    nretries = 0
    while True:
        data = request("retrieve",task_ids=json.dumps(jobids))
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

def searchGlycoCT(seq, delay=0.5, maxretry=1):
    
    if type(seq)!=list:
        seq=[seq]
    main_url = "https://edwardslab.bmcb.georgetown.edu/glylookup/"
    # main_url = "http://localhost:10986/"

    params = {
        "q": json.dumps(seq),
    }
    #print()
    try:
        response1 = requests.post(main_url + "submit", params)
        response_json = response1.json()
        task_ids = list(response_json.values())
    except Exception as e:
        sys.stdout.write("Error: has issue connecting to flask API.")
        sys.stdout.write(str(e))
        sys.exit()

    # the list id for retrieval later
    #print (task_ids,"task_ids here")
    #print ("\n" * 3)

    params = {"q": json.dumps(task_ids)}

    # might require larger wait time in case you send a huge amount of GlycoCTs
    for i in range(maxretry):

        time.sleep(delay)

        response2 = requests.post(main_url+ "retrieve", params)
        results = response2.json()
        #print("results here",results)

        for task_id, res in results.items():
            #print(f"task_id:{task_id}")
            #print(res['result']['error'])
            #print("res:", res["result"]['hits'])
            if res['finished']==True and res["result"]['hits']!=[]:
                accession=res['result']['hits'][0]['accession']
                return accession
            else:
                continue
            #print(f"hits:{res['result']['hits'][0]}")
            #print(f"entire resposne:{res['result']}")
            #for k, v in res.items():
            #    print(f"k:{k} v:{v}")
            #print ("- " * 20)

    return "Not found"


if __name__ == "__main__":
    glycoct = """RES
    1b:x-dgro-dgal-NON-2:6|1:a|2:keto|3:d
    2s:n-acetyl
    3b:x-dglc-HEX-1:5
    4s:n-acetyl
    5b:x-lgal-HEX-1:5|6:d
    6b:x-dglc-HEX-1:5
    7s:n-acetyl
    8b:x-dglc-HEX-1:5
    LIN
    1:1d(5+1)2n
    2:1o(-1+1)3d
    3:3d(2+1)4n
    4:3o(-1+1)5d
    5:3o(-1+1)6d
    6:6d(2+1)7n
    7:1o(-1+1)8d
    """

    glycoct = """
RES
1b:x-dglc-HEX-1:5
2s:n-acetyl
3b:x-dglc-HEX-1:5
4s:n-acetyl
5b:x-dman-HEX-1:5
6b:x-dman-HEX-1:5
7b:x-dglc-HEX-1:5
8s:n-acetyl
9b:x-dgal-HEX-1:5
10b:x-dgro-dgal-NON-2:6|1:a|2:keto|3:d
11s:n-acetyl
12b:x-dglc-HEX-1:5
13s:n-acetyl
14b:x-dgal-HEX-1:5
15b:x-dgro-dgal-NON-2:6|1:a|2:keto|3:d
16s:n-acetyl
17b:x-dman-HEX-1:5
18b:x-dglc-HEX-1:5
19s:n-acetyl
20b:x-dgal-HEX-1:5
21b:x-dgro-dgal-NON-2:6|1:a|2:keto|3:d
22s:n-acetyl
23b:x-lgal-HEX-1:5|6:d
24b:x-dglc-HEX-1:5
25s:n-acetyl
26b:x-lgal-HEX-1:5|6:d
LIN
1:1d(2+1)2n
2:1o(-1+1)3d
3:3d(2+1)4n
4:3o(-1+1)5d
5:5o(-1+1)6d
6:6o(-1+1)7d
7:7d(2+1)8n
8:7o(-1+1)9d
9:9o(-1+2)10d
10:10d(5+1)11n
11:6o(-1+1)12d
12:12d(2+1)13n
13:12o(-1+1)14d
14:14o(-1+2)15d
15:15d(5+1)16n
16:5o(-1+1)17d
17:17o(-1+1)18d
18:18d(2+1)19n
19:18o(-1+1)20d
20:20o(-1+2)21d
21:21d(5+1)22n
22:18o(-1+1)23d
23:17o(-1+1)24d
24:24d(2+1)25n
25:1o(-1+1)26d
"""


    print(*searchGlycoCTnew(glycoct))
