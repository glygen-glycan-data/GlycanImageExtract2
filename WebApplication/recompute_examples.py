#!../.venv/bin/python

import sys, os, glob, json, copy
import time, shutil
import requests

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path to import BKGlycanExtractor
sys.path.append(parent_dir)

from BKGlycanExtractor.compareboxes import CompareBoxes
from BKGlycanExtractor.bbox import BoundingBox

from BKGlycanExtractor.glyomicsclient import ExtractorDevClient, ExtractorClient, APIUnfinishedError

# Should get the port number from the ini file...
extractor = ExtractorDevClient()

patterns = ["*"]
if len(sys.argv) > 1:
    patterns = sys.argv[1:]

tasks = []
filetasks=0
pmidtasks=0
for pat in patterns:
  for resultfile in sorted(glob.glob("static/examples/%s/results.json"%(pat,))):
    basedir = os.path.split(resultfile)[0]
    result = json.loads(open(resultfile).read())
    submission_detail = result['submission_detail']
    inputfilename = submission_detail['filename']
    inputpath = basedir+"/input/"+inputfilename
    submission_type = submission_detail['submission_type']
    if submission_type == "Single-Glycan Image":
        submission_type = "Simple Glycan Image"

    submission_mode = submission_detail["submission_mode"]

    exampledir = os.path.split(basedir)[1]

    processor = submission_detail.get("processor")
    if not processor:
        print(f"Skip {exampledir}: missing processor in submission_detail", file=sys.stderr)
        continue
    
    pmid = None
    is_pmid = bool(submission_detail.get("pmid"))

    if is_pmid:
        pmid = submission_detail["pmid"]
        idx = pmidtasks % 3
    else:
        idx = filetasks % 3

    if idx == 0:
        # file / PMID 
        if is_pmid: 
            aspdf = submission_mode == "PMID-PDF"
            tasks.append((exampledir, extractor.submit_pmid(submission_type, pmid, aspdf)))
        else:
            tasks.append((exampledir, extractor.submit_file(submission_type, inputpath)))

    elif idx == 1:
        # Local
        if is_pmid:
            tasks.append((exampledir, extractor.submit_local(
                submission_type,
                inputpath,
                submission_mode=submission_detail['submission_mode'],   # provide the tasks original submission mode - so that reanalyze can make some helpful distinctions for PMID based jobs for Local submissions
                processor=processor,
                pmid=pmid,
            )))
        else:
            tasks.append((exampledir,extractor.submit_local(
                submission_type,
                inputpath,
                submission_mode='Local',
                processor=submission_detail['processor']
            )))

    else:
        # URL
        url = extractor.makeurl(inputpath)
        kwargs = {}
        if is_pmid:
            kwargs["pmid"] = pmid
        tasks.append((exampledir,extractor.submit_url(submission_type,url, **kwargs)))

    if is_pmid:
        pmidtasks += 1
    else:
        filetasks += 1
    print("Example %s submitted (%s). "%(exampledir,tasks[-1][1]))
    time.sleep(1)

def update_votes(instance):
    result = json.loads(open("static/examples/"+instance+"/results.json").read())
    correct = json.loads(open("static/answers/"+instance+"/correct.json").read())
    correctcnt = 0; incorrectcnt = 0;
    for i,(f1,f2) in enumerate(zip(result["result"]["figures"],correct["result"]["figures"])):
        for j,g1 in enumerate(f1["glycans"]):
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
                incorrectcnt += 1
                print("Warning: No %s figure %s answer matches to glycan %d prediction."%(instance,i,j),file=sys.stderr)
                continue
            g2 = bestg2
            if g1.get("IUPAC"):
                if g1.get("IUPAC") == g2.get("IUPAC","__XXXXXX__"):
                    g1['upvotes'] = 1; g1['downvotes'] = 0;
                    correctcnt += 1
                else:
                    g1['upvotes'] = 0; g1['downvotes'] = 1
                    incorrectcnt += 1
            else:
                if g1.get("composition_str") == g2.get("composition_str","__XXXXXX__"):
                    g1['upvotes'] = 1; g1['downvotes'] = 0;
                    correctcnt += 1
                else:
                    g1['upvotes'] = 0; g1['downvotes'] = 1
                    incorrectcnt += 1
    with open("static/examples/"+instance+"/results.json",'wt') as wh:
        json.dump(result,wh,indent=2)
    return correctcnt,(correctcnt+incorrectcnt)

def add_citation_captions(instance):
    # adds Citation, for each figure --> captions and figure_number
    result = json.loads(open("static/examples/"+instance+"/results.json").read())
    correct = json.loads(open("static/answers/"+instance+"/correct.json").read())

    for key in ("citation","pmid"):
        if key in correct["result"]:
            result["result"][key] = correct["result"][key]

    for f1,f2 in zip(result["result"]["figures"],correct["result"]["figures"]):
        for k in ('figure_number', 'caption'):
            if f2.get(k):
                f1[k] = f2[k]

    with open("static/examples/"+instance+"/results.json", 'wt') as wh:
        json.dump(result, wh, indent=2)
                
for exampledir,taskid in tasks:
    result = {}
    try:
        result = extractor.retrieve(taskid)
    except APIUnfinishedError:
        pass
    if result.get('finished',False):
        shutil.rmtree("static/examples/"+exampledir)
        shutil.copytree("static/files/"+taskid,
                        "static/examples/"+exampledir)
        correct,total = update_votes(exampledir)
        add_citation_captions(exampledir)
        print("Example %s done, %d/%d correct (%s)."%(exampledir,correct,total,taskid))
    else:
        print("Example %s not updated (%s)."%(exampledir,taskid))

