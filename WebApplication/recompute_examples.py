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
for pat in patterns:
  for resultfile in sorted(glob.glob("static/examples/%s/results.json"%(pat,))):
    basedir = os.path.split(resultfile)[0]
    result = json.loads(open(resultfile).read())
    inputfilename = result['submission_detail']['original_file_name']
    inputpath = basedir+"/input/"+inputfilename
    mode = result['submission_detail']['submission_type']
    if mode == "Single-Glycan Image":
        mode = "Simple Glycan Image"
    exampledir = os.path.split(basedir)[1]
    tasks.append((exampledir,extractor.submit_file(mode,inputpath)))
    print("Example %s submitted (%s). "%(exampledir,tasks[-1][1]))
    time.sleep(1)

def update_votes(instance):
    result = json.loads(open("static/examples/"+instance+"/results.json").read())
    correct = json.loads(open("static/answers/"+instance+"/correct.json").read())
    if 'document_metadata' in correct["result"]:
        result["result"]["document_metadata"] = copy.deepcopy(correct["result"]["document_metadata"])
    correctcnt = 0; incorrectcnt = 0;
    for f1,f2 in zip(result["result"]["figure_result"],correct["result"]["figure_result"]):
        for k in ("caption_text","cleaned_caption","figure_name","label"):
            if f2.get(k):
                f1[k] = f2[k]
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
                incorrectcnt += 1
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
        print("Example %s done, %d/%d correct (%s)."%(exampledir,correct,total,taskid))
    else:
        print("Example %s not updated (%s)."%(exampledir,taskid))

