#!../.venv/bin/python

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from BKGlycanExtractor.compareboxes import CompareBoxes
from BKGlycanExtractor.bbox import BoundingBox
from BKGlycanExtractor.glyomicsclient import ExtractorDevClient, APIUnfinishedError

client = ExtractorDevClient(port=10982, verbose=False)

def main():
    parser = argparse.ArgumentParser(
        description="Submit to specific pipelines and with specific parameters."
    )
    parser.add_argument(
        "--manuscript",
        nargs="*",
        default=[],
        help="Manuscripts to submit",
    )
    parser.add_argument(
        "--multiglycan",
        nargs="*",
        default=[],
        help="Multiglycan images to submit",
    )
    parser.add_argument(
        "--simpleglycan",
        nargs="*",
        default=[],
        help="Simple glycan images to submit",
    )
    parser.add_argument(
        "--pipeline",
        nargs="*",
        default=[],
        help="Pipline to run",
    )
    parser.add_argument(
        "--figure_extraction",
        nargs="*",
        default=[],
        help="Figure extraction strategy",
    )
    args = parser.parse_args()

    assert len(args.manuscript) > 0 or len(args.multiglycan) > 0 or len(args.simpleglycan) > 0
    assert len(args.manuscript) + len(args.multiglycan) + len(args.simpleglycan) == max(len(args.manuscript),len(args.multiglycan),len(args.simpleglycan))
    
    if len(args.simpleglycan) > 0:
        submit_type = 'Simple Glycan Image'
        if len(args.pipeline) == 0:
            args.pipeline = [ "SingleGlycanImage-YOLOFinders" ]
        else:
            for p in args.pipeline:
                assert p.startswith("SingleGlycanImage-")
    elif len(args.multiglycan) > 0:
        submit_type = 'Multi-Glycan Image'
        if len(args.pipeline) == 0:
            args.pipeline = [ "MultipleGlycanImage-YOLOFindersV4" ]
        else:
            for p in args.pipeline:
                assert p.startswith("MultipleGlycanImage-")
    else:
        submit_type = "Manuscript"
        if len(args.pipeline) == 0:
            args.pipeline = [ "MultipleGlycanImage-YOLOFindersV4" ]
        else:
            for p in args.pipeline:
                assert p.startswith("MultipleGlycanImage-")

    if len(args.figure_extraction) == 0:
        args.figure_extraction = ['hybrid']

    items = args.manuscript + args.multiglycan + args.simpleglycan
    tasks = dict()
    for it in items:
        for pipeline in args.pipeline:
            for figex in args.figure_extraction:
                if os.path.exists(it):
                    taskid = client.submit_file(submit_type,it,
                                    pipeline_name=pipeline,
                                    image_search_strategy=figex)
                elif re.search(r'^\d+$',it):
                    taskid = client.submit_pmid(submit_type,it,
                                        pipeline_name=pipeline,
                                        image_search_strategy=figex)
                elif re.search(r'^\d+\.pdf$',it.lower()):
                    taskid = client.submit_pmid(submit_type,it.rsplit('.',1)[0],aspdf=True,
                                        pipeline_name=pipeline,
                                        image_search_strategy=figex)
                elif re.search(r'^http'):
                    taskid = client.submit_url(submit_type,it,
                                        pipeline_name=pipeline,
                                        image_search_strategy=figex)
                tasks[taskid] = dict(id=taskid,query=it,pipeline=pipeline,figure_extraction=figex)
                print(f"Task {taskid}: Submitted {it} ({pipeline},{figex}).")

    for tid in tasks:
        while True:
            try:
                result = client.retrieve(tid)
            except APIUnfinishedError:
                pass
            if result.get('finished',False):
                if len(result.get('error',[])) == 0:
                    print(f"Task {tid}: Completed {tasks[tid]["query"]} ({tasks[tid]["pipeline"]},{tasks[tid]["figure_extraction"]}).")
                else:
                    print(f"Task {tid}: Error {tasks[tid]["query"]} ({tasks[tid]["pipeline"]},{tasks[tid]["figure_extraction"]}).")
                break

if __name__ == "__main__":
    sys.exit(main())
