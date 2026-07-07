#!../.venv/bin/python
"""
Submit example jobs across multiple pipelines while keeping the same
submission-mode rotation as recompute_examples.py (Upload / Local / URL / PMID).

Outputs are saved under static/examples/{exampledir}.{pipeline_name}/ so
runs for different pipelines do not overwrite each other or the canonical
static/examples/{exampledir}/ tree.

sgi examples use --single_glycan_pipeline; mgi and mgp examples use
--multi_glycan_pipeline. Omit both to use each job class default pipeline
from the server (task_detail -> ini -> class default).

Example Usage (remember pipeline name prefix should always be either MultipleGlycanImage- or SingleGlycanImage- b/c pipeline name will be validated in processjob.py)
  python compare_pipeline_examples.py mgi1 mgi2

  python compare_pipeline_examples.py mgp mgi sgi \\
    --multi_glycan_pipeline MultipleGlycanImage-A MultipleGlycanImage-B \\
    --single_glycan_pipeline SingleGlycanImage-A SingleGlycanImage-B

  python compare_pipeline_examples.py

  python compare_pipeline_examples.py mgi
"""

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

extractor = ExtractorDevClient(port=10981, verbose=False)

SINGLE_GLYCAN_PREFIX = "SingleGlycanImage"
MULTI_GLYCAN_PREFIX = "MultipleGlycanImage"

SUBMISSION_PIPELINES = {
    "Simple Glycan Image": "single",
    "Multi-Glycan Image": "multi",
    "Manuscript": "multi",
}


def parse_pipeline_list(value):
    if not value:
        return []
    return [p for p in re.split(r"[\s\t]+", value.strip()) if p]


def normalize_pipelines(value):
    if not value:
        return []
    if isinstance(value, list):
        return list(value)
    return parse_pipeline_list(value)


def base_exampledir(exampledir):
    """Answer keys live under static/answers/mgi1/, not mgi1.PipelineName/."""
    return exampledir.split(".", 1)[0]


def is_canonical_example_dir(dirname):
    return "." not in dirname


EXACT_EXAMPLE = re.compile(r"^(sgi|mgi|mgp)\d+$", re.I)


def pattern_to_glob(pat):
    pat = pat.strip()
    if not pat or pat in ("*", "all"):
        return "*"
    if any(c in pat for c in "*?[]"):
        return pat
    if EXACT_EXAMPLE.match(pat):
        return pat
    return pat + "*"


def expand_patterns(patterns):
    expanded = []
    for pat in patterns:
        for part in re.split(r"[\s\t]+", pat.strip()):
            if part:
                expanded.append(part)
    return expanded if expanded else ["*"]


def collect_example_results(patterns):
    seen = set()
    resultfiles = []
    for pat in patterns:
        glob_pat = pattern_to_glob(pat)
        for resultfile in sorted(
            glob.glob("static/examples/%s/results.json" % glob_pat)
        ):
            basedir = os.path.split(resultfile)[0]
            exampledir = os.path.split(basedir)[1]
            if not is_canonical_example_dir(exampledir):
                continue
            if resultfile not in seen:
                seen.add(resultfile)
                resultfiles.append(resultfile)
    return sorted(resultfiles)


def pipelines_for_submission(submission_type, single_pipelines, multi_pipelines):
    kind = SUBMISSION_PIPELINES.get(submission_type)
    if kind == "single":
        return list(single_pipelines) if single_pipelines else [None]
    if kind == "multi":
        return list(multi_pipelines) if multi_pipelines else [None]
    return []


def pipeline_label(pipeline_name):
    return pipeline_name if pipeline_name else "(default)"


def resolved_pipeline_name(result):
    inner = result.get("result") or {}
    name = inner.get("pipeline_name") or result.get("pipeline_name")
    return name or "default"


def job_errors(result):
    errors = result.get("error")
    if errors is None:
        errors = (result.get("result") or {}).get("error", [])
    return errors or []


def job_succeeded(result):
    return result.get("finished", False) and len(job_errors(result)) == 0


def instance_dir(exampledir, pipeline_name):
    return f"{exampledir}.{pipeline_name}"


def figure_list(result_inner):
    if "figure_result" in result_inner:
        return result_inner["figure_result"]
    return result_inner.get("figures", [])


def submit_example(submission_type, submission_mode, pmid, inputpath, idx, pipeline_name):
    if idx == -1:
        aspdf = submission_mode == "PMID.PDF"
        return extractor.submit_pmid(
            submission_type, pmid, aspdf, pipeline_name=pipeline_name
        )
    if idx == 0:
        return extractor.submit_file(
            submission_type, inputpath, pipeline_name=pipeline_name
        )
    if idx == 1:
        return extractor.submit_local(
            submission_type, inputpath, pipeline_name=pipeline_name
        )
    url = extractor.makeurl(inputpath)
    return extractor.submit_url(submission_type, url, pipeline_name=pipeline_name)


def update_votes(instance, exampledir):
    result = json.loads(open("static/examples/" + instance + "/results.json").read())
    answers_dir = base_exampledir(exampledir)
    correct = json.loads(open("static/answers/" + answers_dir + "/correct.json").read())
    correctcnt = 0
    incorrectcnt = 0
    correctdetcnt = 0
    othercnt = 0
    for i, (f1, f2) in enumerate(
        zip(figure_list(result["result"]), figure_list(correct["result"]))
    ):
        for j, g1 in enumerate(f1["glycans"]):
            g1bb = BoundingBox(**dict(zip("xywh", g1["bbox"])))
            bestg2 = None
            bestiou = -1
            for k, g2 in enumerate(f2["glycans"]):
                g2bb = BoundingBox(**dict(zip("xywh", g2["bbox"])))
                iou = CompareBoxes.iou(g1bb, g2bb)
                if iou > 0.4 and iou > bestiou:
                    bestiou = iou
                    bestg2 = g2
                    bestk = k
            if not bestg2:
                incorrectcnt += 1
                print(
                    "Warning: No %s figure %s answer matches to glycan %d predicted box."
                    % (instance, i, j),
                    file=sys.stderr,
                )
                continue
            g2 = bestg2
            if g1.get("IUPAC"):
                if g1.get("IUPAC") == g2.get("IUPAC", "__XXXXXX__"):
                    g1["upvotes"] = 1
                    g1["downvotes"] = 0
                    correctcnt += 1
                elif g1.get("IUPAC") == g2.get("detpart_IUPAC", "__XXXXXX__"):
                    g1["upvotes"] = 2
                    g1["downvotes"] = 0
                    correctdetcnt += 1
                elif not g2.get("IUPAC") and not g2.get("detpart_IUPAC"):
                    print(
                        "Warning: No %s figure %s answer %s IUPAC available to compare predicted glycan %d IUPAC."
                        % (instance, i, bestk, j),
                        file=sys.stderr,
                    )
                    othercnt += 1
                else:
                    print(
                        "Warning: %s figure %s answer %s IUPAC does not match prediction %d IUPAC."
                        % (instance, i, bestk, j),
                        file=sys.stderr,
                    )
                    g1["upvotes"] = 0
                    g1["downvotes"] = 1
                    incorrectcnt += 1
            else:
                if g2.get("IUPAC") or g2.get("detpart_IUPAC"):
                    print(
                        "Warning: %s figure %s answer %s has IUPAC available but prediction %d does not."
                        % (instance, i, bestk, j),
                        file=sys.stderr,
                    )
                    g1["upvotes"] = 0
                    g1["downvotes"] = 1
                    incorrectcnt += 1
                elif g1.get("composition_str") == g2.get("composition_str", "__XXXXXX__"):
                    g1["upvotes"] = 1
                    g1["downvotes"] = 0
                    correctcnt += 1
                elif g1.get("composition_str") == g2.get(
                    "detpart_composition_str", "__XXXXXX__"
                ):
                    g1["upvotes"] = 2
                    g1["downvotes"] = 0
                    correctdetcnt += 1
                else:
                    g1["upvotes"] = 0
                    g1["downvotes"] = 1
                    incorrectcnt += 1
    with open("static/examples/" + instance + "/results.json", "wt") as wh:
        json.dump(result, wh, indent=2)
    return correctcnt, correctdetcnt, (correctcnt + incorrectcnt + correctdetcnt + othercnt)


def add_citation_captions(instance, exampledir):
    result = json.loads(open("static/examples/" + instance + "/results.json").read())
    answers_dir = base_exampledir(exampledir)
    correct = json.loads(open("static/answers/" + answers_dir + "/correct.json").read())

    for key in ("citation", "pmid"):
        if key in correct["result"]:
            result["result"][key] = correct["result"][key]

    for f1, f2 in zip(figure_list(result["result"]), figure_list(correct["result"])):
        for k in ("figure_number", "caption"):
            if f2.get(k):
                f1[k] = f2[k]

    with open("static/examples/" + instance + "/results.json", "wt") as wh:
        json.dump(result, wh, indent=2)


def remove_changable_fields(instance):
    result = json.loads(open("static/examples/" + instance + "/results.json").read())

    result["id"] = instance
    for k in list(result):
        if k in ("task_index", "sessionid") or k.endswith("time"):
            del result[k]

    result["submission_detail"]["id"] = instance
    for k in list(result["submission_detail"]):
        if k in ("task_index", "sessionid") or k.endswith("time"):
            del result["submission_detail"][k]

    result["location"] = "examples"

    for fn in glob.glob("static/examples/" + instance + "/annotated_files/*"):
        os.unlink(fn)
    for fn in glob.glob("static/examples/" + instance + "/output/*.txt"):
        os.unlink(fn)

    with open("static/examples/" + instance + "/results.json", "wt") as wh:
        json.dump(result, wh, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run examples through multiple pipelines without overwriting outputs."
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        default=["*"],
        help="Example selectors (space-separated): mgp, sgi, mgi1, all (default: all)",
    )
    parser.add_argument(
        "--single_glycan_pipeline",
        dest="single_glycan_pipeline",
        nargs="+",
        default=None,
        help="Optional SingleGlycanImage-* pipelines for sgi examples; omit to use server default",
    )
    parser.add_argument(
        "--multi_glycan_pipeline",
        dest="multi_glycan_pipeline",
        nargs="+",
        default=None,
        help="Optional MultipleGlycanImage-* pipelines for mgi/mgp examples; omit to use server default",
    )
    args = parser.parse_args()

    patterns = expand_patterns(args.patterns)

    single_pipelines = normalize_pipelines(args.single_glycan_pipeline)
    multi_pipelines = normalize_pipelines(args.multi_glycan_pipeline)

    for pipeline in single_pipelines:
        if not pipeline.startswith(SINGLE_GLYCAN_PREFIX):
            parser.error(
                "single-glycan pipeline %r must start with %r"
                % (pipeline, SINGLE_GLYCAN_PREFIX)
            )
    for pipeline in multi_pipelines:
        if not pipeline.startswith(MULTI_GLYCAN_PREFIX):
            parser.error(
                "multi-glycan pipeline %r must start with %r"
                % (pipeline, MULTI_GLYCAN_PREFIX)
            )

    tasks = []
    filetasks = 0

    for resultfile in collect_example_results(patterns):
        basedir = os.path.split(resultfile)[0]
        exampledir = base_exampledir(os.path.split(basedir)[1])
        result = json.loads(open(resultfile).read())
        submission_detail = result["submission_detail"]
        inputfilename = submission_detail.get("filename")
        if not inputfilename:
            print(
                "Skipping %s: no filename in submission_detail" % exampledir,
                file=sys.stderr,
            )
            continue
        inputpath = basedir + "/input/" + inputfilename
        submission_type = submission_detail["submission_type"]
        if submission_type == "Single-Glycan Image":
            submission_type = "Simple Glycan Image"

        submission_mode = submission_detail.get("submission_mode")
        pmid = submission_detail.get("pmid")

        applicable = pipelines_for_submission(
            submission_type, single_pipelines, multi_pipelines
        )
        if not applicable:
            print(
                "Skipping %s: no pipeline arg for submission type %r"
                % (exampledir, submission_type),
                file=sys.stderr,
            )
            continue

        if pmid:
            idx = -1
        else:
            idx = filetasks % 3

        for pipeline_name in applicable:
            taskid = submit_example(
                submission_type,
                submission_mode,
                pmid,
                inputpath,
                idx,
                pipeline_name,
            )
            tasks.append((exampledir, pipeline_name, taskid))
            if pipeline_name:
                out_hint = instance_dir(exampledir, pipeline_name)
            else:
                out_hint = instance_dir(exampledir, "<resolved-default>")
            print(
                "Example %s pipeline %s submitted (%s) -> static/examples/%s/"
                % (exampledir, pipeline_label(pipeline_name), taskid, out_hint)
            )
            time.sleep(1)

        if idx != -1:
            filetasks += 1

    if not tasks:
        print("No jobs submitted.", file=sys.stderr)
        print(
            "Patterns tried: %s" % ", ".join(patterns),
            file=sys.stderr,
        )
        return 1

    for exampledir, pipeline_name, taskid in tasks:
        result = {}
        try:
            result = extractor.retrieve(taskid)
        except APIUnfinishedError:
            pass
        resolved_pipeline = pipeline_name or resolved_pipeline_name(result)
        out_instance = instance_dir(exampledir, resolved_pipeline)
        if job_succeeded(result):
            srcdir = "static/files/" + taskid
            if not os.path.isdir(srcdir):
                print(
                    "Example %s pipeline %s not updated (%s): missing %s"
                    % (exampledir, pipeline_label(resolved_pipeline), taskid, srcdir),
                    file=sys.stderr,
                )
                continue
            outpath = "static/examples/" + out_instance
            if os.path.isdir(outpath):
                shutil.rmtree(outpath)
            shutil.copytree(srcdir, outpath)
            correct, correctdet, total = update_votes(out_instance, exampledir)
            add_citation_captions(out_instance, exampledir)
            remove_changable_fields(out_instance)
            print(
                "Example %s pipeline %s done, %d/%d correct, %d/%d detpart correct (%s)."
                % (
                    exampledir,
                    resolved_pipeline,
                    correct,
                    total,
                    correct + correctdet,
                    total,
                    taskid,
                )
            )
        else:
            print(
                "Example %s pipeline %s not updated (%s)."
                % (exampledir, pipeline_label(resolved_pipeline), taskid)
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
