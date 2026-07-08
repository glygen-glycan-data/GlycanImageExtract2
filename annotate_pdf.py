#!.venv/bin/python
import os
import sys
import argparse
import logging
import fitz  # PyMuPDF
import shutil
import json
import time

from BKGlycanExtractor.glyomicsclient import *
from BKGlycanExtractor.bbox import BoundingBox, PDFBoundingBox, PDFConversionContext
from BKGlycanExtractor.image_manager import Image_Manager
from BKGlycanExtractor.compareboxes import *

parser = argparse.ArgumentParser(description="Annotate PDF")

# TODO - if url path is not valid for PDF - state a warning

MATCH_THRESHOLD = 0.3
USE_THRESHOLD = 0.8

parser.add_argument(
    '--pdf',
    type = str,
    nargs = "+",
    # required = True,
    help = 'PDF Manuscript(s), Accpets pdf paths and directories. Required.'
)

parser.add_argument(
    '--pmid',
    type = str,
    nargs = "+",
    # required = True,
    help = 'Pubmed id (Note that the Pubmed resources should be open access).'
)

parser.add_argument(
    '--json',
    type = str,
    nargs = "+",
    default = None,
    help = 'JSON format extractor result(s).'
)

parser.add_argument(
    '--taskid',
    type = str,
    default = None,
    nargs = "+",
    help = 'Task ID(s).'
)

parser.add_argument(
    '--extractorurl',
    type = str,
    default = 'https://extractor.glyomics.org/',
    help = 'Extractor URL.'
)

parser.add_argument(
    '--resubmit',
    action = 'store_true',
    default = False,
    help = 'Resubmit analysis, even if results JSON is present.'
)

parser.add_argument(
    '--manual',
    type = str,
    nargs = "+",
    default = None,
    help = 'Manual YOLO-format bbox file'
)

parser.add_argument(
    '--manual-page',
    type = int,
    default = None,
    help = 'input page number of png in the PDF'
)

parser.add_argument(
    '--manual-remove-border',
    action='store_true',
    default=False,
    help='Remove border from manual YOLO txt using the paired PNG size.'
)

parser.add_argument(
    '--manual-border-size',
    type=int,
    default=100,
    help='Border size in pixels to remove when --manual-remove-border is set.'
)

parser.add_argument(
    '--figure-match-iou-threshold',
    type=float,
    default=0.8,
    help='Minimum IoU for a manual box to count as matched when ranking figures.'
)

parser.add_argument(
    '--figure-match-min-ratio',
    type=float,
    default=0.2,
    help='Optional minimum matched_ratio required before applying a manual file to a figure.'
)
parser.add_argument(
    '--figure-match-min-matched',
    type=int,
    default=2,
    help='Minimum matched manual boxes required before applying a multi-box manual file to a figure.'
)

def normalize_manual_args(manual_args):
    if not manual_args:
        return manual_args

    normalized = []
    i = 0
    while i < len(manual_args):
        manual_file = manual_args[i]
        if os.path.basename(manual_file) == "classes.txt":
            i += 1
            continue
        if os.path.exists(manual_file):
            normalized.append(manual_file)
            i += 1
            continue

        rebuilt = manual_file
        recovered = False
        for j in range(i + 1, len(manual_args)):
            rebuilt += " " + manual_args[j]
            if os.path.basename(rebuilt) == "classes.txt":
                recovered = True
                i = j + 1
                break
            if os.path.exists(rebuilt):
                normalized.append(rebuilt)
                print(f"[MANUAL] recovered spaced path: {rebuilt}")
                recovered = True
                i = j + 1
                break

        if not recovered:
            print(f"[MANUAL] skip missing manual file: {manual_file}")
            i += 1

    return normalized

args = parser.parse_args()
args.manual = normalize_manual_args(args.manual)
if not 0.0 <= args.figure_match_iou_threshold <= 1.0:
    parser.error('--figure-match-iou-threshold must be between 0 and 1.')
if not 0.0 <= args.figure_match_min_ratio <= 1.0:
    parser.error('--figure-match-min-ratio must be between 0 and 1.')
if args.figure_match_min_matched < 1:
    parser.error('--figure-match-min-matched must be at least 1.')
page_number_manual = args.manual_page
print(
    "[RUN] annotate_pdf="
    f"{os.path.abspath(__file__)} "
    f"figure_match_iou_threshold={args.figure_match_iou_threshold:.3f} "
    f"figure_match_min_ratio={args.figure_match_min_ratio:.3f} "
    f"figure_match_min_matched={args.figure_match_min_matched} "
    f"manual_remove_border={args.manual_remove_border} "
    f"manual_border_size={args.manual_border_size} "
    f"manual_files={len(args.manual or [])}"
)
class InputItem:
    '''
    Stores the input item and its metadata (so that user can submit both pmid and pdf's at the same time via cmd line args)
    Note: can also be extended to support file_url's - API framework supports the upload_file request.

    This class helps in recognising the type of submission while using APIFramework/glyomics client.
    '''

    def __init__(self, input_type, value, index):
        # validate type
        if input_type not in ('pdf', 'pmid'):
            raise ValueError(f"Invalid type: {input_type}. Must be 'pdf' or 'pmid'")

        self.type = input_type
        self.value = value
        self.index = index

        # get basename only for pdf's
        if self.type == 'pdf':
            self.basename = os.path.splitext(value)[0]
        else:
            self.basename = f'PMID-{value}'

    def is_pdf(self):
        return self.type == 'pdf'

    def is_pmid(self):
        return self.type == 'pmid'

    def get_annotated_filename(self):
        '''Returns the expected annotated PDF filename for this input item.'''
        if self.type == 'pdf':
            return f'{self.basename}.annotated.pdf'
        elif self.type == 'pmid':
            return f'{self.basename}.annotated.pdf'
        return None

def build_input_items(pdf_list=None, pmid_list=None):
    '''Build InputItem list from PDF and PMID lists.'''
    
    input_items = []
    
    if pdf_list:
        if len(pdf_list) != len(set(pdf_list)):
            print("Provided PDF's are not unique")
            sys.exit(1)
        for pdf in pdf_list:
            input_items.append(InputItem('pdf', pdf, len(input_items)))
    
    if pmid_list:
        if len(pmid_list) != len(set(pmid_list)):
            print("Provided PMID's are not unique")
            sys.exit(1)
        for pmid in pmid_list:
            input_items.append(InputItem('pmid', pmid, len(input_items)))
    
    return input_items

# def bbox_xywh_to_xyxy(box):
#     x, y, w, h = box
#     return (x, y, x + w, y + h)

# def iou_xywh(a, b):
#     ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(a)
#     bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(b)

#     inter_x1 = max(ax1, bx1)
#     inter_y1 = max(ay1, by1)
#     inter_x2 = min(ax2, bx2)
#     inter_y2 = min(ay2, by2)

#     inter_w = max(0.0, inter_x2 - inter_x1)
#     inter_h = max(0.0, inter_y2 - inter_y1)
#     inter = inter_w * inter_h

#     area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
#     area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

#     union = area_a + area_b - inter
#     return 0.0 if union <= 0 else inter / union


def score_figure_by_manual_coverage(manual_boxes, pred_boxes, iou_threshold):
    total_manual_boxes = len(manual_boxes)
    matched_ious = []
    best_ious = []

    for manual_box in manual_boxes:
        best_iou = max(
            (CompareBoxes.iou(manual_box, pred_box) for pred_box in pred_boxes),
            default=0.0
        )
        best_ious.append(best_iou)
        if best_iou >= iou_threshold:
            matched_ious.append(best_iou)

    matched_manual_boxes = len(matched_ious)
    matched_ratio = (
        matched_manual_boxes / total_manual_boxes
        if total_manual_boxes else 0.0
    )
    mean_iou_of_matched_boxes = (
        sum(matched_ious) / matched_manual_boxes
        if matched_manual_boxes else 0.0
    )
    max_iou_of_manual_boxes = max(best_ious, default=0.0)
    mean_best_iou_of_manual_boxes = (
        sum(best_ious) / total_manual_boxes
        if total_manual_boxes else 0.0
    )

    return {
        "total_manual_boxes": total_manual_boxes,
        "matched_manual_boxes": matched_manual_boxes,
        "matched_ratio": matched_ratio,
        "mean_iou_of_matched_boxes": mean_iou_of_matched_boxes,
        "max_iou_of_manual_boxes": max_iou_of_manual_boxes,
        "mean_best_iou_of_manual_boxes": mean_best_iou_of_manual_boxes
    }

def build_manual_box(fig_w, fig_h, xc, yc, mw, mh):
    box = BoundingBox(
        image_width=fig_w,
        image_height=fig_h,
        rcx=xc, rcy=yc, rw=mw, rh=mh
    )
    box.normalize()
    return box

def get_png_size(png_path):
    from PIL import Image

    with Image.open(png_path) as img:
        return img.size

def load_manual_boxes(manual_file, remove_border=False, border=100, announce=True):
    manual_boxes = []

    if not os.path.exists(manual_file):
        if announce:
            print(f"[MANUAL] skip missing manual file: {manual_file}")
        return manual_boxes

    w_new = None
    h_new = None
    w_orig = None
    h_orig = None
    png_path = None

    if remove_border:
        png_path = os.path.splitext(manual_file)[0] + ".png"
        if not os.path.exists(png_path):
            if announce:
                print(
                    f"[MANUAL] skip {manual_file}: paired PNG not found for "
                    f"border removal (expected {png_path})"
                )
            return manual_boxes
        w_new, h_new = get_png_size(png_path)
        w_orig = w_new - 2 * border
        h_orig = h_new - 2 * border
        if w_orig <= 0 or h_orig <= 0:
            raise ValueError(
                f"Border {border}px is too large for image size {w_new}x{h_new}"
            )

    with open(manual_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xc, yc, w, h = parts
            xc, yc, w, h = map(float, (xc, yc, w, h))

            if remove_border:
                xc_abs = xc * w_new
                yc_abs = yc * h_new
                w_abs = w * w_new
                h_abs = h * h_new

                xc_abs -= border
                yc_abs -= border

                xc = xc_abs / w_orig
                yc = yc_abs / h_orig
                w = w_abs / w_orig
                h = h_abs / h_orig

            manual_boxes.append((xc, yc, w, h))

    if remove_border and announce:
        print(f"[MANUAL] removed {border}px border using {png_path}")

    return manual_boxes

def match_and_merge(manual_boxes, pred_boxes, pred_raw):

    pairs = []
    for i, m in enumerate(manual_boxes):
        for j, p in enumerate(pred_boxes):
            iou = CompareBoxes.iou(m, p)
            pairs.append((iou, i, j))

    pairs.sort(reverse=True)

    matched_m = set()
    matched_p = set()
    matches = []

    for iou, i, j in pairs:
        if iou < MATCH_THRESHOLD:
            continue
        if i in matched_m or j in matched_p:
            continue
        matches.append((i, j, iou))
        matched_m.add(i)
        matched_p.add(j)

    merged = []

    # 🔥 统计
    TP = 0
    FP = 0
    FN = 0

    # Case 2: matched
    for i, j, iou in matches:
        # if iou > USE_THRESHOLD:
        #     g = pred_raw[j].copy()
        #     g["source"] = "pred"
        #     merged.append(g)
        #     TP += 1
        # else:
        #     box = manual_boxes[i]
        #     merged.append({
        #         "bbox": box.bbox(),
        #         "confidence": 1.0,
        #         "source": "manual_low_iou"
        #     })
        #     FN += 1   # 👉 这里算 FN（prediction 不够好）
        if iou > USE_THRESHOLD:
            # 🔥 用 manual
            box = manual_boxes[i]
            merged.append({
                "bbox": box.bbox(),
                "confidence": 1.0,
                "source": "manual_high_iou"
            })
            TP += 1
        else: 
            merged.append({
                "bbox": manual_boxes[i].bbox(),
                "confidence": 1.0,
                "source": "manual_low_iou"
            })
            # 也保留 pred
            g = pred_raw[j].copy()
            g["source"] = "pred_low_iou"
            merged.append(g)
            FN += 1

    # Case 1: manual only
    for i in range(len(manual_boxes)):
        if i not in matched_m:
            box = manual_boxes[i]
            merged.append({
                "bbox": box.bbox(),
                "confidence": 1.0,
                "source": "manual_only"
            })
            FN += 1

    # Case 3: pred only
    for j in range(len(pred_boxes)):
        if j not in matched_p:
            g = pred_raw[j].copy()
            g["source"] = "pred_only"
            merged.append(g)
            FP += 1

    # 重新编号
    for idx, g in enumerate(merged):
        g["fig_glycan_count"] = idx + 1

    stats = {
        "TP": TP,
        "FP": FP,
        "FN": FN
    }

    return merged, stats
# GLOBLE
total_TP = 0
total_FP = 0
total_FN = 0
# draw manual YOLO boxes once per figure (if provided)
drew_manual_boxes = False

# Using Image Manager to gets paths of all pdf's from a directory 
# TODO Image_Manager class name - should probably be changed to File_Manager to make the class name sound more relevant, but the Image_Manager classname
# is being used in a couple of places, so need to make these updates in the all places
if args.pdf:
    pdf_manager = Image_Manager(args.pdf, pattern='*.pdf', exclude='*.annotated.pdf')
    args.pdf = pdf_manager.images

# Build unified input items list
input_items = build_input_items(args.pdf, args.pmid)

total_inputs = len(input_items)
if args.json is not None:
    assert len(args.json) == total_inputs, f"Number of JSON files ({len(args.json)}) must match number of inputs ({total_inputs})"
if args.taskid is not None:
    assert len(args.taskid) == total_inputs, f"Number of task IDs ({len(args.taskid)}) must match number of inputs ({total_inputs})"
if args.resubmit:
    assert not args.json, "Cannot use --resubmit with --json"
    assert not args.taskid, "Cannot use --resubmit with --taskid"

client = ExtractorClient(apiurl=args.extractorurl)

needsresults = set()
all_json_data = {}
resultfilename = {}

for i, item in enumerate(input_items):
    if item.is_pdf():
        if not os.path.exists(item.value):
            print(f"Error: PDF file not found: {item.value}")
            sys.exit(1)
            # assert os.path.exists(pdf)
    if item.is_pmid():
        # TODO: validate if pmid is Open Source, else skip and notify user
        pass
    
    if args.json:
        resultfilename[i] = args.json[i]
        assert os.path.exists(resultfilename[i])
    else:
        if item.is_pdf():
            resultfilename[i] = item.basename+".results.json"
        elif item.is_pmid():   # pmid
            resultfilename[i] = f"PMID-{item.value}.results.json"

    if not os.path.exists(resultfilename[i]) or args.resubmit:
        if args.taskid:
            taskid = args.taskid[i]
        elif item.is_pmid():
            print(f"PMID {item.value} submitted for analysis")
            taskid = client.submit_pmid(item.value, curation_task=True)
        else:
            print(os.path.split(item.value)[1],"PDF submitted for analysis.")
            taskid = client.submit_manuscript_file(item.value, curation_task=True)

        json_data = client.retrieve_once(taskid,asis=True)
        with open(resultfilename[i],'w') as f:
            json.dump(json_data,f,indent=2)
    else:
        with open(resultfilename[i], 'r') as f:
            json_data = json.load(f)
        if not json_data.get('finished',False):
            tmp_json_data = client.retrieve_once(json_data['id'],asis=True)
            if 'submission_detail' not in tmp_json_data:
                # result file has non-existent taskid
                if item.is_pmid():
                    print(f"PMID {item.value} resubmitted for analysis (bad taskid).")
                    taskid = client.submit_pmid(item.value)
                else:
                    print(os.path.split(item.value)[1],"resubmitted for analysis (bad taskid).")
                    taskid = client.submit_manuscript_file(item.value, curation_task=True)

                json_data = client.retrieve_once(taskid,asis=True)
                with open(resultfilename[i],'w') as f:
                    json.dump(json_data,f,indent=2)
    all_json_data[i] = json_data
    if not json_data.get('finished',False):
        needsresults.add(i)

completed = set()
while True:
    for i in sorted(needsresults):
        if i in completed:
            continue
        taskid = all_json_data[i]['id']
        json_data = client.retrieve_once(taskid,asis=True)
        input_item = json_data['submission_detail']['filename']
        if json_data.get('finished',False):
            all_json_data[i] = json_data
            completed.add(i)
            if json_data['state'] == "Complete":
                print(input_item,"analysis complete.")
            elif json_data['state'] == "Error":
                print(input_item,"analysis error.")
            # basename = os.path.splitext(args.pdf[i])[0]
            with open(resultfilename[i],'w') as wh:
                wh.write(json.dumps(json_data))
            print("Wrote results JSON:",resultfilename[i])
        else:
            if json_data.get('status'):
                print(input_item,"analysis in progress:",json_data['status'])
            elif json_data['state'] == "Running":
                print(input_item,"analysis in progress.")
            else:
                pass # print(pdf,"analysis queued.")
    if completed == needsresults:
        break
    time.sleep(15)

original_figure_glycans = {}
for i, input_item in enumerate(input_items):
    if all_json_data[i].get('state') == "Error":
        continue
    original_figure_glycans[i] = [
        [g.copy() for g in result.get("glycans", [])]
        for result in all_json_data[i]['result']['figures']
    ]

for manual_file in args.manual or []:

    manual_boxes = load_manual_boxes(
        manual_file,
        remove_border=args.manual_remove_border,
        border=args.manual_border_size
    )
    
    for i,input_item in enumerate(input_items):

        if all_json_data[i].get('state') == "Error":
            print(input_item.value,"skipping due to analysis error.")
            continue

        best = None
        if manual_boxes:
            # Score every figure, including figures with no prediction boxes.
            # candidates = []
            

            # for ridx, result in enumerate(all_json_data[i]['result']['figures']):
            #     pred_n = len(result.get("glycans", []))
            #     if pred_n == m and pred_n > 0:
            #         candidates.append(ridx)
            candidates = list(range(len(all_json_data[i]['result']['figures'])))
            figure_scores = []
            
            
            
            for ridx in candidates:
                result = all_json_data[i]['result']['figures'][ridx]
                pdf_context_instance = PDFConversionContext.from_result_dict(result)

                # predicted boxes in figure-pixel xywh: glycan["bbox"]
                # **************
                # pred_xywh = [g["bbox"] for g in result["glycans"]]
                fig_w = result["width"]
                fig_h = result["height"]
                pred_xywh = [
                    BoundingBox(
                        image_width=fig_w,
                        image_height=fig_h,
                        x=g["bbox"][0],
                        y=g["bbox"][1],
                        w=g["bbox"][2],
                        h=g["bbox"][3]
                    ) for g in original_figure_glycans[i][ridx]
                ]
                
                # manual_xywh = []
                # for (xc, yc, mw, mh) in manual_boxes:
                #     x0 = (xc - mw/2) * fig_w
                #     y0 = (yc - mh/2) * fig_h
                #     w  = mw * fig_w
                #     h  = mh * fig_h
                #     manual_xywh.append((x0, y0, w, h))
                
                # manual_xywh = [
                #     BoundingBox(
                #         image_width=fig_w,
                #         image_height=fig_h,
                #         rcx=xc, rcy=yc, rw=mw, rh=mh
                #     ).bbox()
                #     for (xc, yc, mw, mh) in manual_boxes
                # ]
                manual_xywh = [
                    build_manual_box(fig_w, fig_h, xc, yc, mw, mh)
                    for (xc, yc, mw, mh) in manual_boxes
                ]
                score = score_figure_by_manual_coverage(
                    manual_xywh,
                    pred_xywh,
                    args.figure_match_iou_threshold
                )
                score["figure_result_index"] = ridx
                score["page_number"] = result["page_number"]
                figure_scores.append(score)

                score_key = (
                    score["matched_ratio"],
                    score["mean_iou_of_matched_boxes"],
                    score["matched_manual_boxes"]
                )
                if best is None or score_key > best["score_key"]:
                    best = {**score, "score_key": score_key}

            rejected_best = None
            if best is not None and (
                best["matched_manual_boxes"] < min(args.figure_match_min_matched, best["total_manual_boxes"])
                or best["matched_ratio"] < args.figure_match_min_ratio
            ):
                required_matches = min(
                    args.figure_match_min_matched,
                    best["total_manual_boxes"]
                )
                if best["matched_manual_boxes"] < required_matches:
                    reason = (
                        f"matched_manual_boxes={best['matched_manual_boxes']} "
                        f"< {required_matches}"
                    )
                else:
                    reason = (
                        f"matched_ratio={best['matched_ratio']:.3f} "
                        f"< {args.figure_match_min_ratio:.3f}"
                    )
                print(f"[AUTO] No matching figure found for {manual_file}: {reason}.")
                rejected_best = {
                    key: value
                    for key, value in best.items()
                    if key != "score_key"
                }
                best = None

            best_matching_figure = None
            if best is not None:
                best_matching_figure = {
                    key: value
                    for key, value in best.items()
                    if key != "score_key"
                }
            print(json.dumps({
                "manual_file": manual_file,
                "figure_match_min_ratio": args.figure_match_min_ratio,
                "best_matching_figure": best_matching_figure,
                "rejected_best_figure": rejected_best,
                "figure_scores": figure_scores
            }, indent=2))

            best_ridx = best["figure_result_index"] if best else None
            if manual_boxes and best_ridx is not None:
                if (
                    best["matched_manual_boxes"] < min(args.figure_match_min_matched, best["total_manual_boxes"])
                    or best["matched_ratio"] < args.figure_match_min_ratio
                ):
                    required_matches = min(
                        args.figure_match_min_matched,
                        best["total_manual_boxes"]
                    )
                    print(
                        f"[AUTO] skip merge for {manual_file}: "
                        f"matched_manual_boxes={best['matched_manual_boxes']} "
                        f"required_matches={required_matches} "
                        f"matched_ratio={best['matched_ratio']:.3f}."
                    )
                    continue

                best_result = all_json_data[i]['result']['figures'][best_ridx]
                fig_w = best_result["width"]
                fig_h = best_result["height"]

                # new_glycans = []

                # for gid, (xc, yc, mw, mh) in enumerate(manual_boxes):

                #     x0 = (xc - mw/2) * fig_w
                #     y0 = (yc - mh/2) * fig_h
                #     w  = mw * fig_w
                #     h  = mh * fig_h

                #     new_glycans.append({
                #         "bbox": [x0, y0, w, h],
                #         "fig_glycan_count": gid + 1,
                #         "confidence": 1.0,
                #         "source": "manual"
                #     })
                # new_glycans = [
                #     {
                #         "bbox": BoundingBox(
                #             image_width=fig_w,
                #             image_height=fig_h,
                #             rcx=xc, rcy=yc, rw=mw, rh=mh
                #         ).bbox(),
                #         "fig_glycan_count": gid + 1,
                #         "confidence": 1.0,
                #         "source": "manual"
                #     }
                #     for gid, (xc, yc, mw, mh) in enumerate(manual_boxes)
                # ]

                # # 替换 prediction
                # best_result["glycans"] = new_glycans
                
                pred_raw = original_figure_glycans[i][best_ridx]

                pred_boxes = [
                    BoundingBox(
                        image_width=fig_w,
                        image_height=fig_h,
                        x=g["bbox"][0],
                        y=g["bbox"][1],
                        w=g["bbox"][2],
                        h=g["bbox"][3]
                    )
                    for g in pred_raw
                ]

                manual_bb = [
                    build_manual_box(fig_w, fig_h, xc, yc, mw, mh)
                    for (xc, yc, mw, mh) in manual_boxes
                ]

                merged_glycans, stats = match_and_merge(manual_bb, pred_boxes, pred_raw)
                total_TP += stats["TP"]
                total_FP += stats["FP"]
                total_FN += stats["FN"]
                best_result["glycans"] = merged_glycans
                
                print(f"[STATS] TP={stats['TP']} FP={stats['FP']} FN={stats['FN']}")
                best_result["glycans"] = merged_glycans

                print("[MANUAL] replaced predicted boxes with manual boxes")
                if best is not None:
                    print(
                        "[AUTO] manual matched to figure_result "
                        f"index={best['figure_result_index']} "
                        f"page={best['page_number']} "
                        f"matched_ratio={best['matched_ratio']:.3f} "
                        "mean_iou_of_matched_boxes="
                        f"{best['mean_iou_of_matched_boxes']:.3f}"
                    )
                    page_number_manual = best["page_number"]
                    
                else:
                    print("[AUTO] No matching figure found for manual boxes (count gating failed).")

manual_boxes = []
best_ridx = None

for i,input_item in enumerate(input_items):

    if all_json_data[i].get('state') == "Error":
        print(input_item.value,"skipping due to analysis error.")
        continue

    # doc = fitz.open(input_item.value)
    basename = input_item.basename
    annotated_path = basename + ".annotated.pdf"

    doc = fitz.open(input_item.value)

    # if os.path.exists(basename + ".annotated.pdf") or \
    #     os.path.exists(basename + ".annotated.tsv"):
    #     print(f'{basename}.pdf,skipping due to presence of output files.')
    #     continue

    image_data = []

    anyvotes = False

    for ridx, result in enumerate(all_json_data[i]['result']['figures']):        
        fig_num = result["image_count"]
        taskid = all_json_data[i]['id']

        # page_num - 1, because semantics counts page number starting from 1
        # but fitz accesses page numbers starting from 0
        page = doc[result["page_number"]-1]   

        # add figure boxes on the pdf with a fig: <fig_number> comment
        try:

            fig_annot = page.add_rect_annot(result["pdf_fig_bbox"])
            fig_annot.set_colors(stroke=(0, 0, 1)) 
            fig_annot.set_border(width=0.5) 
                        
            # set fig id
            content = (
                f"fig:{result['image_count']}\n"
            )
            xref = result.get("xref", None)
            if xref is not None and xref > 0:
                content += f"xref: {xref}\n"

            fig_annot.set_info(content=content)
            fig_annot.update()

            pdf_context_instance = PDFConversionContext.from_result_dict(result)

            for glycan in result["glycans"]:
                pdf_gly_box = pdf_context_instance.to_pdf_bbox(glycan["bbox"])
                gly_annot = page.add_rect_annot(pdf_gly_box.bbox())

                gid = f"G{fig_num}.{glycan['fig_glycan_count']}"
                url = client.url() + f"/result/{taskid}#glycan-{fig_num}-{glycan['fig_glycan_count']}"
                content = (
                    f"id: {gid}\n"
                    f"url: {url}\n"
                )

                gly_annot.set_info(content=content)
                source = glycan.get("source", "pred")
                print(
                source,
                glycan.get("fig_glycan_count")
            )
                # if source == "manual_high_iou":
                #     color = (1, 0, 0)      # 红
                # elif source == "manual_low_iou":
                #     color = (1, 1, 0)      # 黄
                # elif source == "manual_only":
                #     color = (1, 1, 0)      # 黄
                # elif source == "pred_low_iou":
                #     color = (1, 0.4, 0.8)      # 粉色
                #     width = 2
                # elif source == "pred_only":
                #     color = (1, 0.4, 0.8)      # 粉色
                #     width = 2
                # elif source == "pred":
                #     color = (0, 1, 0)
                # else:
                #     color = (0, 0, 1)
                if source.startswith("manual"):
                    color = (0, 1, 0)      # 绿色
                    width = 1

                elif source.startswith("pred"):
                    color = (1, 0, 0)      # 红色
                    width = 2

                else:
                    color = (0, 0, 1)      # 蓝色（异常情况）
                    width = 1
                gly_annot.set_colors(stroke=color)
                gly_annot.set_border(width=0.5) 
                gly_annot.update()

                votes = glycan.get('upvotes',0)-glycan.get('downvotes',0)
                if votes != 0:
                    anyvotes = True

                image_data.append({
                    "ID": gid,
                    "xref": result.get("xref"),
                    "page_num": result["page_number"],
                    "fig_num": fig_num,
                    "accession": glycan.get('accession', ''),
                    "iupac": glycan.get('IUPAC', ''),
                    "composition": glycan.get('composition_str', ''),
                    'wurcs': glycan.get('WURCS', ''),
                    'votes': votes,
                    "url": url,
                })

            # if (not drew_manual_boxes) and manual_boxes and result["page_number"] == page_number_manual:

            if manual_boxes and ridx == best_ridx:
                # print(
                #     "DRAW",
                #     manual_file,
                #     "best_ridx=",
                #     best_ridx,
                #     "ridx=",
                #     ridx,
                #     "page=",
                #     result["page_number"]
                # )
                fig_px_w = result["width"]
                fig_px_h = result["height"]

                for (xc, yc, mw, mh) in manual_boxes:
                    # --- BLUE: original manual box ---
                    x0_px = (xc - mw / 2) * fig_px_w
                    y0_px = (yc - mh / 2) * fig_px_h
                    w_px  = mw * fig_px_w
                    h_px  = mh * fig_px_h

                    # pdf_box_blue = pdf_context_instance.to_pdf_bbox((x0_px, y0_px, w_px, h_px))
                    # blue_annot = page.add_rect_annot(fitz.Rect(*pdf_box_blue.bbox()))
                    # blue_annot.set_colors(stroke=(0, 0, 1))
                    # blue_annot.set_border(width=1)
                    # blue_annot.set_info(content="manual box (original)")
                    # blue_annot.update()
                drew_manual_boxes = True

        except Exception as e:
            print(f"\nException occured while drawing bounding box on pdf: {e}")
    # doc.save(
    #     basename + ".annotated.pdf"
    #     # ,
    #     # incremental=True,
    #     # encryption=fitz.PDF_ENCRYPT_KEEP
    # )
    tmp_annotated_path = annotated_path + ".tmp"
    if os.path.exists(tmp_annotated_path):
        os.remove(tmp_annotated_path)
    doc.save(
        tmp_annotated_path,
        garbage=4,
        deflate=True
    )
    doc.close()
    os.replace(tmp_annotated_path, annotated_path)
    print("Wrote annotated PDF:",basename + ".annotated.pdf") 
    wh = open(basename + ".annotated.tsv",'w')
    headers = "ID xref page_num fig_num accession iupac composition wurcs votes url".split()
    if not anyvotes:
        headers.remove("votes")    
    print("\t".join(headers),file=wh)
    for row in image_data:
        print("\t".join(map(str,map(row.get,headers))),file=wh)
    wh.close()
    print("Wrote annotation table:",basename + ".annotated.tsv")

print("\n===== FINAL METRICS =====")
print(f"TP: {total_TP}")
print(f"FP: {total_FP}")
print(f"FN: {total_FN}")

# precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
# recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0

# print(f"Precision: {precision:.3f}")
# print(f"Recall: {recall:.3f}")
'''
Storing the XREF in the figures annotation - because XREF is a figure property and not an individual
annotations (eg. glycan) property.
Eg. for a figure with/without glycan annotations -  will still need xref (if present, so that the 
image can be extracted in its original format without having to specify a fixed dpi) information during
extract_annotations stage to save the image (or else a default dpi will be used) and a good place to store this information would be in the
figures annotations information itself.
The TSV file generated only stores information about the glycan (monos, root, links) annotations, so the xref can be tracked via
the figure annotation in the pdf and this ensures that the dimensions of the figure remain consistent during any extraction activity.
'''

# for ridx, result in enumerate(all_json_data[i]['result']['figures']):
#     cnt = sum(
#         1 for g in result["glycans"]
#         if "source" in g
#     )
#     print(ridx, result["page_number"], cnt)
