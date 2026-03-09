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

parser = argparse.ArgumentParser(description="Annotate PDF")

# TODO - if url path is not valid for PDF - state a warning

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
    default = None,
    help = 'Manual YOLO-format bbox file'
)

parser.add_argument(
    '--manual-page',
    type = int,
    default = None,
    help = 'input page number of png in the PDF'
)
args = parser.parse_args()
page_number_manual = args.manual_page
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

def bbox_xywh_to_xyxy(box):
    x, y, w, h = box
    return (x, y, x + w, y + h)

def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = bbox_xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = bbox_xywh_to_xyxy(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union

def best_assignment_mean_iou(manual_xywh_list, pred_xywh_list):

    n = len(manual_xywh_list)
    if n == 0:
        return 0.0, []
    if len(pred_xywh_list) != n:
        return -1.0, []

    # precompute iou matrix
    iou_mat = [[iou_xywh(manual_xywh_list[i], pred_xywh_list[j]) for j in range(n)] for i in range(n)]

    # dp[mask] = best total iou for assigning first k manuals where k = popcount(mask)
    dp = [-1.0] * (1 << n)
    parent = [None] * (1 << n)
    dp[0] = 0.0

    for mask in range(1 << n):
        k = mask.bit_count()
        if k >= n:
            continue
        base = dp[mask]
        if base < 0:
            continue
        # assign manual k to some unused pred j
        for j in range(n):
            if not (mask & (1 << j)):
                nmask = mask | (1 << j)
                score = base + iou_mat[k][j]
                if score > dp[nmask]:
                    dp[nmask] = score
                    parent[nmask] = (mask, k, j)

    full = (1 << n) - 1
    total = dp[full]
    if total < 0:
        return -1.0, []

    # reconstruct
    assignment = []
    cur = full
    while cur != 0:
        pmask, mi, pj = parent[cur]
        assignment.append((mi, pj))
        cur = pmask
    assignment.reverse()
    return total / n, assignment


manual_boxes = []
if args.manual and os.path.exists(args.manual):
    with open(args.manual, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, w, h = map(float, parts)
            manual_boxes.append((xc, yc, w, h))
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
        input_item = json_data['submission_detail']['original_file_name']
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

for i,input_item in enumerate(input_items):

    if all_json_data[i].get('state') == "Error":
        print(input_item.value,"skipping due to analysis error.")
        continue

    # original_filepath = all_json_data[i]['result']['abs_original_filepath']
    # doc = fitz.open(original_filepath)
    
    best = None  # (best_score, best_result_index, best_page_number, best_assignment)
    if manual_boxes:
        m = len(manual_boxes)
        # 收集所有候选 figure：预测框数量==manual数量
        candidates = []
        for ridx, result in enumerate(all_json_data[i]['result']['figure_result']):
            pred_n = len(result.get("glycans", []))
            if pred_n == m and pred_n > 0:
                candidates.append(ridx)

        for ridx in candidates:
            result = all_json_data[i]['result']['figure_result'][ridx]
            pdf_context_instance = PDFConversionContext.from_result_dict(result)

            # predicted boxes in figure-pixel xywh: glycan["bbox"] (你现在就是这么用的)
            pred_xywh = [g["bbox"] for g in result["glycans"]]

            fig_w = result["width"]
            fig_h = result["height"]
            manual_xywh = []
            for (xc, yc, mw, mh) in manual_boxes:
                x0 = (xc - mw/2) * fig_w
                y0 = (yc - mh/2) * fig_h
                w  = mw * fig_w
                h  = mh * fig_h
                manual_xywh.append((x0, y0, w, h))

            mean_iou, assignment = best_assignment_mean_iou(manual_xywh, pred_xywh)

            if best is None or mean_iou > best[0]:
                best = (mean_iou, ridx, result["page_number"], assignment)

        if best is not None:
            print(f"[AUTO] manual matched to figure_result index={best[1]} page={best[2]} meanIoU={best[0]:.3f}")
            page_number_manual = best[2]   
            
        else:
            print("[AUTO] No matching figure found for manual boxes (count gating failed).")
    doc = fitz.open(input_item.value)
    basename = input_item.basename

    # if os.path.exists(basename + ".annotated.pdf") or \
    #     os.path.exists(basename + ".annotated.tsv"):
    #     print(f'{basename}.pdf,skipping due to presence of output files.')
    #     continue

    image_data = []

    anyvotes = False

    for result in all_json_data[i]['result']['figure_result']:
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
            # draw manual YOLO boxes once per figure (if provided)
            drew_manual_boxes = False
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
                gly_annot.set_border(width=0.5) 
                gly_annot.update()

            
            if (not drew_manual_boxes) and manual_boxes and result["page_number"] == page_number_manual:
                fig_px_w = result["width"]
                fig_px_h = result["height"]

                for (xc, yc, mw, mh) in manual_boxes:
                    # --- BLUE: original manual box ---
                    x0_px = (xc - mw / 2) * fig_px_w
                    y0_px = (yc - mh / 2) * fig_px_h
                    w_px  = mw * fig_px_w
                    h_px  = mh * fig_px_h

                    pdf_box_blue = pdf_context_instance.to_pdf_bbox((x0_px, y0_px, w_px, h_px))
                    blue_annot = page.add_rect_annot(fitz.Rect(*pdf_box_blue.bbox()))
                    blue_annot.set_colors(stroke=(0, 0, 1))
                    blue_annot.set_border(width=1)
                    blue_annot.set_info(content="manual box (original)")
                    blue_annot.update()
                drew_manual_boxes = True
    
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
                    # uncomment the below if we decide to use bounding box info from the semnatics -
                    # for now it is decided to use info that is extracted during the extarct_figures step
                    # using the fitz model to extract co-ordinates of the annotations on the pdf
                    # "_gly_bbox": glycan.get("bbox"),  
                    # "_fig_width": result['width'],     
                    # "_fig_height": result['height']    
                })

        except Exception as e:
            print(f"\nException occured while drawing bounding box on pdf: {e}")

    doc.save(basename + ".annotated.pdf")
    doc.close()
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