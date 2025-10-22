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
from BKGlycanExtractor.bbox import BoundingBox

parser = argparse.ArgumentParser(description="Annotate PDF")

parser.add_argument(
    '--pdf',
    type = str,
    nargs = "+",
    required = True,
    help = 'PDF Manuscript(s). Required.'
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


args = parser.parse_args()

args.pdf = [ f for f in args.pdf if not f.endswith('.annotated.pdf') ]

if len(args.pdf) != len(set(args.pdf)):
    print("PDFs should be unique!",file=sys.stderr)
    sys.exit(1)

if args.json is not None:
    assert len(args.json) == len(args.pdf)
if args.taskid is not None:
    assert len(args.taskid) == len(args.pdf)
if args.resubmit:
    assert not args.json
    assert not args.taskid

client = ExtractorClient(apiurl=args.extractorurl)

needsresults = set()
all_json_data = {}
resultfilename = {}
for i,pdf in enumerate(args.pdf):
    assert os.path.exists(pdf)
    basename = os.path.splitext(pdf)[0]
    
    if args.json:
        resultfilename[i] = args.json[i]
        assert os.path.exists(resultfilename[i])
    else:
        resultfilename[i] = basename+".results.json"
    if not os.path.exists(resultfilename[i]) or args.resubmit:
        if args.taskid:
            taskid = args.taskid[i]
        else:
            print(os.path.split(pdf)[1],"submitted for analysis.")
            taskid = client.submit_manuscript_file(pdf)
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
                print(os.path.split(pdf)[1],"resubmitted for analysis (bad taskid).")
                taskid = client.submit_manuscript_file(pdf)
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
        pdf = json_data['submission_detail']['original_file_name']
        if json_data.get('finished',False):
            all_json_data[i] = json_data
            completed.add(i)
            if json_data['state'] == "Complete":
                print(pdf,"analysis complete.")
            elif json_data['state'] == "Error":
                print(pdf,"analysis error.")
            basename = os.path.splitext(args.pdf[i])[0]
            with open(resultfilename[i],'w') as wh:
                wh.write(json.dumps(json_data))
            print("Wrote results JSON:",resultfilename[i])
        else:
            if json_data.get('status'):
                print(pdf,"analysis in progress:",json_data['status'])
            elif json_data['state'] == "Running":
                print(pdf,"analysis in progress.")
            else:
                pass # print(pdf,"analysis queued.")
    if completed == needsresults:
        break
    time.sleep(15)

for i,pdf in enumerate(args.pdf):

    if all_json_data[i].get('state') == "Error":
        print(pdf,"skipping due to analysis error.")
        continue

    doc = fitz.open(pdf)
    basename = os.path.splitext(pdf)[0]

    if os.path.exists(basename + ".annotated.pdf") or \
        os.path.exists(basename + ".annotated.tsv"):
        print(pdf,"skipping due to presence of output files.")
        continue

    image_data = []

    anyvotes = False
    for result in all_json_data[i]['result']['figure_result']:
        fig_num = result["figure_num"]
        taskid = all_json_data[i]['id']

        # page_num - 1, because semantics counts page number starting from 1
        # but fitz accesses page numbers starting from 0
        page = doc[result["page_num"]-1]   

        fig_pdf_x, fig_pdf_y, fig_pdf_w, fig_pdf_h = result["pdf_fig_bbox"]

        fig_px_w = result["width"]
        fig_px_h = result["height"]

        # Since the original image dimensions might be scaled on the PDF page - we
        # need to make adjustments to map these image pixels wrt the page
        x_scale = fig_pdf_w / float(fig_px_w) 
        y_scale = fig_pdf_h / float(fig_px_h)

        for glycan in result["glycans"]:
            x0, y0, w, h = glycan["bbox"]

            x1 = x0 + w
            y1 = y0 + h

            pdf_x0 = fig_pdf_x + x0 * x_scale
            pdf_x1 = fig_pdf_x + x1 * x_scale
            pdf_y0 = fig_pdf_y + y0 * y_scale
            pdf_y1 = fig_pdf_y + y1 * y_scale

            gly_box = fitz.Rect((pdf_x0, pdf_y0, pdf_x1, pdf_y1))
            annot = page.add_rect_annot(gly_box)

            gid = f"G{fig_num}.{glycan['fig_glycan_count']}"
            url = client.url() + f"/result/{taskid}#glycan-{fig_num}-{glycan['fig_glycan_count']}"
            content = (
                f"id: {gid}\n"
                f"url: {url}\n"
            )

            annot.set_info(content=content)

            annot.update()

            votes = glycan.get('upvotes',0)-glycan.get('downvotes',0)
            if votes != 0:
                anyvotes = True

            image_data.append({
                "ID": gid,
                "xref": result["xref"],
                "page_num": result["page_num"],
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