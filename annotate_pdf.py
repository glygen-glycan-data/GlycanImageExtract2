#!.venv/bin/python
import os
import sys
import argparse
import logging
import fitz  # PyMuPDF
import shutil
import json


from BKGlycanExtractor.glyomicsclient import *
from BKGlycanExtractor.bbox import BoundingBox

parser = argparse.ArgumentParser(description="Start")

parser.add_argument(
    '--pdf',
    type = str,
    required = True,
    help = 'PDF Manuscript. Required.'
)

parser.add_argument(
    '--json',
    type = str,
    default = None,
    help = 'JSON format extractor result.'
)

parser.add_argument(
    '--taskid',
    type = str,
    default = None,
    help = 'Task ID.'
)

parser.add_argument(
    '--extractorurl',
    type = str,
    default = 'https://extractor.glyomics.org/',
    help = 'Extractor URL.'
)

args = parser.parse_args()
basename = os.path.splitext(args.pdf)[0]
client = ExtractorClient(apiurl=args.extractorurl)
if not args.json and not args.taskid:
    if os.path.exists(basename+".results.json"):
        with open(basename+".results.json", 'r') as f:
            json_data = json.load(f)
    else:
        json_data = client.analyze_manuscript_file(args.pdf)
elif args.taskid:
    json_data = client.retrieve(args.taskid)
elif args.json:
    with open(args.json, 'r') as file:
        json_data = json.load(file)

taskid = json_data['id']

figure_results = json_data['result']['figure_result']
basename = os.path.splitext(args.pdf)[0]

if not os.path.exists(basename+".results.json"):
    with open(basename+".results.json",'w') as wh:
        wh.write(json.dumps(json_data))
    print("Wrote results JSON:",basename+".results.json")

doc = fitz.open(args.pdf)

image_data = []

anyvotes = False

for result in figure_results:
    fig_num = result["figure_num"]

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

wh = open(os.path.join(basename + ".annotated.tsv"),'w')
headers = "ID xref page_num fig_num accession iupac composition wurcs votes url".split()
if not anyvotes:
    headers.remove("votes")    
print("\t".join(headers),file=wh)
for row in image_data:
    print("\t".join(map(str,map(row.get,headers))),file=wh)
wh.close()
print("Wrote annotation table:",basename + ".annotated.tsv")