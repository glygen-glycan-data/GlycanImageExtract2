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

for figure in figure_results:
    page_num = figure['image_page']
    page = doc[page_num]


    info = page.get_image_info(xrefs=True)
    # ensures that we get the correct figure bbox wrt to the entire pdf page
    bbox = next(i["bbox"] for i in info if i["xref"] == figure["xref"])
    fig_bbox = fitz.Rect(bbox)
    # page.add_rect_annot(fig_bbox).update()    # draw rect around entire figure for sanity check


    # figure pixel size used in semantics
    fig_px_w = figure["width"] 
    fig_px_h = figure["height"]   


    if (fig_bbox.height <= 60 or fig_bbox.width <= 60) and (fig_bbox.height*fig_bbox.width <= 360):
        continue

    # scales from figure image pixels -> displayed figure on page
    x_scale = fig_bbox.width  / float(fig_px_w)
    y_scale = fig_bbox.height / float(fig_px_h)

    
    for glycan in figure['glycans']:
        glycan_box = BoundingBox(bbox=glycan['bbox'])
        x0, y0, x1, y1 = glycan_box.corners()
        x,y,w,h = glycan_box.bbox()

        # normalize bbox ordering just in case
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))

        # map top-left-origin pixels -> page
        pdf_x0 = fig_bbox.x0 + x0 * x_scale
        pdf_x1 = fig_bbox.x0 + x1 * x_scale
        pdf_y0 = fig_bbox.y0 + y0 * y_scale
        pdf_y1 = fig_bbox.y0 + y1 * y_scale

        glycan_rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)

        annot = page.add_rect_annot(glycan_rect)

        gid = f"G{figure['figure_count']}.{glycan['fig_glycan_count']}"
        url = client.url() + f"/result/{taskid}#glycan-{figure['figure_count']}-{glycan['fig_glycan_count']}"
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
            "image_index": figure['figure_count'],
            "page_number": page_num,
            "accession": glycan.get('accession', ''),
            "iupac": glycan.get('IUPAC', ''),
            "composition": glycan.get('composition_str', ''),
            'wurcs': glycan.get('WURCS', ''),
            'votes': votes,
            "url": url,
            "image_width": fig_bbox.width,
            "image_height": fig_bbox.height,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })

doc.save(basename + ".annotated.pdf")
doc.close()
print("Wrote annotated PDF:",basename + ".annotated.pdf")   

wh = open(os.path.join(basename + ".annotated.tsv"),'w')
headers = "ID page_number composition iupac wurcs accession votes url image_index image_width image_height x y w h".split()
if not anyvotes:
    headers.remove("votes")    
print("\t".join(headers),file=wh)
for row in image_data:
    print("\t".join(map(str,map(row.get,headers))),file=wh)
wh.close()
print("Wrote annotation table:",basename + ".annotated.tsv")