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
glycan_idx = 1
page_idx = 0
image_counter = 0
good_image_counter = 0
for page_num, page in enumerate(doc):
    images = page.get_images(full=True)
    # print(page_num+1,images)
    if not images:
        continue
    for img in images:
        xref = img[0]
        image_name = img[7]

        # Get bounding box of the image on the page
        # Need another way to do this, we get an error if image_name is not unique...
        try:
            fig_bbox = page.get_image_bbox(image_name) 
        except:
            image_counter += 1
            continue

        # page.draw_rect(fig_bbox, color=(1, 0, 0), width=1)  # red border around image
        # fig_rect = fitz.Rect(fig_bbox.x0, fig_bbox.y0, 
        #                      fig_bbox.x0 + fig_bbox.width - 1, 
        #                      fig_bbox.y0 + fig_bbox.height -1)
        # annot = page.add_rect_annot(fig_rect)
        # annot.set_info(content=str(image_counter))
        # annot.update()

        # print(page_num+1,image_counter,img,fig_bbox.height,fig_bbox.width,fig_bbox.height * fig_bbox.width)
        
        # immitate the image counting in processjob - we need a more formal way to track each figure...
        if (fig_bbox.height <= 60 or fig_bbox.width <= 60) and (fig_bbox.height*fig_bbox.width <= 360):
            # image_counter += 1
            continue

        # Extract image info
        img_info = doc.extract_image(xref)
        # img_bytes = img_info["image"]

        img_width = img_info["width"]
        img_height = img_info["height"]

        # image_filename = f"page_{page_num+1}_img_{image_counter+1}.png"

        x_scale = fig_bbox.width / img_width
        y_scale = fig_bbox.height / img_height

        image_glycan_idx = 1
        if image_counter >= len(figure_results):
            image_counter += 1
            continue

        results = figure_results[image_counter]
        json_glycans = results['glycans']

        if len(json_glycans) == 0:
            image_counter += 1
            continue

        for g_id, glycan in enumerate(json_glycans):
            x0, y0, x1, y1 = BoundingBox(bbox=glycan['bbox']).corners()

            pdf_x0 = fig_bbox.x0 + x0 * x_scale
            pdf_x1 = fig_bbox.x0 + x1 * x_scale
            pdf_y0 = fig_bbox.y0 + y0 * y_scale
            pdf_y1 = fig_bbox.y0 + y1 * y_scale

            glycan_rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
            # page.draw_rect(glycan_rect, color=(0, 0, 1), width=1)  # blue box

            url = client.url() + f"/result/{taskid}#glycan-{image_counter}-{g_id+1}"
           
            annot = page.add_rect_annot(glycan_rect)

            gid = f"G{good_image_counter+1}.{image_glycan_idx}"
            content = (
                f"id: {gid}\n"
                f"url: {url}\n"
            )

            annot.set_info(content=content)

            annot.update()

            glycan_result = json_glycans[g_id]

            iupac = glycan_result.get('IUPAC','')
            composition = glycan_result.get('composition_str','')
            accession = glycan_result.get('accession','')
            wurcs = glycan_result.get('WURCS','')
            x,y,w,h = glycan_result.get('bbox','')
            votes = glycan_result.get('upvotes',0)-glycan_result.get('downvotes',0)
            if votes != 0:
                anyvotes = True

            image_data.append({
                "ID": gid,
                "image_index": image_counter,
                "page_number": page_num+1,
                "accession": accession,
                "iupac": iupac,
                "composition": composition,
                'wurcs': wurcs,
                'votes': votes,
                "url": url,
                "image_width": img_width,
                "image_height": img_height,
                "x": x,
                "y": y,
                "w": w,
                "h": h 
            })

            glycan_idx += 1
            image_glycan_idx += 1

        image_counter += 1
        good_image_counter += 1

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