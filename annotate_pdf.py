#!.venv/bin/python
import os
import sys
import argparse
import logging
import fitz  # PyMuPDF
import shutil
import pandas as pd
import json


from BKGlycanExtractor import Image_Manager, Config_Manager
from BKGlycanExtractor.distproc import DistributedProcessing as dp
from BKGlycanExtractor.glycanfinding import KnownGlycanBoxes


script_dir = os.path.dirname(os.path.realpath(__file__))

# Go one level up (GlycanImageExtract2)
parent_dir = os.path.abspath(os.path.join(script_dir, '.'))

external_path = os.path.join(parent_dir, 'WebApplication')
sys.path.append(external_path)

 
parser = argparse.ArgumentParser(description="Start")

parser.add_argument(
    '--pipeline',
    type = str,
    required = True,
    help = 'Pipeline to execute on images. Required.'
)

parser.add_argument(
    '--pdf',
    type = str,
    required = True,
    help = 'Path where the PDF file is located. Required.'
)

parser.add_argument(
    '--json',
    type = str,
    default = None,
    required = True,
    help = 'Path to the json file generated'
)

args = parser.parse_args()

config = Config_Manager()
pipeline = config.get_pipeline(args.pipeline)

# Read json file
with open(args.json, 'r') as file:
    json_data = json.load(file)

taskid = json_data['id']

figure_results = json_data['result']['figure_result']


if os.path.exists("temp_dir"):
    shutil.rmtree("temp_dir")
os.makedirs("temp_dir",exist_ok=True)


basename = os.path.splitext(os.path.basename(args.pdf))[0]

doc = fitz.open(args.pdf)

image_data = []

glycan_idx = 1
page_idx = 0
image_counter = 0
for page_num, page in enumerate(doc):
    print("page",page)
    images = page.get_images(full=True)
    if not images:
        continue
    for img in images:
        xref = img[0]
        image_name = img[7]
        
        # Get bounding box of the image on the page
        fig_bbox = page.get_image_bbox(image_name) 
        page.draw_rect(fig_bbox, color=(1, 0, 0), width=1)  # red border around image

        # Extract image info
        img_info = doc.extract_image(xref)
        img_bytes = img_info["image"]

        img_width = img_info["width"]
        img_height = img_info["height"]

        image_filename = f"page_{page_num+1}_img_{image_counter+1}.png"
        image_path = os.path.join("temp_dir", image_filename)
        with open(image_path, "wb") as f:
            f.write(img_bytes)

        results = pipeline.run(image_path)

        img_width = results.semantics["width"]
        img_height = results.semantics["height"]

        x_scale = fig_bbox.width / img_width
        y_scale = fig_bbox.height / img_height
        

        json_glycans = figure_results[image_counter]['glycans']

        for g_id, glycan in enumerate(results.semantics["glycans"]):
            x0, y0, x1, y1 = glycan.semantics["box"].corners()  # in image pixels

            pdf_x0 = fig_bbox.x0 + x0 * x_scale
            pdf_x1 = fig_bbox.x0 + x1 * x_scale
            pdf_y0 = fig_bbox.y0 + y0 * y_scale
            pdf_y1 = fig_bbox.y0 + y1 * y_scale

            glycan_rect = fitz.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
            page.draw_rect(glycan_rect, color=(0, 0, 1), width=1)  # blue box

            url = f"extractor.glyomics.org/result/{taskid}#glycan-{image_counter}-{g_id+1}"
           
            annot = page.add_rect_annot(glycan_rect)

            content = (
                f"id: {glycan_idx}\n"
                f"url: {url}\n"
            )

            annot.set_info(content=content)

            annot.update()

            glycan_result = json_glycans[g_id]

            iupac = glycan_result.get('IUPAC','')
            composition = glycan_result.get('composition_str','')
            accession = glycan_result.get('accession','')
            wurcs = glycan_result.get('WURCS','')
            x0,y0,x1,y1 = glycan_result.get('bbox','')

            image_data.append({
                "ID": glycan_idx,
                "row_id": f"{image_counter}-{g_id+1}",
                "page_no": page_num+1,
                "accession (if available)": accession,
                "iupac": iupac,
                "composition": composition,
                'wurcs': wurcs,
                "url": url,
                "image_width": img_width,
                "image_height": img_width,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1
            })

            glycan_idx += 1

        image_counter += 1


doc.save(basename + "_annotated.pdf")
doc.close()

df = pd.DataFrame(image_data)
df.to_csv(os.path.join(basename + "_annotated.tsv"), sep="\t", index=False)

shutil.rmtree("temp_dir")


