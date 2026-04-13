#!.venv/bin/python
"""
Extracts figures and annotation information present on them.

This program assumes that all figures are annotated with 
a rectangle/box and comment mentioning the figure number eg. fig:<figure_number>

For each associated annotated figures, if there are annotations present on them (eg. glycan is annotated with an id - required),
then that information will be exctracted.

The information about the figure extraction and other associated information will be extracted and stored in the output file.

Input: Accepts a folder with annotated pdf's and their associated TSV's

Output: Figures and their semantic files (map files).
Semantics file contains info about the figure, glycan, info from TSV file (class, ID, accession, iupac, etc)

Note: Figures with no annotations will also be stored along with a semantics/map file (containing only figure dimensions)
"""

import os
import argparse
import fitz
import glob
import shutil
import csv
import traceback
from BKGlycanExtractor.pdfhandler import STANDARD_DPI, POINTS_PER_INCH, PDFHandler
from BKGlycanExtractor.image_manager import Manuscript_Manager

parser = argparse.ArgumentParser(description="Extract annotated figures and comments from PDFs")

parser.add_argument(
    "-m", "--manuscripts", 
    type=str, 
    nargs="+",
    required=True,
    help="Folder(s) containing PDFs with matching TSV files"
)
    
parser.add_argument(
    "-o", "--output", 
    type=str, 
    required=True, 
    help="Folder name to store output images with semantics" 
)

parser.add_argument(
    "-F", "--force",
    action = 'store_true',
    default = False,
    help = 'Reprocess PDFs and overwrite output images and semantics.'
)

args = parser.parse_args()

MIN_FIGURE_SIZE = 90

def parse_comment(comment):
    """Parse annotation comment into (glycan_id, url).

    2 different formats of comments exist in the annotated pdf, need to handle both the types to extract id and url
    
    Expected formats:
    - Multi-line with prefixes, e.g.:
        id: G123
        url: http://example.com
    - Single-line (which is the id), e.g.:
        G123
    """
    comment = (comment or "").strip()

    if not comment:
        return {}

    comment_dict = {}
    lines = [line.strip() for line in comment.splitlines() if line.strip()]
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            comment_dict[key.strip().lower()] = value.strip()
        else:   # case where is single line is present - will probably be an id (id's are compulsory)
            comment_dict['id'] = lines[0]
    return comment_dict

def load_tsv_data(tsv_path):
    """Load TSV into dictionary keyed by ID"""
    tsv_data = {}
    if os.path.exists(tsv_path):
        with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                row_id = row.get('ID')
                if row_id:
                    assert row_id not in tsv_data
                    tsv_data[row_id] = row
    return tsv_data

# TODO write a static method for this in bbox class
def pixel_coordinates(doc, page, annot_box, fig_box, xref=None, dpi=STANDARD_DPI):

    # Scale: from xref image size if available, else from DPI
    scale_x = scale_y = None

    if xref is not None and xref > 0:
        try: 
            pix = fitz.Pixmap(doc, xref)
            scale_x = pix.width / fig_box.width
            scale_y = pix.height / fig_box.height
        except Exception as e:
            pass

    if scale_x is None or scale_y is None:
        pixels_per_point = float(dpi) / POINTS_PER_INCH
        scale_x = scale_y = pixels_per_point

    # Convert annotation coordinates to pixel coordinates
    px_gly_x0 = round((annot_box.x0 - fig_box.x0) * scale_x)
    px_gly_y0 = round((annot_box.y0 - fig_box.y0) * scale_y)
    px_gly_w = round(annot_box.width * scale_x)
    px_gly_h = round(annot_box.height * scale_y)

    return [px_gly_x0, px_gly_y0, px_gly_w, px_gly_h]

def process_figure_annotation(annot_box, figure_box, page, tsv_data, comment_map, **kwargs):
    '''method to process information about an annotation'''
    return {
        'ID': comment_map['id'],
        'url': comment_map.get('url'),
        'pdf_fig_bbox': figure_box,
        **{k: v.strip() for k, v in tsv_data.items() 
            if k in ['class','accession', 'iupac', 'composition', 'wurcs'] and v is not None},
        **kwargs,
    }

def extract_annotations(output_dir, pdf_path, tsv_path):
    """
    Main extraction method.
    Extracts figures and associated annotation information.
    """

    # load all the existing tsv data as a dict
    # key: id, val: all other data
    tsv_data = load_tsv_data(tsv_path)

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc.pages(), 1):
    
            figure_annotations = []
            other_annotations = []

            # collect all annotations that exist on the page - along with the comment on them
            # and segregate them into two different list's - figure_annotations, other_annotations
            for annotation in page.annots():
                comment = annotation.info.get("content") or annotation.info.get("subject") or ""
                comment_dict = parse_comment(comment)  # annotated_comments_dict

                if comment_dict.get('fig'):
                    figure_annotations.append((annotation, comment_dict))
                else:
                    other_annotations.append((annotation, comment_dict))


            # Save all the figures on this page (discard if height and width is too small)
            # fig_num = 1
            for figure_annot, fig_comments in figure_annotations:
                fig_num = fig_comments['fig']
                xref = fig_comments.get('xref', None)
                xref = int(xref) if xref is not None else None
                pdf_fig_box = figure_annot.rect
                
                if (pdf_fig_box.height > MIN_FIGURE_SIZE and pdf_fig_box.width > MIN_FIGURE_SIZE):
                    figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{fig_num}.png"
                    figure_path = os.path.join(output_dir, figure_filename)

                    # create semantic (map) file for all the corresponding figures
                    semantics_file = figure_path.rsplit('.', 1)[0] + '_map.txt'
                    with open(semantics_file, 'w') as sem_file:
                        
                        try:
                            imgdata = PDFHandler.save_image(doc, page, pdf_fig_box, figure_path, xref=xref, dpi=STANDARD_DPI, annots=False)
                            sem_file.write(f'##### WHOLEIMAGE: {round(imgdata['height'])} x {round(imgdata['width'])} (height x width)\n')
                            # sem_file.write(f'##### IMAGE_DPI: {dpi}\n')
                            if xref is not None and xref > 0:
                                sem_file.write(f'##### IMAGE_XREF: {xref}\n')
                        except Exception as e:
                            # traceback.print_exc()
                            print("Couldnt extract figure:", fig_num, e)

                        # verify that the other annotation intersects with current figure_annotation
                        # and only then accept it as a part of an annotation that exists on the current figure 
                        for annotation, comment_map in other_annotations:
                            annotation_box = annotation.rect
                            # instead of intersects - you can you IOU as well to check
                            if pdf_fig_box.intersects(annotation_box):
                                glycan_id = comment_map['id']

                                tsv_row_data = {}
                                if glycan_id in tsv_data:
                                    tsv_row_data = tsv_data[glycan_id]

                                gly_bbox = pixel_coordinates(doc, page, annotation_box, pdf_fig_box, xref=xref)

                                data = process_figure_annotation(annotation_box, pdf_fig_box, page, tsv_row_data, comment_map)
                                data.update({'figure_num': fig_num, 'page_num': page_num, 'figure_name': figure_filename, 'figure_path': figure_path})
                                data.update({'gly_bbox': gly_bbox})

                                write_semantics(sem_file, data)

def write_semantics(semantics_file, glycan_data):
    x, y, w, h = glycan_data['gly_bbox']
    semantics_file.write(f"### GLYCAN: {x} {y} {w} {h} (bbox: x y w h)\n")
    
    # add other key-value pairs from TSV file
    for key in glycan_data.keys():
        if key == 'gly_bbox':
            continue
        value = glycan_data.get(key)
        if value:  # Checks: not None, not empty, not just whitespace
            semantics_file.write(f"# {key}: {value}\n")

output_folder = args.output
if os.path.isdir(output_folder) and args.force:
    # ensure everything is computed again
    shutil.rmtree(output_folder)

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

pdf_files = Manuscript_Manager(args.manuscripts)
for pdf_path in pdf_files:

    pdf_dir,pdf_file = os.path.split(pdf_path)
    pdf_basename,pdf_extn = pdf_file.rsplit('.',1)

    # check if a corresponding tsv file exists for the pdf
    tsv_path = os.path.join(pdf_dir,pdf_basename + '.tsv')

    if not os.path.exists(tsv_path):
        print(f"  Skipping:   {pdf_file} - no matching TSV found.")
        continue

    if pdf_basename.endswith(".annotated"):
        pdf_basename = pdf_basename.rsplit(".",1)[0]

    output_dir = os.path.join(output_folder, pdf_basename)

    if os.path.exists(output_dir):
        print(f"  Skipping: {pdf_file} - already processed.")
        continue
    
    os.makedirs(output_dir)

    print("Processing:", os.path.split(pdf_path)[1])

    # main step for extraction
    extract_annotations(output_dir,pdf_path,tsv_path)




