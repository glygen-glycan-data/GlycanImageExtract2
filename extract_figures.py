"""
Extracts figures and annotation information present on them.

Input: Accepts a folder with annotated pdf's and their associated TSV's
Note: Only accepts files with the extension: *.annotated.*

Output: Figures and their semantic files (map files).
Semantics file contains info about the figure, glycan, info from TSV file (class, ID, xref, accession, iupac, etc)

Note: Figures with no annotations will also be stored along with a semantics/map file (containing only figure dimensions)
"""

import os
import argparse
import fitz
import glob
import shutil
import csv

parser = argparse.ArgumentParser(description="Extract annotated figures and comments from PDFs")

parser.add_argument(
    "-f", "--folder", 
    type=str, 
    required=True,
    help="Folder containing PDFs and matching TSV files"
)
    
parser.add_argument(
    "-o", "--output_dir", 
    type=str, 
    required=True, 
    help="Provide folder name to store output" 
)

args = parser.parse_args()

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

    if comment is None:
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

def save_figure(page, figure, figure_path):
    pdf_fig_box = fitz.Rect(figure["bbox"])
    zoom_x = figure["width"] / pdf_fig_box.width
    zoom_y = figure["height"] / pdf_fig_box.height
    mat = fitz.Matrix(zoom_x, zoom_y)

    pix = page.get_pixmap(matrix=mat, clip=pdf_fig_box, annots=False)
    pix.save(figure_path)
    return pix.width, pix.height

def load_tsv_data(tsv_path):
    """Load TSV into dictionary keyed by ID"""
    tsv_data = {}
    if os.path.exists(tsv_path):
        with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                row_id = row.get('ID')
                if row_id:
                    tsv_data[row_id] = row
    return tsv_data

def pixel_coordinates(page,annot_box,figure,fig_box):
    '''
    figure["width"], figure["height"] - are the original pixel dimensions
    fig_box - is the PDF page bounding box in points 

    return List[gly_bbox], fig_width, fig_height
    '''

    fig_box = fitz.Rect(figure["bbox"])

    # The division yields the zoom factor that restores the original pixel dimensions.
    zoom_x = figure["width"] / fig_box.width
    zoom_y = figure["height"] / fig_box.height
    mat = fitz.Matrix(zoom_x, zoom_y)
    
    pix = page.get_pixmap(matrix=mat, clip=fig_box, alpha=False, annots=False)
    px_fig_width, px_fig_height = pix.width, pix.height
    
    # Calculate scaling factors
    x_scale = fig_box.width / float(px_fig_width)
    y_scale = fig_box.height / float(px_fig_height)
    
    # Convert annotation coordinates to pixel coordinates
    px_gly_x0 = round(abs(annot_box.x0 - fig_box.x0) / x_scale)
    px_gly_x1 = round(abs(annot_box.x1 - fig_box.x0) / x_scale)
    px_gly_y0 = round(abs(annot_box.y0 - fig_box.y0) / y_scale)
    px_gly_y1 = round(abs(annot_box.y1 - fig_box.y0) / y_scale)
    
    px_gly_w = abs(px_gly_x1 - px_gly_x0)
    px_gly_h = abs(px_gly_y1 - px_gly_y0)

    return [px_gly_x0, px_gly_y0, px_gly_w, px_gly_h], px_fig_width, px_fig_height

def process_figure_annotation(annotation, figure, page, tsv_data, comment_dict, **kwargs):
    '''method to process information about an annotation'''
    annot_box = annotation.rect
    xref = figure.get("xref")
    fig_box = fitz.Rect(figure["bbox"])
    
    # convert's pdf coordinates to pixel coordinates - output - glycan_bbox, fig_width, fig_height 
    gly_bbox, px_fig_width, px_fig_height = pixel_coordinates(page, annot_box, figure, fig_box)

    return {
        'ID': comment_dict['id'],
        'url': comment_dict.get('url'),
        'xref': xref,
        'gly_bbox': gly_bbox,
        'fig_width': px_fig_width,
        'fig_height': px_fig_height,
        **{k: v.strip() for k, v in tsv_data.items() 
            if k in ['class','accession', 'iupac', 'composition', 'wurcs'] and v is not None},
        **kwargs,
    }


def extract_annotations(output_dir, pdf_path, tsv_path):
    """
    Main extraction method.
    Extracts figures and associated annotation information.
    """

    tsv_data = load_tsv_data(tsv_path)

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc.pages(), 1):
            page_info = page.get_image_info(xrefs=True)
            page_annotations = list(page.annots() or [])

            # Save all the figures on this page (discard if height and width is too small)
            fig_num = 1
            for figure in page_info:
                pdf_fig_box = fitz.Rect(figure["bbox"])
                
                if (pdf_fig_box.height > 90 and pdf_fig_box.width > 90):
                    figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{fig_num}.png"
                    figure_path = os.path.join(output_dir, figure_filename)
                    px_fig_height, px_fig_width = save_figure(page, figure, figure_path)

                    # create semnatic (map) file for all the corresponding figures
                    semantics_file = figure_path.rsplit('.', 1)[0] + '_map.txt'
                    with open(semantics_file, 'w') as sem_file:
                        sem_file.write(f'##### WHOLEIMAGE: {round(px_fig_height)} x {round(px_fig_width)} (height x width)\n')
                        
                        # verify that the annotation intersects with figure
                        for annotation in page_annotations:
                            annotation_box = annotation.rect
                            # instead of intersects - you can you IOU as well to check
                            if pdf_fig_box.intersects(annotation_box):
                                comment = annotation.info.get("content") or annotation.info.get("subject") or ""
                                comment_dict = parse_comment(comment)  # annotated_comments_dict
                                glycan_id = comment_dict['id']

                                tsv_row_data = {}
                                if glycan_id in tsv_data:
                                    tsv_row_data = tsv_data[glycan_id]

                                metadata = {'figure_num': fig_num, 'page_num': page_num, 'figure_name': figure_filename, 'figure_path': figure_path}
                                
                                data = process_figure_annotation(annotation, figure, page, tsv_row_data, comment_dict)
                                data.update(metadata)
                                write_semantics(sem_file, data)
          
                    fig_num += 1



def write_semantics(semantics_file, glycan_data):
    x, y, w, h = glycan_data['gly_bbox']
    semantics_file.write(f"### GLYCAN: {x} {y} {w} {h} (bbox: x y w h)\n")
    
    # add other key-value pairs from TSV file
    for key in glycan_data.keys():
        if key not in ['page_num', 'fig_num']:
            value = glycan_data.get(key)
            if value:  # Checks: not None, not empty, not just whitespace
                semantics_file.write(f"# {key}: {value}\n")

input_folder = args.folder
output_folder = args.output_dir
# accepts annotated pdf's only - because annotate_pdf step generates pdf's with the extension .annotated.pdf
# so it is safe to accept only those kind of pdf's
pdf_files = glob.glob(os.path.join(input_folder, "*.annotated.pdf")) 

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

print("\nStarting Process...")
for pdf_path in pdf_files:
    print("Processing PDF:", pdf_path)
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    tsv_path = os.path.join(input_folder, f"{file_name}.tsv")

    output_dir = os.path.join(output_folder, file_name)

    if not os.path.exists(output_dir):
        # shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(tsv_path):
        print(f"Skipping {file_name}: no matching TSV found.")
        continue

    # main step for extraction
    extract_annotations(output_dir,pdf_path,tsv_path)



