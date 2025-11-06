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

def save_figure(page, annot_rect, figure_path, scale):
    """
    Save pixmap of whatever is under the annotation rectangle (annot_rect)
    annot_rect: annotation in pdf coordinates
    scale: multiplier (1.0 = 72 DPI, 2.0 = 144 DPI, etc.)
    """
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, clip=annot_rect, annots=False)
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

def pixel_coordinates(page, annot_box, fig_box, scale):
    # Calculate pixel dimensions directly
    # px_fig_width = round(fig_box.width * scale)
    # px_fig_height = round(fig_box.height * scale)

    # Convert annotation coordinates to pixel coordinates
    px_gly_x0 = round((annot_box.x0 - fig_box.x0) * scale)
    px_gly_y0 = round((annot_box.y0 - fig_box.y0) * scale)
    px_gly_w = round(annot_box.width * scale)
    px_gly_h = round(annot_box.height * scale)

    return [px_gly_x0, px_gly_y0, px_gly_w, px_gly_h]

def process_figure_annotation(annot_box, figure_box, page, tsv_data, comment_map, scale, **kwargs):
    '''method to process information about an annotation'''
    
    # convert's pdf coordinates to pixel coordinates - output - glycan_bbox, fig_width, fig_height 
    gly_bbox = pixel_coordinates(page, annot_box, figure_box, scale)

    return {
        'ID': comment_map['id'],
        'url': comment_map.get('url'),
        # 'xref': xref,
        'gly_bbox': gly_bbox,
        **{k: v.strip() for k, v in tsv_data.items() 
            if k in ['class','accession', 'iupac', 'composition', 'wurcs'] and v is not None},
        'scale': scale,
        **kwargs,
    }


def extract_annotations(output_dir, pdf_path, tsv_path, scale=1.0):
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
                    figure_annotations.append((annotation, comment_dict.get('fig')))
                else:
                    other_annotations.append((annotation, comment_dict))


            # Save all the figures on this page (discard if height and width is too small)
            # fig_num = 1
            for figure_annot, fig_num in figure_annotations:
                # pdf_fig_box = fitz.Rect(figure["bbox"])
                pdf_fig_box = figure_annot.rect
                
                if (pdf_fig_box.height > 90 and pdf_fig_box.width > 90):
                    figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{fig_num}.png"
                    figure_path = os.path.join(output_dir, figure_filename)
                    px_fig_height, px_fig_width = save_figure(page, pdf_fig_box, figure_path, scale)

                    # create semantic (map) file for all the corresponding figures
                    semantics_file = figure_path.rsplit('.', 1)[0] + '_map.txt'
                    with open(semantics_file, 'w') as sem_file:
                        sem_file.write(f'##### WHOLEIMAGE: {round(px_fig_height)} x {round(px_fig_width)} (height x width)\n')
                        
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

                                metadata = {'fig_width': px_fig_width,'fig_height': px_fig_height,'figure_num': fig_num, 'page_num': page_num, 'figure_name': figure_filename, 'figure_path': figure_path}
                                
                                data = process_figure_annotation(annotation_box, pdf_fig_box, page, tsv_row_data, comment_map, scale)
                                data.update(metadata)
                                write_semantics(sem_file, data)
          
                    # fig_num += 1


def write_semantics(semantics_file, glycan_data):
    x, y, w, h = glycan_data['gly_bbox']
    semantics_file.write(f"### GLYCAN: {x} {y} {w} {h} (bbox: x y w h)\n")
    
    # add other key-value pairs from TSV file
    for key in glycan_data.keys():
        value = glycan_data.get(key)
        if value:  # Checks: not None, not empty, not just whitespace
            semantics_file.write(f"# {key}: {value}\n")

input_folder = args.folder
output_folder = args.output_dir
pdf_files = glob.glob(os.path.join(input_folder, "*.pdf")) 


if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
os.mkdir(output_folder)

print("\nStarting Process...")
for pdf_path in pdf_files:
    pdf_basename = os.path.basename(pdf_path).rsplit('.',1)[0]

    # check if a corresponding tsv file exists for the pdf
    tsv_path = os.path.join(input_folder, pdf_basename + '.tsv')

    if not os.path.exists(tsv_path):
        print(f"\nSkipping PDF: {pdf_path} - no matching TSV found.")
        continue

    print("\nProcessing PDF:", pdf_path)

    output_dir = os.path.join(output_folder, pdf_basename)

    if not os.path.exists(output_dir):
        # shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    # main step for extraction
    # scale 3.0 --> DPI = 216 - generallu good
    # scale 4.0 --> DPI = 288 - if there are very small objects (glycans in manuscripts can be small)
    extract_annotations(output_dir,pdf_path,tsv_path,scale=4.0)



