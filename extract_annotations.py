#!.venv/bin/python
"""
Extracts figures and annotation information present on them.

This program assumes that all figures are annotated with 
a rectangle/box and comment mentioning the figure number eg. fig:<figure_number>

For each associated annotated figures, if there are annotations present on them (eg. glycan is annotated with an id - required),
then that information will be exctracted.

The information about the figure extraction and other associated information will be extracted and stored in the output file.

Input: Accepts a folder with annotated pdf's and their associated TSV's and optionally JSON file i.e results.json (if you want to extract components).
Provide extraction type through command line arguments - glycans or components (monos/root/links - all in one map file)

For component modes, one map file is written per glycan with GLYCAN + # metadata + m/l
lines. Mono id 1 is the root by convention.

Output: Figures and their semantic files (map files).
Semantics file contains info about the figure, glycan, info from TSV file (class, ID, accession, iupac, etc)

Note: Figures with no annotations will also be stored along with a semantics/map file (containing only figure dimensions)
"""

import os
import argparse
import fitz
import shutil
import csv
import tempfile
import cv2
from BKGlycanExtractor.semantics import FigureSemantics
from BKGlycanExtractor.pdfhandler import STANDARD_DPI, POINTS_PER_INCH, PDFHandler
from BKGlycanExtractor.image_manager import Manuscript_Manager

MIN_FIGURE_SIZE = 90

parser = argparse.ArgumentParser(description="Extract annotated figures and comments from PDFs")


MIN_FIGURE_SIZE = 90
parser.add_argument(
    "-m", "--manuscripts", 
    type=str, 
    nargs="+",
    required=True,
    help="Folder(s) containing PDFs with matching TSV files"
)

parser.add_argument(
    "-e", "--extract",
    type = str,
    required = False,
    default = 'glycans',
    help = "Type of extraction. Default: glycans. Options: glycans, components"

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
        **{k: v.strip() for k, v in tsv_data.items() if v is not None},
        **kwargs,
    }

def write_figure_header(sem_file, imgdata, xref=None):
    sem_file.write(
        f'##### WHOLEIMAGE: {round(imgdata["height"])} x {round(imgdata["width"])} (height x width)\n'
    )
    if xref is not None and xref > 0:
        sem_file.write(f'##### IMAGE_XREF: {xref}\n')

# fields to skip when writing glycan-level # metadata from JSON
GLYCAN_META_SKIP = {
    'box', 'image', 'links', 'monos', 'root', 'undirected_links',
    'rejected_monos', 'rejected_roots', 'rejected_undirected_links',
    'non_tree_links', 'center', 'glycans', 'log', 'glycan_errors', 'squiggle',
    'extracted_image'
}

def write_metadata(sem_file, data, skip_keys=None, prefix="##"):
    # prefix "##" = ignored by get_known_data (use for mono/link detail)
    # prefix "#"  = glycan-level key-value pairs parsed by get_known_data
    skip = set(skip_keys or ()) | {'box', 'image', 'links', 'monos', 'iupac', 'wurcs', 'composition_str', 'linkexpl', 'extracted_image_path', 'glyImage', 'image_name'}
    for key, value in data.items():
        if key in skip or not value:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        sem_file.write(f"{prefix} {key}: {value}\n")

def write_mono_data(sem_file, mono):
    mid = mono.id()
    symbol = mono.symbol()
    anomer = mono.get('anomer') or '?'
    # Only write the corners we have (x_min,y_min) and (x_max,y_max).
    # get_known_data uses data_points[4:-1] as coords, and last char is probably the radius,
    # so a trailing char for radius here can be '_' while writing to the map file
    x1, y1, x2, y2 = mono.get('box').corners()
    sem_file.write(f"m\t{mid}\t{symbol}\t{anomer}\t{x1},{y1}\t{x2},{y2}\t_\n")
    write_metadata(sem_file, dict(mono.items()))

def write_link_data(sem_file, link):
    # parser: data_points[1]=id1, [2]=carbon, [4]=id2
    # randimgs / SVG format: l  fromid  parent_bond  child_bond  toid
    id1 = link.from_id()
    id2 = link.to_id()
    parent_bond = link.get('parent_bond') or '?'
    child_bond = link.get('child_bond') or '?'
    sem_file.write(f"l\t{id1}\t{parent_bond}\t{child_bond}\t{id2}\n")
    write_metadata(sem_file, dict(link.items()))

def extract_figure(doc, page, figure_data, xref=None):
    # similar to processjob: pixmap of pdf_fig_bbox at calculated dpi, then load as cv2 image.
    pdf_bbox = figure_data.get('pdf_fig_bbox')
    if pdf_bbox is None:
        raise ValueError("figure JSON missing pdf_fig_bbox")
    clip = fitz.Rect(pdf_bbox.bbox() if hasattr(pdf_bbox, 'bbox') else pdf_bbox)

    if figure_data.get('xref') is not None:
        xref = figure_data.get('xref')
    try:
        xref = int(xref) if xref is not None else None
    except (TypeError, ValueError):
        xref = None

    # processjob already has dpi on the figure dict, but if its not present re-compute it 
    dpi = figure_data.get('dpi') or PDFHandler.calculate_dpi({
        'xref': xref,
        'width': figure_data.get('width'),
        'height': figure_data.get('height'),
        'pdf_fig_width': figure_data.get('pdf_fig_width') or (
            pdf_bbox.width() if hasattr(pdf_bbox, 'width') else None),
        'pdf_fig_height': figure_data.get('pdf_fig_height') or (
            pdf_bbox.height() if hasattr(pdf_bbox, 'height') else None),
    }, doc) or STANDARD_DPI
    dpi = int(dpi)

    pix = page.get_pixmap(clip=clip, dpi=dpi)

    tmp_path = os.path.join(tempfile.gettempdir(), f"_temp_figure{os.getpid()}.png")
    pix.save(tmp_path)
    img = cv2.imread(tmp_path)
    os.unlink(tmp_path)
    if img is None:
        raise RuntimeError("failed to read extracted figure pixmap")

    target_w = figure_data.get('width')
    target_h = figure_data.get('height')
    if target_w and target_h and (img.shape[1] != int(target_w) or img.shape[0] != int(target_h)):
        img = cv2.resize(img, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA)
    return img, dpi

def glycan_extraction(figure_details, other_annotations, tsv_data, output_dir):
    doc = figure_details["doc"]
    page = figure_details["page"]
    page_num = figure_details["page_num"]
    pdf_fig_box = figure_details["pdf_fig_box"]
    image_count = figure_details["image_count"]
    xref = figure_details["xref"]

    figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{image_count}.png"
    figure_path = os.path.join(output_dir, figure_filename)
    semantics_file = figure_path.rsplit('.', 1)[0] + '_map.txt'
    with open(semantics_file, 'w') as sem_file:
        try:
            imgdata = PDFHandler.save_image(
                doc, page, pdf_fig_box, figure_path,
                xref=xref, dpi=STANDARD_DPI, annots=False
            )
            write_figure_header(sem_file, imgdata, xref=xref)
        except Exception as e:
            print("Couldnt extract figure:", image_count, e)
        for annotation, comment_map in other_annotations:
            pdf_glycan_box = annotation.rect
            if pdf_fig_box.intersects(pdf_glycan_box):
                glycan_id = comment_map['id']
                tsv_row_data = tsv_data.get(glycan_id, {})
                gly_bbox = pixel_coordinates(
                    doc, page, pdf_glycan_box, pdf_fig_box, xref=xref
                )
                data = process_figure_annotation(
                    pdf_glycan_box, pdf_fig_box, page, tsv_row_data, comment_map
                )
                data.update({
                    'figure_num': image_count,
                    'page_num': page_num,
                    'figure_name': figure_filename,
                    'figure_path': figure_path,
                    'gly_bbox': gly_bbox,
                })
                write_semantics(sem_file, data)


def component_extraction(figure_details, other_annotations, output_dir, json_path, tsv_data=None):

    doc = figure_details["doc"]
    page = figure_details["page"]
    page_num = figure_details["page_num"]
    pdf_fig_box = figure_details["pdf_fig_box"]
    image_count = figure_details["image_count"]
    xref = figure_details.get("xref")
    tsv_data = tsv_data or {}
    
    figure_data = FigureSemantics.read_json(json_path, page_number=page_num, image_count=image_count)

    if figure_data is None:
        return 

    # Rebuild the same figure PNG processjob used (JSON pdf_fig_bbox + dpi/xref),
    # then crop each glycan with JSON glycan.bbox (figure-pixel coords).
    try:
        fig_img, dpi = extract_figure(doc, page, figure_data, xref=xref)
    except Exception as e:
        print(f"Couldnt extract figure {image_count} on page {page_num}: {e}")
        return

    for annotation, comment_map in other_annotations:
        pdf_glycan_box = annotation.rect
        if not pdf_fig_box.intersects(pdf_glycan_box):
            continue
        glycan_id = comment_map['id']
        glycan = figure_data.glycan(glycan_id)
        if not glycan:
            print(f"Skipping glycan {glycan_id} on page {page_num} and figure {image_count}")
            continue
        if glycan.get('upvotes', 0) != 1:
            continue

        gly_box = glycan.get('box')
        if gly_box is None:
            print(f"Skipping glycan {glycan_id}: missing bbox in JSON")
            continue
        glycan_img = gly_box.crop(fig_img)
        if glycan_img is None or glycan_img.size == 0:
            print(f"Skipping glycan {glycan_id}: empty crop {gly_box.bbox()}")
            continue

        glycan_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{image_count}_{glycan_id}.png"
        figure_path = os.path.join(output_dir, glycan_filename)
        semantics_file = figure_path.rsplit('.', 1)[0] + '_map.txt'
        with open(semantics_file, 'w') as sem_file:
            try:
                cv2.imwrite(figure_path, glycan_img)
                out_h, out_w = glycan_img.shape[:2]
                write_figure_header(sem_file, {"height": out_h, "width": out_w})
            except Exception as e:
                print(f"Couldnt extract figure: {image_count}, glycan: {glycan_id}, exception: {e}")
                continue

            sem_file.write(f"### GLYCAN: 0 0 {out_w} {out_h} (bbox: x y w h)\n")

            meta = {}
            tsv_row = tsv_data.get(glycan_id, {})
            for k, v in tsv_row.items():
                if v is not None and str(v).strip():
                    meta[k] = str(v).strip()
            if comment_map.get('id'):
                meta['ID'] = comment_map['id']
            if comment_map.get('url'):
                meta['url'] = comment_map['url']
            for k, v in glycan.items():
                if k in GLYCAN_META_SKIP or not v:
                    continue
                if isinstance(v, (dict, list, tuple)):
                    continue
                meta[k] = v
            meta['figure_num'] = image_count
            meta['page_num'] = page_num
            meta['figure_name'] = glycan_filename
            meta['dpi'] = dpi
            write_metadata(sem_file, meta, skip_keys=GLYCAN_META_SKIP, prefix="#")

            for mono in glycan.monos():
                write_mono_data(sem_file, mono)
            for link in glycan.all_links():
                write_link_data(sem_file, link)
 
def extract_annotations(output_dir, pdf_path, tsv_path, json_path=None):
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
            glycan_annotations = []

            # collect all annotations that exist on the page - along with the comment on them
            # and segregate them into two different list's - figure_annotations, glycan_annotations
            for annotation in page.annots():
                comment = annotation.info.get("content") or annotation.info.get("subject") or ""
                comment_dict = parse_comment(comment)  # annotated_comments_dict

                if comment_dict.get('fig'):
                    figure_annotations.append((annotation, comment_dict))
                else:
                    glycan_annotations.append((annotation, comment_dict))

            # Save all the figures on this page (discard if height and width is too small)
            # image_count = 1
            for figure_annot, fig_comments in figure_annotations:
                image_count = fig_comments['fig']
                xref = fig_comments.get('xref', None)
                xref = int(xref) if xref is not None else None
                pdf_fig_box = figure_annot.rect

                figure_details = dict(
                    doc=doc, page=page, page_num=page_num,
                    pdf_fig_box=pdf_fig_box, image_count=image_count, xref=xref,
                )
                
                # glycan extraction
                if args.extract == 'glycans':
                    # It is optional for a curator to verify the files for glycan extraction - so there is a minimum figure
                    # size requirement to avoid little piece meal figures if any
                    if (pdf_fig_box.height > MIN_FIGURE_SIZE and pdf_fig_box.width > MIN_FIGURE_SIZE):
                        glycan_extraction(figure_details, glycan_annotations, tsv_data, output_dir)
                    else:
                        print(f"Skipping figure {image_count} on page {page_num}, doesn't meet the minumum size requirement of height {MIN_FIGURE_SIZE}, width {MIN_FIGURE_SIZE}\n")
                
                # component extraction
                else:
                    # curator verifies and upvotes glycans that are good - and only those will be picked for training data
                    # components write one full map (GLYCAN + m + l)
                    component_extraction(
                        figure_details, glycan_annotations, output_dir, json_path,
                        tsv_data=tsv_data
                    )


def write_semantics(semantics_file, data, *, label="GLYCAN", bbox_key="gly_bbox", skip_keys=None):
    skip = set(skip_keys or ())
    skip.add(bbox_key)
    x, y, w, h = data[bbox_key]
    semantics_file.write(f"### {label}: {x} {y} {w} {h} (bbox: x y w h)\n")
    for key, value in data.items():
        if key in skip:
            continue
        if value:
            semantics_file.write(f"# {key}: {value}\n")

output_folder = args.output
if os.path.isdir(output_folder) and args.force:
    # ensure everything is computed again
    shutil.rmtree(output_folder)

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

assert args.extract in ('glycans', 'components')

pdf_files = Manuscript_Manager(args.manuscripts)
for pdf_path in pdf_files:

    pdf_dir,pdf_file = os.path.split(pdf_path)
    pdf_basename,pdf_extn = pdf_file.rsplit('.',1)

    # check if a corresponding tsv file exists for the pdf
    tsv_path = os.path.join(pdf_dir,pdf_basename + '.tsv')

    if not os.path.exists(tsv_path):
        print(f"  Skipping:   {pdf_file} - no matching TSV found.")
        continue

    if pdf_basename.rsplit('.',1)[-1] in ("annotated","annotated_Manual"):
        pdf_basename = pdf_basename.rsplit(".",1)[0]

    # if extraction type is glycans - then JSON file is optional. Otherwise required
    json_path = os.path.join(pdf_dir , 'results.json')
    if args.extract != 'glycans':
        if not os.path.exists(json_path):
            print(f"  Skipping:   {pdf_file} - no matching JSON found.")
            continue
            
    output_dir = os.path.join(output_folder, pdf_basename)

    if os.path.exists(output_dir):
        print(f"  Skipping: {pdf_file} - already processed.")
        continue
    
    os.makedirs(output_dir)

    print("Processing:", os.path.split(pdf_path)[1])

    # main step for extraction
    extract_annotations(output_dir,pdf_path,tsv_path,json_path)


