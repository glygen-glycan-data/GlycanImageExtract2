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
import cv2
from BKGlycanExtractor.semantics import ManuscriptSemantics
from BKGlycanExtractor.pdfhandler import STANDARD_DPI, POINTS_PER_INCH, PDFHandler
from BKGlycanExtractor.image_manager import Manuscript_Manager
from BKGlycanExtractor.glycanannotator import Config_Manager

MIN_FIGURE_SIZE = 90

# fields to skip when writing glycan-level metadata from JSON
GLYCAN_META_SKIP = {
    'box', 'image', 'links', 'monos', 'root', 'undirected_links',
    'rejected_monos', 'rejected_roots', 'rejected_undirected_links',
    'non_tree_links', 'center', 'glycans', 'log', 'glycan_errors', 'squiggle',
    'extracted_image', 'iupac', 'wurcs', 'composition_str', 'linkexpl',
    'extracted_image_path', 'glyImage', 'image_name'
}

parser = argparse.ArgumentParser(description="Extract annotated figures and comments from PDFs")

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

    if xref is not None and int(xref) > 0:
        try: 
            pix = fitz.Pixmap(doc, int(xref))
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

def write_figure_header(semantics_file, fig_data):
    semantics_file.write(
        f'##### WHOLEIMAGE: {round(fig_data["height"])} x {round(fig_data["width"])} (height x width)\n'
    )

def write_glycans(annotated_pdf, page, page_num, pdf_fig_box, fig_comments,
                 glycan_annotations, tsv_data, output_dir):
    
    # pdf_fig_box - is fitz.rect format which is [x1, y1, x2, y2]
    image_count = fig_comments['fig']
    xref = fig_comments.get('xref')
    dpi = fig_comments.get('dpi') or STANDARD_DPI

    figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{image_count}.png"   # default is png file but PDFHandler.save_image() will decide what format the original image was embedded in the pdf and accordingly get png, jpeg
    figure_path = os.path.join(output_dir, figure_filename)
    
    # Save Glycan Figure (Figure can have single/multiple glycans)
    try:
        fig_data = PDFHandler.save_image(
            annotated_pdf, page, pdf_fig_box, figure_path,
            xref=xref, dpi=dpi, annots=False
        )
        if not fig_data:
            raise RuntimeError("save_image returned None")
        figure_path = fig_data.get("image_path", figure_path)
        figure_filename = os.path.basename(figure_path)
    except Exception as e:
        print("Couldnt extract figure:", image_count, e)
        return
    
    # Write glycan(s) detials into a _map.txt file
    # glycan details are from TSV file and annotated pdf
    semantics_file = figure_path.rsplit('.', 1)[0] + '_map.txt'
    with open(semantics_file, 'w') as map_file:
        write_figure_header(map_file, fig_data)
        for annotation, comment_dict in glycan_annotations:
            pdf_glycan_box = annotation.rect
            if not pdf_fig_box.intersects(pdf_glycan_box):
                continue

            glycan_id = comment_dict.get('id')
            if not glycan_id:
                print(f"Skipping annotation on figure {image_count}: no id")
                continue

            tsv_row_data = tsv_data.get(glycan_id, {})
            # Note: pixel_coordinates method is needed because glycan extractions work without a json document, so it doesnt get the refernce bbox
            # and needs to compute it based on the annotated pdf based rectangle. Access to json document would directly provide glycan bbox in pixel coordinates.
            gly_bbox = pixel_coordinates(
                annotated_pdf, page, pdf_glycan_box, pdf_fig_box, xref=xref, dpi=dpi
            )

            data = {
                'xref': xref,   # if xref is already present in tsv, then tsv will overwrite this
                'dpi': dpi,     # if dpi is already present in tsv, then tsv will overwrite this
                **{k: str(v).strip() for k, v in tsv_row_data.items() if v is not None},
                'pdf_fig_bbox': pdf_fig_box,
                'figure_name': figure_filename,
                'figure_path': figure_path,
                'gly_bbox': gly_bbox,
            }

            x, y, w, h = gly_bbox
            map_file.write(f"### GLYCAN: {x} {y} {w} {h} (bbox: x y w h)\n")
            for key, value in data.items():
                if key in GLYCAN_META_SKIP:
                    continue
                if value:
                    map_file.write(f"# {key}: {value}\n")


def collect_page_annotations(page):
    figure_annotations, glycan_annotations = [], []
    for annotation in page.annots() or []:
        comment_dict = parse_comment(
            annotation.info.get("content") or annotation.info.get("subject") or ""
        )
        if comment_dict.get('fig'):
            figure_annotations.append((annotation, comment_dict))
        else:
            glycan_annotations.append((annotation, comment_dict))
    return figure_annotations, glycan_annotations

def extract_glycans(annotated_pdf, tsv_data, output_dir):
    for page_num, page in enumerate(annotated_pdf.pages(), 1):
        figure_annotations, glycan_annotations = collect_page_annotations(page)
        for figure_annot, fig_comments in figure_annotations:
            pdf_fig_box = figure_annot.rect
            image_count = fig_comments['fig']
            if pdf_fig_box.height <= MIN_FIGURE_SIZE or pdf_fig_box.width <= MIN_FIGURE_SIZE:
                print(f"Skipping figure {image_count} on page {page_num}, doesn't meet the minumum size requirement of height {MIN_FIGURE_SIZE}, width {MIN_FIGURE_SIZE}\n")
                continue

            write_glycans(
                annotated_pdf, page, page_num, pdf_fig_box, fig_comments,
                glycan_annotations, tsv_data, output_dir
            )

def _load_image(figure_path, target_w=None, target_h=None):
    """
    Load a saved figure and optionally resize to canvas size provided by the JSON file.
    Reusable whenever saved PDF pixels must match pipeline width/height.
    Using this because sometimes there are inconsistencies when there is no xref for a figure - e.g figcap examples
    """

    img = cv2.imread(figure_path)
    if img is None:
        raise RuntimeError(f"failed to read saved figure: {figure_path}")
    if target_w and target_h:       
        target_w, target_h = int(target_w), int(target_h)       # pixel based images have width, height as integers
        h, w = img.shape[:2]
        if (w, h) != (target_w, target_h):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return img

def get_clean_image_finder(pipeline_name):
    if not pipeline_name:
        return None
    # Single-glycan pipelines: no CleanImage step
    if pipeline_name.startswith("SingleGlycanImage"):
        return None
    # Multiple-glycan pipelines: load CleanImage from that pipeline's figure_steps
    if pipeline_name.startswith("MultipleGlycanImage"):
        config = Config_Manager()
        pipeline_config = config.get_config("Pipeline:" + pipeline_name)

        figure_steps = pipeline_config.get("figure_steps")

        if not figure_steps:
            return None

        for s in figure_steps.split(","):
            s = s.strip()
            if s and s.startswith('CleanImage'):
                return config.get_finder(s)
    return None

def write_metadata(map_file, data, skip_keys=GLYCAN_META_SKIP, prefix="#"):
    skip = set(skip_keys)
    for key, value in data.items():
        if key in skip or not value:
            continue
        if isinstance(value, (dict, list, tuple)):
            continue
        map_file.write(f"{prefix} {key}: {value}\n")

def write_mono_data(map_file, mono):
    mid = mono.id()
    symbol = mono.symbol()
    anomer = mono.get('anomer') or '?'
    # Only write the corners we have (x_min,y_min) and (x_max,y_max).
    # get_known_data uses data_points[4:-1] as coords, and last char is probably the radius,
    # so a trailing char for radius here can be '_' while writing to the map file
    x1, y1, x2, y2 = mono.get('box').corners()
    map_file.write(f"## MONO: {mono.bbox()} (bbox: x y w h)\n")
    write_metadata(map_file, dict(mono.items()))
    map_file.write(f"m\t{mid}\t{symbol}\t{anomer}\t{x1},{y1}\t{x2},{y2}\t_\n")

def write_link_data(map_file, link):
    # parser: data_points[1]=id1, [2]=carbon, [4]=id2
    # randimgs / SVG format: l  fromid  parent_bond  child_bond  toid
    id1 = link.from_id()
    id2 = link.to_id()
    parent_bond = link.get('parent_bond') or '?'
    child_bond = link.get('child_bond') or '?'
    map_file.write(f"## LINK: {link.bbox()} (bbox: x y w h)\n")
    write_metadata(map_file, dict(link.items()))
    map_file.write(f"l\t{id1}\t{parent_bond}\t{child_bond}\t{id2}\n")

def extract_components(annotated_pdf, json_data, tsv_data, output_dir):
    '''
    For each glycan, extracts monos, root and link information and adds it to a map.txt file 
    and saves glycan image
    '''

    # Determine the pipeline used for extraction from provided json file
    # and access the Clean Image Step from the pipleine that was originally
    pipeline_name = json_data.get('pipeline_name')
    clean_image_step = get_clean_image_finder(pipeline_name)

    # For each figure with multiple glycan --> extract each glycans component information
    # get the entire figure --> based on the glycan annotations clip the glycan image out of figure and save component info
    for figure in json_data.figures():
        page_number = figure.get('page_number')
        image_count = figure.get('image_count')
        page = annotated_pdf[page_number - 1]

        pdf_fig_bbox = figure.get('pdf_fig_bbox')   # [x1, y1, x2, y2]

        xref = figure.get('xref')
        dpi = figure.get('dpi') or STANDARD_DPI

        figure_path = os.path.join(
            output_dir, f"{os.path.basename(output_dir)}_p{page_number}_f{image_count}.png",
        )

        try:
            # save the entire figure with glycan(s) - so that you can clip individuals glycans from it
            # and at the end the figure will be deleted 
            figure_details = PDFHandler.save_image(
                annotated_pdf, page, pdf_fig_bbox, figure_path,
                xref=xref, dpi=dpi, annots=False,
            )

            if not figure_details:
                raise RuntimeError("save_image returned None")
            
            figure_path = figure_details.get("image_path", figure_path)
            fig_ext = os.path.splitext(figure_path)[1] or ".png"
            figure_filename = os.path.basename(figure_path)

            glycan_figure = _load_image(
                figure_path,
                target_w=figure.get('width'),
                target_h=figure.get('height'),
            )

        except Exception as e:
            print(f"Couldnt extract figure {image_count} on page {page_number}: {e}")
            continue
        
        # save glycan components info and clip the glycan from the entire figure
        try:
            for glycan in figure.glycans():
                gid = glycan.get('GID')
                tsv_row = tsv_data.get(gid, {})

                # only accept glycans which were voted as GOOD
                if int(tsv_row.get('votes') or 0) != 1:
                    continue

                gly_box = glycan.get('box')
                if gly_box is None:
                    print(f"Skipping glycan {gid}: missing bbox in JSON")
                    continue
                
                # get glycan image and clean it
                glycan_img = gly_box.crop(glycan_figure)
                
                if glycan_img is None or glycan_img.size == 0:
                    print(f"Skipping glycan {gid}: empty crop {gly_box.bbox()}")
                    continue

                if clean_image_step is not None:
                    _cropped, glycan_img = clean_image_step.process_image(glycan_img)

                gx, gy, gw, gh = gly_box.bbox()
                glycan_filename = (f"{os.path.basename(output_dir)}_p{page_number}_f{image_count}_{gid}{fig_ext}")
                glycan_image_path = os.path.join(output_dir, glycan_filename)
                semantics_file = glycan_image_path.rsplit('.', 1)[0] + '_map.txt'

                try:
                    cv2.imwrite(glycan_image_path, glycan_img)
                except Exception as e:
                    print(f"Couldnt extract glycan image from figure: {image_count}, glycan id: {gid}, exception: {e}")
                    continue

                with open(semantics_file, 'w') as map_file:
                    out_h, out_w = glycan_img.shape[:2]
                    write_figure_header(map_file, {"height": out_h, "width": out_w})
                    map_file.write(f"### GLYCAN: 0 0 {out_w} {out_h} (bbox: x y w h)\n")

                    meta = {}
                    for k, v in tsv_row.items():
                        if v is not None and str(v).strip():
                            meta[k] = str(v).strip()
                    for k, v in glycan.items():
                        if k in GLYCAN_META_SKIP or not v:
                            continue
                        if isinstance(v, (dict, list, tuple)):
                            continue
                        meta[k] = v
                    meta['figure_number'] = image_count
                    meta['page_number'] = page_number
                    meta['figure_name'] = glycan_filename
                    # meta['dpi'] = dpi
                    write_metadata(map_file, meta,  prefix="#")

                    for mono in glycan.monos():
                        write_mono_data(map_file, mono)
                    for link in glycan.all_links():
                        write_link_data(map_file, link)
        finally:
            if figure_path and os.path.isfile(figure_path):
                os.remove(figure_path)

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
        print(f"  Skipping:   {pdf_file} - no matching TSV found. Require {tsv_path}")
        continue

    if pdf_basename.rsplit('.',1)[-1] in ("annotated","annotated_Manual"):
        pdf_basename = pdf_basename.rsplit(".",1)[0]

    # if extraction type is glycans - then JSON file is optional. Otherwise required
    json_path = os.path.join(pdf_dir , 'results.json')
    if args.extract != 'glycans':
        if not os.path.exists(json_path):
            print(f"  Skipping:   {pdf_file} - no matching JSON found. Require {json_path}")
            continue
            
    output_dir = os.path.join(output_folder, pdf_basename)

    if os.path.exists(output_dir):
        print(f"  Skipping: {pdf_file} - already processed.")
        continue
    
    os.makedirs(output_dir)

    print("Processing:", os.path.split(pdf_path)[1])

    # main step for extraction, based on extraction type - 'glycans' or 'component' - monos, root, links
    # extract_annotations(output_dir,pdf_path,tsv_path,json_path,extraction_type=args.extract)

    # load all the existing tsv data as a dict
    # key: id, val: all other data
    tsv_data = load_tsv_data(tsv_path)

    with fitz.open(pdf_path) as annotated_pdf:
        if args.extract == 'glycans':
            extract_glycans(annotated_pdf, tsv_data, output_dir)
        else:
            json_data = ManuscriptSemantics.read_json(json_path)
            if json_data is None:
                print(f"Skipping: could not read {json_path}")
                continue
            extract_components(annotated_pdf, json_data, tsv_data, output_dir)


