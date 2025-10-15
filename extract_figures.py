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
    required =True, 
    help="New folder created/provided to store output files" 
)

args = parser.parse_args()

def parse_comment(comment: str):
    """Parse annotation comment into (glycan_id, url).

    2 different formats of comments exist, need to handle both the types to extract id and url
    
    Expected formats:
    - Multi-line with prefixes, e.g.:
        IDX: G123
        URL: http://example.com
    - Single-line fallback, e.g.:
        G123
    """
    comment = (comment or "").strip()
    glycan_id, url = None, None

    if not comment:
        return glycan_id, url

    lines = [line.strip() for line in comment.splitlines() if line.strip()]
    for line in lines:
        lower = line.lower()
        if lower.startswith("id:"):
            glycan_id = line.split(":", 1)[1].strip()
        elif lower.startswith("url:"):
            url = line.split(":", 1)[1].strip()

    # fallback: single-line comment with no "id:" or "url:"
    if not glycan_id and not url and len(lines) == 1:
        glycan_id = lines[0]

    return glycan_id, url


def extract_annotated_images(output_dir,pdf_path):
    """Extract figures and annotations from a PDF file
    
    This function iterates through annotations first, then finds the figure
    each annotation belongs to by checking for intersection."""

    metadata = []
    #Track which figures have been saved to avoid duplicates
    saved_figures = {}

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            # Get all figures on current page
            figures = page.get_image_info(xrefs=True)

            # Get all annotations on the page
            annotations = list(page.annots())
            #print("annotations:",annotations)

            # glycan_metadata = {}
            for annot in annotations:
                    #print("annot:",annot)
                    #print("annot type:", annot.type)
                    
                    if annot.type[1] not in ['Square', 'Rect']:
                        continue
                        
                    annot_box = annot.rect

                    # Find which figure this annotation intersects with
                    matching_figure = None
                    overlap_ratio = None
                    fig_num = None
                        
                    for idx, figure in enumerate(figures,1):
                        fig_box = fitz.Rect(figure["bbox"])

                        # Check if annotation intersects with this figure
                        # Using intersection area to determine if annotation is on this figure
                        if fig_box.intersects(annot_box):
                            # Calculate intersection area as a percentage of annotation area
                            intersection = fig_box & annot_box
                            annot_area = annot_box.get_area()
                        
                            if annot_area > 0:
                                overlap_ratio = intersection.get_area() / annot_area
                                # If annotation is mostly (>50%) inside the figure, consider it a match
                                if overlap_ratio > 0.5:
                                    matching_figure = figure
                                    fig_num = idx
                                    break
                
                    # If no matching figure found then consider the image itself as a glycan and the dimensions as the figure diemnsions as well
                    if not matching_figure:
                        xref = annot.xref
                        # print(f"Warning: Annotation at {annot_box} on page {page_num} doesn't match any figure and overlap {overlap_ratio}")
                        comment = annot.info.get("content") or annot.info.get("subject") or ""
                        glycan_id, url = parse_comment(comment)
                        #print("----->>",glycan_id, annot.rect)

                        # Create filename for standalone annotated glycan image
                        figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_annot{xref}.png"
                        figure_path = os.path.join(output_dir, figure_filename)

                        # Save cropped annotation area as image
                        try:
                            pix = page.get_pixmap(clip = annot_box)
                            px_fig_width, px_fig_height = pix.width, pix.height
                        except:
                            px_fig_height = px_fig_width = 0
                        
                        # Fill in bounding box dimensions
                        # the glycan box starts at the top-left corner of the cropped region, so x0, y0 = 0
                        px_gly_x0 = 0
                        px_gly_y0 = 0
                        px_gly_w = int(annot_box.width)
                        px_gly_h = int(annot_box.height)
                        
                
                    # Extract figure information (including xref)
                    if matching_figure:
                        xref = matching_figure.get("xref")
                        fig_box = fitz.Rect(matching_figure["bbox"])

                        # Create filename
                        figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{fig_num}.png"
                        figure_path = os.path.join(output_dir, figure_filename)
                
                        # Get original figure dimensions in pixels
                        # Remove the scaling factor added to the figures while pasting them into the PDF
                        zoom_x = matching_figure["width"] / fig_box.width
                        zoom_y = matching_figure["height"] / fig_box.height
                        mat = fitz.Matrix(zoom_x, zoom_y)
                        pix = page.get_pixmap(matrix=mat, clip=fig_box, alpha=False)
                        px_fig_width = pix.width
                        px_fig_height = pix.height
                
                        # Calculate scaling factors
                        x_scale = fig_box.width / float(px_fig_width)
                        y_scale = fig_box.height / float(px_fig_height)
                
                        # Convert annotation coordinates to pixel coordinates relative to the figure
                        px_gly_x0 = round((annot_box.x0 - fig_box.x0) / x_scale)
                        px_gly_x1 = round((annot_box.x1 - fig_box.x0) / x_scale)
                        px_gly_y0 = round((annot_box.y0 - fig_box.y0) / y_scale)
                        px_gly_y1 = round((annot_box.y1 - fig_box.y0) / y_scale)
                
                        px_gly_w = abs(px_gly_x1 - px_gly_x0)
                        px_gly_h = abs(px_gly_y1 - px_gly_y0)
                
                        # Parse annotation comment
                        comment = annot.info.get("content") or annot.info.get("subject") or ""
                        glycan_id, url = parse_comment(comment)
                
                    annotation_data = {
                            'ID': glycan_id,
                            'url': url,
                            'xref': xref,
                            "gly_bbox": [px_gly_x0, px_gly_y0, px_gly_w, px_gly_h],
                            "comment": comment,
                            "fig_width": px_fig_width,
                            "fig_height": px_fig_height,
                            "figure_name": figure_filename,
                            "figure_path": figure_path,
                            "page_num": page_num,
                            "fig_num": fig_num
                        }

                    metadata.append(annotation_data)

                    # Save the figure only once if not already saved
                    figure_key = (page_num, fig_num)
                    if figure_key not in saved_figures:
                        if (px_fig_width > 60 and px_fig_height > 60):
                            pix.save(figure_path)
                            saved_figures[figure_key] = True
                           
    return metadata


def to_int(v):
    try: return int(v)
    except: return None

def merge_glycan_data_with_tsv(output_dir, glycan_data, tsv_path):
    existing_rows = {}
    existing_order = []

    # track the existing rows via column name 'ID' (which is unique)
    if os.path.exists(tsv_path):
        with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
            r = csv.DictReader(f, delimiter='\t')
            existing_order = r.fieldnames or []
            for row in r:
                rid = row.get('ID')
                if rid:
                    existing_rows[rid] = row

    # below code parses all the annotations that exist in the Manuscript
    # 1) if an annotation matches with an already existing ID from with TSV, then merge the data
    # 2) if a new manual annotation was added (ID doesnt exist in the TSV) - it will be added as a new row in the TSV file
    new_fields = []
    seen = set()
    for item in glycan_data:
        rid = item.get('ID')
        if not rid: 
            continue
        # step to merge new data with existing data from the TSV file
        for k in item:
            if k not in existing_order and k not in seen:
                seen.add(k)
                new_fields.append(k)


        if rid in existing_rows:
            existing_rows[rid].update(item)
        else:
            existing_rows[rid] = dict(item)

    # final header: ID, new fields, then existing fileds from provided TSV
    final_fields = ['ID'] if ('ID' in existing_order or any('ID' in d for d in glycan_data)) else []
    final_fields += [c for c in new_fields if c != 'ID' and c not in final_fields]
    final_fields += [c for c in existing_order if c not in ['ID', 'x', 'y', 'w', 'h', 'image_index', 'image_width', 'image_height'] and c not in final_fields]

    # write merged
    tsv_filename = os.path.basename(tsv_path).rsplit('.', 1)[0]
    merged_tsv_path = os.path.join(output_dir, f"{tsv_filename}_merged.tsv")

    with open(merged_tsv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=final_fields, delimiter='\t')
        w.writeheader()
        for row in existing_rows.values():
            w.writerow({col: row.get(col, '') for col in final_fields})

    print(f"Wrote merged TSV: {merged_tsv_path}")
    return merged_tsv_path

        
input_folder = args.folder
output_folder = args.output_dir
pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

for pdf_path in pdf_files:
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]
    tsv_path = os.path.join(input_folder, f"{file_name}.tsv")

    output_dir = os.path.join(output_folder, file_name)

    if not os.path.exists(output_dir):
        # shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(tsv_path):
        print(f"Skipping {file_name}: no matching TSV found.")
        continue

    # main steps for extraction and merging
    metadata = extract_annotated_images(output_dir,pdf_path)
    merge_glycan_data_with_tsv(output_dir,metadata,tsv_path)


