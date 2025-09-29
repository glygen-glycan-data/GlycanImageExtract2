import os
import argparse
import fitz
import glob
import shutil
import csv


def parse_comment(comment: str):
    """Parse annotation comment into (glycan_id, url).

    2 different formats of comments exist, need to handle both the types to extract id and url
    
    Expected formats:
    - Multi-line with prefixes, e.g.:
        ID: G123
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
    """Extract figures and annotations from a PDF file"""

    metadata = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, 1):
            figures = page.get_image_info(xrefs=True)

            # Get all annotations on the page
            annotations = list(page.annots())

            # get all the figures on the current page
            for fig_num, figure in enumerate(figures, 1):
                xref = figure["xref"]

                # basics details about the figure
                base_figure = doc.extract_image(xref)
                figure_bytes = base_figure["image"]
                figure_ext = base_figure["ext"]

                px_fig_width = base_figure["width"]
                px_fig_height = base_figure["height"]
                
                # Create filename
                figure_filename = f"{os.path.basename(output_dir)}_p{page_num}_f{fig_num}.png"
                figure_path = os.path.join(output_dir, figure_filename)

                annots_in_figure = False

                fig_box = fitz.Rect(figure["bbox"])

                x_scale = fig_box.width  / float(px_fig_width)
                y_scale = fig_box.height / float(px_fig_height)

                # glycan_metadata = {}
                for annot in annotations:
                    if fig_box.intersects(annot.rect):
                        annots_in_figure = True
                        
                        annot_box = annot.rect

                        px_gly_x0 = round((annot_box.x0 - fig_box.x0) / x_scale)
                        px_gly_x1 = round((annot_box.x1 - fig_box.x0) / x_scale)
                        px_gly_y0 = round((annot_box.y0 - fig_box.y0) / y_scale)
                        px_gly_y1 = round((annot_box.y1 - fig_box.y0) / y_scale)

                        px_gly_w = abs(px_gly_x1-px_gly_x0)
                        px_gly_h = abs(px_gly_y1-px_gly_y0)

                        comment = annot.info.get("content") or annot.info.get("subject") or ""
                        glycan_id, url = parse_comment(comment)

                        annotation_data = {
                            'ID': glycan_id,
                            'url': url,
                            'xref': xref,
                            "gly_bbox": [px_gly_x0, px_gly_y0, px_gly_w, px_gly_h],
                            # "comment": comment,
                            "fig_width": px_fig_width,
                            "fig_height": px_fig_height,
                            "figure_name": figure_filename,
                            "figure_path": figure_path,
                            "page_num": page_num,
                            "fig_num": fig_num
                        }

                        metadata.append(annotation_data)

                # Only save the figure if it has annotations
                if annots_in_figure:
                    with open(figure_path, "wb") as figure_file:
                        figure_file.write(figure_bytes)
                    
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

    # final header: ID, new fields, then existing fileds from provided TSV
    final_fields = ['ID'] if ('ID' in existing_order or any('ID' in d for d in glycan_data)) else []
    final_fields += [c for c in new_fields if c != 'ID' and c not in final_fields]
    final_fields += [c for c in existing_order if c != 'ID' and c not in final_fields]

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

        
def main(input_folder):
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

    for pdf_path in pdf_files:
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        tsv_path = os.path.join(input_folder, f"{file_name}.tsv")

        output_dir = os.path.splitext(os.path.basename(pdf_path))[0]

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        if not os.path.exists(tsv_path):
            print(f"Skipping {file_name}: no matching TSV found.")
            continue

        # main steps for extraction and merging
        metadata = extract_annotated_images(output_dir,pdf_path)
        merge_glycan_data_with_tsv(output_dir,metadata,tsv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract annotated figures and comments from PDFs")
    parser.add_argument("-f", "--folder", type=str, required=True,
                        help="Folder containing PDFs and matching TSV files")
    args = parser.parse_args()
    main(args.folder)