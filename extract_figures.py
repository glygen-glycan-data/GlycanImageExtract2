import os
import argparse
import fitz
import pandas as pd
import glob
import shutil


def scale_annot_to_image(annot_rect, fig_rect, fig_width, fig_height):
    """
    Convert an annotation rectangle in PDF coordinates to
    pixel coordinates within the cropped figure image.
    """
    scale_x = fig_width / fig_rect.width
    scale_y = fig_height / fig_rect.height

    x1 = (annot_rect.x0 - fig_rect.x0) * scale_x
    y1 = (annot_rect.y0 - fig_rect.y0) * scale_y
    x2 = (annot_rect.x1 - fig_rect.x0) * scale_x
    y2 = (annot_rect.y1 - fig_rect.y0) * scale_y

    return int(x1), int(y1), int(x2), int(y2)


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


def extract_annotated_figures(page, page_num, output_dir):
    """Extract figures on a page that contain annotations."""
    all_metadata = []

    # candidate figure regions = image blocks
    figures = [fitz.Rect(b["bbox"]) for b in page.get_text("dict")["blocks"] if b["type"] == 1]
    if not figures:
        return all_metadata

    annots = list(page.annots() or [])

    for fig_index, fig_rect in enumerate(figures, start=1):
        fig_annots = [a for a in annots if fig_rect.intersects(a.rect)]
        if not fig_annots:
            continue

        # render figure
        pix = page.get_pixmap(clip=fig_rect, annots=False)
        figure_filename = f"{os.path.basename(output_dir)}_p{page_num+1}_f{fig_index}.png"
        figure_path = os.path.join(output_dir, figure_filename)
        pix.save(figure_path)

        # collect metadata
        for annot in fig_annots:
            x1, y1, x2, y2 = scale_annot_to_image(annot.rect, fig_rect, pix.width, pix.height)
            comment = annot.info.get("content") or annot.info.get("subject") or ""
            glycan_id, url = parse_comment(comment)

            metadata = {
                "id": glycan_id,
                # "page_num": page_num + 1,
                "fig_width": pix.width,
                "fig_height": pix.height,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "pdf_coordinates": [annot.rect.x0, annot.rect.y0,
                                    annot.rect.x1, annot.rect.y1],
                "comment": comment,
                "figure_name": figure_filename,
                "figure_path": figure_path,
                # "url": url,
            }
            all_metadata.append(metadata)

    return all_metadata


def extract_annotations(pdf_path):
    """Extract figures and annotations from a PDF file into a list of metadata dicts."""
    output_dir = os.path.splitext(os.path.basename(pdf_path))[0]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    all_metadata = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            all_metadata.extend(extract_annotated_figures(page, page_num, output_dir))
    return all_metadata


def merge_with_existing(df, tsv_path):
    """Merge extracted data with an existing TSV (glycan mapping)."""
    glycan_df = pd.read_csv(tsv_path, sep="\t")
    df["id"] = df["id"].str.strip()
    glycan_df["ID"] = glycan_df["ID"].str.strip()

    # Outer join
    merged = glycan_df.merge(df, how="outer", left_on="ID", right_on="id")

    merged["ID"] = merged["ID"].fillna(merged["id"])

    # Drop the duplicate id col
    merged = merged.drop(columns=["id"])

    return merged


def main(input_folder):
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

    for pdf_path in pdf_files:
        file_name = os.path.splitext(os.path.basename(pdf_path))[0]
        tsv_path = os.path.join(input_folder, f"{file_name}.tsv")

        if not os.path.exists(tsv_path):
            print(f"Skipping {file_name}: no matching TSV found.")
            continue

        metadata = extract_annotations(pdf_path)
        df = pd.DataFrame(metadata)

        merged_tsv_path = os.path.join(file_name, f"{file_name}_merged.tsv")
        merged_df = merge_with_existing(df, tsv_path)
        merged_df.to_csv(merged_tsv_path, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract annotated figures and comments from PDFs")
    parser.add_argument("-f", "--folder", type=str, required=True,
                        help="Folder containing PDFs and matching TSV files")
    args = parser.parse_args()
    main(args.folder)