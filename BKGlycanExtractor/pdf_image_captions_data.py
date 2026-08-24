import os
import re
import sys
import time
import json
import shutil
# from difflib import SequenceMatcher

from PDFigCapX.code import xpdf_process
from .bbox import PDFBoundingBox 

class PDFiguesCaptionsData:

    @staticmethod
    def clean_captions(caption):

        if not caption or not isinstance(caption, str):
            return {}

        text = caption.replace('\n', ' ')
        text = text.strip()

        pattern = re.compile(
            r'^'
            r'(?:Supplementary|Suppl\.?|Supp\.?|Extended\s+Data\s+)?'  # optional prefix
            r'(Figure|Fig|FIG)s?'   # allow "Figures"
            r'(\.?)'
            r'\s*'                  # allow "Fig.1" (zero or more space)
            r'([A-Za-z]?\d+(?:[\.\-]\d+)?[a-zA-Z]?)'   # S1, 1, 1.1, 1a
            r'\s*'
            r'([\.\:\-\—\–]?)'      # optional separator: . : - — –
            r'\s*'
            r'(.*)$',
            re.IGNORECASE | re.DOTALL
        )

        match = pattern.match(text) 

        if match:
            prefix = match.group(1)   # "Figure", "Fig", "FIG", "Figures", etc.
            # dot = match.group(2)      # "." or ""
            figure_number = match.group(3)   # "1", "1a", "S1", "1.1", etc.
            # match.group(4) = separator (., :, -, —) optional to use
            rest = match.group(5)     # caption text

            caption = rest.strip()
            caption = re.sub(r'^[\.:\-—–]\s*', '', caption)
            caption = re.sub(r'^\s+', '', caption)

            return {"caption": caption, "figure_number": figure_number, 
                    "figure_label": prefix}

        return {}

    @staticmethod
    def figures_info(pdf,page_dpi):
        '''TODO accepts a pdf - use Image Manager (if multiple pdf's, folders of pdfs??)'''

        basename = os.path.splitext(pdf)[0]
        output_json = basename + '_figures.json'
    
        try:
            figures, info = xpdf_process.figures_captions_list(pdf, page_dpi)
        except Exception as exc:
            print("figures_captions_list failed for {}: {}".format(pdf, exc))
            raise

        summary = {
            "filename": pdf,
            "page_dimensions": {
                "width": info.get("page_width"),
                "height": info.get("page_height"),
            },
            "figures": [],
        }
        image_count = 0


        for page_name, entries in figures.items():
            page_no = int(page_name[4:-4])
            for image_number, box in enumerate(entries, start=1):
                image_count += 1
                x,y,w,h = box[0]    # note: x,y,w,h --> in pdf points
                # x0,y0,x1,y1 = x,y,x+w,y+h
                pdf_box = PDFBoundingBox(x=x,y=y,w=w+1,h=h+1, page_width=info["page_width"], page_height=info["page_height"])
                pdf_box.normalize()

                caption_box, caption = (box[1] if box[1] else (None, []))

                if caption_box:
                    c_x, c_y, c_w, c_h = caption_box
                    caption_box = PDFBoundingBox(x=c_x,y=c_y,w=c_w+1,h=c_h+1, page_width=info["page_width"], page_height=info["page_height"])
                    caption_box.normalize()
                
                #  clean up caption text
                figure_caption = ''.join(caption)      # the text string is in a list, standardize it to be a simple clean string 
                caption_dict = PDFiguesCaptionsData.clean_captions(figure_caption)

                summary["figures"].append({
                    "image_count": image_count,
                    "page_number": page_no,
                    "image_number": image_number,
                    # "pdf_box": pdf_box,
                    "pdf_fig_bbox": pdf_box.bbox(),        # x0,y0,x1,y1
                    "bbox": pdf_box.bbox(),                 # x0,y0,x1,y1
                    **caption_dict,
                    "pdf_fig_width": pdf_box.width(),
                    "pdf_fig_height": pdf_box.height(), 
                    "page_width": info["page_width"],
                    "page_height": info["page_height"],
                    "width": pdf_box.bbox()[2] * info['png_ratio'],
                    "height": pdf_box.bbox()[3] * info['png_ratio'],
                })

        with open(output_json, "w") as fh:
            json.dump(summary, fh, indent=2)

        # delete the folder which contains extra data (folder with all flattened pages and intermediate json file)
        # note that athejson file with the figures data will still continue to exist and will be named as: <pdf_name>_figures.json
        pages_dir = pdf.rsplit('.')[0]

        if os.path.exists(pages_dir) and os.path.isdir(pages_dir):
            shutil.rmtree(pages_dir)

        return output_json