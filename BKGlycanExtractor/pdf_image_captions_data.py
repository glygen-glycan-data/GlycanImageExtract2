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
    def clean_captions(caption_text):

        if not caption_text or not isinstance(caption_text, str):
            return {
                'label': None,
                'full_caption_text': caption_text,
                'cleaned_caption': False
            }

        text = caption_text.replace('\n', ' ')
        text = text.strip()

        # Match prefix (with optional period) + number, then extract caption from remainder
        # Pattern captures: (prefix, optional_period, number, everything_after)
        pattern = r'^(Figure|Fig|FIG)(\.?)\s+(\d+(?:[\.\-]\d+)?[a-zA-Z]?)\s*(.*)$'

        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            prefix = match.group(1)  # "Figure", "Fig", "FIG", etc.
            prefix_cap = (prefix[0].upper() + prefix[1:]) if prefix else ""
            number = match.group(3)  # "1", "1a", "1.1", etc.
            rest = match.group(4)  # Everything after the number (may include separators)

            # Construct normalized label
            label = f"{prefix_cap} {number}".strip()
            
            # Extract caption by removing common separators and whitespace
            # Handles: ". ", ": ", "- ", "— ", "– ", ".", "-", etc.
            caption = rest.strip()
            # Remove leading separators (period, colon, dashes) and their whitespace
            caption = re.sub(r'^[\.:\-—–]\s*', '', caption)  # Remove separator + optional whitespace
            caption = re.sub(r'^\s+', '', caption)  # Remove any remaining leading whitespace
            

            return {
                'label': label,
                'caption_text': caption,
                'full_caption_text': caption_text,
                'cleaned_caption': True
            }
        
        # If no pattern matched, return the full text as caption
        return {
            'label': None,
            'full_caption_text': caption_text,
            'cleaned_caption': False
        }

    
    @staticmethod
    def figures_info(pdf,page_dpi):
        '''TODO accepts a pdf - use Image Manager (if multiple pdf's, folders of pdfs??)'''

        # pdf_full_path = os.path.join(self.input_path, pdf)
        # basepath = os.path.dirname(pdf)
        # basename = os.path.splitext(os.path.basename(pdf))[0]
        basename = os.path.splitext(pdf)[0]
        output_json = basename + '_figures.json'
    
        data = {}

        data[pdf] = {}
        data[pdf]['figures'] = []
        data[pdf]['pages_annotated'] = []
        pdf_flag = 0

        if pdf_flag == 0:
            try:
                figures, info = xpdf_process.figures_captions_list(pdf, page_dpi)
            except Exception as exc:
                print("figures_captions_list failed for {}: {}".format(pdf, exc))
                raise
    
            data[pdf]['fig_no'] = info['fig_no_est']

            # output_file_path = os.path.join(self.output_path, pdf[:-4])
            # if not os.path.isdir(output_file_path):
            #     os.mkdir(output_file_path)      

            # if not flag:
            #     # continue  # or handle failure
            #     # TODO log the error
            #     # TODO remove the data dict
            #     pass

            summary = {
                "filename": pdf,
                "total_pages": info.get('page_no'),
                "figure_count_estimate": info.get("fig_no_est", 0),
                "page_dimensions": {
                    "width": info.get("page_width"),
                    "height": info.get("page_height"),
                },
                "figures": [],
            }
            image_count = 0


            for page_name, entries in figures.items():
                page_no = int(page_name[4:-4])
                for figure_number, box in enumerate(entries, start=1):
                    image_count += 1
                    x,y,w,h = box[0]    # note: x,y,w,h --> in pdf points
                    # x0,y0,x1,y1 = x,y,x+w,y+h
                    pdf_box = PDFBoundingBox(x=x,y=y,w=w+1,h=h+1, page_width=info["page_width"], page_height=info["page_height"])
                    pdf_box.normalize()

                    caption_box, caption_text = (box[1] if box[1] else (None, []))

                    if caption_box:
                        c_x, c_y, c_w, c_h = caption_box
                        caption_box = PDFBoundingBox(x=c_x,y=c_y,w=c_w+1,h=c_h+1, page_width=info["page_width"], page_height=info["page_height"])
                        caption_box.normalize()
                    
                    #  clean up caption text
                    figure_caption = ''.join(caption_text)      # the text string is in a list, standardize it to be a simple clean string 
                    caption_dict = PDFiguesCaptionsData.clean_captions(figure_caption)
                    if figure_caption is not None:
                        # Remove newlines and normalize whitespace
                        figure_caption = figure_caption.replace('\n', ' ')

                        # Remove "Figure X." prefix - split on first '.' and take everything after
                        # if '.' in figure_caption:
                        #     parts = figure_caption.split('.', 1)
                        #     if len(parts) > 1:
                        #         figure_name = parts[0].strip()
                        #         figure_caption = parts[1].strip()
                        
                        # Clean up extra whitespace
                        figure_caption = ' '.join(figure_caption.split())


                    summary["figures"].append({
                        "image_count": image_count,
                        "page_number": page_no,
                        "image_number": figure_number,
                        # "pdf_box": pdf_box,
                        "pdf_fig_bbox": pdf_box.bbox(),        # x0,y0,x1,y1
                        "bbox": pdf_box.bbox(),                 # x0,y0,x1,y1
                        "caption_text": figure_caption,
                        # "figure_name": figure_name,
                        "pdf_fig_width": pdf_box.width(),
                        "pdf_fig_height": pdf_box.height(), 
                        "page_width": info["page_width"],
                        "page_height": info["page_height"],
                        "width": pdf_box.bbox()[2] * info['png_ratio'],
                        "height": pdf_box.bbox()[3] * info['png_ratio'],
                        **caption_dict
                    })

            with open(output_json, "w") as fh:
                json.dump(summary, fh, indent=2)

            # delete the folder which contains extra data (folder with all flattened pages and intermediate json file)
            # note that athejson file with the figures data will still continue to exist and will be named as: <pdf_name>_figures.json
            pages_dir = pdf.rsplit('.')[0]

            if os.path.exists(pages_dir) and os.path.isdir(pages_dir):
                shutil.rmtree(pages_dir)

        return output_json