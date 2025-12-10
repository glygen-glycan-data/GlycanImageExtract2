import os
import sys
import json
import time
import shutil
import fitz

from BKGlycanExtractor.bbox import PDFBoundingBox
from BKGlycanExtractor.pdfhandler import PDFHandler, CompoundPDFImageFilter, PDFXRefImageFilter, PDFImageSizeFilter
from BKGlycanExtractor.compareboxes import CompareBoxes

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to sys.path to import PDFFigCapX_mine
sys.path.append(parent_dir)
from PDFigCapX.code import xpdf_process


class FigCapX_Search:
    
    @staticmethod
    def figures_info(pdf):
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
            flag = 0
            wrong_count = 0
            info = {'fig_no_est': 0}
            figures = {}
            while flag==0 and wrong_count<5:
                try:
                    figures, info = xpdf_process.figures_captions_list(pdf)
                    flag = 1

                except Exception as exc:
                    wrong_count = wrong_count +1
                    time.sleep(5)
                    print("Retrying figures_captions_list for {} ({})".format(pdf, exc))
    
            data[pdf]['fig_no'] = info['fig_no_est']

            # output_file_path = os.path.join(self.output_path, pdf[:-4])
            # if not os.path.isdir(output_file_path):
            #     os.mkdir(output_file_path)      

            if not flag:
                # continue  # or handle failure
                # TODO log the error
                # TODO remove the data dict
                pass

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
                        "figure_number": figure_number,
                        # "pdf_box": pdf_box,
                        "pdf_fig_bbox": pdf_box.bbox(),        # x0,y0,x1,y1
                        "caption_text": figure_caption,
                        # "figure_name": figure_name,
                        "pdf_fig_width": pdf_box.width(),
                        "pdf_fig_height": pdf_box.height(), 
                        "page_width": info["page_width"],
                        "page_height": info["page_height"],
                        # "xres": info["xres"],
                        # "yres": info["yres"],
                        **({"caption_bbox": caption_box.bbox()} if caption_box else {}),
                    })

            with open(output_json, "w") as fh:
                json.dump(summary, fh, indent=2)

            # delete the folder which contains extra data (folder with all flattened pages and intermediate json file)
            # note that athejson file with the figures data will still continue to exist and will be named as: <pdf_name>_figures.json
            pages_dir = pdf.rsplit('.')[0]

            if os.path.exists(pages_dir) and os.path.isdir(pages_dir):
                shutil.rmtree(pages_dir)

        return output_json


class PDF_Figure_Search:
    dpi=300

    def xref_figure_info(self, input_filepath):
        """
        Extract figure metadata from PDF using fitz - based on XREF of the images
        Stores per page information (about figures, etc) in a data structure.
        
        Information from the data structure about the pdf metadata - can be used to find images in the pdf and later find glycans using downstream methods.

        Sample Data structure - pdf_metadata
        pdf_metadata = {

            page_num (int): {
                image_number (int): {
                    rest other key-value pairs about the figure
                }
            },
            ...  
        }
        """

        pdf_metadata = {}

        pdf = PDFHandler(input_filepath)

        filter = CompoundPDFImageFilter(
            PDFXRefImageFilter(min_xref=1),
            PDFImageSizeFilter(width=90,height=90)
        )

        for fig_metadata in pdf.figures(filter=filter):
            page_number = fig_metadata['page_number']
            image_number = fig_metadata['image_number']

            if page_number not in pdf_metadata:
                pdf_metadata[page_number] = {}

            pdf_metadata[page_number][image_number] = {k:v for k,v in fig_metadata.items() if k in ('bbox', 'width', 'height', 'xref', 'pdf_fig_bbox', 'pdf_fig_width', 'pdf_fig_height', 'page_width', 'page_height', 'pdf_fig_height', 'image_count', 'dpi')}
            pdf_metadata[page_number][image_number].update({'page_number': page_number, 'image_number': image_number})

            self.log_file.write(
                f"\nXREF: {fig_metadata.get('xref')}, Page number: {page_number}, Image number: {image_number},  bbox: {fig_metadata["pdf_fig_bbox"]}, Width: {fig_metadata['pdf_fig_width']}, Height: {fig_metadata['pdf_fig_height']}\n"
            )

        return pdf_metadata

    def figcap_figure_info(self, input_filepath):
        """
        Extract figure metadata from PDF using PDFigCapX repository
        Stores per page information (about figures, etc) in a data structure.
        
        Information from the data structure about the pdf metadata - can be used to find images in the pdf and later find glycans using downstream methods.

        Sample Data structure - pdf_metadata
       
        pdf_metadata = {

            page_num (int): {
                figure_number (int): {
                    rest other key-value pairs about the figure
                }
            },
            ...  
        }
        """
        
        pdf_metadata = {}
        figures_data = {}       # data obtained from PDFigCapX repository

        json_data_path = FigCapX_Search.figures_info(input_filepath) 
        with open(json_data_path) as f:
            figures_data = json.load(f)

        try:
            # delete the json file after loading the semantics data obtained from PDFigCapX
            os.remove(json_data_path)
        except OSError as e:
            print("\nCould not delete json file obtained from PDFigCapX")

        doc = fitz.open(input_filepath)

        for result in figures_data.get("figures", {}):
            try:
                page_num = result['page_number']
                figure_num = result['figure_number']

                if page_num not in pdf_metadata:
                    pdf_metadata[page_num] = {}

                image_count = result['image_count']

                page = doc[page_num-1]

                # TODO: edge case - verify that the bbox contains an image  - see PDF with PMID 20060370
                # IDEA 1: if the xref based figure extration doesnt contain a co-inciding box - then drop the box
                # IDEA 2: everything has a xref (i.e the text blocks, headings, etc) - so if a bounding box is drawn in the proximity of the text block xref - eliminate the box

                # TODO edge case - sometimes the caption bbox encompasses the figure as well - so make sure that the
                # caption bbox always starts below the figure

                # dpi = PDFHandler.calculate_dpi(result)
        
                figure_metadata = {
                    "page_number": page_num,
                    "image_number": figure_num,      # number based on per page
                    "image_count": image_count,   # cumulative count
                    **{k:v for k,v in result.items() if k in ('pdf_fig_bbox', 'pdf_fig_width', 'pdf_fig_height', 'caption_bbox', 'figure_name','caption_text', 'page_width', 'page_height')},
                    # "dpi": dpi if dpi else None
                }

                pdf_metadata[page_num][figure_num] = {**figure_metadata}

            except Exception as e:
                self.log_file.write(
                    f"\nException occured while extracting a figure from the pdf: {e}."
                )
                print("Exception occured in figCap extraction:", e)

        return pdf_metadata

    def find_containment_groups(self, all_boxes_lookup, merged_figures, matched_box_keys, **kwargs):
        '''
            Selects the biggest boxes and eliminates all the other boxes that are 
            contained within in.
            
            args:
            all_boxes_lookup: dict
                key: (figure_no, figure_type --> either figcap or xref_fig)
                value: info about the figure
        '''
        # Find all containment relationships between boxes.
        containment_groups = {}
        all_container_keys = set()

        # Sort keys by area (larger boxes first) for efficient processing
        sorted_figures = dict(sorted(
            all_boxes_lookup.items(),
            key=lambda item: item[1]['box'].area(),
            reverse=True
        ))


        for i, ((fig_no1, fig_type1), fig_info1) in enumerate(sorted_figures.items()):
            box1 = fig_info1['box']

            for j, ((fig_no2, fig_type2), fig_info2) in enumerate(sorted_figures.items()):
                if i >= j:   
                    continue 
                
                try:
                    box2 = fig_info2['box']

                    # get containment relationship --> this is 
                    # supposed to return (contained_box, container_box) pair
                    containment = CompareBoxes.get_containment(box1,box2)

                    if containment is None:
                        continue
                    
                    contained_box, container_box = containment

                    # Determine which id corresponds to container
                    if container_box is box1:
                        container_key = (fig_no1, fig_type1)
                        contained_key = (fig_no2, fig_type2)
                    elif container_box is box2:
                        container_key = (fig_no2, fig_type2)
                        contained_key = (fig_no1, fig_type1)

                    # add to containment groups
                    if container_key not in containment_groups:
                        containment_groups[container_key] = []
                        all_container_keys.add(container_key)

                    if contained_key not in containment_groups[container_key]:
                        containment_groups[container_key].append(contained_key)
                    
                    matched_box_keys.add(contained_key)         # so these contained_key related boxes should not appear in the merged_figure metadata anymore as have been replaced by on large container box

                except Exception as e:
                    print("\nException occuered while finding containment:", e)

        # finally add all the final container keys in the matched box keys
        # adding this after the for loops ends --> so that the big container is free to grab 'n' no. of contained boxes within it
        matched_box_keys.update(all_container_keys)

        # clean the data before adding it to merged_figures (i.e bring to the the standard convention followed throughout the code)
        for key, val in all_boxes_lookup.items():
            if key in containment_groups:
                page_number = val['page_number']
                image_number = val['image_number']

                if page_number not in merged_figures:
                    merged_figures[page_number] = {}

                merged_figures[page_number][image_number] = val
                merged_figures[page_number][image_number].update({'merge_type': 'containment'})

    def match_by_iou(self, all_boxes_lookup, merged_figures, matched_box_keys, iou_threshold=0.8, **kwargs):
        
        # Collect all possible matches with their IOU scores
        all_matches = []  # List of (iou_score, (fig_no1, fig_type1), (fig_no2, fig_type2))
                
        for i, ((fig_no1, fig_type1), fig_info1) in enumerate(all_boxes_lookup.items()):
            if (fig_no1, fig_type1) in matched_box_keys:
                continue

            pdf_fig_box1 = fig_info1['box']
                    
            for j, ((fig_no2, fig_type2), fig_info2) in enumerate(all_boxes_lookup.items()):

                if (fig_no2, fig_type2) in matched_box_keys or fig_type1 == fig_type2:
                    continue

                try:
                    pdf_fig_box2 = fig_info2['box']

                    iou = CompareBoxes.iou(pdf_fig_box1, pdf_fig_box2)
                    if iou >= iou_threshold:
                        # Store match with IOU score
                        all_matches.append((iou, (fig_no1, fig_type1), (fig_no2, fig_type2)))
                except Exception as e:
                    print("\nException occured while checking IOU threshold:", e)

        # Sort by IOU (highest first) - greedy matching
        all_matches.sort(reverse=True, key=lambda x: x[0])
        
        # Process matches in order of best IOU first
        for iou, (fig_no1, fig_type1), (fig_no2, fig_type2) in all_matches:
            # Skip if either box is already matched
            if (fig_no1, fig_type1) in matched_box_keys or (fig_no2, fig_type2) in matched_box_keys:
                continue
            
            try:
                # Mark as matched
                matched_box_keys.add((fig_no1, fig_type1))
                matched_box_keys.add((fig_no2, fig_type2))

                fig_info1 = all_boxes_lookup[(fig_no1, fig_type1)]
                fig_info2 = all_boxes_lookup[(fig_no2, fig_type2)]

                # Merge boxes - by taking union of the boxes
                merged_bbox = CompareBoxes.union_pdf_boxes(fig_info1['box'], fig_info2['box'])      # method from compareboxes - which returns a bbox (doesnt return box object - circular import problem , will think about what to do...)
                merged_box = PDFBoundingBox(bbox=merged_bbox, page_width=fig_info1['page_width'], page_height=fig_info1['page_height'])
                
                # page_number from both the extraction methods will be the same,
                # but the image_number might be different (b/c how the different extraction method identified images) 
                # --> so take image_number based on the length of images on the page
                page_number = fig_info2['page_number']

                # clean the data before adding it to merged_figures (i.e bring to the the standard convention followed throughout the code)
                if page_number not in merged_figures:
                    merged_figures[page_number] = {}
                
                # image_number = len(merged_figures[page_number]) + 1
                existing_image_numbers = list(merged_figures[page_number].keys())
                if existing_image_numbers:
                    image_number = max(existing_image_numbers) + 1
                else:
                    image_number = 1

                dpi1 = fig_info1.get('dpi')
                dpi2 = fig_info2.get('dpi')

                dpi_values = [d for d in [dpi1, dpi2] if d is not None]
                dpi = max(dpi_values) if dpi_values else DPI   # default is 150

                merged_entry = {}
                merged_entry.update(fig_info1)
                merged_entry.update(fig_info2)

                merged_entry['pdf_fig_bbox'] = merged_bbox
                merged_entry['pdf_fig_height'] = merged_box.height()
                merged_entry['pdf_fig_width'] = merged_box.width()
                merged_entry['page_number'] = page_number
                merged_entry['image_number'] = image_number
                merged_entry['merge_type'] = 'iou'
                merged_entry['merged_iou'] = iou
                merged_entry['dpi'] = dpi

                merged_figures[page_number][image_number] = merged_entry

            except Exception as e:
                print("\nException occured while matching and merging boxes based on IOU:", e)
            
    def merge_figures(self, xref_figures, figcap_figures, **kwargs):
        '''
        Approach:
        For each page, this method matches figures based on
        1) Containment - selects the biggest boxes and eliminates all the other boxes that are contained within in - method find_containment_groups()
        2) IOU - if boxes from the different figure identification methods overlap based on an IOU threshold, then the union of both boxes will be taken - method match_by_iou()
        3) Rest of the unmatched boxes from both figure identification methods are included.
        '''
    
        # convert all the fig boxes into pdf box objects
        all_boxes_lookup = {}

        for figcap_fig_no, figcap_fig_data in figcap_figures.items():
            figcap_fig_data['box'] = PDFBoundingBox(bbox=figcap_fig_data['pdf_fig_bbox'], page_width=figcap_fig_data['page_width'], page_height=figcap_fig_data['page_height'])
            figcap_fig_data['extraction_type'] = 'figcap'
            all_boxes_lookup[(figcap_fig_no, 'figcap')] = figcap_fig_data
        
        for xref_fig_no, xref_fig_data in xref_figures.items():
            xref_fig_data['box'] = PDFBoundingBox(bbox=xref_fig_data['pdf_fig_bbox'], page_width=xref_fig_data['page_width'], page_height=xref_fig_data['page_height'])
            xref_fig_data['extraction_type'] = 'xref'
            all_boxes_lookup[(xref_fig_no, 'xref')] = xref_fig_data

        if not all_boxes_lookup:
            return []

        # track matched boxes
        merged_figures = {}
        matched_box_keys = set()

        # Step 1 - find if box(es) is/are contained another big box - when comparing from figcap_figures & xref_figures
        # because there are some edge cases where multiple small boxes are identified in one 
        # method (xref_based) and another method creates one single box large box encompassing the different components (figCapX) - so this logic should handle 
        # containment cases and the above edge cases
        self.find_containment_groups(all_boxes_lookup, merged_figures, matched_box_keys)        

        # step 2: for any remaining boxes - check if boxes can be unionized/merged based
        # on IOU
        self.match_by_iou(all_boxes_lookup, merged_figures, matched_box_keys, iou_threshold=0.8)

        # step 3: rest of the unmatched items will be added to the merged figures as individual boxes for the page 
        # because one of the figure extraction methods could have FN's
        for (fig_no, fig_type), fig_info in all_boxes_lookup.items():
            if (fig_no, fig_type) not in matched_box_keys:
                matched_box_keys.add((fig_no, fig_type))
                page_number = fig_info['page_number']
                image_number = fig_info['image_number']
                if page_number not in merged_figures:
                    merged_figures[page_number] = {}
                merged_figures[page_number][image_number] = fig_info
                merged_figures[page_number][image_number].update({'merge_type': 'regular'})

        return merged_figures

    def merge_pdf_fig_info(self, xref_pdf_metadata, figcap_pdf_metadata):
        '''
        Approach:
        For each page --> merge_figures method is supposed matches figures based on
        1) Containment - selects the biggest box and eliminates all the other boxes that are contained within in --> method find_containment_groups()
        2) IOU - if boxes from the different figure metadata extraction methods (xref_based vs PDFfigCapX) overlap based on an IOU threshold, then the union of both boxes will be taken - method match_by_iou()
        3) Rest of the unmatched boxes from both figure identification methods are included.
        '''
        
        merged_pdf_info = {}
        global_image_count = 1
    
        # since page_nos are the key's for xref_pdf_metadata & figcap_pdf_metadata - some page_no's may or
        # may not be present in one of the dicts based on the method used for extraction - so need to
        # handle these cases and consider the data from both the cases and use them - get keys (page_no's from both dicts)
        all_page_nos = xref_pdf_metadata.keys() | figcap_pdf_metadata.keys()
        
        for pg_no in all_page_nos:
            xref_figures = xref_pdf_metadata.get(pg_no, {})     # can have multiple figures on the page
            figcap_figures = figcap_pdf_metadata.get(pg_no, {})   # can have multiple figures on the page

            # if both thr dicts have data - need to process the data --> to unionize/merge their results
            # else consider the data from whichever dict provides it

            if not xref_figures and not figcap_figures:
                continue
            else:
                # need to verify that proper image_number's are applied to each page in a sorted manner,
                # using containment, iou and other methods sometimes messes up the ordering of images
                merged_figures = self.merge_figures(xref_figures,figcap_figures)

                # get the figures for this page
                page_figures = merged_figures.get(pg_no, {})

                if not page_figures:
                    continue

                # sort figures by position on page
                figures_list = list(page_figures.items())

                def get_sort_key(item):
                    img_num, img_info = item
                    bbox = img_info.get('pdf_fig_bbox', [0, 0, 0, 0])
                    
                    if len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                        
                        top_y = min(y1, y2)  # Top edge (smaller y = top)
                        left_x = min(x1, x2)
                        return (top_y, left_x)  # Ascending (smaller y = top first)
    
                    return (float('inf'), float('inf'))

                # Sort figures by position
                sorted_figures = sorted(figures_list, key=get_sort_key)

                # Initialize page dict if not exists
                if pg_no not in merged_pdf_info:
                    merged_pdf_info[pg_no] = {}

                # Renumber sequentially and add to merged_pdf_info
                # Note: global_image_count persists across pages for unique numbering
                image_number = 1    # image_number per page - so the counter refreshes for every new page
                for old_image_number, image_info in sorted_figures:
                    merged_pdf_info[pg_no][image_number] = image_info
                    image_info['image_count'] = global_image_count
                    # image_info['dpi'] = image_info.get('dpi',PDF_Figure_Search.dpi)   # dpi obtained is not doing great
                    image_info['dpi'] = PDF_Figure_Search.dpi
                    image_number += 1
                    global_image_count += 1

        return merged_pdf_info
    

if __name__ == '__main__':
    fs = FigCapX_Search()
    pdf_path = sys.argv[1]      # pdf path
    fig_json_path = fs.figures_info(pdf_path)