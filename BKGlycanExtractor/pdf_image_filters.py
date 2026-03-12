import os
import sys

from BKGlycanExtractor.compareboxes import CompareBoxes

class ImageFilter:
    '''
    PDF Image Filters
    System for filtering and merging PDF images (fitz and figcap based)
    '''
    def apply(self, pdf_page_number, fitz_figures, figcap_figures, 
                merged_figures, merged_figures_keys):
        raise NotImplementedError


class DetectFragmentedFitz(ImageFilter):
    '''
    Detects if fitz images are fragmented (many small pieces that should be replaced by figcap).
    If no fragments on the page - add the fitz figures(s) to merged_figures.

    Strategy: If there are many small fitz images (>5) that cover a large portion of the page
    (width, height, or total area), they are likely fragments.

    NOTE: This filter only FLAGS fragmentation - it doesn't drop images. Individual fragment
    handling is done by FilterFitzByFigcapContainers, which can identify which specific fitz
    images are fragments vs good fitz images by checking if they are contained in figcap images.

    This allows good images and fragments to coexist on the same page, so good images will
    be processed normally, while fragments will be handled by FilterFitzByFigcapContainers.
    '''

    def __init__(self, min_fragments=5, page_coverage_threshold=0.4, width_coverage_threshold=0.6, height_coverage_threshold=0.6):
        self.min_fragments = min_fragments
        self.page_coverage_threshold = page_coverage_threshold
        self.width_coverage_threshold = width_coverage_threshold
        self.height_coverage_threshold = height_coverage_threshold

    def _is_fragmented(self, fitz_figures):
        '''
        Detect if fitz figures are fragmented.
        '''
        if len(fitz_figures) < self.min_fragments:
            return False

        bboxes = []
        for fitz_fig_no, fitz_fig_data in fitz_figures.items():
            bboxes.append(fitz_fig_data['pdf_fig_bbox'])
            page_width = fitz_fig_data['page_width']
            page_height = fitz_fig_data['page_height']

        if len(bboxes) < self.min_fragments:
            return False

        # get union of all fitz pdf fig bboxes
        union_bbox = CompareBoxes.union_pdf_boxes(bboxes)
        union_bbox_width = union_bbox[2] - union_bbox[0]
        union_bbox_height = union_bbox[3] - union_bbox[1]
        union_bbox_area = union_bbox_width * union_bbox_height

        # calculate the coverage of the unioned box on the page
        page_area = page_width * page_height

        width_coverage = union_bbox_width / page_width if page_width > 0 else 0
        height_coverage = union_bbox_height / page_height if page_height > 0 else 0
        area_coverage = union_bbox_area / page_area if page_area > 0 else 0

        # check if coverage threhsolds are met
        # Note: page_coverage_threshold is 0.4 (small) because the page includes margins
        if (area_coverage >= self.page_coverage_threshold or
            width_coverage >= self.width_coverage_threshold or
            height_coverage >= self.height_coverage_threshold):
            return True

        return False

    def apply(self, pdf_page_number, fitz_figures, figcap_figures, 
            merged_figures, merged_figures_keys):
        '''
        Detect if fitz figures are fragmented piece meals.

        - If NOT fragmented: Add all good fitz images are added to merged_figures.
        - If fragmented: Don't add them (let FilterFitzByFigcapContainers handle individual
          fragment detection by checking containment in figcap).
          This allows good images and fragments to coexist on the same page.

        '''

        if pdf_page_number not in merged_figures:
            merged_figures[pdf_page_number] = {}

        is_fragmented = self._is_fragmented(fitz_figures)

        if not is_fragmented:
            for fitz_fig_no, fitz_fig_data in sorted(fitz_figures.items(), key=lambda k: k[0]):
                fitz_key = (fitz_fig_no, 'fitz')
                if fitz_key not in merged_figures_keys:
                    merged_figures[pdf_page_number][fitz_key] = fitz_fig_data.copy()
                    merged_figures[pdf_page_number][fitz_key].update({
                        'merge_type': 'fitz'
                    })

                    merged_figures_keys.add(fitz_key)

        return merged_figures, merged_figures_keys

class FilterFitzByFigcapContainers(ImageFilter):
    '''
    Filter: If multiple fitz images (3+) are contained within a figcap image, use figcap because the 
    fitz images are likely to be piece meals/fragements.

    Strategy:
    - Count how many fitz images are contained in the figcap box
    - Only use figcap if 3+ fitz images (fragments) are contained within the figcap box 
    - If 1-2 fitz boxes are contained within figcap box, they might be legitimate separate images, so keep fitz
    - Mark contained fitz images as matched (so they dont get added later)
    - If any contained fitz is already in merged_figures --> drop figcap (fitz has priority)

    This ensures:
    - Only use figcap when there are clearly multiple fragments (3+)
    - Fitz always has priority when it's a good image (already in merged_figures)
    '''

    def __init__(self, min_contained_fitz = 3):
        '''
        min_contained_fitz: Minimum number of fitz images/bboxes that must be contained in a figcap bbox to consider using figcap
        '''
        self.min_contained_fitz = min_contained_fitz

    def apply(self, pdf_page_number, fitz_figures, figcap_figures, 
                merged_figures, merged_figures_keys):
        '''
        Filter out fitz figures that are contained within figcap figures.
        '''

        figcap_to_fitz_matches: dict[int, list[(int,str)]] = {}

        if pdf_page_number not in merged_figures:
            merged_figures[pdf_page_number] = {}

        # Step 1: If figcap is contained inside a fitz box or intersects with a fitz box that was already matched in the past (in merged_figures),
        # mark the figcap as merged so it doesn't get added to merged figures later.
        for figcap_fig_no, figcap_fig_data in figcap_figures.items():
            figcap_key = (figcap_fig_no, figcap_fig_data.get('figure_type', 'figcap'))
            if figcap_key in merged_figures_keys:
                continue
            figcap_box = figcap_fig_data.get('box')
            
            for fitz_fig_no, fitz_fig_data in fitz_figures.items():
                fitz_key = (fitz_fig_no, 'fitz')
                # Only consider fitz that is already in merged_figures (matched)
                if fitz_key not in merged_figures.get(pdf_page_number, {}):
                    continue
                fitz_box = fitz_fig_data.get('box')

                # if figcap intersects with fitz, drop figcap because fitz image
                # was already matched earlier and priority is given to fitz.
                if CompareBoxes.have_intersection(fitz_box, figcap_box):
                    merged_figures_keys.add(figcap_key)

                    # if the figcap image has a caption - use that caption for the matched fitz image
                    if "caption" in figcap_fig_data:
                        fitz_fig_data.update({"caption": figcap_fig_data["caption"]})      
                    break
                
        # Step 2: Collect info about all the fitz matches contained inside the figcap box
        for figcap_fig_no, figcap_fig_data in figcap_figures.items():
            figcap_box = figcap_fig_data.get('box')
                        
            for fitz_fig_no, fitz_fig_data in sorted(fitz_figures.items(), key=lambda k: k[0]):
                fitz_box = fitz_fig_data.get('box')
                                
                fitz_key = (fitz_fig_no, 'fitz')
                
                # Only check containment for fitz that haven't been matched yet
                if fitz_key not in merged_figures_keys:                    
                    containment = CompareBoxes.get_containment(figcap_box, fitz_box)

                    if containment is not None:
                        if figcap_fig_no not in figcap_to_fitz_matches:
                            figcap_to_fitz_matches[figcap_fig_no] = []
                        figcap_to_fitz_matches[figcap_fig_no].append(
                            (fitz_fig_no, fitz_fig_data.get('figure_type', 'fitz'))
                        )

        # Step 3: Check the fitz vs figcap matches - if 3+ fitz images are contained in the same figcap box,
        # it is likely a piece meal image, so consider taking the figcap image over the fitz fragments
        for figcap_fig_no, fitz_matches in figcap_to_fitz_matches.items():
            if len(fitz_matches) >= self.min_contained_fitz:
                # Mark fitz figures as used, in merged_figures_keys
                for fitz_fig_no, fig_type in fitz_matches:
                    fitz_key = (fitz_fig_no, fig_type)
                    merged_figures_keys.add(fitz_key)
                    
                    # Remove from merged_figures if present (so that it doesn't display in the final results)
                    if fitz_key in merged_figures[pdf_page_number]:
                        del merged_figures[pdf_page_number][fitz_key]
                
                # Add the figcap container
                figcap_key = (figcap_fig_no, 'figcap')
                if figcap_key not in merged_figures_keys:
                    merged_figures[pdf_page_number][figcap_key] = figcap_figures[figcap_fig_no]
                    merged_figures[pdf_page_number][figcap_key].update({
                        # 'merge_type': 'figcap_container'
                        'merge_type': 'figcap'
                    })
                    merged_figures_keys.add(figcap_key)
            else:
                # Mark the container as merged - so that it doesn't get used later
                merged_figures_keys.add((figcap_fig_no, 'figcap'))
        
        return merged_figures, merged_figures_keys

class MergeByIOU(ImageFilter):
    '''
    Filter: Merge by Intersection over Union (IOU), finding common matches 
    between fitz and figcap images.

    Stratergy: Only merges matching pairs (IOU > threshold), prioritizes fitz.
    - For each fitz figure, try to find matching figcap (IOU > threshold)
    - If match found, add fitz to the merged_images (fitz has priority)
    - Mark both fitz and figcap as matched.
    - Does NOT add unmatched figures, thats handled by RegularMerge filter.
    '''

    def __init__(self, iou_threshold = 0.8):
        self.iou_threshold = iou_threshold

    def apply(self, pdf_page_number, fitz_figures, figcap_figures, 
            merged_figures, merged_figures_keys):
        '''
        Find image matches between fitz and figcap images using IOU.
        Keep the matched fitz image and filter out the corresponding figcap image.
        '''

        if pdf_page_number not in merged_figures:
            merged_figures[pdf_page_number] = {}
        
        try:
            for fitz_fig_no, fitz_fig_data in sorted(fitz_figures.items(), key=lambda k: k[0]):
                fitz_key = (fitz_fig_no, fitz_fig_data['figure_type'])
                
                for figcap_fig_no, figcap_fig_data in figcap_figures.items():
                    figcap_key = (figcap_fig_no, figcap_fig_data['figure_type'])
                    
                    fitz_box = fitz_fig_data['box']
                    figcap_box = figcap_fig_data['box']
                    
                    iou = CompareBoxes.iou(fitz_box, figcap_box)
                    if iou >= self.iou_threshold:
                        # if there is a match - check if the fitz image has already been merged before,
                        # if true, then mark the figcap image as merged,
                        # else, add the fitz image in merged figures and mark fitz and figcap image as merged

                        if fitz_key in merged_figures_keys:
                            merged_figures_keys.add(figcap_key)     # mark the figcap figure as matched 

                            # if the figcap image has a caption - use that caption for the matched fitz image
                            if "caption" in figcap_fig_data:
                                # fitz_fig_data.update({"caption": figcap_fig_data["caption"]})
                                merged_figures[pdf_page_number][fitz_key].update({
                                    "caption": figcap_fig_data["caption"],
                                    "figure_number": figcap_fig_data["figure_number"],
                                    })
                        else:
                            merged_figures_keys.add(fitz_key)
                            merged_figures_keys.add(figcap_key)
                            
                            merged_figures[pdf_page_number][fitz_key] = fitz_fig_data
                            merged_figures[pdf_page_number][fitz_key].update({
                                # 'merge_type': 'iou',
                                'merge_type': 'fitz'    # means the fitz based and figcap image matched based on IOU, but we are using all the properties from the fitz image + figcap captions
                                # 'merge_iou': iou,
                                **{k: v for k, v in figcap_fig_data.items() if k in (
                                    'caption', 'figure_number'
                                )}
                            })
        except Exception as e:
            print("\nException occurred while matching and merging boxes based on IOU:", e)
        
        return merged_figures, merged_figures_keys
            
class RegularMerge(ImageFilter):
    '''
    Regular merge of unmatched figures.
    Can be used for 'fitz' or 'figcap' figures or 'both'
    '''

    def __init__(self, image_source='both'):
        self.image_source = image_source

    def _merge_figures(self, pdf_page_number, page_figures, merged_figures, merged_figures_keys):
        if pdf_page_number not in merged_figures:
            merged_figures[pdf_page_number] = {}

        for fig_no, fig_data in sorted(page_figures.items(), key=lambda k: k[0]):
            key = (fig_no, fig_data['figure_type'])

            if key in merged_figures_keys:
                continue

            merged_figures[pdf_page_number][key] = fig_data.copy()
            merged_figures[pdf_page_number][key].update({
                # 'merge_type': 'regular_' + fig_data['figure_type']
                'merge_type': fig_data['figure_type']
            })
            merged_figures_keys.add(key)

        return merged_figures, merged_figures_keys

    def apply(self, pdf_page_number, fitz_figures, figcap_figures, 
            merged_figures, merged_figures_keys):
        
        if self.image_source in ('fitz', 'both'):
            merged_figures, merged_figures_keys = self._merge_figures(
                pdf_page_number, fitz_figures, merged_figures, merged_figures_keys
            )
        
        if self.image_source in ('figcap', 'both'):
            merged_figures, merged_figures_keys = self._merge_figures(
                pdf_page_number, figcap_figures, merged_figures, merged_figures_keys
            )

        return merged_figures, merged_figures_keys

class ImageFilterPipeline:
    """
    Pipeline to apply multiple filters in sequence.
    Users can add/remove/reorder filters as needed.
    """
    
    def __init__(self, filters = []):
        self.filters = filters
    
    def add_filter(self, filter_instance: ImageFilter):
        self.filters.append(filter_instance)
    
    def apply(self, pdf_page_number, fitz_figures, figcap_figures,
                merged_figures, merged_figures_keys):
        '''
        Apply all filters in sequence.
        '''

        if merged_figures is None:
            merged_figures = {}
        if merged_figures_keys is None:
            merged_figures_keys = []
        
        for filter_instance in self.filters:
            merged_figures, merged_figures_keys = filter_instance.apply(
                pdf_page_number, fitz_figures, figcap_figures, 
                merged_figures, merged_figures_keys
            )

        # adding a clean up step which removes (not so important keys, so that it is not carried 
        # foward in the json results) the 'figure_type'
        for page_num in merged_figures:
            for fig_key in merged_figures[page_num]:
                merged_figures[page_num][fig_key].pop('figure_type', None)

        return merged_figures
