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

class FilterFitzByFigcapContainers(ImageFilter):
    '''
    Filter: If fitz images are piece meal parts of a larger figcap based image, 
    do not consider the fitz images.
    '''

    def apply(self, pdf_page_number, fitz_figures, figcap_figures, 
                merged_figures, merged_figures_keys):
        '''Filter out fitz figures that are contained within figcap figures.'''

        figcap_to_fitz_matches: dict[int, list[(int,str)]] = {}

        if pdf_page_number not in merged_figures:
            merged_figures[pdf_page_number] = {}
        
        for figcap_fig_no, figcap_fig_data in figcap_figures.items():
            figcap_box = figcap_fig_data['box']
            
            for fitz_fig_no, fitz_fig_data in sorted(fitz_figures.items(), key=lambda k: k[0]):
                fitz_box = fitz_fig_data['box']
                
                containment = CompareBoxes.get_containment(figcap_box, fitz_box)
                if containment is None:
                    continue
                
                if figcap_fig_no not in figcap_to_fitz_matches:
                    figcap_to_fitz_matches[figcap_fig_no] = []
                figcap_to_fitz_matches[figcap_fig_no].append((fitz_fig_no, fitz_fig_data['figure_type']))
        
        for figcap_fig_no, fitz_matches in figcap_to_fitz_matches.items():
            if len(fitz_matches) >= 3:
                # Mark fitz figures as used, in merged_figures_keys
                for fitz_fig_no, fig_type in fitz_matches:
                    fitz_key = (fitz_fig_no, fig_type)
                    merged_figures_keys.add(fitz_key)

                    # Remove from merged_figures (so that it doesnt display in the final results)
                    if fitz_key in merged_figures[pdf_page_number]:
                        del merged_figures[pdf_page_number][fitz_key]
                
                # Add the figcap container
                figcap_key = (figcap_fig_no, 'figcap')
                if figcap_key not in merged_figures_keys:
                    merged_figures[pdf_page_number][figcap_key] = figcap_figures[figcap_fig_no]
                    merged_figures[pdf_page_number][figcap_key].update({
                        'merge_type': 'container'
                    })
                    merged_figures_keys.add(figcap_key)
            else:
                # mark the container as merged - so that it doesnt get used later.
                merged_figures_keys.add((figcap_fig_no, 'figcap'))
        
        return merged_figures, merged_figures_keys

class MergeByIOU(ImageFilter):
    '''
    Filter: Merge by Intersection over Union (IOU), finding common matches 
    between fitz and figcap images.
    '''
    def __init__(self, iou_threshold = 0.8):
        self.iou_threshold = iou_threshold

    def apply(self, pdf_page_number, fitz_figures, figcap_figures, 
            merged_figures, merged_figures_keys):
        '''
        Find image matches between fitz and figcap images using IOU.
        Keep the matched fitz image and filter out the corresponding figcap image.
        '''
        figure_matches: dict[int, list[int]] = {}

        if pdf_page_number not in merged_figures:
            merged_figures[pdf_page_number] = {}
        
        try:
            for fitz_fig_no, fitz_fig_data in sorted(fitz_figures.items(), key=lambda k: k[0]):
                fitz_key = (fitz_fig_no, fitz_fig_data['figure_type'])
                
                if fitz_key in merged_figures_keys:
                    continue
                
                for figcap_fig_no, figcap_fig_data in figcap_figures.items():
                    figcap_key = (figcap_fig_no, figcap_fig_data['figure_type'])
                    
                    if figcap_key in merged_figures_keys:
                        continue
                    
                    fitz_box = fitz_fig_data['box']
                    figcap_box = figcap_fig_data['box']
                    
                    iou = CompareBoxes.iou(fitz_box, figcap_box)
                    if iou >= iou_threshold:
                        if fitz_fig_no not in figure_matches:
                            figure_matches[fitz_fig_no] = []
                        figure_matches[fitz_fig_no].append(figcap_fig_no)
                        
                        merged_figures_keys.add(fitz_key)
                        merged_figures_keys.add(figcap_key)
                        
                        merged_figures[pdf_page_number][fitz_key] = fitz_fig_data
                        merged_figures[pdf_page_number][fitz_key].update({
                            'merge_type': 'iou',
                            'merge_iou': iou,
                            **{k: v for k, v in figcap_fig_data.items() if k in (
                                'label', 'caption_text', 'full_caption_text', 'cleaned_caption'
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
            
            if key not in merged_figures_keys:
                # Use original key (prevents collisions, preserves info)
                merged_figures[pdf_page_number][key] = fig_data
                merged_figures[pdf_page_number][key].update({
                    'merge_type': 'regular'
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
        
        return merged_figures, merged_figures_keys
