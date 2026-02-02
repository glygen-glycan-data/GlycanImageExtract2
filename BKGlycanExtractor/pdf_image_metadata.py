import os
import sys
import json
import time
import shutil
import fitz
import copy

from BKGlycanExtractor.bbox import PDFBoundingBox
from BKGlycanExtractor.pdfhandler import PDFHandler, CompoundPDFImageFilter, PDFXRefImageFilter, PDFImageSizeFilter, PDFLargeImageSizeFilter
from BKGlycanExtractor.compareboxes import CompareBoxes
from BKGlycanExtractor.pdf_image_captions_data import PDFiguesCaptionsData
from BKGlycanExtractor.pdf_image_filters import ImageFilterPipeline, DetectFragmentedFitz, FilterFitzByFigcapContainers, MergeByIOU, RegularMerge

class ImageSearch:

    @staticmethod
    def search_method(search_type='fitz'):
        if search_type == 'fitz':
            return FitzImageSearch()
        elif search_type == 'figcap':
            return FigCapImageSearch()
        elif search_type == 'hybrid':
            return HybridImageSearch()

    def get_metadata(self):
        raise NotImplementedError

class FitzImageSearch:
    def get_metadata(self, input_filepath):
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
            PDFImageSizeFilter(width=120,height=120)
        )

        for fig_metadata in pdf.figures(filter=filter):
            page_number = fig_metadata['page_number']
            image_number = fig_metadata['image_number']

            if page_number not in pdf_metadata:
                pdf_metadata[page_number] = {}

            pdf_metadata[page_number][image_number] = {k:v for k,v in fig_metadata.items() if k in 
                ('page_number','image_number','image_count','bbox', 
                'width', 'height', 'xref', 'pdf_fig_bbox', 'pdf_fig_width', 
                'pdf_fig_height', 'page_width', 'page_height', 'dpi'
            )}
          
            # self.log_file.write(
            #     f"\nXREF: {fig_metadata.get('xref')}, Page number: {page_number}, Image number: {image_number},  bbox: {fig_metadata["pdf_fig_bbox"]}, Width: {fig_metadata['pdf_fig_width']}, Height: {fig_metadata['pdf_fig_height']}\n"
            # )

        return pdf_metadata

class FigCapImageSearch:
    def get_metadata(self, input_filepath):
        """
        Extract figure metadata from PDF using PDFigCapX (i.e PDFiguesCaptionsData class) repository
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

        # PDFigCapX repository - saves the pdf pages as a pixmap and processing it using openCV to identify the different components present on the page.
        # The higher the dpi, better the resolution, the more accurate the figure detection will be.
        page_dpi = 300    

        json_data_path = PDFiguesCaptionsData.figures_info(input_filepath,page_dpi) 
        with open(json_data_path) as f:
            figures_data = json.load(f)

        pdf = PDFHandler(input_filepath)

        filter = CompoundPDFImageFilter(
            PDFImageSizeFilter(width=90,height=90),
            PDFLargeImageSizeFilter(page_coverage_threshold=0.75)
        )

        for fig_metadata in pdf.figures(figures_data, filter=filter):
            page_number = fig_metadata['page_number']
            image_number = fig_metadata['image_number']

            if page_number not in pdf_metadata:
                pdf_metadata[page_number] = {}

            pdf_metadata[page_number][image_number] = {k:v for k,v in fig_metadata.items() if k in 
                ('page_number','image_number','image_count','bbox', 'width', 
                'height', 'xref', 'pdf_fig_bbox', 'pdf_fig_width', 'pdf_fig_height', 
                'page_width', 'page_height', 'caption_bbox', 'figure_name',
                'width', 'height', 'label', 'caption_text', 'full_caption_text', 
                'cleaned_caption', 'dpi'
            )} 

        try:
            # delete the json file after loading the semantics data obtained from PDFigCapX
            os.remove(json_data_path)
        except OSError as e:
            print("\nCould not delete json file obtained from PDFigCapX")

        return pdf_metadata
    
class HybridImageSearch:
    '''
    Combination of FigCapImageSearch + FitzImageSearch.
    Uses composition to combine both search methods.
    '''

    def __init__(self, fitz_searcher=None, figcap_searcher=None):
        '''
        Initialize with searcher instances.
        If not provided create default instances
        '''
        self.fitz_searcher = fitz_searcher or FitzImageSearch()
        self.figcap_searcher = figcap_searcher or FigCapImageSearch()

    def get_metadata(self, input_filepath):
        # get metadata from each individual pdf image search strategy
        fitz_based_metadata = self.fitz_searcher.get_metadata(input_filepath)
        figcap_based_metadata = self.figcap_searcher.get_metadata(input_filepath)

        # merge the metadata - to pick the best information from both the strategies
        return self._merge_metadata(fitz_based_metadata, figcap_based_metadata)

    def _merge_metadata(self, fitz_based_metadata, figcap_based_metadata):
        '''
        Primary matching is done based on fitz metadata
        Secondary - Images from figCapX and fitz that did not find any match will be considered

        Note: the metadata received as function parameters are assumed to filtered (eg. image size filter, etc)
        based on PDFHandler class 
        '''

        all_merged_figures = {}

        pdf_page_nos = fitz_based_metadata.keys() | figcap_based_metadata.keys()
        for pg_no in pdf_page_nos:
            fitz_figures = fitz_based_metadata.get(pg_no, {})
            figcap_figures = figcap_based_metadata.get(pg_no, {})

            if not fitz_figures and not figcap_figures:
                continue

            # all the merged figures on the given page number
            merged_figures, merged_figures_keys = self._merge_figures_metadata_from_page(pg_no, fitz_figures, figcap_figures)

            all_merged_figures.update(merged_figures)

        # process/clean all the figures data in the pdf - which will sequentially number the figures, etc
        self._process_merged_figures_data(all_merged_figures)

        return all_merged_figures

    def _merge_figures_metadata_from_page(self, pdf_page_number, fitz_figures, figcap_figures):
        # initilaze state
        merged_figures = {pdf_page_number: {}}
        merged_figures_keys = set()

        # format the inputs
        fitz_figures = self._format_page_figs_metadata(fitz_figures, 'fitz')
        figcap_figures = self._format_page_figs_metadata(figcap_figures, 'figcap')

        # execute filter pipeline
        pipeline = ImageFilterPipeline([
            DetectFragmentedFitz(),
            FilterFitzByFigcapContainers(),
            MergeByIOU(iou_threshold=0.8),
            RegularMerge(image_source='fitz'),
            RegularMerge(image_source='figcap'),
        ])

        merged_figures, merged_figures_keys = pipeline.apply(
            pdf_page_number, fitz_figures, figcap_figures, merged_figures, merged_figures_keys
        )

        return merged_figures, merged_figures_keys

    # Default vertical tolerance (PDF points) for treating boxes as the same "row" when sorting L→R, T→B
    DEFAULT_ROW_TOLERANCE = 30


    def sort_figures_reading_order(self,
        items,
        row_tolerance = DEFAULT_ROW_TOLERANCE):
        """
        Sort figure items top-to-bottom, then left-to-right (reading order).

        Boxes within `row_tolerance` vertical distance are treated as on the same row,
        so the right-hand box is not ordered before the left-hand box when it is
        only slightly higher.
        """

        def key(item):
            key_tuple, figure_data = item  
            bbox = figure_data['pdf_fig_bbox']
            x0, y0 = bbox[0], bbox[1]
            row = round(y0 / row_tolerance) * row_tolerance
            return (row, x0)

        return sorted(items, key=key)


    def _process_merged_figures_data(self, all_merged_figures):
        '''
        - Images on each page are sorted based on bbox position and renumbered sequentially
        - Provides sequential image count for the entire pdf
        - Converts from (fig_no, figure_type) keys to sequential image_number keys
        '''
        image_count = 1
        
        for page_no in sorted(all_merged_figures.keys()):
            merged_figures = all_merged_figures[page_no]
            image_number = 1
            
            sorted_figures = self.sort_figures_reading_order(list(merged_figures.items()))
            
            # Rebuild with sequential image_number keys
            new_merged_figures = {}
            for old_key, figure in sorted_figures:
                figure['image_number'] = image_number
                figure['image_count'] = image_count
                
                new_merged_figures[image_number] = figure
                
                image_number += 1
                image_count += 1
            
            all_merged_figures[page_no] = new_merged_figures
        
        return all_merged_figures

    def _format_page_figs_metadata(self, page_figures, figure_type='regular'):
        '''
        converts all the fig boxes into pdf box objects
        '''

        formatted_figures = copy.deepcopy(page_figures)

        for fig_no, fig_data in formatted_figures.items():
            fig_data['box'] = PDFBoundingBox(bbox=fig_data['pdf_fig_bbox'], page_width=fig_data['page_width'], page_height=fig_data['page_height'])
            fig_data['figure_type'] = figure_type
        
        return formatted_figures

if __name__ == '__main__':
    fs = FigCapX_Search()
    pdf_path = sys.argv[1]      # pdf path
    fig_json_path = fs.figures_info(pdf_path)