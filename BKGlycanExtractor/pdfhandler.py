import fitz, os, os.path, re, difflib, traceback, sys

from . pmc_details import PMCData
from .bbox import PDFBoundingBox
from .compareboxes import CompareBoxes


# if more constants are added, then create a Enum class 
STANDARD_DPI = 300
XREF_MATCH_IOU_THRESHOLD = 0.8

class PDFHandler(object):
    def __init__(self,filepath):
        self.doc = fitz.open(filepath)
        self.dir,self.base = os.path.split(filepath)
        self.base,self.extn = self.base.rsplit('.',1)
    
    def make_figure_path(self,image):
        return os.path.join(self.dir,self.base+"-xref"+str(image['xref'])+"."+image.get("ext","png"))

    def pages(self):
        return self.doc.pages()
    
    def images_per_page(self,page):
        return page.get_image_info(xrefs=True)
    
    @staticmethod
    def image_dimensions(image_info):
        x0,y0,x1,y1 = image_info.get('bbox') or image_info.get('pdf_fig_bbox')      # fitz based bbox is [x0,y0,x1,y1]
        return abs(x0-x1)+1, abs(y0-y1)+1         # returns width, height

    @staticmethod
    def page_dimensions(image_info):
        if image_info.get('page_width') and image_info.get('page_height'):
            return image_info.get('page_width'), image_info.get('page_height')
        return None, None
    
    @staticmethod
    def image_bbox(image_info):
        return image_info.get('bbox') or image_info.get('pdf_fig_bbox')       # returns fitz based bbox is [x0,y0,x1,y1]
    
    @staticmethod
    def create_box(bbox):
        return fitz.Rect(bbox)
    
    def path_without_annotations(self):
        '''
        Path to an annotation free copy of this pdf, or the original path when it has
        no annotations. FigCapX detects figures from the rendered page, so boxes drawn
        on a figure shift the detected pdf_fig_bbox on a re-run.
        Caller deletes the copy when it differs from the original.
        '''
        doc = fitz.open(self.doc.name)
        try:
            removed = False
            for page in doc:
                for annot in list(page.annots() or []):
                    page.delete_annot(annot)
                    removed = True
            if not removed:
                return self.doc.name
            output_path = self.doc.name.replace('.pdf', '.noannots.pdf')
            doc.save(output_path, garbage=3, clean=True)
            return output_path
        finally:
            doc.close()
            
    def _with_ext(self, path, ext):
        if path.endswith('.' + ext):
            return path
        return path.rsplit('.', 1)[0] + '.' + ext
    
    def _write_bytes(self, image, image_path):
        # ensure that the provided image_path extension matches with the original 
        # image bytes extension, if not update it to the orginal
        image_path = self._with_ext(image_path, image['ext'])
        with open(image_path, 'wb') as fh:
            fh.write(image['image'])
        return dict(width=image.get('width'), height=image.get('height'), image_path=image_path)
    
    def _save_pixmap(self, pix, image_path):
        try:
            pix.save(image_path)
        except ValueError:
            # unsupported colorspace (e.g. CMYK) --> convert and retry
            pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(image_path)
        return dict(width=pix.width, height=pix.height, image_path=image_path)
    
    @staticmethod
    def _to_xref(value):
        '''xref may arrive as a string (pdf comments) or None - normalize to int/None'''
        try:
            xref = int(value) if value is not None else None
        except (TypeError, ValueError):
            print(f"invalid xref {value!r}")
            return None
        return xref if xref and xref > 0 else None

    def resolve_xref(self, page, pdf_fig_bbox, original_xref=None,
                 iou_threshold=XREF_MATCH_IOU_THRESHOLD):
        """
        Find the image xref on the current given page whose bbox matches provided pdf_fig_bbox.

        Note: Annotating a pdf and saving a copy renumbers its objects, so the xref stored
        in the json/pdf comments can point at a non-image object even though the
        figure itself did not move. 
        The bbox (pdf_fig_bbox) is the stable authority, so it is used for sanity check, by looking up
        the current xref's on the page and comparing it with the provided xref. By comparing the bounding
        boxes of the provided xref vs the bounding boxes on the currently on the page using IOU or containment -
        it is possible to determine if the provided xref is good, or an updated xref should be used or fallback to
        clipping the image using pdf_fig_bbox.

        returns the resolved xref
        """

        if page is None or pdf_fig_bbox is None:
            return None
        try:
            fig_box = PDFBoundingBox(bbox=list(fitz.Rect(pdf_fig_bbox)))
        except (ValueError, TypeError):
            print(f"invalid figure bbox {pdf_fig_bbox!r}")
            return None
        original_xref = self._to_xref(original_xref)
        matches = []
        for image_info in self.images_per_page(page):
            page_xref = self._to_xref(image_info.get('xref'))
            if page_xref is None:
                continue
            try:
                img_box = PDFBoundingBox(bbox=list(image_info['bbox']))
            except (KeyError, ValueError, TypeError):
                continue
            # a figure bbox drawn slightly bigger/smaller than the embedded image
            # lowers the iou, so full containment (either way) also counts as a match
            if CompareBoxes.iou(fig_box, img_box) >= iou_threshold \
                    or CompareBoxes.get_containment(fig_box, img_box) is not None:
                matches.append(page_xref)

        # the caller's original xref wins whenever it still matches, so a pdf that was never
        # rewritten keeps extracting the exact same image as before
        if original_xref in matches:
            return original_xref
        
        # exactly one image fits the bbox - it must be this figure.
        # several images fit (panels inside the figure box), so which one is the
        # figure cannot be decided here - clipping the bbox gives the whole figure
        return matches[0] if len(matches) == 1 else None
    
    def _get_image_by_xref(self, image_path, xref):        
        # 1) original embedded bytes
        try:
            image_info = self.doc.extract_image(xref)
            return self._write_bytes(image_info, image_path)
        except (ValueError, RuntimeError, OSError) as e:
            print(f"extract_image(xref={xref}) failed: {type(e).__name__}: {e}")
        
        # 2)rasterize that image object - using Pixmap (uses xref)
        try:
            image_info = fitz.Pixmap(self.doc, xref)     
            return self._save_pixmap(image_info, image_path)  
        except (ValueError, RuntimeError, OSError) as e:
            print(f"Pixmap(xref={xref}) failed: {type(e).__name__}: {e}")
        
        # both xref based extraction failed; so caller so fallback to clipping the image uisng pdf_fig_box
        return None
    

    def write_image(self, image, image_path=None, image_annotations=True):
        if image_path is None:
            image_path = self.make_figure_path(image)

        result = None

        try:
            # 1) raw bytes of the image are already on dict (i.e image dict), so directly save it
            if image.get('image') is not None and image.get('ext'):
                return self._write_bytes(image, image_path)
            
            pdf_fig_bbox = image.get('pdf_fig_bbox')
            page_number = image.get('page_number')
            page = None
            
            if page_number is not None:
                try:
                    page = self.doc[int(page_number) - 1]
                except (TypeError, ValueError, IndexError):
                    page = None

            # 2) try to extract raw bytes (embedded orginal form of image) using xref,
            # if not possible, then attempt fitz Pixmap extraction
            original_xref = self._to_xref(image.get('xref'))

            resolved_xref = None
            if original_xref is not None:
                if page is not None and pdf_fig_bbox is not None:
                    resolved_xref = self.resolve_xref(page, pdf_fig_bbox, original_xref=original_xref)
                else:
                    resolved_xref = original_xref  # no bbox to validate against
                if resolved_xref and resolved_xref > 0:
                    result = self._get_image_by_xref(image_path, resolved_xref)

                    # print("original vs new xref", original_xref, resolved_xref)

            # 3) clip the image based on the bbox provided - generally used
            # for figcap extractions when original xref is not available
            if result is None and pdf_fig_bbox is not None and page is not None:
                try:
                    clip = fitz.Rect(pdf_fig_bbox)
                    if not clip.is_empty:
                        dpi = STANDARD_DPI
                        image_info = page.get_pixmap(
                            clip=clip, dpi=dpi, annots=image_annotations
                        )
                        # print("USED CLIP")
                        result = self._save_pixmap(image_info, image_path)
                except (ValueError, TypeError, IndexError, RuntimeError) as e:
                    print(f"clip failed: {type(e).__name__}: {e}")

            if result is None:
                raise RuntimeError("embedded image and clip image - both failed")
            
            return result
        except OSError as e:
            print(f"write_image I/O error: {e}")
            traceback.print_exc()
            return None
        except RuntimeError as e:
            print(f"write_image failed: {e}")
            traceback.print_exc()
            return None

    def figures(self,images_data=None,filter=None):
        # if images_data is provided (not None) --> it is from figcap 
        # else fitz based image data will be created
        if images_data is not None:         
            '''Generator that yields image metadata when the data is already provided (images_data)'''
            for image_info in images_data.get('figures', {}):
                if filter is None or filter.keep(image_info):
                    image_info['dpi'] = STANDARD_DPI
                    yield image_info
        else:
            image_count = 1
            for page_number,page in enumerate(self.pages(),1):
                images = self.images_per_page(page)         # image identification is based on xrefs
                figure_number = None
                caption = None
                figure_label = None
                if len(images) == 1:
                    blocks = list(self.find_text_blocks(page=page_number))
                    # print(blocks)
                    if len(blocks) == 1:
                        m = re.search(r'^(\w+(\s+\w+)*) (\w+)\. (.*)$',blocks[0])
                        if m:
                            figure_label = m.group(1)
                            figure_number = m.group(3)
                            caption = m.group(4)
                            # print(figure_label, figure_number, caption, file=sys.stderr)
                for image_number,image in enumerate(images,1):
                    try:
                        image.update(self.doc.extract_image(image['xref']))
                    except ValueError:
                        pass
                    pdf_fig_width, pdf_fig_height = self.image_dimensions(image)
                    image['page_number'] = page_number
                    image['image_number'] = image_number                # image count per page
                    image['pdf_fig_bbox'] = self.image_bbox(image)      # pdf_fig_bbox - x0,y0,x1,y1
                    image['pdf_fig_width'] = pdf_fig_width
                    image['pdf_fig_height'] = pdf_fig_height
                    image['page_width'] = page.rect.width
                    image['page_height'] = page.rect.height
                    if figure_number:
                        image['figure_number'] = figure_number
                    if figure_label:
                        image['figure_label'] = figure_label
                    if caption:
                        image['caption'] = caption
                        
                    if filter is None or filter.keep(image):
                        image['image_count'] = image_count                  # total image count so far
                        image['dpi'] = STANDARD_DPI
                        image_count += 1
                        yield image

    doi_regex = re.compile(r'(doi: *|://doi.org/|\b)(10.\d{4,9}/[-._;()/:a-zA-Z0-9]+)',re.IGNORECASE)
    def find_dois(self):
        dois = {}
        for page_number,page in enumerate(self.pages(),1):
            text = page.get_text()
            for match in self.doi_regex.finditer(text):
                if match:
                    doi = match.group(2)
                    if doi not in dois:
                        dois[doi] = dict(doi=doi,count=1,index=len(dois)+1,page=page_number)
                    else:
                        dois[doi]['count'] += 1

        return sorted([t for t in dois.values() ],key=lambda t: t['index'])

    def find_doi(self):
        for doi in self.find_dois():
            return doi['doi']
        return None

    def find_text_blocks(self,page=None,pages=None):
        if page:
            pages = [ page ]
        if pages:
            pages = set(pages)
        for page_number,thepage in enumerate(self.pages(),1):
            if page and page_number not in pages:
                continue
            text = thepage.get_text('blocks')
            for tb in thepage.get_text('blocks'):
                yield " ".join(tb[4].split())

    def get_citation(self):
        
        dois = self.find_dois()
        for doi in dois:
            title = None
            ids = PMCData.lookup(doi=doi['doi'])
            if ids is not None and ids.get('pmid'):
                pmid = ids.get('pmid')
                cite = PMCData.citation_details(pmid)
                if cite and cite.get('title'):
                    title = " ".join(cite.get('title').split()).rstrip('.')
            
            if title:
                # print(title)
                for i,tb in enumerate(self.find_text_blocks(pages=(1,2,3))):
                    ratio = difflib.SequenceMatcher(None,title,tb).ratio()
                    # print(ratio,tb)
                    if ratio >= 0.8 or title in tb:
                        cite['title_match_ratio'] = ratio
                        return cite

        return None                 
    
class PDFImageFilter(object):
    def keep(self,image):
        raise NotImplementedError

class CompoundPDFImageFilter(PDFImageFilter):
    def __init__(self,*filters):
        self._filters = filters

    def keep(self,image):
        for f in self._filters:
            if not f.keep(image):
                return False
        return True

class PDFXRefImageFilter(PDFImageFilter):
    def __init__(self,min_xref=1):
        self._min_xref = min_xref

    def keep(self,image):
        if image['xref'] >= self._min_xref:
            return True
        return False

class PDFImageSizeFilter(PDFImageFilter):
    def __init__(self,width=90,height=90,area=None):
        self._width = width
        self._height = height
        if area is not None:
            self._area = area
        else:
            self._area = width*height
    
    def keep(self,image):
        width,height = PDFHandler.image_dimensions(image)
        area = width*height;
        if width >= self._width and height >= self._height:
            return True
        if area >= self._area:
            return True
        return False

class PDFLargeImageSizeFilter(PDFImageFilter):
    def __init__(self, page_coverage_threshold=0.80):
        self._page_coverage_threshold = page_coverage_threshold

    def keep(self, image):
        '''Returns False if image covers more than threshold of page area.'''
        try:
            width, height = PDFHandler.image_dimensions(image)
            page_width, page_height = PDFHandler.page_dimensions(image)
            
            page_area = page_width * page_height
            image_area = width * height
            coverage = image_area / page_area if page_area > 0 else 0
            
            # Return False to filter out full-page images
            return coverage < self._page_coverage_threshold
            
        except (KeyError, ValueError, ZeroDivisionError):
            return True 

if __name__ == "__main__":

    import sys
    import searchpmc

    print(sys.argv[1])
    pdf = PDFHandler(sys.argv[1])
    cite = pdf.get_citation()
    if cite:
        print(cite['ascii_citation'])
        # print(cite)

    filter = CompoundPDFImageFilter(
        PDFXRefImageFilter(min_xref=1),
        PDFImageSizeFilter(width=90,height=90)
        )
    for fig in pdf.figures(filter=filter):
        print(fig)
        pdf.write_image(fig)
