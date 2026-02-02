import fitz, os, os.path

# if more constants are added, then create a Enum class 
STANDARD_DPI = 300
POINTS_PER_INCH = 72.0

class PDFHandler(object):
    def __init__(self,filename):
        self.doc = fitz.open(filename)
        self.dir,self.base = os.path.split(filename)
        self.base,self.extn = self.base.rsplit('.',1)
    
    def make_figure_filename(self,image):
        return os.path.join(self.dir,self.base+"-"+str(image['xref'])+".png")

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

    @staticmethod
    def calculate_dpi(image_info, doc=None):
        '''
            - If xref available: Extract pixel dimensions from xref, then calculate effective DPI based on how the image
            is displayed in the pdf
            - Otherwise: Use provided width/height from image_info, then calculate effective DPI

            Args:
                image_info: Dict with keys:
                    - 'xref': xref of embedded image (checked first if available)
                    - 'width', 'height': pixel dimensions (used if no xref)
                    - 'pdf_fig_width', 'pdf_fig_height': bbox dimensions in PDF points (required)
                doc: PyMuPDF Document object (required if using xref)

            This ensures reproducible extractions - same DPI whether you have xref or not.
        '''
        
        width_px = 0
        height_px = 0
        
        # 1) Check if xref is available and use it
        xref = image_info.get('xref')
        if xref and xref > 0:
            if doc is None:
                # Can't use xref without doc, fall through to provided dimensions
                pass
            else:
                try:
                    native_pix = fitz.Pixmap(doc, xref)
                    width_px = native_pix.width
                    height_px = native_pix.height
                    del native_pix  
                except Exception as e:
                    print(f"Warning: Failed to extract image from xref {xref}: {e}")
        
        # 2) Use provided dimensions if xref didn't work or wasn't available
        if width_px == 0 or height_px == 0:
            width_px = image_info.get('width', 0)
            height_px = image_info.get('height', 0)
        
        if width_px == 0 or height_px == 0:
            return None

        # Get PDF bbox dimensions (in points)
        pdf_fig_width = image_info.get('pdf_fig_width')
        pdf_fig_height = image_info.get('pdf_fig_height')
        
        if not pdf_fig_width or not pdf_fig_height:
            return None

        # Convert points to inches (72 points = 1 inch)
        width_in = pdf_fig_width / 72.0
        height_in = pdf_fig_height / 72.0

        # Calculate effective DPI
        if width_in > 0 and height_in > 0:
            dpi_x = width_px / width_in
            dpi_y = height_px / height_in
            return int((dpi_x + dpi_y) / 2)  # dpi should be an integer

        return None

    def write_image(self,image,filename=None):
        if filename is None:
            filename = self.make_figure_filename(image)
        pic = fitz.Pixmap(self.doc, image['xref'])
        failed = False
        try:
            pic.save(filename)
        except ValueError:
            failed = True
        if failed:
            # If save fails, try to convert to RGB and try again
            pic = fitz.Pixmap(fitz.csRGB, pic)
            pic.save(filename)

    @staticmethod
    def save_image(doc, page, pdf_fig_bbox, image_path, xref=None, dpi=STANDARD_DPI, annots=True):
        pix = None

        if xref is not None:
            xref = int(xref)

        if xref is not None and xref > 0:
            pix = fitz.Pixmap(doc, xref)
        else:
            pix = page.get_pixmap(clip=pdf_fig_bbox, dpi=dpi, annots=annots)   # pdf_fig_bbox - [x0,y0,x1,y1]
                
        try:
            pix.save(image_path)
        except Exception as e:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(image_path)
        return pix
    
    def figures(self,images_data=None,filter=None):
        if images_data is not None:         # images_data is provided by figcap 
            '''Generator that yields image metadata when the data is already provided (images_data)'''
            for image_info in images_data.get('figures', {}):
                if filter is None or filter.keep(image_info):
                    # image_info['dpi'] = PDFHandler.calculate_dpi(image_info, self.doc) or self.STANDARD_DPI
                    yield image_info
        else:
            image_count = 1
            for page_number,page in enumerate(self.pages(),1):
                images = self.images_per_page(page)         # image identification is based on xrefs
                for image_number,image in enumerate(images,1):
                    pdf_fig_width, pdf_fig_height = self.image_dimensions(image)
                    image['page_number'] = page_number
                    image['image_number'] = image_number                # image count per page
                    image['pdf_fig_bbox'] = self.image_bbox(image)      # pdf_fig_bbox - x0,y0,x1,y1
                    image['pdf_fig_width'] = pdf_fig_width
                    image['pdf_fig_height'] = pdf_fig_height
                    image['page_width'] = page.rect.width
                    image['page_height'] = page.rect.height
                        
                    if filter is None or filter.keep(image):
                        image['image_count'] = image_count                  # total image count so far
                        # image['dpi'] = PDFHandler.calculate_dpi(image, self.doc) or self.STANDARD_DPI
                        image_count += 1
                        yield image
                                    
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

    pdf = PDFHandler(sys.argv[1])
    filter = CompoundPDFImageFilter(
        PDFXRefImageFilter(min_xref=1),
        PDFImageSizeFilter(width=90,height=90)
        )
    for fig in pdf.figures(filter=filter):
        print(fig)
        pdf.write_image(fig)