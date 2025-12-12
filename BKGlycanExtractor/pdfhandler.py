
import fitz, os, os.path

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
        x0,y0,x1,y1 = image_info['bbox']      # fitz based bbox is [x0,y0,x1,y1]
        return abs(x0-x1)+1, abs(y0-y1)+1         # returns width, height
    
    # TODO need to create a class which verifies or converts bbox and box version automatcailly 
    @staticmethod
    def image_bbox(image_info):
        return image_info['bbox']       # returns fitz based bbox is [x0,y0,x1,y1]


    @staticmethod
    def create_box(bbox):
        return fitz.Rect(bbox)

    @staticmethod
    def calculate_dpi(image_info):
        """
        Calculate DPI from image info.
        Uses xres/yres if available in image_info.
        """
        # Check if xres/yres are directly available (PyMuPDF sometimes includes this)
        if 'xres' in image_info and 'yres' in image_info:
            # Use average if x and y differ slightly
            dpi_x = image_info['xres']
            dpi_y = image_info['yres']
            
            return int(max(dpi_x, dpi_y))
        
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
    
    def figures(self,filter=None):
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

                dpi = self.calculate_dpi(image)
                if dpi is not None:
                    image['dpi'] = dpi
                    
                if filter is None or filter.keep(image):
                    image['image_count'] = image_count                  # total image count so far
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