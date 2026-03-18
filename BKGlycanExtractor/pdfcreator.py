
import fitz  # PyMuPDF
import os

class PDFCreator(object):
    PAGE_WIDTH = 612.0
    PAGE_HEIGHT = 792.0
    MARGIN = 72.0  
    CAPTION_SPACE = 60.0 

    def __init__(self):
        self.clear()

    def clear(self):
        self.images = []
        self.text = []

    def add_image(self,imagefile,caption=None):
        assert os.path.exists(imagefile)
        self.images.append((imagefile,caption))
    
    def add_text(self,text):
        self.text.append(text)
    
    def write(self,outfile):
        doc = fitz.open()
        scale = 1e+20

        for img_path, caption in self.images:
            img_doc = fitz.open(img_path)
            img_w = img_doc[0].rect.width
            img_h = img_doc[0].rect.height
            img_doc.close()

            avail_w = self.PAGE_WIDTH - (2 * self.MARGIN)
            avail_h = self.PAGE_HEIGHT - (2 * self.MARGIN) - self.CAPTION_SPACE

            scale = min(avail_w / img_w, avail_h / img_h, scale)

        for img_path, caption in self.images:

            img_doc = fitz.open(img_path)
            img_w = img_doc[0].rect.width
            img_h = img_doc[0].rect.height
            img_doc.close()
        
            new_w = img_w * scale
            new_h = img_h * scale

            x0 = (self.PAGE_WIDTH - new_w) / 2.0
            y0 = self.MARGIN
            x1 = x0 + new_w
            y1 = y0 + new_h

            img_rect = fitz.Rect(x0, y0, x1, y1)

            page = doc.new_page(width=self.PAGE_WIDTH, height=self.PAGE_HEIGHT)

            # Insert the image losslessly. Passing 'filename' prevents PyMuPDF from 
            # re-encoding the JPEG, storing the exact raw binary stream inside the PDF.
            page.insert_image(img_rect, filename=img_path)

            if caption:
                # Define a bounding box for the caption text just below the image
                caption_rect = fitz.Rect(x0, 
                                         y1 + 5, 
                                         x1, 
                                         self.PAGE_HEIGHT - 20)

                # Insert the caption text, horizontally centered
                rc = page.insert_textbox(
                    caption_rect, 
                    caption, 
                    fontsize=10, 
                    fontname="helv", 
                    align=fitz.TEXT_ALIGN_LEFT
                )
                
                if rc < 0:
                    print(f"Warning: Caption for {img_path} was STILL too long to fit in the bounding box.")
                
        # Save and close the generated document
        doc.save(outfile, garbage=4, deflate=True)
        doc.close()

if __name__ == "__main__":

    import sys

    pdfwriter = PDFCreator()

    pdffile = sys.argv[1]
    imagefiles = sys.argv[2:]

    for i,ifn in enumerate(imagefiles):
        pdfwriter.add_image(ifn,"Figure %d. Lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum, lorum ipsum."%(i+1,))
    
    pdfwriter.write(pdffile)

