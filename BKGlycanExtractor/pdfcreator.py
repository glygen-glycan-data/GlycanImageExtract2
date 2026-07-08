
import fitz  # PyMuPDF
import os, sys
from . pmc_details import PMCData

class PDFCreator(object):
    PAGE_WIDTH = 612.0
    PAGE_HEIGHT = 792.0
    MARGIN = 72.0  
    CAPTION_SPACE = 60.0 
    TITLE_SPACE = 50.0

    def __init__(self,pmid=None):
        self.clear()
        self.citation = {}
        if pmid:
            self.citation = PMCData.citation_details(pmid)

    def clear(self):
        self.images = []
        self.text = []

    def add_image(self,imagefile,caption=None,figure_number=None,allow_upscale=True):
        assert os.path.exists(imagefile)
        self.images.append((imagefile,caption,figure_number,allow_upscale))
    
    def add_text(self,text):
        self.text.append(text)
    
    def write(self,outfile):
        # Replaced document wide image scaling with per image scaling and 
        # an allow_upscale flag so full page figures still fit the page while single glycan 
        # images are no longer stretched.

        doc = fitz.open()
        # scale = 1e+20
        avail_w = self.PAGE_WIDTH - (2 * self.MARGIN)
        avail_h = self.PAGE_HEIGHT - (2 * self.MARGIN) - self.CAPTION_SPACE

        if self.citation.get('title'):
            page = doc.new_page(width=self.PAGE_WIDTH, height=self.PAGE_HEIGHT)
            title_rect = fitz.Rect(self.MARGIN, 
                                   self.MARGIN,
                                   self.PAGE_WIDTH-self.MARGIN,
                                   self.MARGIN+2*self.TITLE_SPACE)
        
            rc = page.insert_textbox(
                title_rect, 
                self.citation['title'], 
                fontsize=16, 
                fontname="helv", 
                align=fitz.TEXT_ALIGN_CENTER
            )
        
            if self.citation.get('citation'):
                author_rect = fitz.Rect(self.MARGIN, 
                       self.MARGIN+2*self.TITLE_SPACE,
                       self.PAGE_WIDTH-self.MARGIN,
                       self.MARGIN+5*self.TITLE_SPACE)
                rc = page.insert_textbox(
                    author_rect, 
                    self.citation['ascii_citation'],
                    fontsize=12, 
                    fontname="helv", 
                    align=fitz.TEXT_ALIGN_LEFT
                )

            if self.citation.get('doi'):
                doi_rect = fitz.Rect(self.MARGIN, 
                                     self.MARGIN+5*self.TITLE_SPACE,
                                     self.PAGE_WIDTH-self.MARGIN,
                                     self.MARGIN+6*self.TITLE_SPACE)

                rc = page.insert_textbox(
                    doi_rect, 
                    "PMID:"+str(self.citation["pmid"]) + " doi:"+self.citation['doi'], 
                    fontsize=12, 
                    fontname="helv", 
                    align=fitz.TEXT_ALIGN_CENTER
                )

        for img_path, caption, figure_number, allow_upscale in self.images:

            img_doc = fitz.open(img_path)
            img_w = img_doc[0].rect.width
            img_h = img_doc[0].rect.height
            img_doc.close()
            fit_scale = min(avail_w / img_w, avail_h / img_h)
            max_upscale = float('inf') if allow_upscale else 1.0  # if allow_upscale is False (simple glycan image) --> never enlarge the image
            scale = min(fit_scale, max_upscale)
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
            
            # outlining the figures doesn't seem to help. 
            # fig_annot = page.add_rect_annot([ x0-1, y0-1, x1+1, y1+1 ])
            # fig_annot.set_colors(stroke=(0, 0, 0))
            # fig_annot.set_border(width=1)
            # fig_annot.update()

            if caption:
                if figure_number:
                    full_caption = f"Figure {figure_number}. {caption}"
                else:
                    full_caption = f"{caption}"

                # Define a bounding box for the caption text just below the image
                caption_rect = fitz.Rect(x0, 
                                         y1 + 5, 
                                         x1, 
                                         self.PAGE_HEIGHT - 20)

                # Insert the caption text, horizontally centered
                rc = page.insert_textbox(
                    caption_rect, 
                    full_caption, 
                    fontsize=10, 
                    fontname="helv", 
                    align=fitz.TEXT_ALIGN_LEFT
                )
                
                if rc < 0:
                    rc = page.insert_textbox(
                        caption_rect, 
                        full_caption[:200]+"...", 
                        fontsize=10, 
                        fontname="helv", 
                        align=fitz.TEXT_ALIGN_LEFT
                    )
                
        # Save and close the generated document
        doc.save(outfile, garbage=4, deflate=True)
        doc.close()

if __name__ == "__main__":

    import sys

    pdfwriter = PDFCreator()

    pdffile = sys.argv[1]
    sys.argv.pop(1)
    
    if len(sys.argv) <= 1:
        sys.exit(1)

    pmid = None
    if not os.path.exists(sys.argv[1]):
        try:
            pmid = int(sys.argv[1])
            sys.argv.pop(1)
        except:
            pass
    
    pdfwriter = PDFCreator(pmid)
    
    i = 1; j = 1
    while i < len(sys.argv):
        assert os.path.exists(sys.argv[i])
        if (i+1) < len(sys.argv) and not os.path.exists(sys.argv[i+1]):
            caption = f"Figure {j}. {sys.argv[i+1]}"
            pdfwriter.add_image(sys.argv[i],caption)
            i += 2; j += 1
        else:
            pdfwriter.add_image(sys.argv[i])
            i += 1; j += 1
    
    pdfwriter.write(pdffile)

