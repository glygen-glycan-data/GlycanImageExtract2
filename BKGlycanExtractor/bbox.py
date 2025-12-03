
__all__ = [ 'BoundingBox' ]

import copy

def hasall(dct,*keys):
    return all(map(lambda k: dct.get(k) is not None, keys))

from . compareboxes import CompareBoxes

class BaseBoundingBox:
    '''
    Abstarct Base class for bounding boxes

    Defines the common interface that all bounding box class implementations must provide.
    '''

    def area(self):
        raise NotImplementedError
    
    def corners(self):
        """
        Return corner coordinates.
        For pixel boxes: (x, y, x+w-1, y+h-1)
        For PDF boxes: (x0, y0, x1, y1)
        """
        raise NotImplementedError
    
    def bbox(self):
        """
        Return bounding box
        For pixel boxes: (x, y, w, h)
        For PDF boxes: (x0, y0, x1, y1)
        """
        raise NotImplementedError
    
    def center(self):
        """Return the center point as (cx, cy)."""
        raise NotImplementedError
    
    def width(self):
        raise NotImplementedError
    
    def height(self):
        raise NotImplementedError



class BoundingBox(BaseBoundingBox): 
    '''
    Pixel-based bounding box using (x, y, w, h) format with integer (pixels) coordinates.

    Internal representation has: x, y, w, h as integers.
    '''

    reserved_kwargs = set("""
       image image_width image_height
       x y w h
       x y width height
       bbox
       x1 y1 x2 y2
       rx ry rw rh
       rcx rcy rw rh
    """.split())
    def __init__(self, **kwargs):

        self.set_image_dimensions(**kwargs)

        # 
        # absolute dimensions should be integers (or convert to ints)
        # image width, height should be integers (or convert to ints)
        #
        # corner to width conversions
        #     x1 + w = x2 + 1; y1 + w = y2 + 1;
        #
        # relative dimensions should be floats (or convert to floats)
        #
        # primary units are absolute, x,y,w,h
        #
        if hasall(kwargs,'x','y','w','h'):
            self.x = int(kwargs['x'])
            self.y = int(kwargs['y'])
            self.w = int(kwargs['w'])
            self.h = int(kwargs['h'])
        elif hasall(kwargs,'x','y','width','height'):
            self.x = int(kwargs['x'])
            self.y = int(kwargs['y'])
            self.w = int(kwargs['width'])
            self.h = int(kwargs['height'])
        elif hasall(kwargs,'bbox'):
            self.x = int(kwargs['bbox'][0])
            self.y = int(kwargs['bbox'][1])
            self.w = int(kwargs['bbox'][2])
            self.h = int(kwargs['bbox'][3])
        elif hasall(kwargs,'x1','y1','x2','y2'):
            self.x = int(kwargs['x1'])
            self.y = int(kwargs['y1'])
            self.w = int(kwargs['x2'])-int(kwargs['x1'])+1
            self.h = int(kwargs['y2'])-int(kwargs['y1'])+1
        elif hasall(kwargs,'rx','ry','rw','rh'):
            if self.imwidth is None or self.imheight is None:
                raise ValueError("required arguments missing")
            self.x = int(round(self.imwidth*float(kwargs['rx'])))
            self.y = int(round(self.imheight*float(kwargs['ry'])))
            self.w = int(round(self.imwidth*float(kwargs['rw'])))
            self.h = int(round(self.imheight*float(kwargs['rh'])))
        elif hasall(kwargs,'rcx','rcy','rw','rh'):
            if self.imwidth is None or self.imheight is None:
                raise ValueError("required arguments missing")
            self.x = int(round(self.imwidth*(float(kwargs['rcx'])-float(kwargs['rw'])/2)))
            self.y = int(round(self.imheight*(float(kwargs['rcy'])-float(kwargs['rh'])/2)))
            self.w = int(round(self.imwidth*float(kwargs['rw'])))
            self.h = int(round(self.imheight*float(kwargs['rh'])))
        else:
            raise ValueError("required arguments missing")

        self.data = dict()
        for k,v in kwargs.items():
            if k not in self.reserved_kwargs:
                self.data[k] = copy.deepcopy(v)

    def set_image_dimensions(self,**kwargs):
        if hasall(kwargs,'image'):
            image = kwargs["image"]
            # cv2 image1!
            height, width, channels = image.shape
            self.imwidth = int(width)
            self.imheight = int(height)
        elif hasall(kwargs,'image_width','image_height'):
            self.imwidth = int(kwargs['image_width'])
            self.imheight = int(kwargs['image_height'])
        else:
            self.imwidth = None
            self.imheight = None

    def set(self,key,value):
        self.data[key] = value

    def has(self,key):
        return key in self.data

    def get(self,key,default=None):
        return self.data.get(key,default)

    # method to return a dict of items that the box object contains except the dimensions
    def items(self):
        # return self.data    # any changes made will to the data passed from here will reflect back to the object.....users can utilize getters/setters to make changes instead
        return {**self.data}

    def update(self,**kwargs):
        self.data.update(kwargs)

    def clone(self):
        return BoundingBox(image_width=self.imwidth, image_height=self.imheight,
                           x=self.x, y=self.y, w=self.w, h=self.h, **self.data)

    # not necessarily integers!
    def center(self):
        return (self.x+self.w/2,self.y+self.h/2)

    def width(self):
        return self.w
    
    def height(self):
        return self.h

    def corners(self):
        return (self.x,self.y,self.x+self.w-1,self.y+self.h-1)
	
    def area(self):
        return self.w * self.h

    def bbox(self):
        return (self.x,self.y,self.w,self.h)
    
    def update_bbox(self,**kwargs):
        """Update bounding box values dynamically if provided."""
        if 'x' in kwargs:
            self.x = int(kwargs['x'])
        if 'y' in kwargs:
            self.y = int(kwargs['y'])
        if 'w' in kwargs:
            self.w = int(kwargs['w'])
        if 'h' in kwargs:
            self.h = int(kwargs['h'])
 
    def tolist(self,*extra_keys):
        return list(self.bbox()) + [ self.data.get(k) for k in extra_keys ]

    def center_relative(self):
        assert(self.imwidth is not None and self.imheight is not None)
        return ((self.x+self.w/2)/self.imwidth,
                (self.y+self.h/2)/self.imheight,
                self.w/self.imwidth,
                self.h/self.imheight)

    def corners_relative(self):
        assert(self.imwidth is not None and self.imheight is not None)
        return (self.x/self.imwidth,
                self.y/self.imheight,
                (self.x+self.w-1)/self.imwidth,
                (self.y+self.h-1)/self.imheight)

    def contains(self,b):
        return CompareBoxes.is_contained_in(b,self)

    def __str__(self):
        x1,y1,x2,y2 = self.corners()
        retval = "[ "
        retval += "(%s"%(x1,)
        retval += ", %s)"%(y1,)
        retval += ", (%s"%(x2,)
        retval += ", %s)"%(y2,)
        for k,v in sorted(self.data.items()):
            retval += ", " + k + ": " + str(v)
        retval += " ]"
        return retval
    
    def __repr__(self):
        return str(self)

    def crop(self,image):
        (x1, y1, x2, y2) = self.corners()
        return image[y1:y2, x1:x2].copy()

    def pad(self, padding):
        self.x -= int(padding)
        self.y -= int(padding)
        self.w += int(2*padding)
        self.h += int(2*padding)
        self.normalize()

    def shift(self, dx=0, dy=0):
        self.x += int(dx)
        self.y += int(dy)
        self.normalize()

    def pad_relative(self, padding):
        # assert 0 <= padding <= 1 # permit larger relative padding
        self.x -= int(round(padding*self.w))
        self.y -= int(round(padding*self.h))
        self.w += int(round(2*padding*self.w))
        self.h += int(round(2*padding*self.h))
        self.normalize()

    def normalize(self):
        if self.imwidth is None or self.imheight is None:                                                                          
            raise RuntimeError("Image dimensions not provided.")                                          
        x1,y1,x2,y2 = self.corners()                                                                      
        x1 = max(x1,0)                                                                                    
        y1 = max(y1,0)                                                                                    
        x2 = min(x2,self.imwidth-1)                                                                       
        y2 = min(y2,self.imheight-1)                                                                      
        self.update_bbox(x=x1,y=y1,w=(x2-x1+1),h=(y2-y1+1))

    def to_pdf_bbox(self, pdf_page_width: float, pdf_page_height: float, pdf_fig_bbox):
        """
        Convert to PDFBoundingBox using page dimensions.

        pdf_page_width, pdf_page_height -- refers to the pdf page

        pdf_fig_bbox - helps to calculate the scale
                    
        Returns: PDFBoundingBox with coordinates in PDF space
        """

        if self.imwidth is None or self.imheight is None:
            raise ValueError("Image dimensions required for conversion")

        pdf_x1_orig, pdf_y1_orig, pdf_x2_orig, pdf_y2_orig = pdf_fig_bbox
        pdf_bbox_width = pdf_x2_orig - pdf_x1_orig
        pdf_bbox_height = pdf_y2_orig - pdf_y1_orig
        
        # Convert pixel coordinates to PDF coordinates
        scale_x = pdf_bbox_width / self.imwidth
        scale_y = pdf_bbox_height / self.imheight

        x1 = pdf_x1_orig + (self.x * scale_x)
        y1 = pdf_y1_orig + (self.y * scale_y)
        x2 = pdf_x1_orig + ((self.x + self.w-1) * scale_x)
        y2 = pdf_y1_orig + ((self.y + self.h-1) * scale_y)

        
        return PDFBoundingBox(
            page_width=pdf_page_width,
            page_height=pdf_page_height,
            x1=x1, y1=y1, x2=x2, y2=y2,
            **self.data
        )

    # below here needs to be fixed, commenting for now 
    #
    #    def annotate(self, image, defaulttext='', colour=(0,255,0)):
    #        self.to_four_corners()
    #        p1 = (self.x, self.y)
    #        p2 = (self.x2, self.y2)
    #        cv2.rectangle(image, p1, p2, colour, 3)
    #        if hasattr(self, 'class_dictionary'):
    #            text = self.class_dictionary[self.get_class()]
    #        else:
    #            text = defaulttext
    #        cv2.putText(image, text, p1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colour)
    #        return image
    #
    #    # adjust image for whitespace padding
    #    def fix_image(self):
    #        assert self.cen_x is not None
    #        assert self.w is not None
    #        half_white_space = int(self.whitespace/2)
    #        self.cen_x = self.cen_x - half_white_space
    #        self.cen_y = self.cen_y - half_white_space
    #        self.rel_cen_x = float(self.cen_x/self.imwidth)
    #        self.rel_cen_y = float(self.cen_y/self.imheight)
    #        self.rel_w = float(self.w/self.imwidth)
    #        self.rel_h = float(self.h/self.imheight)
    #        
    #    # pad borders by 20% to protect from cropping issues
    #    def pad_borders(self):
    #        assert self.x is not None
    #        assert self.w is not None
    #        self.x = self.x - int(0.2*self.w)
    #        self.y = self.y - int(0.2*self.h)
    #        self.w = int(1.4*self.w)
    #        self.h = int(1.4*self.h) 
    #
    #    # fix borders 
    #    # so box cannot be outside image boundaries 
    #    # once whitespace is removed
    #    def fix_borders(self):
    #        assert self.x is not None
    #        assert self.w is not None
    #        if self.x < 0:
    #            self.x = 0
    #        if self.y < 0:
    #            self.y = 0
    #        if self.x+self.w >= self.imwidth:
    #            self.w = int(self.imwidth-self.x)
    #        if self.y+self.h >= self.imheight:
    #            self.h = int(self.imheight-self.y)
    #
    #    # check if the bounding box is too large
    #    def is_entire_image(self):
    #        assert self.w is not None
    #        if self.w*self.h > 0.8*0.8*self.imwidth*self.imheight:
    #            self.x = 0
    #            self.y = 0
    #            self.w = self.imwidth
    #            self.h = self.imheight
    #        else:
    #            pass 
    #
    #    #add whitespace to image
    #    def reset_image(self,white_space):
    #        assert self.cen_x is not None
    #        self.white_space = white_space
    #        half_white_space = int(white_space/2)
    #        self.cen_x = self.cen_x + half_white_space
    #        self.cen_y = self.cen_y + half_white_space
    #        self.imwidth = self.imwidth + self.white_space
    #        self.imheight = self.imheight + self.white_space
    #


class PDFBoundingBox(BaseBoundingBox):
    '''
    PDF coordinate bounding box using (x1, y1, x2, y2) format with float coordinates.
    
    Internal representation: x1, y1, x2, y2 as floats
    PDF coordinates are in points (1/72 inch) and are typically floats.
    '''

    reserved_kwargs = set("""
       page_width page_height
       bbox
       x1 y1 x2 y2
       x y w h
       width height
    """.split())


    def __init__(self, **kwargs):
        self.set_page_dimensions(**kwargs)
        
        # PDF bbox format: x1, y1, x2, y2 (corner coordinates as floats)
        if hasall(kwargs, 'x1', 'y1', 'x2', 'y2'):
            self.x1 = float(kwargs['x1'])
            self.y1 = float(kwargs['y1'])
            self.x2 = float(kwargs['x2'])
            self.y2 = float(kwargs['y2'])
        elif hasall(kwargs, 'bbox'):
            # bbox can be [x1, y1, x2, y2] or [x, y, w, h]
            bbox = kwargs['bbox']
            if len(bbox) == 4:
                # Try to determine format: if x2 > x1 and y2 > y1, assume corners
                # Otherwise assume x, y, w, h
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    # Likely x1, y1, x2, y2 format
                    self.x1 = float(bbox[0])
                    self.y1 = float(bbox[1])
                    self.x2 = float(bbox[2])
                    self.y2 = float(bbox[3])
                else:
                    # Likely x, y, w, h format - convert to corners
                    self.x1 = float(bbox[0])
                    self.y1 = float(bbox[1])
                    self.x2 = float(bbox[0]) + float(bbox[2]) - 1
                    self.y2 = float(bbox[1]) + float(bbox[3]) - 1
            else:
                raise ValueError(f"bbox must have 4 elements, got {len(bbox)}")
        elif hasall(kwargs, 'x', 'y', 'w', 'h'):
            # Convert from x, y, w, h to x1, y1, x2, y2
            self.x1 = float(kwargs['x'])
            self.y1 = float(kwargs['y'])
            self.x2 = float(kwargs['x']) + float(kwargs['w']) - 1
            self.y2 = float(kwargs['y']) + float(kwargs['h']) - 1
        elif hasall(kwargs, 'x', 'y', 'width', 'height'):
            self.x1 = float(kwargs['x'])
            self.y1 = float(kwargs['y'])
            self.x2 = float(kwargs['x']) + float(kwargs['width']) - 1
            self.y2 = float(kwargs['y']) + float(kwargs['height']) - 1
        else:
            raise ValueError("required arguments missing: need (x1,y1,x2,y2) or (x,y,w,h) or bbox")
        
        # Ensure x1 < x2 and y1 < y2
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
        
        self.data = dict()
        for k, v in kwargs.items():
            if k not in self.reserved_kwargs:
                self.data[k] = copy.deepcopy(v)
    
    def set_page_dimensions(self, **kwargs):
        """Set PDF page dimensions (in points)."""
        if hasall(kwargs, 'page_width', 'page_height'):
            self.page_width = float(kwargs['page_width'])
            self.page_height = float(kwargs['page_height'])
        else:
            self.page_width = None
            self.page_height = None
    
    def set(self, key, value):
        self.data[key] = value
    
    def has(self, key):
        return key in self.data
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def items(self):
        return {**self.data}
    
    def update(self, **kwargs):
        self.data.update(kwargs)
    
    def clone(self):
        return PDFBoundingBox(
            page_width=self.page_width,
            page_height=self.page_height,
            x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
            **self.data
        )
    
    def center(self):
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    
    def width(self):
        return self.x2 - self.x1 + 1
    
    def height(self):
        return self.y2 - self.y1 + 1
    
    def corners(self):
        """Return corner coordinates as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def area(self):
        return self.width() * self.height()
    
    def bbox(self):
        """Return bounding box in PDF format: (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def bbox_xywh(self):
        """TEST: Return bounding box in (x, y, w, h) format."""
        return (self.x1, self.y1, self.width(), self.height())
    
    def update_bbox(self, **kwargs):
        if 'x1' in kwargs:
            self.x1 = float(kwargs['x1'])
        if 'y1' in kwargs:
            self.y1 = float(kwargs['y1'])
        if 'x2' in kwargs:
            self.x2 = float(kwargs['x2'])
        if 'y2' in kwargs:
            self.y2 = float(kwargs['y2'])
        
        # Ensure x1 < x2 and y1 < y2
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1
    
    def tolist(self, *extra_keys):
        """Return bbox as list plus extra data values."""
        return list(self.bbox()) + [self.data.get(k) for k in extra_keys]
    
    def __str__(self):
        retval = "[ "
        retval += "(%s" % self.x1
        retval += ", %s)" % self.y1
        retval += ", (%s" % self.x2
        retval += ", %s)" % self.y2
        for k, v in sorted(self.data.items()):
            retval += ", " + k + ": " + str(v)
        retval += " ]"
        return retval
    
    def __repr__(self):
        return str(self)
    
    def normalize(self):
        """Normalize bounding box to be within page boundaries."""
        if self.page_width is None or self.page_height is None:
            raise RuntimeError("Page dimensions not provided.")
        
        self.x1 = max(self.x1, 0.0)
        self.y1 = max(self.y1, 0.0)
        self.x2 = min(self.x2, self.page_width)
        self.y2 = min(self.y2, self.page_height)
        
        # Ensure valid box
        if self.x1 >= self.x2:
            self.x1 = 0.0
            self.x2 = self.page_width
        if self.y1 >= self.y2:
            self.y1 = 0.0
            self.y2 = self.page_height
    
    def pad(self, padding: float):
        self.x1 -= padding
        self.y1 -= padding
        self.x2 += padding
        self.y2 += padding
        self.normalize()
    
    def shift(self, dx: float = 0.0, dy: float = 0.0):
        """Shift the bounding box by dx, dy (in PDF points)."""
        self.x1 += dx
        self.y1 += dy
        self.x2 += dx
        self.y2 += dy
        self.normalize()
    
    def pad_relative(self, padding):
        w = self.width()
        h = self.height()
        self.x1 -= padding * w
        self.y1 -= padding * h
        self.x2 += padding * w
        self.y2 += padding * h
        self.normalize()
    
    # TODO NEED TO TEST
    def to_pixel_bbox(self, image_width: int, image_height: int):
        """
        Convert to BoundingBox (pixel coordinates) using image dimensions.

        image_width, image_height --> int in pixels
        
            
        Returns: BoundingBox with coordinates in pixel space
            
        Note: Conversion from float PDF coordinates to integer pixels may lose precision.
        The conversion uses rounding to minimize error.
        """
        if self.page_width is None or self.page_height is None:
            raise ValueError("Page dimensions required for conversion")
        
        # Convert PDF coordinates to pixel coordinates
        scale_x = image_width / self.page_width
        scale_y = image_height / self.page_height
        
        # Convert corners, then to x, y, w, h
        # Use rounding to minimize conversion error
        x1_px = round(self.x1 * scale_x)
        y1_px = round(self.y1 * scale_y)
        x2_px = round(self.x2 * scale_x)
        y2_px = round(self.y2 * scale_y)
        
        # Convert to x, y, w, h format
        x = int(x1_px)
        y = int(y1_px)
        w = int(x2_px) - int(x1_px) + 1  # +1 to include both endpoints
        h = int(y2_px) - int(y1_px) + 1
        
        return BoundingBox(
            image_width=image_width,
            image_height=image_height,
            x=x, y=y, w=w, h=h,
            **self.data
        )






