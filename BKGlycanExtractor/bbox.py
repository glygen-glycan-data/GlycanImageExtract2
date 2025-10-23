
__all__ = [ 'BoundingBox' ]

import copy

def hasall(dct,*keys):
    return all(map(lambda k: dct.get(k) is not None, keys))

class BoundingBox: 
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
