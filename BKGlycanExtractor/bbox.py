
__all__ = [ 'BoundingBox' ]

import cv2
import copy

def hasall(dct,*keys):
    return all(map(lambda k: dct.get(k) is not None, keys))

class BoundingBox: 
    reserved_kwargs = set("""
       image image_width image_height
       x y w h
       x y width height
       x1 y1 x2 y2
       rx ry rw rh
       rcx rcy rw rh
    """.split())
    def __init__(self, **kwargs):

        self.set_image_dimensions(**kwargs)

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
        elif hasall(kwargs,'x1','y1','x2','y2'):
            self.x = int(kwargs['x1'])
            self.y = int(kwargs['y1'])
            self.w = int(kwargs['x2']-kwargs['x1'])
            self.h = int(kwargs['y2']-kwargs['y1'])
        elif hasall(kwargs,'rx','ry','rw','rh'):
            if self.imwidth is None or self.imheight is None:
                raise ValueError("required arguments missing")
            self.x = int(self.imwidth*kwargs['rx'])
            self.y = int(self.imwidth*kwargs['ry'])
            self.w = int(self.imwidth*kwargs['rw'])
            self.h = int(self.imwidth*kwargs['rh'])
        elif hasall(kwargs,'rcx','rcy','rw','rh'):
            if self.imwidth is None or self.imheight is None:
                raise ValueError("required arguments missing")
            self.x = int(self.imwidth*(kwargs['rcx']-kwargs['rw']/2))
            self.y = int(self.imheight*(kwargs['rcy']-kwargs['rh']/2))
            self.w = int(self.imwidth*kwargs['rw'])
            self.h = int(self.imheight*kwargs['rh'])
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
            self.imwidth = width
            self.imheight = height
        elif hasall(kwargs,'image_width','image_height'):
            self.imwidth = kwargs['image_width']
            self.imheight = kwargs['image_height']
        else:
            self.imwidth = None
            self.imheight = None

    def set(self,key,value):
        self.data[key] = value

    def has(self,key):
        return key in self.data

    def get(self,key,default=None):
        return self.data.get(key,default)

    def clone(self):
        return BoundingBox(image_width=self.imwidth, image_height=self.imheight,
                           x=self.x, y=self.y, w=self.w, h=self.h, **self.data)

    def center(self):
        return (int(self.x+self.w/2),int(self.y+self.h/2))

    def corners(self):
        return (self.x,self.y,self.x+self.w,self.y+self.h)
	
    def area(self):
        return (self.w+1) * (self.h+1)

    def bbox(self):
        return (self.x,self.y,self.w,self.h)
 
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
                (self.x+self.w)/self.imwidth,
                (self.y+self.h)/self.imheight)

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

    def crop(self,image):
        (x1, y1, x2, y2) = self.corners()
        return image[y1:y2, x1:x2].copy()

    def pad(self, padding):
        self.x -= padding
        self.y -= padding
        self.w += 2*padding
        self.h += 2*padding
        self.normalize()

    def shift(self, dx=0, dy=0):
        self.x += dx
        self.y += dy
        self.normalize()

    def pad_relative(self, padding):
        assert 0 <= padding <= 1
        self.x -= padding*self.w
        self.y -= padding*self.h
        self.w += 2*padding*self.w
        self.h += 2*padding*self.h
        self.normalize()

    def normalize(self):
        self.x = max(self.x,0)
        self.y = max(self.y,0)
        if self.imwidth is not None:
            self.w = min(self.w,self.imwidth-self.x)
        if self.imheight is not None:
            self.h = min(self.h,self.imheight-self.y)

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
