class BoundingBox:
    def __init__(self, image, **kwargs):
        height, width, channels = image.shape
        self.imwidth = width
        self.imheight = height
        self.rel_cen_x = kwargs.get("rel_cen_x", None)
        self.rel_cen_y = kwargs.get("rel_cen_y", None)
        self.rel_w = kwargs.get("rel_w", None)
        self.rel_h = kwargs.get("rel_h", None)
        self.cen_x = kwargs.get("cen_x", None)
        self.cen_y = kwargs.get("cen_y", None)
        self.w = kwargs.get("width", None)
        self.h = kwargs.get("height", None)
        self.x = kwargs.get("x", None)
        self.y = kwargs.get("y", None)
        self.x2 = kwargs.get("x2",None)
        self.y2 = kwargs.get("y2", None)
    def area(self):
        return (self.w+1)*(self.h+1)
    def center_to_corner(self):
        assert self.cen_x is not None
        self.x = int(self.cen_x - self.w/2)
        self.y = int(self.cen_y - self.h/2)
    def pad_borders(self):
        assert self.x is not None
        self.x = self.x - int(0.2*self.w)
        self.y = self.y - int(0.2*self.h)
        self.w = int(1.4*self.w)
        self.h = int(1.4*self.h) 
    def set_name(self,name):
        self.name = name
    def to_four_corners(self):
        assert self.x is not None
        self.x2 = self.x + self.w
        self.y2 = self.y + self.h
    def to_list(self):
        raise NotImplementedError
    def __str__(self):
        printdict = {"x0": str(self.x), "x2": str(self.x2), "y0": str(self.y), "y2": str(self.y2)}
        #print(printdict)
        printstr = "[ "
        for key, value in printdict.items():
            printstr += key + ": "
            printstr += value + "    "
        printstr += "]"
        return printstr
        
        
class Detected(BoundingBox):
    def __init__(self,image,confidence,white_space = 0,**kwargs):
        super().__init__(image,**kwargs)
        self.confidence = confidence
        self.whitespace = white_space
    def fix_borders(self):
        assert self.x is not None
        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x+self.w >= self.imwidth:
            self.w = int(self.imwidth-self.x)
        if self.y+self.h >= self.imheight:
            self.h = int(self.imheight-self.y)
    def fix_image(self):
        assert self.cen_x is not None
        half_white_space = int(self.whitespace/2)
        self.cen_x = self.cen_x - half_white_space
        self.cen_y = self.cen_y - half_white_space
        self.rel_cen_x = float(self.cen_x/self.imwidth)
        self.rel_cen_y = float(self.cen_y/self.imheight)
        self.rel_w = float(self.w/self.imwidth)
        self.rel_h = float(self.h/self.imheight)
    def is_entire_image(self):
        assert self.w is not None
        if self.w*self.h > 0.8*0.8*self.imwidth*self.imheight:
            self.x = 0
            self.y = 0
            self.w = self.imwidth
            self.h = self.imheight
        else:
            pass    
    def rel_to_abs(self):
        assert self.rel_cen_x is not None
        self.cen_x = int(self.rel_cen_x * (self.imwidth+self.whitespace))
        self.cen_y = int(self.rel_cen_y * (self.imheight+self.whitespace))
        self.w = int(self.rel_w * (self.imwidth+self.whitespace))
        self.h = int(self.rel_h * (self.imheight+self.whitespace))
    def to_list(self):
        [x,y,w,h,confidence] = [self.x, self.y, self.w, self.h, self.confidence]
        return [x,y,w,h,confidence]
    def to_pdf_coords(self):
        assert self.x is not None
        self.x0 = self.x/self.imwidth
        self.y0 = self.y/self.imheight
        self.x1 = self.x2/self.imwidth
        self.y1 = self.y2/self.imheight
        
class Training(BoundingBox):
    def abs_to_rel(self):
        self.rel_cen_x = float(self.cen_x/self.imwidth)
        self.rel_cen_y = float(self.cen_y/self.imheight)
        self.rel_w = float(self.w/self.imwidth)
        self.rel_h = float(self.h/self.imheight)
    def rel_to_abs(self):
        assert self.rel_cen_x is not None
        self.cen_x = int(self.rel_cen_x*self.imwidth)
        self.cen_y = int(self.rel_cen_y*self.imheight)
        self.w = int(self.rel_w*self.imwidth)
        self.h = int(self.rel_h*self.imheight)
    def reset_image(self,white_space):
        self.white_space = white_space
        half_white_space = int(white_space/2)
        self.cen_x = self.cen_x + half_white_space
        self.cen_y = self.cen_y + half_white_space
        self.imwidth = self.imwidth + self.white_space
        self.imheight = self.imheight + self.white_space
    def to_list(self):
        [glycan,rel_cen_x,rel_cen_y,rel_w,rel_h] = [0,self.rel_cen_x,self.rel_cen_y,self.rel_w,self.rel_h]
        return [glycan,rel_cen_x,rel_cen_y,rel_w,rel_h]
