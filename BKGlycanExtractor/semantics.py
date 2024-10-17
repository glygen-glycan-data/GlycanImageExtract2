'''

'''
import cv2
import os
import numpy as np
import json
import copy
from collections import defaultdict

# Base class for any thing (figure, glycan) which has an image with width and height
class Image_Semantics:
    '''
    image can be any of the following formats: img_path, png, cv2, pdf
    '''

    def __init__(self,image):
        self.semantics = {}
        self.semantics['image'] = image
        height, width, _ = image.shape
        self.semantics['height'] = height
        self.semantics['width'] = width

    def image(self):
        return self.semantics['image']

    def width(self):
        return self.semantics['width']

    def height(self):
        return self.semantics['height']

    def set(self,key,value):
        self.semantics[key] = value

    def has(self,key):
        return key in self.semantics

    def get(self,key,default=None):
        return self.semantics.get(key,default)


# Class for whole figure/image containing glycans
class Figure_Semantics(Image_Semantics):
    def __init__(self,image_path,**kwargs):
        image = self.format_image(image_path)
        super().__init__(image)
        self.semantics['image_path'] = os.path.abspath(image_path)
        self.semantics['file_name'] = os.path.basename(image_path) 
        self.semantics['glycans'] = []
        self.semantics.update(copy.deepcopy(kwargs))

    def tojson(self):
        data = {}
        for k,v in self.semantics.items():
            if k not in ('image','box','glycans'):
                data[k] = v
        data['glycans'] = [ json.loads(gly.tojson()) for gly in self.glycans() ]
        return json.dumps(data,indent=2,sort_keys=True)

    def format_image(self,image):
        if isinstance(image, str):
            if (image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg')):
                glycan_image = cv2.imread(image)
                return glycan_image
            elif image.endswith('.pdf'):
                raise ValueError("Can't handle PDF format: "+image)
                return pdf
        elif isinstance(image, np.ndarray):
            return image
        raise ValueError("Can't handle image format: "+image)

    def image_path(self):
        return self.semantics['image_path']

    def glycans(self):
        return self.semantics['glycans']

    def clear_glycans(self):
        self.semantics['glycans'] = []

    def add_glycan(self,box,**kwargs):
        if kwargs.get('id') is None:
            if len(self.glycans()) == 0:
                kwargs['id'] = 1
            else:
                kwargs['id'] = max(gly.semantics['id'] for gly in self.glycans())+1
        box.set_image_dimensions(image_width=self.width(),image_height=self.height())
        gly = Glycan_Semantics(image=box.crop(self.image()),box=box,**kwargs)
        self.semantics['glycans'].append(gly)

class Glycan_Semantics(Image_Semantics):
    def __init__(self,image,box,**kwargs):
        super().__init__(image)  
        self.semantics['box'] = box
        self.semantics['bbox'] = box.bbox()
        self.semantics['monos'] = {}
        self.semantics.update(copy.deepcopy(kwargs))

    def clear_monos(self):
        self.semantics['monos'] = {}

    def add_mono(self,symbol,box,**kwargs):
        if kwargs.get('id') is None:
            if len(self.monosaccharides()) == 0:
                kwargs['id'] = 1
            else:
                kwargs['id'] = max(self.semantics['monos'])+1
        mono = dict(symbol=symbol,box=box,bbox=box.bbox(),center=box.center(),links=[],**kwargs)
        assert kwargs['id'] not in self.semantics['monos']
        self.semantics['monos'][kwargs['id']] = mono

    def monosaccharides(self):
        return self.semantics['monos'].values()

    def mono_boxes(self):
        boxes = []
        for id,mono in self.semantics['monos'].items():
            boxes.append(mono['box'])
        return boxes

    def add_root(self,root_id=None):
        self.semantics['root'] = root_id 

    def add_link(self,id,link_ids):
        assert id is not None
        self.semantics['monos'][id]['links'] = link_ids

    def get_links(self,id):
        return list(self.semantics['monos'][id]['links'])

    def tojson(self):
        data = {}
        for k,v in self.semantics.items():
            if k not in ('image','box','monos'):
                data[k] = v
        data['monos'] = []
        for mono in self.semantics['monos'].values():
            monodict = {}
            for k,v in mono.items():
                if k not in ('image','box'):
                    monodict[k] = v
            data['monos'].append(monodict)
        return json.dumps(data,indent=2,sort_keys=True)

    def image_path(self):
        # for single glycan images, the glycan image "has" a path
        return self.semantics.get('image_path',None)

    def composition(self):
        count = defaultdict(int)
        for m in self.monosaccharides():
            sym = m['symbol']
            if sym not in count:
                count[sym] = 1
            else:
                count[sym] += 1
        return count

    mono_syms = ["GlcNAc","GalNAc","Man","Gal","Glc","Fuc","NeuAc","NeuGc"]

    def compstr(self):
        comp = self.composition()
        retval = ""
        for sym in self.mono_syms:
            if comp[sym] > 0:
                retval += sym + "(" + str(comp[sym]) + ")"
        return retval
