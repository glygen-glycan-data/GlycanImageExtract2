import cv2
import os
import numpy as np
import json
import copy
import random
from . bbox import BoundingBox, PDFBoundingBox
from collections import defaultdict, deque
try:
    from . lineno import callsig
except ImportError:
    pass

class Semantics(object):
    def __init__(self,kvdict=None,**kwargs):
        self._semantics = {}
        if kvdict is not None:
            self.update(kvdict)
        self.update(**kwargs)

    def set(self,key,value):
        self._semantics[key] = value
    
    def unset(self,key):
        if key in self._semantics:
            del self._semantics[key]

    def append(self,key,value):
        if not key in self._semantics:
            self._semantics[key] = []
        self._semantics[key].append(value)

    def has(self,key):
        return key in self._semantics

    def get(self,key,default=None):
        return self._semantics.get(key,default)

    def update(self,kvdict=None,**kwargs):
        if kvdict is not None:
            for k,v in kvdict:
                self.set(k,v)
        for k,v in kwargs.items():
            self.set(k,v)

    def items(self):
        return self._semantics.items()
        # return dict(self._semantics)

    def keys(self):
        return self._semantics.keys()

    def values(self):
        return self._semantics.values()

    # make semantics objects behave like dictionaries
    def __getitem__(self,key):
        if not self.has(key):
            raise KeyError(key)
        return self.get(key)

    def __contains__(self,key):
        return self.has(key)

    def __iter__(self):
        return self.keys()
    
    def __str__(self):
        return "<"+self.__class__.__name__+" instance>"

    def __repr__(self):
        return self.__str__()

    def log(self,message):
        self.append('log',"[%s] %s"%(callsig(1),message))

# We make the box optional so that even those semantics classes that
# *might* have a box can be derived from it to handle their optional
# box...
class BoxPredictionSemantics(Semantics):
    def __init__(self, *, classlabel=None, box=None, confidence=None, **kwargs):
        super().__init__(**kwargs)
        if box is not None:
            self.set_box(box)
        if confidence is not None:
            self.set('confidence',float(confidence))
        if classlabel is not None:
            self.set('classlabel',classlabel)

    def set_box(self,box):
        self.set('box',box)
        self.set('bbox',box.bbox())
        self.set('center',box.center())
        self.set('width',box.width())
        self.set('height',box.height())

        # # add confidence and classlabel - for box_to_object))
        # confidence = box.get('confidence')
        # classlabel = box.get('classlabel')

        # if confidence is not None:
        #     self.set('confidence', float(confidence))

        # if classlabel is not None:
        #     self.set('classlabel', classlabel)

    def box(self):
        return self.get('box')
    
    def corners(self):
        if self.get('box'):
            return self.get('box').corners()
        return None
    
    def iou(self,b):
        if self.get('box') and b.get('box'):
            return self.get('box').iou(b.get('box'))
        return None

    def bbox(self):
        return self.get('bbox')

    def center(self):
        return self.get('center')

    def width(self):
        return self.get('width')

    def height(self):
        return self.get('height')

    def confidence(self):
        return self.get('confidence')

    def classlabel(self):
        return self.get('classlabel')
    
    def __str__(self):
        return f"[Box: classlabel: {self['classlabel']}, box: {self['box']}, confidence: {self.get('confidence')}]"
        
class LinkSemantics(BoxPredictionSemantics):
    def __init__(self, *, from_id, to_id, **kwargs):
        super().__init__(from_id=int(from_id),to_id=int(to_id),**kwargs)

    def from_id(self):
        return self['from_id']

    def to_id(self):
        return self['to_id']
    
    def parent_bond(self):
        return self.get('parent_bond')
    
    def child_bond(self):
        return self.get('child_bond')

# Mono
# Required parameters: symbol: str, box: BoundingBox
# Optional parameters: id: int, confidence: float, classlabel: str
class MonoSemantics(BoxPredictionSemantics):
    def __init__(self, *, symbol, box, id=None, **kwargs):
        super().__init__(symbol=symbol,box=box,**kwargs)
        if id is not None:
            self.set('id',int(id))
        self.reset_links()

    def reset_links(self):
        self.set('links',[])

    def add_link(self, link: LinkSemantics):
        if link.from_id() != self.id():
            raise ValueError(f"Bad from_id for link: {link.from_id()}")
        self.append('links',link)

    def links(self):
        return self['links']

    def remove_link(self, to_id):
        links = list(self.links())
        retval = None
        self.reset_links()
        for l in links:
            if l.to_id() != to_id:
                self.add_link(l)
            else:
                retval = l
        if retval is None:
            raise ValueError(f"Cannot remove link with id: {to_id}")
        return retval                

    def symbol(self):
        return self['symbol']

    def id(self):
        return self.get('id')

    def has_id(self):
        return (self.get('id') is not None)

    def confidence(self):
        return self.get('confidence')

    def anomer(self):
        return self.get('anomer')

    def __str__(self):
        if self.has_id():
            res = f"[Mono:{self.id()}: classlabel: {self['classlabel']}, symbol: {self['symbol']}, box: {self['box']}, confidence: {self.get('confidence')}]"
        else:
            res = f"[Mono: classlabel: {self['classlabel']}, symbol: {self['symbol']}, box: {self['box']}, confidence: {self.get('confidence')}]"
        return res

# Root
# Required parameters: mono_id: int
# Optional parameters: box: BoundingBox, confidence: float, classlabel: str
class RootSemantics(BoxPredictionSemantics):
    def __init__(self, *, mono_id, **kwargs):
        super().__init__(mono_id=int(mono_id),**kwargs)

    def mono_id(self):
        return self['mono_id']

# UndirectedLink
# Required parameters: mono_id1: int, mono_id2: int, 
# Optional parameters: box: BoundingBox, confidence: float, classlabel: str
class UndirectedLinkSemantics(BoxPredictionSemantics):
    def __init__(self, *, mono_id1, mono_id2, **kwargs):
        super().__init__(**kwargs)
        self.set_mono_ids(mono_id1,mono_id2)

    def set_mono_ids(self,mono_id1,mono_id2):
        if mono_id1 == mono_id2:
            raise ValueError(f"Bad monosaccharide ids for undirected link: {(mono_id1,mono_id2)}")
        self.set('mono_ids',tuple(sorted(map(int, [mono_id1,mono_id2]))))

    def mono_ids(self):
        return self['mono_ids']
    
    def parent_bond(self):
        return self.get('parent_bond')
    
    def anomer(self):
        return self.get('anomer')
    
    def creates_cycle(self,ulinks):
        # try to find a path from one mono id to the other
        fromto = defaultdict(set)
        for ul in ulinks:
            ids = ul.mono_ids()
            fromto[ids[0]].add(ids[1])
            fromto[ids[1]].add(ids[0])
        selfids = self.mono_ids()
        toexplore = set([selfids[0]])
        found = set()
        while len(toexplore) > 0 and selfids[1] not in found:
            u = toexplore.pop()
            found.add(u)
            for v in fromto[u]:
                if v not in toexplore and v not in found:
                    toexplore.add(v)
        if selfids[1] in found:
            return True
        return False

# Base class for any thing (figure, glycan) which has an image with width and height
class ImageSemantics(BoxPredictionSemantics):
    def __init__(self,image_path=None,image=None,**kwargs):
        super().__init__(**kwargs)
        if image_path:
            self.set_image_path(image_path)
        elif image:
            self.set_image(image)

    def set_image(self, image):
        height, width, _ = image.shape
        self.set('image',image)
        self.set('width',width)
        self.set('height',height)

    def set_image_path(self,image_path):
        assert os.path.exists(image_path)
        image = self.read_image(image_path)
        self.set_image(image)
        abs_image_path = os.path.abspath(image_path)
        if abs_image_path.startswith(os.getcwd()):
            self.set('image_path',abs_image_path[len(os.getcwd())+1:])
        else:
            self.set('image_path',abs_image_path)
        self.set('file_name',os.path.basename(image_path))

    # Do we need to handle the numpy array stuff here?
    def read_image(self,image):
        if isinstance(image, str):
            if (image.endswith('.png') or image.endswith('.jpg') or image.endswith('.jpeg')):
                glycan_image = cv2.imread(image)
                return glycan_image
            elif image.endswith('.pdf'):
                raise ValueError("Can't handle PDF format: ",image)
                return pdf
        elif isinstance(image, np.ndarray):
            return image
        raise ValueError("Can't handle image format: ",image)

    def image(self):
        return self.get('image')

    def image_path(self):
        return self.get('image_path')
    
    def filename(self):
        return self.get('file_name')

    def width(self):
        return self.get('width')

    def height(self):
        return self.get('height')

# Class for whole figure/image containing glycans
# add pdf data - like xref, page_no, etc
# find_glycans in processjob() - can then get all these details (webApplication)
class FigureSemantics(ImageSemantics):

    JSON_SKIP_KEYS = {'glycans', 'image', 'box', 'processed_image', 'pdf_fig_box'}

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.reset_glycans()

    # semantics writer
    def tojson(self):
        data = {}
        
        for k, v in self._semantics.items():
            if k in ('image', 'processed_image', 'box', 'glycans', 'pdf_fig_box'):
                continue
            data[k] = v

        # Recursively convert glycans to dicts
        data['glycans'] = [json.loads(gly.tojson()) for gly in self.glycans()]
        
        return json.dumps(data, indent=2, sort_keys=True)
    
    @classmethod
    def read_json(cls, json_file, **kwargs):
        with open(json_file, 'r') as f:
            return cls.from_json_data(json.load(f), **kwargs)
    

    @classmethod
    def from_json_data(cls, figure_data, **kwargs):
        if not figure_data:
            return None
        
        figure_obj = cls()     # FigureSemantics instance
        
        if figure_data.get('pdf_fig_bbox') is not None:
            figure_obj.set('pdf_fig_box', PDFBoundingBox(bbox=figure_data['pdf_fig_bbox']))
            figure_obj.set('pdf_fig_bbox', figure_data['pdf_fig_bbox'])

        # Keep figure metadata so we can reconstruct the same PNG
        # (processjob: page.get_pixmap(clip=pdf_fig_bbox, dpi=...)).
        for key, value in figure_data.items():
            if key in cls.JSON_SKIP_KEYS:
                continue
            figure_obj.set(key, figure_data[key])

        for glycan in figure_data.get('glycans', []):
            figure_obj.add_glycan(GlycanSemantics.from_json_data(glycan))

        return figure_obj  # this contains pdf_fig_bbox, image_path and glycans details
    
    def reset_glycans(self):
        self.set('glycans',[])
    
    def set_glycans(self, accepted, rejected=[]):
        self.reset_glycans()
        for glycan in accepted:
            self.add_glycan(glycan)
        for glycan in rejected:
            self.append('rejected_glycans',glycan)

    def rejected_glycans(self):
        return self.get('rejected_glycans',[])

    def add_glycan(self,glycan):
        self.append('glycans',glycan)

    def glycans(self):
        return self['glycans']
    
    def glycan(self, gid):
        for glycan in self.glycans():
            if glycan.get('GID') and glycan.get('GID') == gid:
                return glycan
        return None

    def random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def annotate(self,x1,y1,x2,y2,**kwargs):
        font_scale = kwargs.get('font_scale',0.5)
        color = kwargs.get('color',(0,255,0))
        thickness = kwargs.get('thickness',1)
        text = kwargs.get('text','')
        textanchor = kwargs.get('textanchor',"TR")
        if textanchor == "TR":
            xt=kwargs.get('xt',x2)
            yt=kwargs.get('yt',y1)
        elif textanchor == "BR":
            xt=kwargs.get('xt',x2)
            yt=kwargs.get('yt',y2)
        xtoff=kwargs.get('xtoff',0)
        ytoff=kwargs.get('ytoff',0)
        xt += xtoff
        yt += ytoff

        # uncomment below to apply random colors for monos, links, root
        # # Define overlay for transparency
        # overlay = image.copy()
        # cv2.rectangle(overlay,(x1,y1),(x2,y2),color=color,thickness=thickness)
        # # Apply the overlay with transparency
        # alpha = 0.5  # Transparency factor
        # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.rectangle(self.image(),(x1,y1),(x2,y2),color=color,thickness=thickness)
        if text:
            cv2.putText(self.image(),org=(xt,yt),fontFace=cv2.FONT_HERSHEY_PLAIN,text=text,fontScale=font_scale,thickness=1,color=(0,0,0),lineType=cv2.LINE_AA)

    def make_filename(self,filename=None,outdir=None,basename=None,extension=None,overwrite=False,filename_template=None):
        if filename is None or outdir is None:
            idr,ifn = os.path.split(self.image_path())
            iba,iex = ifn.rsplit('.',1)
            if outdir is None:
                outdir = idr
            if basename is None:
                basename = iba
            if extension is None:
                extension = iex
            if filename_template is None:
                filename_template = '%(basename)s.%(extension)s'
            if filename is None:
                filename = filename_template%dict(basename=basename,extension=extension)
        if not overwrite and self.image_path() == os.path.join(outdir,filename):
            raise IOError("Will not overwrite file %s, use overwrite=True to override."%(self.image_path(),))
        return os.path.join(outdir,filename)

    def write_json(self,**kwargs):
        wh = open(self.make_filename(extension='json',**kwargs),'w')
        wh.write(self.tojson())
        wh.close()

    def annotate_glycans(self,color=(0,255,0)):
        for glycan in self.glycans():
            x1,y1,x2,y2 = glycan.box().corners()
            self.annotate(x1,y1,x2,y2,color=color) # green for glycan

    def annotate_monos(self,color=(128, 0, 128),root_color=(0, 100, 0),alternative_color=(0, 165, 255)):
        for glycan in self.glycans():
            # monosaccharides and root labelling
            root_id = None
            if glycan.has_root():
                root_id = glycan.root().mono_id()
            # print("root_id",root_id)
            for mono in glycan.monos():
                x1,y1,x2,y2 = mono.box().corners()
                text = mono.classlabel() + ":" + str(mono.id())
                # color = (128, 0, 128) # purple for monos
                # if mono['id'] == root_id:
                #     color = (0, 100, 0) # dark green for root
                # if mono.get('alternative') is not None:
                #     color = (0, 165, 255) # orange for alternatives
                self.annotate(x1,y1,x2,y2,text=text,xtoff=2,ytoff=-2,color=color,thickness=1)   
                color1 = color
                if mono.id() == root_id:
                    color1 = root_color if root_color else color # dark green for root
                # if mono.get('alternative') is not None:
                #     color1 = alternative_color if alternative_color else color # orange for alternatives
                self.annotate(x1,y1,x2,y2,text=text,xtoff=2,ytoff=-2,color=color1,thickness=1)   

    def annotate_root(self,color=(0, 100, 0),labels=False):
        for glycan in self.glycans():
            # monosaccharides and root labelling
            root = glycan.root()
            if not root:
                return
            box = root.corners()
            x_min,y_min,x_max,y_max = box.corners()

            text=''                                                                                       
            if labels:
                text = root.get('classlabel','')
            self.annotate(x_min,y_min,x_max,y_max,color=color,text=text,thickness=1)

    def annotate_links(self,color=(255, 255, 0),labels=False):
        for glycan in self.glycans():
            for link in glycan.undirected_links():
                box = link.box()
                x_min,y_min,x_max,y_max = box.corners()
                text=''
                if labels:
                    text = str(link.classlabel())
                self.annotate(x_min,y_min,x_max,y_max,color=color,text=text,thickness=1)

    def annotate_boxes(self,boxes,colors=None,color=(255, 255, 0),labels=False,label=None,anchor="TR"):
        for box in boxes:
            x_min,y_min,x_max,y_max = box.corners()
            text=''
            if label:
                text = label
            if labels:
                text = str(box.get('classlabel',''))
            if not colors:
                colors = [ color ]
            color = colors[box.get('classid',0)%len(colors)]
            self.annotate(x_min,y_min,x_max,y_max,color=color,text=text,thickness=1,textanchor=anchor)

    def write_image(self,**kwargs):
        cv2.imwrite(self.make_filename(**kwargs), self.image())

class ManuscriptSemantics(Semantics):
    '''
    Supports Multiple figures
    Should have a reader and writer - it should be able to read and write all 3 semantics
    formats - i.e JSON, TODO yolo txt data and _map.txt files.
    '''

    JSON_SUBMISSION_SKIP_KEYS = {'result'}

    JSON_RESULT_SKIP_KEYS = {'figures'}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result_keys = set()   # notes all the keys present inside the 'result' i.e result['figures']
        self.reset_figures()    

    def reset_figures(self):
        self.set('figures', [])

    def add_figure(self, figure: FigureSemantics):
        self.append('figures', figure)

    def figures(self):
        return self.get('figures', [])

    def tojson(self):
        data = {}
        for key, value in self.items():
            if key == 'figures' or key in self._result_keys:
                continue
            data[key] = value

        result = {key: self.get(key) for key in self._result_keys if self.has(key)}
        result['figures'] = [json.loads(fig.tojson()) for fig in self.figures()]

        data['result'] = result

        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def read_json(cls, json_file):
        with open(json_file, 'r') as f:
            return cls.from_json_data(json.load(f))
    
    @classmethod
    def from_json_data(cls, json_data):
        if not json_data or json_data['state'] == 'Error':
            return None

        manuscript_obj = cls()

        # copy all keys from json, except 'result'
        for key, value in json_data.items():
            if key in cls.JSON_SUBMISSION_SKIP_KEYS:
                continue
            manuscript_obj.set(key, value)

        # result.* --> copy everything except 'figures'
        result = json_data.get('result', {})
        for key, value in result.items():
            if key in cls.JSON_RESULT_SKIP_KEYS:
                continue
            if manuscript_obj.has(key):
                continue
            manuscript_obj.set(key, value)
            manuscript_obj._result_keys.add(key)

        for figure in result.get('figures', []):
            figure_obj = FigureSemantics.from_json_data(figure)

            if figure_obj is not None:
                manuscript_obj.add_figure(figure_obj)

        return manuscript_obj 


class GlycanSemantics(ImageSemantics):
    
    JSON_GLYCAN_SKIP_KEYS = {'image_path', 'bbox', 'box', 'image', 'monos', 'root',
            'undirected_links', 'rejected_monos', 'rejected_roots', 'rejected_undirected_links'}
    
    JSON_MONO_SKIP_KEYS = {'bbox', 'box', 'image', 'links'}

    JSON_ROOT_SKIP_KEYS = {'bbox', 'box', 'image'}

    JSON_ULINK_SKIP_KEYS = {'bbox', 'box', 'image', 'mono_ids'}

    
    def __init__(self,*,figure=None,box,**kwargs):
        super().__init__(box=box,**kwargs)  
        if figure is not None:      # its possible to have glycan information without the glycan image stored locally (CV version) for annotated pdf's
            self.set_image(box.crop(figure))
        self.reset_monos()
        self.reset_root()
        self.reset_undirected_links()

    def reset_monos(self):
        self.set('monos',{})
        self.unset('rejected_monos')

    def monos(self):
        return list(self['monos'].values())

    def monoids(self):
        return list(self['monos'].keys())

    def has_mono(self,id):
        return (id in self['monos'])

    def mono(self,id):
        return self['monos'][id]

    def add_mono(self, mono: MonoSemantics):
        # if mono has an id, it cannot already be in the dictionary
        # if it doesn't have an id, we must give it a unique one
        if mono.get('id') is not None:
            if mono.get('id') in self['monos']:
                raise ValueError("Repeated monosaccharide id %s"%(mono.get('id'),))
        else:
            if len(self['monos']) == 0:
                mono.set('id',1)
            else:
                mono.set('id',max(self['monos'])+1)
        self['monos'][mono.get('id')] = mono

    def remove_mono(self,id):
        m = self.mono(id)
        del self['monos'][id]
        return m

    def set_monos(self, accepted, rejected=[]):
        self.reset_monos()
        for m in accepted:
            self.add_mono(m)
        for m in rejected:
            self.append('rejected_monos',m)

    def rejected_monos(self):
        return self.get('rejected_monos',[])

    def mono_boxes(self,include_rejected=False):
        for m in self.monos():
            yield m.box()
        if include_rejected:
            for m in self.rejected_monos():
                yield m.box()
    
    def mono_count(self):
        return len(self.monoids())

    def reset_root(self):
        self.set('root',None)
        self.unset('rejected_roots')

    def root(self):
        return self.get('root')

    def has_root(self):
        return self.get('root') is not None

    def set_root(self,root):
        if not self.has_mono(root.mono_id()):
            raise ValueError(f"Invalid monosaccharide id for root: {root.get('mono_id')}")
        self.set('root',root)

    def set_roots(self,accepted,rejected=[]):
        self.reset_root()
        if accepted is not None:
            self.set_root(accepted)
        for r in rejected:
            self.append('rejected_roots',r)

    def reset_undirected_links(self):
        self.set('undirected_links',[])

    def undirected_links(self):
        return self['undirected_links']

    def add_undirected_link(self,link: UndirectedLinkSemantics):
        mono_ids = link.mono_ids()
        if not self.has_mono(mono_ids[0]) or not self.has_mono(mono_ids[1]):
            raise ValueError(f"Invalid monosaccharide ids for link: {link.get('mono_ids')}")
        self.append('undirected_links',link)

    def set_undirected_links(self, accepted, rejected=[]):
        self.reset_undirected_links()
        for link in accepted:
            self.add_undirected_link(link)
        for link in rejected:
            self.append('rejected_undirected_links',link)

    def add_glycan_error(self,error_msg):
        self.append('glycan_errors',error_msg)
        self.log(error_msg)

    def glycan_errors(self):
        return self.get('glycan_errors',[])

    def has_glycan_errors(self):
        return bool(len(self.glycan_errors()) > 0)

    def get_logs(self):
        return self.get('log',[])

    def build_adjacency_list(self):
        adj = defaultdict(list)

        for link in self.undirected_links():
            id1, id2 = link.mono_ids()
            # print("--->>",link.items())
            link_without_ids = {k: v for k, v in link.items() if k != "mono_ids"}
            adj[id1].append((id2, link_without_ids))
            adj[id2].append((id1, link_without_ids))

        return adj
                    
    # to log links which either create a cycle or are extra w.r.t number of monos
    def link_cycle(self,data):
        self.append('non_tree_links',data)

    def reset_all_links(self):
        for mono_id in self.monoids():
            self.mono(mono_id).reset_links()

    def all_links(self):
        return [ link for m in self.monos() for link in m.links() ]

    def mono_links(self,id):
        return self.mono(id).links()

    def remove_link(self,from_id,to_id):
        mono=self.mono(from_id)
        return mono.remove_link(to_id)
       
    def add_link(self,from_id,to_id,**kwargs):
        mono = self.mono(from_id)
        link = LinkSemantics(from_id=from_id,to_id=to_id,**kwargs)
        tomono = self.mono(to_id)
        if link.get('anomer'): #copied over in kwargs
            tomono.set('anomer',link.get('anomer'))
            link.unset('anomer')
        if tomono.symbol() in ("NeuAc","NeuGc"):
            link.set('child_bond',2)
        else:
            link.set('child_bond',1)
        mono.add_link(link)

    def create_links(self):
        adj = self.build_adjacency_list()
        root_id = self.root().mono_id()

        visited = set()
        queue = deque([root_id])

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            for neighbor, link_info in adj[node]:
                if neighbor not in visited:
                    self.add_link(node, neighbor, **link_info) 
                    queue.append(neighbor)
               
    # semantics writer
    def tojson(self):
        data = self.remove_binary_values(copy.deepcopy(self._semantics))
        data['monos'] = sorted(data['monos'].values(),key=lambda m: m['id'])  
        return json.dumps(data, sort_keys=True)
    
    # semantics reader
    @classmethod
    def from_json_data(cls, glycan):
        glycan_box = BoundingBox(bbox=glycan['bbox'])
        
        glycan_kwargs = {k: v for k, v in glycan.items() if k not in cls.JSON_GLYCAN_SKIP_KEYS}
        glycan_obj = cls(box=glycan_box, **glycan_kwargs)

        if glycan.get('image_path'):
            glycan_obj.set('image_path', glycan['image_path'])

        # Mono Semantics (includes Link Semantics as well)
        accepted_monos = [cls._mono_fromjson(m) 
                          for m in glycan.get('monos', [])
                        ]
                
        # Rejected Mono Semantics (doesnt include Link Semantics)
        rejected_monos = [cls._mono_fromjson(m, include_links=False)
                          for m in glycan.get('rejected_monos', [])
                        ]

        glycan_obj.set_monos(accepted_monos, rejected_monos)

        # Root Semantics
        accepted_root = cls._root_fromjson(glycan['root']) if glycan.get('root') else None
        rejected_roots = [cls._root_fromjson(r) for r in glycan.get('rejected_roots', [])]
        glycan_obj.set_roots(accepted_root, rejected_roots)

        # TODO - add undirected links?
        # # undirected links
        # accepted_ulinks = [
        #     cls._undirected_link_fromjson(u)
        #     for u in glycan.get('undirected_links', [])
        # ]
        # rejected_ulinks = [
        #     cls._undirected_link_fromjson(u)
        #     for u in glycan.get('rejected_undirected_links', [])
        # ]
        # glycan_obj.set_undirected_links(accepted_ulinks, rejected_ulinks)

        return glycan_obj

    @staticmethod
    def _mono_fromjson(mono, *, include_links=True):
        mono_box = BoundingBox(bbox=mono['bbox'])

        # rebuild these, everything else comes from JSON
        kwargs = {k: v for k, v in mono.items() if k not in GlycanSemantics.JSON_MONO_SKIP_KEYS}
        symbol = kwargs.pop('symbol')
        mono_id = kwargs.pop('id', None)
        
        mono_obj = MonoSemantics(symbol=symbol, box=mono_box, id=mono_id, **kwargs)
        
        if include_links:
            for link in mono.get('links', []):
                link_skip = {'bbox', 'box', 'image'}
                link_kwargs = {k: v for k, v in link.items() if k not in link_skip}
                from_id = link_kwargs.pop('from_id')
                to_id = link_kwargs.pop('to_id')

                if link.get('bbox'):
                    link_kwargs['box'] = BoundingBox(bbox=link['bbox'])

                mono_obj.add_link(
                    link=LinkSemantics(from_id=from_id, to_id=to_id, **link_kwargs)
                )

        return mono_obj
    
    @staticmethod
    def _root_fromjson(root):
        kwargs = {k: v for k, v in root.items() if k not in GlycanSemantics.JSON_ROOT_SKIP_KEYS}
        mono_id = kwargs.pop('mono_id')
        if root.get('bbox'):
            kwargs['box'] = BoundingBox(bbox=root['bbox'])
        return RootSemantics(mono_id=mono_id, **kwargs)

    @staticmethod
    def _undirected_link_fromjson(ulink):
        kwargs = {
            k: v for k, v in ulink.items()
            if k not in GlycanSemantics.JSON_ULINK_SKIP_KEYS
        }
        mono_id1, mono_id2 = ulink['mono_ids']
        if ulink.get('bbox'):
            kwargs['box'] = BoundingBox(bbox=ulink['bbox'])
        return UndirectedLinkSemantics(mono_id1=mono_id1, mono_id2=mono_id2, **kwargs)

    def remove_binary_values(self, data):
        def is_serializable(val):
            try:
                json.dumps(val)
                return True
            except (TypeError, ValueError):
                return False

        if isinstance(data, (MonoSemantics, RootSemantics, LinkSemantics, UndirectedLinkSemantics)):
            return self.remove_binary_values(data._semantics)

        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                cleaned = self.remove_binary_values(v)
                if is_serializable(cleaned):
                    result[k] = cleaned
            return result

        elif isinstance(data, list):
            return [self.remove_binary_values(v) for v in data]

        else:
            return data  # return as-is, will be checked for serializability in parent

    def image_path(self):
        # for single glycan images, the glycan image "has" a path
        return self.get('image_path',None)
            
    def semantic_compare(self,**kwargs):
        return GlycanCompare(**kwargs)

    def composition(self):
        count = defaultdict(int)
        for m in self.monos():
            count[m.get('symbol')] += 1
        return count
    
    def compstr(self):
        # labels here must be a superset of all supported symbols!!!!!
        labels = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc","Xyl"]
        mono_syms = labels

        comp = self.composition()
        retval = ""
        for sym in mono_syms:
            if comp[sym] > 0:
                retval += sym + "(" + str(comp[sym]) + ")"
        return retval
        
    def IUPAC(self):
        root = self.root()
        if not root:
            return None

        # adds directed links in semantics
        self.create_links()

        root_id = root.mono_id()
        iupac = []
        adj = self.build_adjacency_list()
        visited = set()

        self.generate_iupac(iupac, adj, visited, -1, root_id)
       
        iupac = iupac[::-1] # IUPAC sequences are read in reverse order
        return ''.join(iupac)

    def find_link_info(self, parent_id, child_id): # Helper function for IUPAC generation with YOLO linkages
        for link in self.mono_links(parent_id):
            if link.to_id() == child_id:
                return link
        return None
    
    def generate_iupac(self,iupac, adj, visited, parent, u):
        visited.add(u)

        # Get the current node's data
        symbol = self.mono(u).symbol()
        #extension = '?1-?' if symbol not in ['NeuAc', 'NeuGc'] else '?2-?' #Old line before YOLO linkage detection
        
        # determine the extension based on the link class label 
        if parent != -1: 
            link_info = self.find_link_info(parent, u)
            if link_info:
                a = self.mono(u).get('anomer','?')
                pb = link_info.get('parent_bond','?')
                cb = link_info.get('child_bond','?')
                extension = f'{a}{cb}-{pb}'
            else:
                # Fallback to generic linkages
                extension = '?1-?' if symbol not in ['NeuAc', 'NeuGc'] else '?2-?'
        
        data = symbol + extension if parent != -1 else symbol

        # Append the current node's data to the result
        iupac.append(data)

        # Filter adjacent nodes to only include unvisited ones
        filtered_adj = [v[0] for v in adj[u] if v[0] not in visited]

        # Use monosaccharide positions, if possible, for branch order
        uxy = self.mono(u).center()
        scale = (self.mono(u).width()+self.mono(u).height())/2 #average of width + height
        approx = round(0.2*scale) #pixel to tolerance for "equal"
        adjxy = [ self.mono(v).center() for v in filtered_adj ]
        
        # print(self.mono(u).get('symbol'),[self.mono(v).get('symbol') for v in filtered_adj])

        # figure out if they are all on one side of u
        dircnt = defaultdict(int)
        for vxy in adjxy:
            if (vxy[0] - uxy[0]) > approx:
                dircnt['right'] += 1
            elif (uxy[0] - vxy[0]) > approx:
                dircnt['left'] += 1
            if (vxy[1] - uxy[1]) > approx:
                dircnt['down'] += 1
            elif (uxy[1] - vxy[1]) > approx:
                dircnt['up'] += 1
      
        xyorder = [0]*len(adjxy)
        if len(adjxy) > 1 and max(dircnt.values()) == len(adjxy):
            # all are on one side
            dirn = max(dircnt.items(),key=lambda t: t[1])[0]
            if dirn in ("up","down"):
                cy = sum(vxy[1] for vxy in adjxy)/len(adjxy)
                maxdel = max(abs(vxy[1]-cy) for vxy in adjxy)
                # check they are all in a "line"
                if maxdel <= approx:
                    if dirn == "up":
                        xyorder = [ vxy[0] for vxy in adjxy ]
                    if dirn == "down":
                        xyorder = [ -vxy[0] for vxy in adjxy ]
            else: # left, right
                cx = sum(vxy[0] for vxy in adjxy)/len(adjxy)
                maxdel = max(abs(vxy[0]-cx) for vxy in adjxy)
                # check they are all in a "line"
                if maxdel <= approx:
                    if dirn == "left":
                        xyorder = [ -vxy[1] for vxy in adjxy ]
                    if dirn == "right":
                        xyorder = [ vxy[1] for vxy in adjxy ]

        branch_strings = []
        for i,v in enumerate(filtered_adj):
            branch_iupac = []
            self.generate_iupac(branch_iupac, adj, visited, u, v)  # Recurse for each child
            
            # Convert the branch into a single string
            branch_str = ''.join(branch_iupac[::-1])  # Reverse the list and join it into a string
            branch_strings.append((i,branch_str))

        # Sort branches lexicographically after recursion
        branch_strings.sort(key=lambda bs: (xyorder[bs[0]],bs[1][-1],bs[1]))

        if len(branch_strings) > 0:
            for idx, branch in branch_strings[:-1]:
                iupac.append('(' + branch + ')')
            iupac.append(branch_strings[-1][1])

        return iupac


    def glycan_orientation(self):
        root = self.root()

        if not root:
            return "RL"

        root_id = root.get('mono_id')

        # determine the position of the next connected element from the root
        # do not take fucose into account
        # depending on which side the the next element is - that will be the orientation

        root_mono = self.mono(root_id)
        root_box = root_mono.get('box')
        root_links = self.mono_links(root_id)

        if not root_links:
            return "BT"

        for link in root_links:
            linked_mono = self.mono(link.to_id())
            sym = linked_mono.symbol()

            if sym == 'Fuc' and len(root_links) > 1:
                continue

            linked_mono_box = linked_mono.get('box')

            r_x, r_y = root_box.center()
            l_x, l_y = linked_mono_box.center()

            dx = l_x - r_x  # Difference in X
            dy = l_y - r_y  # Difference in Y

            if abs(dx) > abs(dy):  # If movement in X is more dominant
                if dx > 0:
                    # print("orientation","LR")
                    return "LR"  # Moving right
                else:
                    # print("orientation","RL")
                    return "RL"  # Moving left
            else:  # If movement in Y is more dominant
                if dy > 0:
                    # print("orientation","TB")
                    return "TB"  # Moving downward
                else:
                    # print("orientation","BT")
                    return "BT"  # Moving upward
        
        # catch all default to avoid error...
        return "BT"


if __name__ == "__main__":

    # simple testing code...

    import sys
    from bbox import BoundingBox

    figure = FigureSemantics(sys.argv[1])
    box = BoundingBox(x=20,y=20,w=figure.width()-40,h=figure.height()-40)
    glycan = GlycanSemantics(figure=figure,box=box)
    
    pos = (5,3); m1box = BoundingBox(x=pos[0]*20,y=pos[1]*20,w=20,h=20)
    pos = (4,3); m2box = BoundingBox(x=pos[0]*20,y=pos[1]*20,w=20,h=20)
    pos = (3,3); m3box = BoundingBox(x=pos[0]*20,y=pos[1]*20,w=20,h=20)
    pos = (2,2); m4box = BoundingBox(x=pos[0]*20,y=pos[1]*20,w=20,h=20)
    pos = (2,4); m5box = BoundingBox(x=pos[0]*20,y=pos[1]*20,w=20,h=20)

    m1 = MonoSemantics(symbol="GlcNAc",classlabel="GlcNAc",confidence=0.99,box=m1box)
    m2 = MonoSemantics(symbol="GlcNAc",classlabel="GlcNAc",confidence=0.99,box=m2box)
    m3 = MonoSemantics(symbol="Man",classlabel="Man",confidence=0.99,box=m3box)
    m4 = MonoSemantics(symbol="Man",classlabel="Man",confidence=0.99,box=m4box)
    m5 = MonoSemantics(symbol="Man",classlabel="Man",confidence=0.99,box=m5box)

    glycan.add_mono(m1)
    glycan.add_mono(m2)
    glycan.add_mono(m3)
    glycan.add_mono(m4)
    glycan.add_mono(m5)

    # ids will be from order added...
    l1 = UndirectedLinkSemantics(mono_id1=1,mono_id2=2,parent_bond=4,anomer='b',classlabel="link",confidence=0.99)
    l2 = UndirectedLinkSemantics(mono_id1=2,mono_id2=3,parent_bond=4,anomer='b',classlabel="link",confidence=0.99)
    l3 = UndirectedLinkSemantics(mono_id1=3,mono_id2=4,parent_bond=3,anomer='a',classlabel="link",confidence=0.99)
    l4 = UndirectedLinkSemantics(mono_id1=3,mono_id2=5,parent_bond=6,anomer='a',classlabel="link",confidence=0.99)

    glycan.add_undirected_link(l1)
    glycan.add_undirected_link(l2)
    glycan.add_undirected_link(l3)
    glycan.add_undirected_link(l4)

    r = RootSemantics(mono_id=1,classlabel="redend",confidence=0.99)
    
    glycan.set_root(r)
    glycan.create_links()

    figure.add_glycan(glycan)

    print(figure.tojson())

    print(glycan.IUPAC())

