import cv2
import os
import numpy as np
import json
import copy
import random
import math
from collections import defaultdict, deque, Counter
from . lineno import callsig
import importlib
import shutil


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

    def set_image(self, image):
        self.semantics['image'] = image

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

    def log(self,message):
        if 'log' not in self.semantics:
            self.semantics['log'] = []
        self.semantics['log'].append("[%s] %s"%(callsig(1),message))

# Class for whole figure/image containing glycans
class Figure_Semantics(Image_Semantics):
    def __init__(self,image_path,**kwargs):
        image = self.format_image(image_path)
        super().__init__(image)
        self.semantics['image_path'] = os.path.abspath(image_path)
        if self.semantics['image_path'].startswith(os.getcwd()):
            self.semantics['image_path'] = self.semantics['image_path'][len(os.getcwd())+1:]
        self.semantics['file_name'] = os.path.basename(image_path) 
        self.semantics['glycans'] = []
        self.semantics.update(copy.deepcopy(kwargs))

    def tojson(self):
        data = {}
        for k,v in self.semantics.items():
            if k not in ('image','processed_image','box','glycans'):
                data[k] = v
        data['glycans'] = [ json.loads(gly.tojson()) for gly in self.glycans() ]
        return json.dumps(data,sort_keys=True)

    def format_image(self,image):
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

    def random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))

    def annotate(self,x1,y1,x2,y2,**kwargs):
        font_scale = kwargs.get('font_scale',0.5)
        color = kwargs.get('color',(0,255,0))
        thickness = kwargs.get('thickness',1)
        text = kwargs.get('text','')
        xt=kwargs.get('xt',x2)
        yt=kwargs.get('yt',y1)
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
            raise IOError("Will not overwrite image file %s, use overwrite=True to override."%(self.image_path(),))
        return os.path.join(outdir,filename)

    def write_json(self,**kwargs):
        wh = open(self.make_filename(extension='json',**kwargs),'w')
        wh.write(self.tojson())
        wh.close()

    def annotate_glycans(self,color=(0,255,0)):
        for glycan in self.semantics['glycans']:
            # glycan annotation
            x1,y1,x2,y2 = glycan.glycan_box().corners()
            self.annotate(x1,y1,x2,y2,color=color) # green for glycan

    def annotate_monos(self,color=(128, 0, 128),root_color=(0, 100, 0),alternative_color=(0, 165, 255)):
        for glycan in self.semantics['glycans']:
            # monosaccharides and root labelling
            root_id = None
            if glycan.root():
                root_id = glycan.root()['mono_id']
            # print("root_id",root_id)
            for mono in glycan.monosaccharides():
                x1,y1,x2,y2 = mono['box'].corners()
                text = mono.get('classlabel','') + ":" + str(mono.get('id'))
                # color = (128, 0, 128) # purple for monos
                # if mono['id'] == root_id:
                #     color = (0, 100, 0) # dark green for root
                # if mono.get('alternative') is not None:
                #     color = (0, 165, 255) # orange for alternatives
                self.annotate(x1,y1,x2,y2,text=text,xtoff=2,ytoff=-2,color=color,thickness=1)   
                color1 = color
                if mono['id'] == root_id:
                    color1 = root_color if root_color else color # dark green for root
                if mono.get('alternative') is not None:
                    color1 = alternative_color if alternative_color else color # orange for alternatives
                self.annotate(x1,y1,x2,y2,text=text,xtoff=2,ytoff=-2,color=color1,thickness=1)   

    def annotate_links(self,color=(255, 255, 0),labels=False):
        for glycan in self.semantics['glycans']:
            # monosaccharides and root labelling
            for link in glycan.undirected_links():
                box = link.get('box')
                x_min,y_min,x_max,y_max = box.corners()

                text=''
                if labels:
                    text = link.get('classlabel','')
                self.annotate(x_min,y_min,x_max,y_max,color=color,text=text,thickness=1)

    def write_image(self,**kwargs):
        cv2.imwrite(self.make_filename(**kwargs), self.image())

class Glycan_Semantics(Image_Semantics):

    def __init__(self,image,box,**kwargs):
        super().__init__(image)  
        self.semantics['box'] = box
        self.semantics['bbox'] = box.bbox()
        self.semantics['monos'] = {}
        self.semantics['non_tree_links'] = []
        self.semantics['glycan_errors'] = []
        self.semantics['undirected_links'] = []
        self.semantics.update(kwargs)

    def glycan_box(self):
        return self.semantics['box']

    def glycan(self):
        return [self.semantics]

    # to log links which either create a cycle or are extra w.r.t number of monos
    def link_cycle(self,data):
        self.semantics['non_tree_links'].append(data)

    def glycan_error(self,error_msg):
        self.semantics['glycan_errors'].append(error_msg)

    def get_glycan_errors(self):
        return self.semantics['glycan_errors']
        
    def image(self):
        return self.semantics['image']

    def clear_monos(self):
        self.semantics['monos'] = {}

    def clear_undirected_links(self):
        self.semantics['undirected_links'] = []

    def add_mono(self,classlabel,symbol,box,**kwargs):
        if kwargs.get('id') is None:
            if len(self.monosaccharides()) == 0:
                kwargs['id'] = 1
            else:
                kwargs['id'] = max(self.semantics['monos'])+1
        mono = dict(classlabel=classlabel,symbol=symbol,box=box,bbox=box.bbox(),center=box.center(),**kwargs)
        assert mono['id'] not in self.semantics['monos']
        self.semantics['monos'][mono['id']] = mono

    def monosaccharides(self):
        return list(self.semantics['monos'].values())

    def monosaccharideids(self):
        return list(self.semantics['monos'].keys())

    def monosaccharide(self,id):
        assert id is not None
        return self.semantics['monos'][id]

    def delete_mono(self,id):
        assert id is not None
        m = self.semantics['monos'][id]
        del self.semantics['monos'][id]
        return m

    def add_alternative_mono(self,id,altm):
        m = self.monosaccharide(id)
        if 'alternative' not in m:
            m['alternative'] = []
        m['alternative'].append(altm)

    def make_alternative_mono(self,id,altid):
        alt = self.delete_mono(altid)
        self.add_alternative_mono(id,alt)

    def mono_boxes(self):
        boxes = []
        for id,mono in self.semantics['monos'].items():
            boxes.append(mono['box'])
            for alt in m.get('alternative',[]):
                boxes.append(alt['box'])
        return boxes

    def set_root(self,root_id,**kwargs):
        self.semantics['root'] = { 'mono_id': root_id, **kwargs}

    def no_root(self):
        if 'root' in self.semantics:
            del self.semantics['root']

    def root(self):
        return self.semantics.get('root',None)

    def add_alternative_root(self,altr):
        r = self.root()
        if 'alternative' not in r:
            r['alternative'] = []
        r['alternative'].append(altr)

    def build_adjacency_list(self):
        adj = defaultdict(list)

        for link in self.undirected_links():
            id1, id2 = link["mono_ids"]
            link_without_ids = {k: v for k, v in link.items() if k != "mono_ids"}
            adj[id1].append((id2, link_without_ids))
            adj[id2].append((id1, link_without_ids))

        return adj
                    
    def clear_links(self,mono_id):
        self.semantics['monos'][mono_id]['links'] = []

    def clear_all_links(self):
        for mono_id in self.monosaccharideids():
            self.clear_links(mono_id)

    def all_links(self):
        # a mono could have multiple links
        return [link for item in self.monosaccharides() if item.get('links') for link in item['links']]  

    def links(self,id):
        return self.semantics['monos'][id].get('links')

    def delete_link(self,fromid,toid):
        new_links = [
            link for link in self.semantics['monos'][fromid]['links']
            if ([link['fromid'], link['toid']] != [fromid, toid])
        ]
        self.semantics['monos'][fromid]['links'] = new_links  

    def undirected_links(self):
        return self.semantics['undirected_links']

    def set_undirected_links(self,links):
        self.semantics['undirected_links'] = links

    def add_undirected_link(self,id1,id2,**kwargs):
        self.semantics['undirected_links'].append({"mono_ids": list(sorted((id1,id2))),**kwargs})

    def add_link(self,id1,id2,**kwargs):
        mono = self.monosaccharide(id1)

        if 'links' not in mono:
            mono['links'] = []

        mono['links'].append({"fromid":id1,"toid":id2, **kwargs})


    def create_links(self):
        adj = self.build_adjacency_list()
        root_id = self.root().get("mono_id")

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
               

    def tojson(self):
        data = self.remove_binary_values(copy.deepcopy(self.semantics))
        data['monos'] = sorted(data['monos'].values(),key=lambda m: m['id'])        
        return json.dumps(data, sort_keys=True)

    def remove_binary_values(self,d):
        if isinstance(d,dict):
            for k,v in list(d.items()):
                if not isinstance(v,list) and not isinstance(v,dict):
                    try:
                        json.dumps(v)
                    except (TypeError,ValueError):
                        del d[k]
                else:
                    v = self.remove_binary_values(v)
        elif isinstance(d,list):
            for v in d:
                v = self.remove_binary_values(v)
        return d

    def image_path(self):
        # for single glycan images, the glycan image "has" a path
        return self.semantics.get('image_path',None)

    def composition(self):
        count = defaultdict(int)
        for m in self.monosaccharides():
            count[m.get('classlabel')] += 1
        return count
    
    def compstr(self):
        # circular import issue for monosaccharideid
        #  need the same labels from the Monosaccideid class - so that any updates to the class varibale labels
        # will also persist here 
        # Mono = importlib.import_module('.monosaccharideid', package='BKGlycanExtractor')
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

        root_id = root.get('mono_id')
        iupac = []
        adj = self.build_adjacency_list()
        visited = set()

        self.generate_iupac(iupac, adj, visited, -1, root_id)
       
        iupac = iupac[::-1] # IUPAC sequences are read in reverse order
        return ''.join(iupac)

    def find_link_info(self, parent_id, child_id): # Helper function for IUPAC generation with YOLO linkages
        for link in self.undirected_links():
            if set(link['mono_ids']) == {parent_id, child_id}:
                return link
        return None
    
    def generate_iupac(self,iupac, adj, visited, parent, u):
        visited.add(u)

        # Get the current node's data
        symbol = self.monosaccharide(u).get('symbol')
        
        # determine the extension based on predicted class label
        if parent != -1: 
            link_info = self.find_link_info(parent, u)
            a = link_info.get('anomer', '?') if link_info.get('anomer') != 'x' else '?'
            c = link_info.get('parent_carbon_bond', '?') if link_info.get('parent_carbon_bond') != 'x' else '?'
            extension = f'{a}1-{c}' if symbol not in ['NeuAc', 'NeuGc'] else f'{a}2-{c}'
        
        data = symbol + extension if parent != -1 else symbol

        # Append the current node's data to the result
        iupac.append(data)

        # Filter adjacent nodes to only include unvisited ones
        filtered_adj = [v[0] for v in adj[u] if v[0] not in visited]

        # Sort the children (branches) lexicographically by their symbol for consistency
        filtered_adj = sorted(filtered_adj, key=lambda x: self.monosaccharide(x).get('symbol'))

        branch_strings = []
        for v in filtered_adj:
            branch_iupac = []
            self.generate_iupac(branch_iupac, adj, visited, u, v)  # Recurse for each child
            
            # Convert the branch into a single string
            branch_str = ''.join(branch_iupac[::-1])  # Reverse the list and join it into a string
            branch_strings.append(branch_str)

        # Sort branches lexicographically after recursion
        branch_strings.sort()

        # Handle parentheses for branches based on the rule
        for idx, branch in enumerate(branch_strings):
            if idx < len(branch_strings) - 1:  # For all branches except the last
                iupac.append('(' + branch + ')')
            else:  # For the last branch
                iupac.append(branch)

        return iupac


    def glycan_orientation(self):
        root = self.root()

        if not root:
            return "RL"

        root_id = root.get('mono_id')

        # determine the position of the next connected element from the root
        # do not take fucose into account
        # depending on which side the the next element is - that will be the orientation

        root_mono = self.monosaccharide(root_id)
        root_box = root_mono.get('box')
        root_links = self.links(root_id)

        if not root_links:
            return "BT"

        for link in root_links:
            fromid, toid = link['fromid'], link['toid']
            linked_mono = self.monosaccharide(toid)
            sym = linked_mono.get('symbol')

            if sym == 'Fuc':
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







        
