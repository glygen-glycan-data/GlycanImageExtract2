import cv2
import os
import numpy as np
import json
import copy
import random
import math
from collections import defaultdict, deque

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

    def annotate_glycans(self):
        for glycan in self.semantics['glycans']:
            # glycan annotation
            x1,y1,x2,y2 = glycan.glycan_box().corners()
            self.annotate(x1,y1,x2,y2,color=(0,255,0)) # green for glycan

    def annotate_monos(self):
        for glycan in self.semantics['glycans']:
            # monosaccharides and root labelling
            root_id = None
            if glycan.root():
                root_id = glycan.root()['mono_id']
            for mono in glycan.monosaccharides():
                x1,y1,x2,y2 = mono['box'].corners()
                text = mono.get('classlabel','') + ":" + str(mono.get('id'))
                color = (128, 0, 128) # purple for monos
                if mono['id'] == root_id:
                    color = (0,0,255) # red for root  
                if mono.get('alternative') is not None:
                    color = (0, 165, 255) # orange for alternatives
                self.annotate(x1,y1,x2,y2,text=text,xtoff=2,ytoff=-2,color=color,thickness=1)   

    def annotate_links(self):
        for glycan in self.semantics['glycans']:
            # monosaccharides and root labelling
            for mono in glycan.monosaccharides():
                for link in mono['links']:
                    toid = link['to']
                    linked_mono = glycan.monosaccharide(toid)
                    _x1,_y1,_x2,_y2 = linked_mono['box'].corners()

                    x_coords = [x1,x2,_x1,_x2]
                    y_coords = [y1,y2,_y1,_y2]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords) 

                    self.annotate(x_min,y_min,x_max,y_max,color=(255, 255, 0),thickness=1)

    def write_image(self,**kwargs):
        cv2.imwrite(self.make_filename(**kwargs), self.image())

class Glycan_Semantics(Image_Semantics):
    def __init__(self,image,box,**kwargs):
        super().__init__(image)  
        self.semantics['box'] = box
        self.semantics['bbox'] = box.bbox()
        self.semantics['monos'] = {}
        self.semantics.update(kwargs)

    def glycan_box(self):
        return self.semantics['box']

    def clear_monos(self):
        self.semantics['monos'] = {}

    def add_mono(self,classid,symbol,box,**kwargs):
        if kwargs.get('id') is None:
            if len(self.monosaccharides()) == 0:
                kwargs['id'] = 1
            else:
                kwargs['id'] = max(self.semantics['monos'])+1
        mono = dict(classid=classid,symbol=symbol,box=box,bbox=box.bbox(),center=box.center(),links=[],**kwargs)
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

    def add_link(self,fromid,toid,**kwargs):
        assert fromid is not None
        assert toid is not None
        assert fromid in self.semantics['monos']
        assert toid in self.semantics['monos']
        self.semantics['monos'][fromid]['links'].append({ 'from': fromid, 'to': toid, **kwargs })

    def clear_links(self,fromid):
        self.semantics['monos'][fromid]['links'] = []

    def clear_all_links(self):
        for fromid in self.monosaccharideids():
            self.clear_links(fromid)

    def links(self,id):
        return self.semantics['monos'][id]['links']

    def all_links(self):
        result = []
        for fromid in self.monosaccharideids():
            for l in self.links(fromid):
                result.append(l)
        return result

    # def tojson(self):
    #     data = {}
    #     for k,v in self.semantics.items():
    #         if k not in ('image','box','monos'):
    #             data[k] = v
    #     data['monos'] = []
    #     for mono in self.semantics['monos'].values():
    #         monodict = {}
    #         for k,v in mono.items():
    #             if k not in ('image','box','links'):
    #                 monodict[k] = v
    #             elif k == 'links' and len(v) > 0:
    #                 monodict[k] = [item if isinstance(item,int) else item[0] for item in v]
    #         data['monos'].append(monodict)

    #     return json.dumps(data,indent=2,sort_keys=True)

    def glycan_orientation(self):
        # print("Orientation",self.semantics['root'])

        root_id = self.semantics['root']

        if root_id == -1:
            return "RL"

        # determine the position of the next connected element from the root
        # do not take fucose into account
        # depending on which side the the next element is - that will be the orientation

        try:
            root_mono = self.semantics['monos'][root_id]
            root_box = root_mono.get('box')

            root_link_ids = root_mono['links']

            for link_id, confidence in root_link_ids:
                linked_mono = self.semantics['monos'][link_id]

                sym = linked_mono.get("symbol")

                if sym == 'Fuc':
                    continue

                linked_mono_box = linked_mono.get('box')


            r_x, r_y = root_box.center()
            l_x, l_y = linked_mono_box.center()


            dx = l_x - r_x  # Difference in X
            dy = l_y - r_y  # Difference in Y

            if abs(dx) > abs(dy):  # If movement in X is more dominant
                if dx > 0:
                    return "LR"  # Moving right
                else:
                    return "RL"  # Moving left
            else:  # If movement in Y is more dominant
                if dy > 0:
                    return "TB"  # Moving downward
                else:
                    return "BT"  # Moving upward
        except:
            return "RL"


    def tojson(self):
        data = self.remove_binary_values(copy.deepcopy(self.semantics))
        data['monos'] = sorted(data['monos'].values(),key=lambda m: m['id'])        
        return json.dumps(data, indent=2, sort_keys=True)

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
    
    @staticmethod
    def composition(monosaccharides):
        count = defaultdict(int)
        for m in monosaccharides:
            sym = Glycan_Semantics.mono_syms[m['classid']]
            if sym not in count:
                count[sym] = 1
            else:
                count[sym] += 1
        return count

    mono_syms = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]
    
    # remove
    @staticmethod
    def all_monos_reachable(data):
        links_adj = Glycan_Semantics.build_adjacency_list(data)
        visited = set()
        source = next(iter(links_adj))
        Glycan_Semantics.DFS(links_adj,visited,-1,source)
        return len(visited)


    # returns true if there is a cycle else false
    @staticmethod
    def DFS(adj, visited, parent, u):
        visited.add(u)

        for v in adj[u]:
            if v == parent:
                continue
            elif v in visited:
                return True
            
            elif Glycan_Semantics.DFS(adj, visited, u, v):
                return True

        return False

    @staticmethod
    def link_count(data):

        links_adj = Glycan_Semantics.build_adjacency_list(data, cleaned=True)

        # no. of links = no. of monos - 1
        # num_links = []
        num_links = 0
        monos_collections = set()

        # links_adj = Utility.build_adjacency_list(pred_data)
        for box_id, linked_id in links_adj.items():
            for link_id in linked_id:
                monos_collections.add(box_id)
                monos_collections.add(link_id)

                num_links += 1

        return int(num_links/2), tuple(sorted(monos_collections))      # because the links are bi-directional, every link is counted twice

    
    @staticmethod
    def compstr(monosaccharides):
        comp = Glycan_Semantics.composition(monosaccharides)
        retval = ""
        for sym in Glycan_Semantics.mono_syms:
            if comp[sym] > 0:
                retval += sym + "(" + str(comp[sym]) + ")"
        return retval

    @staticmethod
    def IUPAC(mono_data):
        iupac = []

        root_id = mono_data.root()

        if not root_id or root_id == -1:
            return None

        adj = Glycan_Semantics.build_adjacency_list(mono_data, cleaned=True)

        other_adj = Glycan_Semantics.build_adjacency_list(mono_data, cleaned=True)

        visited = set()
        ans = []

        monos = mono_data.semantics['monos']
        ans = Glycan_Semantics.generate_iupac(iupac, adj, visited, -1, root_id, monos)
       
        iupac = iupac[::-1] # IUPAC sequence are read in reverse order
        return ''.join(iupac)


    @staticmethod
    def build_adjecency_list(monosaccharides):
        adj = {}
        for id, mono in monosaccharides.items():
            links = mono.get('links')
            # Process links to keep only integer IDs
            filtered_links = []
            for link in links:
                if isinstance(link, list):  # If link contains a list, extract the first element
                    filtered_links.append(link[0])
                else:  # Otherwise, it’s already an integer
                    filtered_links.append(link)
            adj[mono.get('id')] = filtered_links
        return adj


    @staticmethod
    def floor_precision(value, precision):
        scale = 10 ** precision
        return math.floor(value * scale) / scale


    @staticmethod
    def build_adjacency_list(data, cleaned=False):
        # adj = {}
        monos = data.semantics['monos']

        adj = defaultdict(list)
        for id, mono_data in monos.items():
            # links = mono.get('links')
            # Process links to keep only integer IDs
            filtered_links = []
            for link_data in mono_data.get('links'):
                if isinstance(link_data, list):  # Handle case with confidence value - for pred data case
                    linked_id, conf = link_data
                    # filtered_links.append(link[0])
                    try:
                        adj[mono_data['box']].append([monos[linked_id]['box'], Glycan_Semantics.floor_precision(conf,8)])
                    except KeyError as k:
                        print("Key doesnt exist:", k)
                else:  # Handle case without confidence - for known data case
                    linked_id = link_data
                    adj[mono_data['box']].append(monos[linked_id]['box'])

        if cleaned:
            return Glycan_Semantics.adjacenecy_list_cleaned(adj)

        return adj

    @staticmethod
    def adjacenecy_list_cleaned(adj):

        cleaned_adj = {}

        for key, vals in adj.items():
            try:
                linked_ids = [linked_box.get('id') for linked_box in vals]
            except:
                linked_ids = [linked_box.get('id') for linked_box, _ in vals]

            cleaned_adj[key.get('id')] = linked_ids

        return cleaned_adj

            

    @staticmethod
    def generate_iupac(iupac, adj, visited, parent, u, monosaccharides):
        visited.add(monosaccharides[u].get('id'))

        # Get the current node's data
        symbol = monosaccharides[u].get('symbol')
        extension = '?1-?' if symbol not in ['NeuAc', 'NeuGc'] else '?2-?'
        data = symbol + extension if parent != -1 else symbol

        # Append the current node's data to the result
        iupac.append(data)

        # Filter adjacent nodes to only include unvisited ones
        filtered_adj = [v for v in adj[u] if v not in visited]

        # Sort the children (branches) lexicographically by their symbol for consistency
        filtered_adj = sorted(filtered_adj, key=lambda x: monosaccharides[x].get('symbol'))

        branch_strings = []
        for v in filtered_adj:
            branch_iupac = []
            Glycan_Semantics.generate_iupac(branch_iupac, adj, visited, u, v, monosaccharides)  # Recurse for each child
            
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



