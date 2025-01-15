import cv2
import os
import numpy as np
import json
import copy
import random
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

    def random_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))


    def annotate(self,image,x1,y1,x2,y2,**kwargs):
        font_scale = kwargs.get('font_scale',0.5)
        color = kwargs.get('color',(0,255,0))
        thickness = kwargs.get('thickness',1)
        text = kwargs.get('text','')

        # uncomment below to apply random colors for monos, links, root
        # # Define overlay for transparency
        # overlay = image.copy()
        # cv2.rectangle(overlay,(x1,y1),(x2,y2),color=color,thickness=thickness)
        # # Apply the overlay with transparency
        # alpha = 0.5  # Transparency factor
        # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        cv2.rectangle(image,(x1,y1),(x2,y2),color=color,thickness=thickness)
        cv2.putText(image,org=(x1,y1),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,text=text, fontScale=0.5,thickness=1,color=color)


    def label_image(self,image,name='default'):
        assert image is not None

        directory = os.getcwd() + '/annotated_images/'
        if not os.path.exists(directory):
            os.makedirs(directory)

        glycan = self.semantics['glycans'][0]

        # glycan annotation
        x1,y1,x2,y2 = glycan.glycan_box().corners()
        self.annotate(image,*glycan.glycan_box().corners(),color=(0,255,0)) # green for glycan

        # monosaccharides, root and link labelling
        root_id = glycan.root()
        for mono in glycan.monosaccharides():
            x1,y1,x2,y2 = mono['box'].corners()
            text = mono.get('symbol','') + str(mono.get('id'))
            text = str(mono.get('classid')) + '-' + str(mono.get('id'))
            # text = str(mono.get('id'))

            # links
            if len(mono['links']) > 0:
                for link_id in mono['links']:
                    try:
                        id = link_id[0]
                    except:
                        id = link_id

                    linked_mono = glycan.monosaccharide(id)
                    _x1,_y1,_x2,_y2 = linked_mono['box'].corners()

                    x_coords = [x1,x2,_x1,_x2]
                    y_coords = [y1,y2,_y1,_y2]

                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords) 

                    self.annotate(image,x_min,y_min,x_max,y_max,color=(255, 255, 0),thickness=1)
            
            color = (128, 0, 128) # purple for monos
            if mono['id'] == root_id:
                color = (0,0,255) # red for root  

            self.annotate(image,x1,y1,x2,y2,text=text,color=color,thickness=1)   
        cv2.imwrite(directory + name + '.png', image)
        return image



class Glycan_Semantics(Image_Semantics):
    def __init__(self,image,box,**kwargs):
        super().__init__(image)  
        self.semantics['box'] = box
        self.semantics['bbox'] = box.bbox()
        self.semantics['monos'] = {}
        self.semantics.update(copy.deepcopy(kwargs))

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
        assert kwargs['id'] not in self.semantics['monos']
        self.semantics['monos'][kwargs['id']] = mono

    def monosaccharides(self):
        return self.semantics['monos'].values()

    def monosaccharide(self,id):
        assert id is not None
        return self.semantics['monos'][id]

    def mono_boxes(self):
        boxes = []
        for id,mono in self.semantics['monos'].items():
            boxes.append(mono['box'])
        return boxes

    def add_root(self,root_id=None):
        self.semantics['root'] = root_id 

    def root(self):
        return self.semantics.get('root',None)

    def add_link(self,id,link_ids):
        assert id is not None
        self.semantics['monos'][id]['links'] = link_ids

    def links(self,id):
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
                if k not in ('image','box','links'):
                    monodict[k] = v
                elif k == 'links' and len(v) > 0:
                    monodict[k] = [item if isinstance(item,int) else item[0] for item in v]
            data['monos'].append(monodict)

        return json.dumps(data,indent=2,sort_keys=True)

    def image_path(self):
        # for single glycan images, the glycan image "has" a path
        return self.semantics.get('image_path',None)

    def composition(self):
        count = defaultdict(int)
        for m in self.monosaccharides():
            sym = self.mono_syms[m['classid']]
            if sym not in count:
                count[sym] = 1
            else:
                count[sym] += 1
        return count

    mono_syms = ["GlcNAc","NeuAc","Fuc","Man","GalNAc","Gal","Glc","NeuGc"]
    

    def compstr(self):
        comp = self.composition()
        retval = ""
        for sym in self.mono_syms:
            if comp[sym] > 0:
                retval += sym + "(" + str(comp[sym]) + ")"
        return retval

    @staticmethod
    def IUPAC(monosaccharides, root_id):
        iupac = []
        # root_id = monosaccharides['root']

        if root_id == -1:
            return None

        # root_mono = self.monosaccharide(root_id)

        adj = Glycan_Semantics.build_adjecency_list(monosaccharides)

        visited = set()
        ans = []
                
        Glycan_Semantics.DFS(iupac,adj,visited,-1,root_id,monosaccharides)
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
    def DFS(iupac, adj, visited, parent, u, monosaccharides):
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
            Glycan_Semantics.DFS(branch_iupac, adj, visited, u, v, monosaccharides)  # Recurse for each child
            
            branch_str = ''
            for i in range(len(branch_iupac)-1,-1,-1):
                branch_str += branch_iupac[i]

            branch_strings.append(branch_str)  # Collect each branch as a string

        # Sort branches lexicographically after recursion
        branch_strings.sort()

        # If there are multiple branches, open a parenthesis to indicate a branch
        if len(branch_strings) > 1:
            iupac.append(')')

        # Append branches, enclosing only the first branch with parentheses
        for idx, branch in enumerate(branch_strings):
            if idx == 0 and len(branch_strings) > 1:
                iupac.append('(' + branch)  # Close after the first branch
            else:
                iupac.append(branch)



    
    # def IUPAC(self):
    #     iupac = []
    #     root_id = self.root()

    #     if root_id == -1:
    #         return None

    #     # root_mono = self.monosaccharide(root_id)

    #     adj = self.build_adjecency_list()

    #     visited = set()
    #     ans = []
                
    #     self.DFS(iupac,adj,visited,-1,root_id)
    #     iupac = iupac[::-1] # IUPAC sequence are read in reverse order
    #     return ''.join(iupac)



    # def build_adjecency_list(self):
    #     adj = {}
    #     for mono in self.monosaccharides():
    #         links = mono.get('links')
    #         # Process links to keep only integer IDs
    #         filtered_links = []
    #         for link in links:
    #             if isinstance(link, list):  # If link contains a list, extract the first element
    #                 filtered_links.append(link[0])
    #             else:  # Otherwise, it’s already an integer
    #                 filtered_links.append(link)
    #         adj[mono.get('id')] = filtered_links
    #     return adj


    # def DFS(self, iupac, adj, visited, parent, u):
    #     visited.add(self.monosaccharide(u).get('id'))

    #     # Get the current node's data
    #     symbol = self.monosaccharide(u).get('symbol')
    #     extension = '?1-?' if symbol not in ['NeuAc', 'NeuGc'] else '?2-?'
    #     data = symbol + extension if parent != -1 else symbol

    #     # Append the current node's data to the result
    #     iupac.append(data)

    #     # Filter adjacent nodes to only include unvisited ones
    #     filtered_adj = [v for v in adj[u] if v not in visited]

    #     # Sort the children (branches) lexicographically by their symbol for consistency
    #     filtered_adj = sorted(filtered_adj, key=lambda x: self.monosaccharide(x).get('symbol'))

    #     branch_strings = []
    #     for v in filtered_adj:
    #         branch_iupac = []
    #         self.DFS(branch_iupac, adj, visited, u, v)  # Recurse for each child
            
    #         branch_str = ''
    #         for i in range(len(branch_iupac)-1,-1,-1):
    #             branch_str += branch_iupac[i]

    #         branch_strings.append(branch_str)  # Collect each branch as a string

    #     # Sort branches lexicographically after recursion
    #     branch_strings.sort()

    #     # If there are multiple branches, open a parenthesis to indicate a branch
    #     if len(branch_strings) > 1:
    #         iupac.append(')')

    #     # Append branches, enclosing only the first branch with parentheses
    #     for idx, branch in enumerate(branch_strings):
    #         if idx == 0 and len(branch_strings) > 1:
    #             iupac.append('(' + branch)  # Close after the first branch
    #         else:
    #             iupac.append(branch)



