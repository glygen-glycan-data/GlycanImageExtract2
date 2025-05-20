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
                color1 = color
                if mono['id'] == root_id:
                    color1 = root_color if root_color else color # dark green for root
                if mono.get('alternative') is not None:
                    color1 = alternative_color if alternative_color else color # orange for alternatives
                self.annotate(x1,y1,x2,y2,text=text,xtoff=2,ytoff=-2,color=color1,thickness=1)   

    def annotate_links(self,color=(255, 255, 0)):
        for glycan in self.semantics['glycans']:
            # monosaccharides and root labelling
            for link in glycan.undirected_links():

                fromid, toid = link['mono_ids']
                from_mono = glycan.monosaccharide(fromid)
                x1,y1,x2,y2 = from_mono['box'].corners()

                to_mono = glycan.monosaccharide(toid)
                _x1,_y1,_x2,_y2 = to_mono['box'].corners()

                x_coords = [x1,x2,_x1,_x2]
                y_coords = [y1,y2,_y1,_y2]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords) 

                self.annotate(x_min,y_min,x_max,y_max,color=color,thickness=1)

    def write_image(self,**kwargs):
        cv2.imwrite(self.make_filename(**kwargs), self.image())

    def training_data(self,folder_name):
        # need labels - monos need labels file
        # root - need labels file
        # links - need labels file

        # print("\nself.params",pipeline.get_steps("glycan")[0].labels)
        # classid_mappings = {"GlcNAc": 0, "NeuAc":1,"Fuc":2,"Man":3,"GalNAc":4,"Gal":5,"Glc":6,"NeuGc":7, "Xyl": 8}
        # labels = pipeline.get_steps("glycan")[0].labels
        # classid_mappings = {label:idx for idx,label in enumerate(labels)}

        image_path = self.image_path()
        image_filename = os.path.basename(image_path)
        base_filename = os.path.splitext(image_filename)[0]

        # os.makedirs(folder_name, exist_ok=True)
        # metadata_dir = os.path.join(folder_name, "metadata")
        # os.makedirs(metadata_dir, exist_ok=True)
        # metadata_file = os.path.join(metadata_dir, "metadata.txt")

        # print("file",metadata_file)

        # with open(metadata_file,'w') as f:
        #     f.write('labels: ' + ', '.join(labels) + '\n')

            # for k,v in pipeline.get_steps("glycan")[0].params.items():
            #     f.write(f"{k}: {v}\n")

        
        training_file_path = os.path.join(folder_name, base_filename + ".txt")
        with open(training_file_path, 'w') as f:  # Open the file to write annotations
            for glycan in self.semantics['glycans']:
                for mono in glycan.monosaccharides():
                    x,y,w,h = mono['box'].center_relative()
                    class_id = classid_mappings[mono.get('classlabel')]
                    f.write(f"{class_id} {x} {y} {w} {h}\n") 

        shutil.copy(image_path, folder_name)

class Glycan_Semantics(Image_Semantics):

    def __init__(self,image,box,**kwargs):
        super().__init__(image)  
        self.semantics['box'] = box
        self.semantics['bbox'] = box.bbox()
        self.semantics['monos'] = {}
        self.semantics.update(kwargs)

    def glycan_box(self):
        return self.semantics['box']

    def glycan(self):
        return [self.semantics]

    def image(self):
        return self.semantics['image']

    def clear_monos(self):
        self.semantics['monos'] = {}

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


    def find(self, u):
        if u not in self.parent:
            self.parent[u] = u
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False  # cycle detected
        self.parent[pu] = pv
        return True


    def filter_links_to_tree(self):
        links = self.undirected_links()
        # uf = UnionFind()

        kept_links = []
        cycle_links = []

        for link in links:
            id1, id2 = link["mono_ids"]
            if self.union(id1, id2):
                kept_links.append(link)
            else:
                cycle_links.append(link)

        if cycle_links:
            # Drop the link with the lowest confidence
            cycle_links.sort(key=lambda l: l.get('confidence', 0.0))
            dropped = cycle_links[0]
            print(f"Cycle found. Dropping least confident link: {dropped}")
            cycle_links.remove(dropped)
            kept_links += cycle_links  # If you want to keep the rest

        return kept_links


    def build_adjacency_list(self):
        error_msg = ''
        adj = defaultdict(list)
        parent = {}

        def find(u):
            while parent.get(u, u) != u:
                parent[u] = parent.get(parent[u], parent[u])
                u = parent[u]
            return u

        def union(u, v):
            pu, pv = find(u), find(v)
            if pu == pv:
                return False  # cycle detected
            parent[pu] = pv
            return True

        links = self.undirected_links()
        kept_links = []
        cycle_links = []

        for link in links:
            id1, id2 = link["mono_ids"]
            if union(id1, id2):
                kept_links.append(link)
            else:
                cycle_links.append(link)

        # Drop least confident cycle-forming link
        if cycle_links:
            cycle_links.sort(key=lambda l: l.get('confidence', 0.0))
            dropped = cycle_links.pop(0)
            print(f"Cycle found. Dropping least confident link: {dropped}")
            error_msg = f"Cycle found. Dropping least confident link: {dropped}"
            kept_links += cycle_links  # Keep remaining cycle links, optional

        for link in kept_links:
            id1, id2 = link["mono_ids"]
            link_without_ids = {k: v for k, v in link.items() if k != "mono_ids"}
            adj[id1].append((id2, link_without_ids))
            adj[id2].append((id1, link_without_ids))

        return adj, error_msg

    # used after undirected_links are already populated 
    # def build_adjacency_list(self):

    #     adj = defaultdict(list)

    #     for link in self.undirected_links():
    #         id1, id2 = link["mono_ids"]
    #          # removed 'mono_ids' because it has undirected_links which are in sorted order - since we dont want to propagate this pattern in the adjacency list
    #         link_without_ids = {k: v for k, v in link.items() if k != "mono_ids"} 
    #         adj[id1].append((id2, link_without_ids)) 
    #         adj[id2].append((id1, link_without_ids))

    #     return adj

    # build adjacency list from undirected links
    # if root is present - use that as the start to traverse the tree, if not use any random node and traverse the tree.
    # During the traversal if node is already visited - then there must be a cycle
    # At the end - if all nodes are not visited - then the tree must be disjointed

    def traverse_tree(self):
        adj,error_msg = self.build_adjacency_list()
        visited = set()

        root = self.root()

        if root:
            first_node = root['mono_id']
        elif adj:
            first_node = next(iter(adj))
        else:
            return False, "Graph is empty: no root and no nodes in adjacency list."

        stack = [(first_node, -1)]      # current_node, parent

        visited.add(first_node)

        while stack:
            u, parent = stack.pop()
            for v, other_info in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append((v,u))
                else:
                    if v == parent:
                        continue
                    else:
                        return False, f"Cycle detected at node '{v}' (visited from node '{u}')."

        # check if all nodes were visited
        all_nodes = set(m['id'] for m in self.monosaccharides())
        unvisited = all_nodes - visited
        if unvisited:
            return False, f"Tree is disjointed. Unvisited nodes: {', '.join(str(node) for node in unvisited)}"

        print("tree traversal", error_msg)
        return True, error_msg
                    
    def clear_links(self,mono_id):
        self.semantics['monos'][mono_id]['links'] = []

    def clear_all_links(self):
        for mono_id in self.monosaccharideids():
            self.clear_links(mono_id)

    def all_links(self):
        # a mono could have multiple links
        return [link for item in self.monosaccharides() if item.get('links') for link in item['links']]  

    def links(self,id):
        return self.semantics['monos'][id]['links']

    def delete_link(self,fromid,toid):
        new_links = [
            link for link in self.semantics['monos'][fromid]['links']
            if ([link['fromid'], link['toid']] != [fromid, toid])
        ]
        self.semantics['monos'][fromid]['links'] = new_links  

    def undirected_links(self):
        return self.semantics.get('undirected_links')


    def add_undirected_link(self,id1,id2,**kwargs):
        self.semantics['undirected_links'].append({"mono_ids": list(sorted((id1,id2))),**kwargs})

    def add_link(self,id1,id2,**kwargs):
        mono = self.monosaccharide(id1)

        if 'links' not in mono:
            mono['links'] = []

        mono['links'].append({"fromid":id1,"toid":id2, **kwargs})


    def create_links(self):
        adj,_ = self.build_adjacency_list()
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
        # print("IUPAC called",root)
        if not root:
            return None

        # adds directed links in semantics
        self.create_links()

        root_id = root.get('mono_id')
        iupac = []
        adj,_ = self.build_adjacency_list()
        visited = set()

        self.generate_iupac(iupac, adj, visited, -1, root_id)

        # print("iupac",iupac)
       
        iupac = iupac[::-1] # IUPAC sequences are read in reverse order
        return ''.join(iupac)


    def generate_iupac(self,iupac, adj, visited, parent, u):
        visited.add(u)

        # Get the current node's data
        symbol = self.monosaccharide(u).get('symbol')
        extension = '?1-?' if symbol not in ['NeuAc', 'NeuGc'] else '?2-?'
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







        
