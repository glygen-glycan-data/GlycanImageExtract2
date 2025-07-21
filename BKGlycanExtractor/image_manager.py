import os
import re
import sys
import xml.dom.minidom
from cairosvg import svg2png
import cv2
import numpy as np
import random
import traceback
from fnmatch import fnmatch
from . svg_parse_path import get_points

class Image_Manager:
    def __init__(self,glycan_folder,pattern='*.png,*.jpg'):
        self.globs = pattern.split(',')
        self.images = self.get_images(glycan_folder)

    def __iter__(self):
        return iter(self.images)

    def exclude(self,pattern="*.annotated.*"):
        self.images = list(filter(lambda fn: not fnmatch(fn,pattern), self.images))

    def get_images(self,glycan_folder):
        images = []

        if isinstance(glycan_folder,list):
            for image_file in glycan_folder:
                if os.path.is_file(image_file) and self.match_glob(image_file):
                    images.append(image_file)
        elif os.path.isdir(glycan_folder):
            for image_file in os.scandir(glycan_folder):
                if image_file.is_file() and self.match_glob(image_file):
                    images.append(image_file.path)
        return sorted(images)

    def match_glob(self,image_file):
        return any(fnmatch(image_file.name,ext) for ext in self.globs)



class Image_Data:

    def __init__(self,valid_monos):
        self.valid_monos = set(valid_monos)

    color_range_dict = {
        "black_lower" : np.array([0,0,0]),
        "black_upper" : np.array([180,171,235]),
        "red_lower_l" : np.array([0,98,120]),
        "red_upper_l" : np.array([9,255,255]),
        "yellow_lower" : np.array([18,33,222]),
        "yellow_upper" : np.array([30,255,255]),
        "green_lower" : np.array([37,30,70]),
        "green_upper" : np.array([73,255,255]),
        "blue_lower" : np.array([99,123,77]),
        "blue_upper" : np.array([120,255,255]),
        "purple_lower" : np.array([128,58,74]),
        "purple_upper" : np.array([163,255,225]),
        "red_lower_h" : np.array([170,111,135]),
        "red_upper_h" : np.array([180,255,255]),
        "light_blue_lower": np.array([85,41,201]),
        "light_blue_upper": np.array([108,121,255]),
    }

    def generate(self,glycan_folder):
        # use the random SVG image's folder and create data
        img_manager = Image_Manager(glycan_folder,pattern='*.svg')

        if len(img_manager.images) < 1:
            sys.exit("Error: No SVG images were found in the folder provided")

        print("\nCreating PNG and TXT files if they do not exist for the corresponding SVG files...")

        for image_file in img_manager.images:
            self.generate_image(image_file)

    def generate_image(self,image_file,force=False,**kwargs):
        base_name,extn = image_file.rsplit('.', 1)
        assert extn.lower() == "svg"

        png_image = base_name + '.png'
        txt_file = base_name + '_map.txt'

        # create PNG and TXT files only if they do not exist...
        if not force and os.path.exists(png_image):
            return
        if not force and os.path.exists(txt_file):
            return
        
        self.make_semantics_file(image_file,txt_file)
        self.svg_to_png(image_file,png_image)
        self.random_colors(png_image)
        return txt_file

    def parse_clippaths(self,svg):
        cppoints = {}
        for cp in svg.getElementsByTagName('clipPath'):
            clipPathID = cp.getAttribute('id')
            points = self.get_path(cp)
            assert points is not None
            cppoints[clipPathID] = points
        return cppoints

    def get_path(self,e):
        points = None
        for ch in e.childNodes:
            if ch.nodeName == "path":
                points = get_points(ch.getAttribute('d'))
                break
        if points is not None:
            return points[0]
        return None

    def find_clippath(self,e):
        for ch in e.childNodes:
            stylestring = None
            if not hasattr(ch,'getAttribute'):
                continue
            stylestring = ch.getAttribute("style")
            if 'clip-path:url(#' not in stylestring:
                continue
            break
        if stylestring is not None:
            stylestring = stylestring.split("clip-path:url(#",1)[1]
            pathname = stylestring.split(")",1)[0]
            return pathname
        return None

    def find_dimensions(self,e):
        length = None
        for ch in e.childNodes:
            if hasattr(ch,'hasAttribute') and ch.hasAttribute("height"):
                length = int(ch.getAttribute("height"))
                assert length == int(ch.getAttribute("width"))
                cx = int(ch.getAttribute("x")) + length/2
                cy = int(ch.getAttribute("y")) + length/2
                break
        if length == None:
            return None
        return dict(center=(cx,cy),diameter=length)

    def parse_elements(self,svg):
        elements = {}
        for e in svg.getElementsByTagName('g'):

            if not e.hasAttribute('ID'):
                continue

            gid = e.getAttribute("ID")
            e.setIdAttribute("ID")

            data_type = e.getAttribute("data.type")
            
            if data_type == "Monosaccharide":           
                id = int(gid.split(':')[1])

                name = e.getAttribute("data.residueName") 
                if name not in self.valid_monos:
                    raise ValueError("SVG Parser: %s not a valid mono name"%(name,))
                anomer = e.getAttribute("data.residueAnomericState")

                clippath = self.find_clippath(e)
                assert clippath is not None

                dims = self.find_dimensions(e)
                assert dims is not None

                elements[gid] = dict(id=id,datatype=data_type,name=name,anomer=anomer,clippath=clippath,**dims)

            elif data_type == "Linkage":
                fromid,toid = map(int,gid.split(":")[1].split(','))
                parent_bond = e.getAttribute("data.parentPositions")
                child_bond = e.getAttribute("data.childPositions")

                elements[gid] = dict(fromid=fromid,toid=toid,datatype=data_type,parent_bond=parent_bond,child_bond=child_bond)

            elif gid == "r-1:1":
                # Not a monosaccharide, must be the redend squiggle
                id = int(gid.split(':')[1])

                points = self.get_path(e)
                assert points is not None

                elements[gid] = dict(id=id,datatype="RedEndMarker",points=points)

            elif gid == "l-1:1,2":
                # Not a linkage, must be the redend link
                fromid,toid = map(int,gid.split(":")[1].split(','))

                elements[gid] = dict(fromid=fromid,toid=toid,datatype="RedEndLink")
                
        return elements

    def make_semantics_file(self,infile,outfile):
        svg_file = xml.dom.minidom.parse(infile)
        svg = svg_file.getElementsByTagName('svg')[0]
        
        clippaths = self.parse_clippaths(svg)
        elements = self.parse_elements(svg)

        rows = []

        datatype_order = dict(RedEndMarker=1,Monosaccharide=2,Linkage=3)
        for e in sorted(elements.values(),key=lambda e: (datatype_order.get(e['datatype'],0),e.get('id',0),e.get('fromid',0),e.get('toid',0))):
            row = None
            if e['datatype'] == "Monosaccharide":
                row = [ "m", e['id'], e['name'], e['anomer'] ]
                row += [ "%d,%d"%p for p in clippaths[e['clippath']] ]
                row += [ "%d,%d"%e['center'], e['diameter'] ]
            elif e['datatype'] == "Linkage":
                row = [ "l", e['fromid'], e['parent_bond'], e['child_bond'], e['toid'] ]
            elif e['datatype'] == "RedEndMarker":
                row = [ "r", e['id'], "~" ]
                row += [ "%d,%d"%p for p in e['points'] ]
            
            if row is not None:
                rows.append(row)

        with open(outfile, 'w') as of:
            for row in rows:
                print("\t".join(map(str,row)),file=of)

        return

    def randomize_anomers(self,svgdoc,elements,anomers):

        assert all([ (a in ('a','b',' ','?')) for a in anomers ])

        for lgid,e in elements.items():
            if e['datatype'] not in ('Linkage','RedEndLink'):
                continue

            newanomer = random.choice(anomers)
            assert newanomer in ('a','b',' ','?')

            togid = "r-1:%d"%(e['toid'],)
            toele = svgdoc.getElementById(togid)
            toele.setAttribute("data.residueAnomericState",newanomer if newanomer in ('a','b') else "?")

            ligid = lgid.replace('l-1:','li-1:')
            liele = svgdoc.getElementById(ligid)

            teeles = [ te for te in liele.getElementsByTagName('text') if te.firstChild.nodeValue ]
            # last one is anomer
            if newanomer == "a":
                teeles[-1].firstChild.nodeValue = "\u03B1" #alpha
            elif newanomer == "b":
                teeles[-1].firstChild.nodeValue = "\u03B2" #beta
            else:
                teeles[-1].firstChild.nodeValue = newanomer
        return 
    
    def randomize_parent_carbon_bonds(self,svgdoc,elements,carbon_bonds):

        assert all([ (c in ('2','3','4','6','8',' ','?')) for c in carbon_bonds ])

        for lgid,e in elements.items():
            if e['datatype'] not in ('Linkage','RedEndLink'):
                continue

            newcarbon = random.choice(carbon_bonds)
            assert newcarbon in ('2','3','4','6','8',' ','?')

            togid = "l-1:%d,%d"%(e['fromid'],e['toid'])
            toele = svgdoc.getElementById(togid)
            toele.setAttribute("data.parentPositions",newcarbon if newcarbon in ('2','3','4','6','8') else "?")

            ligid = lgid.replace('l-1:','li-1:')
            liele = svgdoc.getElementById(ligid)

            teeles = [ te for te in liele.getElementsByTagName('text') if te.firstChild.nodeValue ]
            # first one is carbon bond
            teeles[0].firstChild.nodeValue = newcarbon
        return 


    def randomize_anomercarbon_pairs(self,svgdoc,elements,anomers,a_carbons=['8'], b_carbons=[' ','?'], x_carbons=['2','3','4','6','?',' ']):
        assert all([ (a in ('a','b',' ','?')) for a in anomers ])

        for lgid,e in elements.items():
            if e['datatype'] not in ('Linkage','RedEndLink'):
                continue

            # randomize anomer
            newanomer = random.choice(anomers)
            assert newanomer in ('a','b',' ','?')

            # logic based carbon selection with goal of raising % of rare linkages bx a8 x3 x4 x6 x2
            if newanomer == "a":
                newcarbon = random.choice(a_carbons)
            elif newanomer == "b":
                newcarbon = random.choice(b_carbons)
            elif newanomer in (' ','?'):
                newcarbon = random.choice(x_carbons)
            else:
                raise ValueError("Unknown anomer %s"%newanomer)

            a_togid = "r-1:%d"%(e['toid'],)
            a_toele = svgdoc.getElementById(a_togid)
            a_toele.setAttribute("data.residueAnomericState",newanomer if newanomer in ('a','b') else "?")

            c_togid = "l-1:%d,%d"%(e['fromid'],e['toid'])
            c_toele = svgdoc.getElementById(c_togid)
            c_toele.setAttribute("data.parentPositions",newcarbon if newcarbon in ('2','3','4','6','8') else "?")

            ligid = lgid.replace('l-1:','li-1:')
            liele = svgdoc.getElementById(ligid)
 
            teeles = [ te for te in liele.getElementsByTagName('text') if te.firstChild.nodeValue ]
            # last one is anomer
            if newanomer == "a":
                teeles[-1].firstChild.nodeValue = "\u03B1" #alpha
            elif newanomer == "b":
                teeles[-1].firstChild.nodeValue = "\u03B2" #beta
            else:
                teeles[-1].firstChild.nodeValue = newanomer

            # first one is carbon bond
            teeles[0].firstChild.nodeValue = newcarbon

        return 

    
    def randomize_linkinfo(self,infile,anomers=None,carbon_bonds=None):

        if anomers is None:
            anomers = ['?',' ','a','a','b','b']
        if carbon_bonds is None:
            carbon_bonds = ['?',' ','2','2','3','3','4','4','6','6','8','8']

        svg_file = xml.dom.minidom.parse(infile)
        svg = svg_file.getElementsByTagName('svg')[0]
        
        elements = self.parse_elements(svg)
        # self.randomize_anomers(svg_file,elements,anomers)
        # self.randomize_parent_carbon_bonds(svg_file,elements,carbon_bonds)
        self.randomize_anomercarbon_pairs(svg_file,elements,anomers)

        with open(infile, 'w') as f:
            svg_file.writexml(f, encoding='UTF-8')

        return

    def svg_parser(self,infile,outfile,**kwargs):

        x = kwargs.get('x',None)
        y = kwargs.get('y',None)
        groups = kwargs.get('groups',None)

        svg_file = xml.dom.minidom.parse(infile)
        svg = svg_file.getElementsByTagName('svg')[0]
        svg_viewbox = svg.getAttribute('viewBox').split()
        svg_width = svg_viewbox[2]
        svg_height = svg_viewbox[3] 
        raw_width = float(svg_width)
        raw_height = float(svg_height)
        width_ratio = x and (x / raw_width) or 1
        height_ratio = y and (y / raw_height) or 1

        if groups:
            elements = [g for g in svg.getElementsByTagName('g') if (g.hasAttribute('ID') and g.getAttribute('ID') in groups)]
            elements.extend([p for p in svg.getElementsByTagName('path') if (p.hasAttribute('ID') and p.getAttribute('ID') in groups)])
        else:
            elements = svg.getElementsByTagName('g')

        parsed_groups = {}
        for e in elements:
            pointset_count = 0
            if e.nodeName == 'g':
                for node in e.childNodes:
                    if node.nodeName == 'defs':
                        clipPaths = node.childNodes
                        for clipPath in clipPaths:
                            if clipPath.nodeName == 'clipPath':
                                clipPathID = clipPath.getAttribute('id')
                                svgpaths = clipPath.childNodes
                                paths = []
                                for path in svgpaths:
                                    if path.nodeName == 'path':
                                        points = get_points(path.getAttribute('d'))
                                        for pointset in points:
                                            paths.append([clipPathID, pointset])
                                            pointset_count += 1
                                parsed_groups[clipPathID] = paths
            else:
                points = get_points(e.getAttribute('d'))
                for pointset in points:
                    paths.append([e.getAttribute('ID'), pointset])
            if e.hasAttribute('transform'):
                for transform in re.findall(r'(\w+)\((-?\d+.?\d*),(-?\d+.?\d*)\)', e.getAttribute('transform')):
                    if transform[0] == 'translate':
                        x_shift = float(transform[1])
                        y_shift = float(transform[2])
                        for path in paths:
                            path[1] = [(p[0] + x_shift, p[1] + y_shift) for p in path[1]]

        groups = {}
        
        for e in elements:
            if e.hasAttribute('ID'):
                data_type = e.getAttribute("data.type")
                gid = e.getAttribute("ID")
                anomer = e.getAttribute("data.residueAnomericState") if e.hasAttribute("data.residueAnomericState") else ""
                parent_bond = e.getAttribute("data.parentPositions") if e.hasAttribute("data.parentPositions") else "" 
                child_bond = e.getAttribute("data.childPositions") if e.hasAttribute("data.childPositions") else ""
                
                if kwargs.get('change_all_links'):
                    anomer, parent_bond = self.get_written_link(kwargs.get('change_all_links'))
                
                if data_type == 'Monosaccharide':

                    if kwargs.get('change_all_links'):
                        self.write_anomer(e,anomer)

                    name = e.getAttribute("data.residueName") 
                    if name not in self.valid_monos:
                        raise ValueError("SVG Parser: %s not a valid mono name"%(name,))

                    stylestring = None
                    for ch in e.childNodes:
                        if not hasattr(ch,'getAttribute'):
                            continue
                        stylestring = ch.getAttribute("style")
                        if 'clip-path:url' not in stylestring:
                            continue
                        break
                    if stylestring is None:
                        continue

                    stylestring = stylestring.split("clip-path:url(#",1)[1]
                    pathname = stylestring.split(")",1)[0]
                    groups[gid] = []           
                    groups[gid].append(str(name))
                    groups[gid].append(anomer) if anomer != " " else groups[gid].append("?")
                    for i in parsed_groups[pathname][0][1]:
                        groups[gid].append(i)

                    for ch in e.childNodes:
                        if hasattr(ch,'hasAttribute') and ch.hasAttribute("height"):
                            length = int(ch.getAttribute("height"))
                            assert length == int(ch.getAttribute("width"))
                            cx = int(ch.getAttribute("x")) + length/2
                            cy = int(ch.getAttribute("y")) + length/2
                            groups[gid].append((cx,cy))
                            groups[gid].append(length)
                            break

                elif data_type == 'Linkage':

                    gid = e.getAttribute("ID")   
                    t = gid.split(':')[1].split(',')
                    if parent_bond != " ": 
                        groups[gid] = ['l', t[0], parent_bond, child_bond, t[1]]  
                    else:
                        groups[gid] = ['l', t[0], "?", child_bond, t[1]]  

                elif gid[0:2] == 'li':
                    if kwargs.get('change_all_links'):
                        self.write_link(e, anomer, parent_bond)

                elif gid == "r-1:1": # reducing-end squiggle
                    for ch in e.childNodes:
                        if hasattr(ch,'hasAttribute') and ch.hasAttribute("d"):
                            points = get_points(ch.getAttribute('d'))
                            groups[gid] = [ "~" ] + points[0]

                # elif gid == "l-1:1,2": # reducing-end squiggle link
                #      groups[gid] = []

        out = []

        for g in groups:

            # linkages
            if g[0][0] == 'l':
                out.append(groups[g])

            # monosaccharides
            if g[0] == 'r':
                i = g.split(':')[-1]
                tmp = ['m',i]
                if groups[g][0] == "~":
                    tmp[0] = 'r'
                for p in groups[g]:
                    if type(p) == tuple:
                        tmp.append(str(int(round(p[0]*width_ratio))) +',' + str(int(round(p[1]*height_ratio))))
                    else:
                        tmp.append(str(p))
                out.append(tmp)

        labelorder = dict(r=0,m=1,l=2)
        def sortkey(l):
            try:
                intval = int(l[2])
                strval = ''
            except ValueError:
                intval = 1e+20
                strval = l[2]
            return labelorder[l[0]],int(l[1]),intval,strval

        out.sort(key=sortkey)   

        with open(outfile, 'w') as of:
            of.write('\n'.join([ "\t".join(line) for line in out]))

        #campbell
        if kwargs.get('change_all_links'):
            # Save the modified SVG back to the original file
            with open(infile, 'w') as f:
                svg_file.writexml(f, encoding='UTF-8')
            print(f"Modified SVG saved to {infile}")

    def svg_to_png(self,infile,outfile):
        svg2png(file_obj=open(infile, "rb"), write_to=outfile)
    
    def random_colors(self,image_file):
        # use heuristic mono finding colour ranges to make ranges of blue/green/red/etc

        image = cv2.imread(image_file)
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, self.color_range_dict['yellow_lower'], self.color_range_dict['yellow_upper'])
        purple_mask = cv2.inRange(hsv, self.color_range_dict['purple_lower'], self.color_range_dict['purple_upper'])
        red_mask_l = cv2.inRange(hsv, self.color_range_dict['red_lower_l'], self.color_range_dict['red_upper_l'])
        red_mask_h = cv2.inRange(hsv, self.color_range_dict['red_lower_h'], self.color_range_dict['red_upper_h'])
        red_mask = red_mask_l + red_mask_h
        green_mask = cv2.inRange(hsv, self.color_range_dict['green_lower'], self.color_range_dict['green_upper'])
        blue_mask = cv2.inRange(hsv, self.color_range_dict['blue_lower'], self.color_range_dict['blue_upper'])
        black_mask = cv2.inRange(hsv, self.color_range_dict['black_lower'], self.color_range_dict['black_upper'])

        # store these mask into array
        mask_array = (red_mask, yellow_mask, green_mask, blue_mask, purple_mask, black_mask)
        mask_array_name = ("red_mask", "yellow_mask", "green_mask", "blue_mask", "purple_mask", "black_mask")
        mask_dict = dict(zip(mask_array_name, mask_array))


        replacement_color_dict = {}

        for mask_name in mask_dict.keys():
            color = mask_name.split("_")[0]
            if color == "black":
                continue
            hsv_range_dict = {"0": [],
                            "1": [],
                            "2": []}
            
            for name in self.color_range_dict.keys():
                if name.startswith(color):
                    if "lower" in name:
                        lowername = name
                        uppername = lowername.replace("lower","upper")
                        for idx in range(0,3):
                            low = self.color_range_dict[lowername][idx]
                            high = self.color_range_dict[uppername][idx]
                            valuerange = range(low, high)
                            [ hsv_range_dict[str(idx)].append(r) for r in valuerange ]
            
            h = random.choice(hsv_range_dict["0"])
            s = random.choice(hsv_range_dict["1"])
            v = random.choice(hsv_range_dict["2"])
            replacement_color_dict[color] = (h,s,v)

        for color, value in replacement_color_dict.items():
            for name, mask in mask_dict.items():
                if name.startswith(color):
                    hsv[mask>0] = value

        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        blur = random.choice([True,False])
        if blur:
            img = cv2.blur(img, (4,4))
        cv2.imwrite(image_file, img)