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
        self.glob = [pattern_type.strip()[1:] if pattern_type.strip().startswith('*') else pattern_type.strip() for pattern_type in pattern.split(',')]
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
        return any(image_file.name.endswith(ext) for ext in self.glob)



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

    def generate_image(self,image_file,force=False):
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

    def parse_clippaths(self,svgdoc):
        svg = svgdoc.getElementsByTagName('svg')[0]
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

    def parse_elements(self,svgdoc):
        svg = svgdoc.getElementsByTagName('svg')[0]
        elements = {}
        for e in svg.getElementsByTagName('g'):

            if not e.hasAttribute('ID'):
                continue

            gid = e.getAttribute("ID")
            e.setIdAttribute("ID")

            data_type = e.getAttribute("data.type")
            
            if data_type == "Monosaccharide":           
                id = int(e.getAttribute("data.residueIndex"))

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
                fromsvgid,tosvgid = map(lambda i: ("r-1:"+i),gid.split(':')[1].split(','))
                fromid=int(e.getAttribute("data.parentResidueIndex"))
                toid=int(e.getAttribute("data.childResidueIndex"))
                parent_bond = e.getAttribute("data.parentPositions")
                child_bond = e.getAttribute("data.childPositions")

                elements[gid] = dict(fromid=fromid,toid=toid,fromsvgid=fromsvgid,tosvgid=tosvgid,datatype=data_type,parent_bond=parent_bond,child_bond=child_bond)

            elif gid == "r-1:1":
                # Not a monosaccharide, must be the redend squiggle
                id = 0

                points = self.get_path(e)
                assert points is not None

                elements[gid] = dict(id=id,datatype="RedEndMarker",points=points)

            elif gid == "l-1:1,2":
                # Not a linkage, must be the redend link
                fromsvgid,tosvgid = map(lambda i: ("r-1:"+i),gid.split(':')[1].split(','))
                fromid = 0
                toid = 1

                elements[gid] = dict(fromid=fromid,toid=toid,fromsvgid=fromsvgid,tosvgid=tosvgid,datatype="RedEndLink")
                
        return elements

    def make_semantics_file(self,svgfile,outfile):
        svgdoc = xml.dom.minidom.parse(svgfile)
        
        clippaths = self.parse_clippaths(svgdoc)
        elements = self.parse_elements(svgdoc)

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

    def blank_redend_marker(self,svgdoc,elements):
        for gid,e in elements.items():
            if e['datatype'] not in ('RedEndLink','RedEndMarker'):
                continue
            ele = svgdoc.getElementById(gid)
            ele.parentNode.removeChild(ele)
        return 

    def blank_unknown_linkinfo(self,svgdoc,elements):
        for lgid,e in elements.items():
            if e['datatype'] not in ('Linkage','RedEndLink'):
                continue

            blank_anomer = False 
            blank_parent_bond = False
            tm = elements[e['tosvgid']]
            if tm['anomer'] == "?":
                blank_anomer = True
            if e['datatype'] == "Linkage" and e['parent_bond'] == "?":
                blank_parent_bond = True

            if not blank_anomer and not blank_parent_bond:
                continue

            ligid = lgid.replace('l-1:','li-1:')
            liele = svgdoc.getElementById(ligid)

            if not liele:
                continue

            teeles = [ te for te in liele.getElementsByTagName('text') if te.firstChild.nodeValue ]
            if blank_parent_bond:
                teeles[0].firstChild.nodeValue = " " # blank
            if blank_anomer:
                teeles[-1].firstChild.nodeValue = " " # blank

        return 

    def randomize_anomers(self,svgdoc,elements,anomers):

        assert all([ (a in ('a','b',' ','?')) for a in anomers ])

        for lgid,e in elements.items():
            if e['datatype'] not in ('Linkage','RedEndLink'):
                continue

            newanomer = random.choice(anomers)
            assert newanomer in ('a','b',' ','?')

            togid = e['tosvgid']
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

            togid = e['tosvgid']
            lele = svgdoc.getElementById(togid)
            lele.setAttribute("data.parentPositions",newcarbon if newcarbon in ('2','3','4','6','8') else "?")

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

    def randomize_linkinfo(self,svgfile,anomers=None,carbon_bonds=None):

        if anomers is None:
            anomers = ['?',' ','a','a','b','b']
        if carbon_bonds is None:
            carbon_bonds = ['?',' ','2','2','3','3','4','4','6','6','8','8']

        svgdoc = xml.dom.minidom.parse(svgfile)
        
        elements = self.parse_elements(svgdoc)
        self.randomize_anomers(svgdoc,elements,anomers)
        self.randomize_parent_carbon_bonds(svgdoc,elements,carbon_bonds)
        # self.randomize_anomercarbon_pairs(svgdoc,elements,anomers)

        with open(svgfile, 'w') as f:
            svgdoc.writexml(f, encoding='UTF-8')

        return

    def randomize_blanks(self,svgfile,unknown,redend):

        svgdoc = xml.dom.minidom.parse(svgfile)
        
        elements = self.parse_elements(svgdoc)
        if random.choice(unknown):
            self.blank_unknown_linkinfo(svgdoc,elements)
        if random.choice(redend):
            self.blank_redend_marker(svgdoc,elements)

        with open(svgfile, 'w') as f:
            svgdoc.writexml(f, encoding='UTF-8')

        return

    def svg_to_png(self,svgfile,outfile):
        svg2png(file_obj=open(svgfile, "rb"), write_to=outfile)
    
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
