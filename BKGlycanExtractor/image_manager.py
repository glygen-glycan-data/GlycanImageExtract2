import os
import re
import sys
import xml.dom.minidom
from cairosvg import svg2png
import cv2
import numpy as np
import random
from .scripts import parse_path
# import parse_path


class Image_Manager:
    def __init__(self,glycan_folder,pattern='*.png,*.jpg,*.txt'):
        self.glob = [pattern_type.strip()[1:] if pattern_type.strip().startswith('*') else pattern_type.strip() for pattern_type in pattern.split(',')]
        self.images = self.get_images(glycan_folder)

    def __iter__(self):
        return iter(self.images)

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
        return images

    def match_glob(self,image_file):
        return any(image_file.name.endswith(ext) for ext in self.glob)



class Image_Data:

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

    def __init__(self,glycan_folder):
        # use the random SVG image's folder and create data
        self.generate(glycan_folder)

    def generate(self,glycan_folder):
        img_manager = Image_Manager(glycan_folder,pattern='*.svg')

        if len(img_manager.images) < 1:
            sys.exit("Error: No SVG images were found in the folder provided")

        print("\nCreating PNG and TXT files if they do not exist for the corresponding SVG files...")

        for image_file in img_manager.images:
            base_name = image_file.rsplit('.', 1)[0]
            png_image = base_name + '.png'
            txt_file = base_name + '_map.txt'

            # create PNG and TXT files only if they do not exist...
            if not os.path.exists(png_image) or not os.path.exists(txt_file):
                self.svg_parser(image_file)
                self.svg_to_png(image_file)

                self.svg_parser(image_file)
                self.svg_to_png(image_file)

                png_image = image_file.rsplit('.',1)[0] + '.png'
                self.random_colors(png_image)
            
        
            
    def svg_parser(self,file,**kwargs):

        x = kwargs.get('x',None)
        y = kwargs.get('y',None)
        groups = kwargs.get('groups',None)

        svg_file = xml.dom.minidom.parse(file)
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
                                        points = parse_path.get_points(path.getAttribute('d'))
                                        for pointset in points:
                                            paths.append([clipPathID, pointset])
                                            pointset_count += 1
                                parsed_groups[clipPathID] = paths
            else:
                points = parse_path.get_points(e.getAttribute('d'))
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
                if data_type == 'Monosaccharide':
                    gid = e.getAttribute("ID")
                    name = e.getAttribute("data.residueName") 

                    first_child = e.firstChild
                    while first_child and first_child.nodeType == first_child.TEXT_NODE:
                        first_child = first_child.nextSibling

                    stylestring = first_child.getAttribute("style")
                    stylestring = stylestring.split("(")[1]
                    stylestring = stylestring.split(")")[0]
                    pathname = stylestring.strip("#")
                    groups[gid] = []           
                    groups[gid].append(str(name))
                    for i in parsed_groups[pathname][0][1]:
                        groups[gid].append(i)

                    if e.childNodes[1].hasAttribute("height"):
                        lenth = int(e.childNodes[1].getAttribute("height"))
                        cx = int(e.childNodes[1].getAttribute("x")) + lenth/2
                        cy = int(e.childNodes[1].getAttribute("y")) + lenth/2
                        groups[gid].append((cx,cy))
                        groups[gid].append(lenth)
                cx = 0
                cy = 0
                lenth = 0
                if data_type == 'Linkage':
                    gid = e.getAttribute("ID")     
                    groups[gid] = []

        out = []

        for g in groups:
            if g[0] == 'l':
                t = g.split(':')[1].split(',')
                out.append('l\t'+t[0]+'\t'+t[1])
            if g[0] == 'r':
                tmp = 'm\t'+g.split(':')[-1]+'\t'
                for p in groups[g]:
                    if type(p) == tuple:
                        tmp += str(int(p[0]*width_ratio)) +',' + str(int(p[1]*height_ratio))+'\t'
                    else:
                        tmp += str(p) + '\t'

                out.append(tmp[:-1])
        out.sort()   

        file_path = file.replace('.svg', '_map.txt')  

        with open(file_path, 'w') as outfile:
            outfile.write('\n'.join(out))



    def svg_to_png(self,file):
        outfile = file.rsplit('.',1)[0] + '.png'
        svg2png(file_obj=open(file, "rb"), write_to=outfile)

    
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







