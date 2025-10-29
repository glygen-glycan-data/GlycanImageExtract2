import importlib
import os.path

from . glycanannotator import Config_Manager, Config
from . yolomodels import YOLOModel
from . compareboxes import CompareBoxes
from . semantics import BoxPredictionSemantics
from . bbox import BoundingBox
from . model_evaluator import BoxCompare

class Finder(object):

    filters = []

    def __init__(self,labels=[]):
        self._labels = list(labels)
        self.params = {}
      
    def execute(self, obj, boxesonly=False):
        if boxesonly:
            return self.find_boxes(obj)
        return self.find_objects(obj)
    
    def clear(self):
        return

    def find_boxes(self, obj):
        raise NotImplementedError

    # same for KnownFinder and YOLOFinder
    def find_objects(self, obj):
        obj_list = []

        boxes = self.find_boxes(obj)

        for box in boxes:
            new_obj = self.box_to_object(box,obj)

            if new_obj is not None:
                obj_list.append(new_obj)
             
        accepted, rejected = self.filter_objects(obj_list)
        self.set_results(obj, accepted, rejected)
        self.log_error(obj, accepted, rejected)
        return accepted

    # same for KnownFinder and YOLOFinder
    def filter_objects(self,object_list):
        accepted = object_list
        rejected_total = []
        for f in self.filters:
            accepted, rejected = f.filter(accepted)
            rejected_total.extend(rejected)
        return accepted, rejected_total

    def get_label(self, index):
        if index < 0 or index >= len(self._labels):
            raise IndexError("Bad label index %s."%(index,))
        return self._labels[index]
    
    def box_label(self, box):
        return self.get_label(box.get('classid'))

    def get_label_index(self, label):
        if label not in self._labels:
            self._labels.append(label)
        return self._labels.index(label)

    def get_param(self, key, default=None):
        return self.params.get(key,default)

    def set_param(self, key, value):
        self.params[key] = value

class KnownFinder(Finder):

    defaults = {
        'boxpadding': 0.0,
    }

    def __init__(self,**kwargs):
        super().__init__()
        self.params.update(dict(
            boxpadding = Config.get_param('boxpadding', Config.FLOAT, kwargs, self.defaults),
        ))

    def write_model(self, finder_name, filename):
        with open(filename, 'w') as wh:
            print(f"[Finder:{finder_name}]",file=wh)
            print(f"class={self.__class__.__name__}",file=wh)
            # might need something more sophistocated if we have
            # params that are not easily output as strings...
            for k,v in self.params.items():
                print(f"{k}={v}",file=wh)
        return

    def write_labels(self, filename):
        with open(filename, 'w') as wh:
            for label in self._labels:
                print(f"{label}",file=wh)
        return

    def get_known_data(self, image_path):
        '''
        Structure to handle multiple glycans in a figure.

        DATA STRUCTURE to store _map.txt file details
        
        map_dict = {
            'figure': {'height': x, 'width': x},
            'glycans': [
                {   
                    'iupac': '',        # Note: these were supposed be key-value pair outside the list of glycan - but had to add them inside - because multiple glycans in a figure exist while doing the build training data task for glycans
                    'composition': '',
                    ...,
                    'bbox': [x,y,w,h],
                    classlabel: label, (if present, else default label = 'glycan')
                    'monos': {
                        1: {'symbol': GlcNac, 'anomer': 'a', 'x_min': 1, 'x_max': 5, 'y_min': 2, 'y_max':6},
                        2: {'symbol': Man, 'anomer': 'a', 'x_min': 1, 'x_max': 5, 'y_min': 2, 'y_max':6}
                    },
                    'links': {
                        (id1,id2): {'carbon_number': 1, ...},
                        (id1,id2): {'carbon_number': 4, ...},
                    },
                    'root': mono_id,
                    'squiggle': {'symbol': '~', 'x_min': 1, 'x_max': 5, 'y_min': 2, 'y_max':6},
                },
                # ... more glycans
            ],
        }
        '''
        
        image_data = image_path.rsplit('.',1)[0] + "_map.txt"

        map_dict = {
            'figure': {},
            'glycans': []
        }

        current_glycan = None
        glycan_count = 0

        with open(image_data, 'r') as file:

            for line in file:
                data_points = line.split()

                if data_points[0] == '#####' and data_points[1] == 'WHOLEIMAGE:':
                    map_dict['figure']['height'] = int(data_points[2])
                    map_dict['figure']['width'] = int(data_points[4])

                elif data_points[0] == '###' and data_points[1] == 'GLYCAN:':
                    # Create new glycan dictionary
                    current_glycan = {
                        'classlabel': 'glycan',    # default label is glycan, if map file contains a CLASS - it will be updated
                        'bbox': list(map(int, data_points[2:6])),
                        'monos': {},
                        'links': {},
                        'root': None
                    }
                    map_dict['glycans'].append(current_glycan)
                    glycan_count += 1

                elif data_points[0] == '###' and data_points[1] == 'CLASS:':
                    if current_glycan is not None:
                        current_glycan['classlabel'] = data_points[2]

                elif data_points[0] == 'm':
                    if current_glycan is not None:
                        mono_id = int(data_points[1])
                        name = data_points[2]
                        anomer = data_points[3]
                        assert anomer in ('a','b','?')
                        x_coords = []
                        y_coords = []

                        for coords in data_points[4:-1]:
                            x,y = map(int,coords.split(','))
                            x_coords.append(x)
                            y_coords.append(y)

                        x_min = int(min(x_coords))
                        y_min = int(min(y_coords))
                        x_max = int(max(x_coords))
                        y_max = int(max(y_coords))

                        mono_data = {mono_id: {'symbol': name, 'anomer': anomer, 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}}

                        current_glycan['monos'].update(mono_data)
                        
                        # Update root to be the minimum mono_id
                        if current_glycan['root'] is None:
                            current_glycan['root'] = mono_id
                        else:
                            current_glycan['root'] = min(current_glycan['root'], mono_id)

                elif data_points[0] == 'l':
                    if current_glycan is not None:
                        mono_id1, mono_id2 = map(int,[data_points[1],data_points[4]])

                        x1_min, x1_max, y1_min, y1_max = [v for k,v in current_glycan['monos'][mono_id1].items() if k in ['x_min', 'x_max', 'y_min', 'y_max']]
                        x2_min, x2_max, y2_min, y2_max = [v for k,v in current_glycan['monos'][mono_id2].items() if k in ['x_min', 'x_max', 'y_min', 'y_max']]

                        x_min, x_max = min(x1_min, x1_max,x2_min, x2_max), max(x1_min, x1_max,x2_min, x2_max)
                        y_min, y_max = min(y1_min, y1_max, y2_min, y2_max), max(y1_min, y1_max, y2_min, y2_max)

                        carbon_number = data_points[2]
                        try:
                            carbon_number = int(carbon_number)
                        except ValueError:
                            pass
                        assert carbon_number in (1,2,3,4,5,6,8,'?')

                        link_data = {
                            (mono_id1, mono_id2): {'carbon_number': carbon_number,'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
                        }

                        current_glycan['links'].update(link_data)

                elif data_points[0] == 'r':
                    if current_glycan is not None:
                        x_coords = []
                        y_coords = []

                        for coords in data_points[4:-1]:
                            x,y = map(int,coords.split(','))
                            x_coords.append(x)
                            y_coords.append(y)

                        x_min = int(min(x_coords))
                        y_min = int(min(y_coords))
                        x_max = int(max(x_coords))
                        y_max = int(max(y_coords))
                        current_glycan['squiggle'] = {'symbol': data_points[2], 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

                elif data_points[0] == '#':
                    # rest of the key-value pairs from map file (like iupac, composition, etc.)
                    key = data_points[1][:-1]
                    value = ' '.join(data_points[2:])
                    current_glycan[key] = value

        # Boolean to indicate if map file contains SGI or MGI
        map_dict["SGI"] = (glycan_count == 1)
        map_dict["glycan_count"] = glycan_count
        
        print("\n----->>>map_dict",map_dict)
        return map_dict

    def find_boxes(self, obj):
        image_path = obj.image_path()
        assert image_path, f"{self.__class__.__name__} can only run on SingleGlycanImage objects"

        map_dict = self.get_known_data(image_path)

        boxes = []
        for b in self.create_boxes(map_dict):
            b.set_image_dimensions(image=obj.image())
            if self.params['boxpadding'] > 1:
                b.pad(self.params['boxpadding'])
            elif self.params['boxpadding'] > 0:
                b.pad_relative(self.params['boxpadding'])
            boxes.append(b)
        return boxes

def toboxes(func):
    def wrapper(self,x,y):
        if isinstance(x,BoxPredictionSemantics):
            x = x.box()
        if isinstance(y,BoxPredictionSemantics):
            y = y.box()
        result = func(self,x,y)
        return result
    return wrapper

class YOLOFinder(YOLOModel,Finder):

    defaults = {
        'conf_threshold': 0.5,
        'boxpadding': 0,
        'expandimage': 0,
        'iou_threshold': 0.4
    }

    def __init__(self,**kwargs):
        weights_file = Config.get_param('weights', Config.CONFIGFILE, kwargs, self.defaults)
        assert weights_file is not None
        labels_file = weights_file.replace("weights","labels")
        labels = [ s.strip() for s in open(labels_file).read().split() ]
        Finder.__init__(self,labels)
        self.params.update(dict(
            config = Config.get_param('config', Config.CONFIGFILE, kwargs, self.defaults),
            weights = weights_file,
            conf_threshold = Config.get_param('conf_threshold', Config.FLOAT, kwargs, self.defaults),
            iou_threshold = Config.get_param('iou_threshold', Config.FLOAT, kwargs, self.defaults),
            boxpadding = Config.get_param('boxpadding', Config.FLOAT, kwargs, self.defaults),
            expandimage = Config.get_param('expandimage', Config.INT, kwargs, self.defaults)
        ))
        YOLOModel.__init__(self,self.params)

    def known_finder(self):
        weights_file = self.params.get("weights",None)
        assert weights_file is not None
        model_file = weights_file.replace("weights","model")
        labels_file = weights_file.replace("weights","labels")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(model_file)
        if not os.path.isfile(labels_file):
            raise FileNotFoundError(labels_file)
        cm = Config_Manager(config_fullpath=model_file)
        f = cm.get_one_finder()
        for label in open(labels_file).read().split():
            f.get_label_index(label)
        return f

    def box_compare(self,**kwargs):
        return BoxCompare(**kwargs)

    def find_boxes(self, obj):
        image = obj.image()
        boxes = self.get_YOLO_output(image)
        # add classlabel here instead of YoloModel
        for box in boxes:
            box.set('classlabel',self.box_label(box))
        return sorted(boxes, key=lambda box: float(box.get('confidence',0.0)), reverse=True)

    @toboxes
    def dist(self,x,y):
        return CompareBoxes.euclidean_distance(x,y)

    @toboxes
    def proximity(self,x,y):
        return CompareBoxes.proximity(x,y)

    @toboxes
    def iou(self,x,y):
        return CompareBoxes.iou(x,y)

    @toboxes
    def intersect(self,x,y):
        return CompareBoxes.have_intersection(x,y)
    
    @toboxes
    def intersection_area(self,x,y):
        return CompareBoxes.intersection_area(x,y)

    def clear(self):
        if isinstance(self,YOLOModel):
            self.clear_model()
 
