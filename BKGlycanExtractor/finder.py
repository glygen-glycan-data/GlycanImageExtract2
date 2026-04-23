import importlib
import os.path

from . glycanannotator import Config_Manager, Config
from . yolomodels import YOLOModel
from . compareboxes import CompareBoxes
from . semantics import BoxPredictionSemantics, Semantics
from . bbox import BoundingBox
from . model_evaluator import BoxCompare

class Finder(object):

    filters = []

    def __init__(self,name=None,cfgmgr=None,labels=[]):
        self._name = name
        self._cfgmgr = cfgmgr
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

    def box_to_object(self, box, semantics):
        raise NotImplementedError
    
    def second_chance_boxes_to_objects(self,second_chance_boxes,obj,obj_list):
        raise NotImplementedError

    # same for KnownFinder and YOLOFinder
    def find_objects(self, obj):
        boxes = self.find_boxes(obj)
        
        obj_list = []
        second_chance_boxes = []
        
        for box in boxes:
            new_obj = self.box_to_object(box,obj)
            if new_obj is not None and isinstance(new_obj, Semantics):
                obj_list.append(new_obj)
            elif new_obj is not None and len(new_obj) > 1:
                # ambiguous case (for links)
                second_chance_boxes.append((box,new_obj))
            else:
                pass
        
        if len(second_chance_boxes) > 0:
            obj_list = self.second_chance_boxes_to_objects(second_chance_boxes,obj,obj_list)
        
        obj_list.sort(key=lambda o: -o.get('confidence',0.0))
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
        super().__init__(name=kwargs.get('name'),
                         cfgmgr=kwargs.get('cfgmgr'))
        self.params.update(dict(
            boxpadding = Config.get_param('boxpadding', Config.FLOAT, kwargs, self.defaults),
        ))

        # generally a label type selected from the TSV file while building training data
        # this will be provided to the respective known finders - create_boxes/find_boxes - so that the labels can be updated
        # based on cmd line args (which is optional) during the activity of buildign training data
        self.label_type = kwargs.get('label_type')
        self.default_label = kwargs.get('default_label','glycan')
        self.exclude_labels = kwargs.get('exclude_labels',[])

    def write_model(self, filename):
        with open(filename, 'w') as wh:
            print(f"[Finder:{self._name}]",file=wh)
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

    def set_label_type(self,label_type):
        self.label_type = label_type

    def set_default_label(self,default_label):
        self.default_label = default_label

    def set_exclude_labels(self,exclude_labels):
        self.exclude_labels = exclude_labels

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

        if not os.path.exists(image_data):
            return None

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
                        'classlabel': self.default_label,
                        'bbox': list(map(int, data_points[2:6])),
                        'monos': {},
                        'links': {},
                        'root': None
                    }
                    map_dict['glycans'].append(current_glycan)
                    glycan_count += 1

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
                # rest of the key-value pairs from map file (like iupac, composition, classlabel, etc.)
                elif data_points[0] == '#':
                    key = data_points[1][:-1]
                    value = ' '.join(data_points[2:])
                    current_glycan[key] = value

        # Boolean to indicate if map file contains SGI or MGI
        map_dict["SGI"] = (glycan_count == 1)
        map_dict["glycan_count"] = glycan_count
        
        return map_dict

    def get_yolo_known_data(self,obj):

        image_path = obj.image_path()
        yolo_boxes_file = image_path.rsplit('.',1)[0] + ".txt"
        if not os.path.exists(yolo_boxes_file):
            return None
        
        height = obj.height()
        width = obj.width()
        map_dict = {
            'figure': {'height': height, 'width': width },
            'glycans': []
        }

        imagedir = os.path.split(image_path)[0]
        classes = open(os.path.join(imagedir,"classes.txt")).read().split()

        glycan_count = 0
        for line in open(yolo_boxes_file):
            classid,rcx,rcy,rw,rh = map(float,line.split()[:5])
            classid = int(classid)
            classlabel = classes[classid]
            bbox = BoundingBox(rcx=rcx,rcy=rcy,rw=rw,rh=rh,image_width=width,image_height=height)
            glycan = {
                'classlabel': classlabel,
                'classid': classid,
                'bbox': bbox.bbox()
            }
            map_dict['glycans'].append(glycan)
            glycan_count += 1
        
        map_dict["SGI"] = False
        map_dict["glycan_count"] = glycan_count
        
        return map_dict

    def find_boxes(self, obj):
        image_path = obj.image_path()
        assert image_path, f"{self.__class__.__name__} can only run on SingleGlycanImage objects"

        map_dict = self.get_known_data(image_path)
        if map_dict is None:
            # try the old-school YOLO format boxes...
            map_dict = self.get_yolo_known_data(obj)
        
        if map_dict is None:
            raise RuntimeError("Can't read known data for image %s."%(image_path,))

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
        Finder.__init__(self,name=kwargs.get('name'),cfgmgr=kwargs.get('cfgmgr'),labels=labels)
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
 
