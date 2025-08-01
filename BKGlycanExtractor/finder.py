'''
Subclasses inheriting from Finder must define labels
'''

import importlib

from .yolomodels import YOLOModel
class Finder(object):
    labels = None

    def __init__(self):
        # self._labels = []
        # if hasattr(self,'labels'):
        #     self._labels = list(self.labels)

        if self.labels is None:
            raise NotImplementedError(f"Class '{self.__class__.__name__}' must define a 'labels' attribute before calling Finder.__init__()")
        self._labels = list(self.labels)
      
    def execute(self, obj, boxesonly=False):
        if boxesonly:
            return self.find_boxes(obj)
        return self.find_objects(obj)

    def find_boxes(self, obj):
        raise NotImplementedError

    def find_objects(self, obj):
        raise NotImplementedError

    def get_label(self, index):
        if index < 0 or index >= len(self._labels):
            raise IndexError("Bad label index %s."%(index,))
        return self._labels[index]

    def get_label_index(self, label):
        if label not in self._labels:
            raise LookupError("Label %s not found."%(label,))
        return self._labels.index(label)

    def get_labels(self):
        return self._labels

 


class KnownFinder(Finder):

    def get(self, key, default=None):
        return getattr(self, key, default)

    
    def get_known_data(self, image_path):
        '''
        DATA STRUCTURE to store _map.txt file details

        map_dict = {
            'monos': {
                1: {'symbol': GlcNac, 'anomer': 'a', 'x_min': 1, 'x_max': 5, 'y_min': 2, 'y_max':6},
                2: {'symbol': Man, 'anomer': 'a', 'x_min': 1, 'x_max': 5, 'y_min': 2, 'y_max':6}
            },
            'links': {
                (id1,id2): {'carbon_bond': 1, ...},
                (id1,id2): {'carbon_bond': 4, ...},
            }
            'root': lowest_mono_id,
            'squiggle': {'symbol': '~', 'x_min': 1, 'x_max': 5, 'y_min': 2, 'y_max':6},
            
            'iupac': '',
            'composition': '',
        }
        '''
        
        image_data = image_path.rsplit('.',1)[0] + "_map.txt"

        map_dict = {
            'monos':{},
            'links': {},
            'root': None
        }

        with open(image_data, 'r') as file:

            root_id = float("inf")

            for line in file:
                data_points = line.split()

                if data_points[0] == 'm':
                    mono_id = int(data_points[1])
                    root_id = min(root_id, mono_id)
                    name = data_points[2]
                    anomer = data_points[3]
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

                    map_dict['monos'].update(mono_data)

                elif data_points[0] == 'l':
                    mono_id1, mono_id2 = map(int,[data_points[1],data_points[4]])

                    x1_min, x1_max, y1_min, y1_max = [v for k,v in map_dict['monos'][mono_id1].items() if k in ['x_min', 'x_max', 'y_min', 'y_max']]
                    x2_min, x2_max, y2_min, y2_max = [v for k,v in map_dict['monos'][mono_id2].items() if k in ['x_min', 'x_max', 'y_min', 'y_max']]

                    x_min, x_max = min(x1_min, x1_max,x2_min, x2_max), max(x1_min, x1_max,x2_min, x2_max)
                    y_min, y_max = min(y1_min, y1_max, y2_min, y2_max), max(y1_min, y1_max, y2_min, y2_max)

                    link_data = {
                        (mono_id1, mono_id2): {'carbon_bond': data_points[2], 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
                    }

                    map_dict['links'].update(link_data)

                elif data_points[0] == 'r':
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
                    map_dict['squiggle'] = {'symbol': data_points[2], 'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

                elif data_points[0] == '#':
                    key = data_points[1][:-1]
                    value = ' '.join(data_points[2:])

                    map_dict[key] = value

            map_dict['root'] = root_id if root_id == float("inf") else root_id

        return map_dict


class YOLOFinder(YOLOModel,Finder):

    filters = []


    def get_known_finder(self, training_file):

        if not os.path.isfile(training_file):
            raise FileNotFoundError()

        model_info = {}

        with open(training_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    model_info['finder'] = line[1:-1].split(":")[1].strip()
                elif line.startswith('class='):
                    model_info['class'] = line.split('=')[1].strip()
                else:
                    print('line',line)
                    key, value = line.split('=')
                    model_info[key.strip()] = float(value.strip())
        
        # Loads class dynamically
        module = importlib.import_module(".pipeline",package="BKGlycanExtractor")
        findercls = getattr(module,model_info.get('class'))
        return findercls(**model_info)

    def find_boxes(self, obj):
        image = obj.image()
        boxes = self.get_YOLO_output(image)
        return sorted(boxes, key=lambda box: float(box.get('confidence',0.0)), reverse=True)


    # since this is the method in the parent class - no filters are inherited, so we fallback to [] + current defined filters
    # or if you need to completely change the order of the filters execution - you can override the method in the child class
    def filter_objects(self,object_list):
        accepted = object_list
        rejected_total = []

        # the filter function returns accepted, rejected but does not have any side effects/mutations on object_list.
        # we dont want data in object_list to get updated.
        # Idea is to let all filters make updates (elsewhere) without 
        # changing the raw object_list - so that we do not muddy up the object_list using filters whose working we are not aware about....
        # but we are aware about which filters are used because of the declarative style of defining the filter names before executing the program.
        for f in getattr(super(), "filters", []) + self.filters:
            # filter - returns two new lists: accepted, rejected - so object_list is not mutated
            accepted, rejected = f.filter(accepted)
            rejected_total.extend(rejected)
            # print("\naccepeted",accepted)

        return accepted, rejected_total