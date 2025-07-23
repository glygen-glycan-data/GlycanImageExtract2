'''
Subclasses inherting from Finder must define labels
'''

from .yolomodels import YOLOModel
import sys

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

    def clear(self):
        if isinstance(self,YOLOModel):
            self.clear_model()
 
