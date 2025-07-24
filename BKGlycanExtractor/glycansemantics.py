"""
class for generating data for IUPAC.
"""
from .finder import Finder
from .glycanannotator import Config


class Glycan_Base(Finder):
    finder_class = 'Glycan_Semantics'
    
    def __init__(self,params):
        self.label_type = params.get('label_type')

    def get_label(self,obj):
        res = obj.compstr()
        if res:
            obj.set('composition_str',res)
        res = obj.IUPAC()
        if res:
            obj.set('IUPAC',res)
        if self.label_type == 'composition':
            return obj.get('composition_str')
        return obj.get('IUPAC')


class YOLO_Glycan(Glycan_Base):

    defaults = {
        'label_type': 'composition'
    }

    def __init__(self,**kwargs):
        params = dict(
           label_type = Config.get_param('label_type', Config.STR, kwargs, self.defaults),
        )
        super().__init__(params)

    def get_confidence(self,obj):
        '''
        Returns the min confidence values after all finders are executed.
        '''
        return min(
            [mono.get('confidence') for mono in obj.monosaccharides() if mono.get('confidence') is not None] +
            ([obj.root().get('confidence')] if obj.root() and obj.root().get('confidence') is not None else []) +
            [link.get("confidence") for link in obj.all_links() if link.get("confidence") is not None],
            default=1.1  # or any appropriate fallback confidence
        )

    def find_objects(self, obj):
        obj.set('classlabel',self.get_label(obj) )
        obj.set('center',obj.glycan_box().center())     # helps for proximity
        obj.set('confidence', self.get_confidence(obj))
        return obj.glycan()

    def find_boxes(self):
        pass


class Known_Glycan(Glycan_Base): 
    defaults = {
        'label_type': 'composition'
    }

    def __init__(self,**kwargs):
        params = dict(
           label_type = Config.get_param('label_type', Config.STR, kwargs, self.defaults),
        )
        super().__init__(params)

    def find_objects(self, obj):
        obj.set('classlabel',self.get_label(obj) )
        obj.set('center',obj.glycan_box().center())     # helps for proximity
        return obj.glycan()
        
    def find_boxes(self):
        pass


    
