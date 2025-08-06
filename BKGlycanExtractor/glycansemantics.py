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
        iupac = obj.IUPAC()
        if iupac:
            obj.set('IUPAC',iupac)
        compstr = obj.compstr()
        if compstr is not None:
            obj.set('composition_str',compstr)
        if self.label_type == 'composition':
            return obj.get('composition_str',"")
        return obj.get('IUPAC',"")


# create two different class for IUPAC AND COMPOSITION - not like this 
# defaults = {
#         'label_type': 'composition'
#     }
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
            [mono.confidence() for mono in obj.monos() if mono.confidence() is not None] +
            ([obj.root().confidence()] if obj.root() and obj.root().confidence() is not None else []) +
            [link.confidence() for link in obj.all_links() if link.confidence() is not None],
            default=1.1  # or any appropriate fallback confidence
        )
    
    # this should also add IUPAC/COMPOSITION in the semnatics - it should be in the pipeline
    def find_objects(self, obj):
        obj.set('classlabel',self.get_label(obj) )
        obj.set('center',obj.center())     # helps for proximity
        obj.set('confidence', self.get_confidence(obj))
        return [ obj ]

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
        obj.set('classlabel',self.get_label(obj))
        obj.set('center',obj.box().center())     # helps for proximity
        return [ obj ]
        
    def find_boxes(self):
        pass


    
