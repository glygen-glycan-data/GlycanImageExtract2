"""
class for generating data for IUPAC.
"""
from .finder import Finder
from .glycanannotator import Config
from .model_evaluator import GlycanCompare

class Glycan_Base(Finder):
    finder_class = 'Glycan_Semantics'
    
    def __init__(self,params):
        self.label_type = params.get('label_type')

    def get_label(self,obj):
        iupac = obj.IUPAC()
        if iupac and not obj.has_glycan_errors():
            obj.set('IUPAC',iupac)
        compstr = obj.compstr()
        if compstr is not None and compstr.strip() != "":
            obj.set('composition_str',compstr)
            
        if self.label_type == 'none':
            return None
        elif self.label_type == 'composition':
            return obj.get('composition_str',"")
        return obj.get('IUPAC',"")

    # add other metadata and lgging details about the glycan
    def add_metadata(self, obj):
        if obj.has_glycan_errors():
            obj.set('glycan_errors', obj.glycan_errors())
            obj.set('log', obj.get_logs())  # maybe keep only glycan errors or logs in the json - currently there are some else checks 

        obj.set('orientation', obj.glycan_orientation())

    def semantic_compare(self,**kwargs):
        return GlycanCompare(**kwargs)


# TODO create two different class for IUPAC AND COMPOSITION - not like this 
# defaults = {
#         'label_type': 'composition'
#     }
# TODO - move iupac() implementation from semnatics.py to this place and set iupac, composition and other
# details here 
class YOLO_Glycan(Glycan_Base):

    defaults = {
        'label_type': 'none'
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
    
    def find_objects(self, obj):
        # IUPAC and composition are set via Glycan_Base class, when get_label function is used

        self.add_metadata(obj)
        
        label = self.get_label(obj)

        if obj.has_glycan_errors():
            return []

        if label:    
            obj.set('classlabel',label)
            obj.set('center',obj.center())     # helps for proximity
            obj.set('confidence', self.get_confidence(obj))
            return [ obj ]
        return []

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


    
