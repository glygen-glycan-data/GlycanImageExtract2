"""
class for generating data for IUPAC.
"""
from .finder import Finder
from .glycanannotator import Config
from .model_evaluator import GlycanCompare
from .object_filters import ObjectFilter, MinConfidence

from collections import defaultdict
import sys

class Glycan_Base(Finder):
    finder_class = 'Glycan_Semantics'
    
    def __init__(self,params):
        self.label_type = params.get('label_type')

    def get_label(self,obj):
        iupac = obj.IUPAC()
        if iupac is not None and iupac.strip() != "":
            obj.set('IUPAC',iupac)
        compstr = obj.compstr()
        if compstr is not None and compstr.strip() != "":
            obj.set('composition_str',compstr)
        obj.set('orientation', obj.glycan_orientation())

        if self.label_type == 'iupac':
            return obj.get('IUPAC',"")
        elif self.label_type == 'composition':
            return obj.get('composition_str',"")
        return None

    # add other metadata and lgging details about the glycan
    def add_metadata(self, obj):
        # if obj.has_glycan_errors():
        # 
        #     obj.set('glycan_errors', obj.glycan_errors())
        #     obj.set('log', obj.get_logs())  # maybe keep only glycan errors or logs in the json - currently there are some else checks 
        # obj.set('orientation', obj.glycan_orientation())
        pass

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
        'label_type': 'none',
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

    def log_errors(self,obj):
        obj.do_sanity_checks()

    def find_objects(self, obj):
        # IUPAC and composition are set via Glycan_Base class, when get_label function is used

        self.add_metadata(obj)
        label = self.get_label(obj)
        self.log_errors(obj)

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

class Figure_Postprocessor(Finder):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs.get('name'),
                         cfgmgr=kwargs.get('cfgmgr'))
        
    def set_results(self, figure_semantics, accepted, rejected):
        figure_semantics.set_glycans(accepted, rejected)

    def find_objects(self, obj):
        # no box steps needed for post-processing, just 
        # filter or manipulate the glycans
        obj_list = obj.glycans()
        obj_list.sort(key=lambda o: -o.get('confidence',0.0))
        accepted, rejected = self.filter_objects(obj_list)
        self.set_results(obj, accepted, rejected)

class GlycanModifierFilter(ObjectFilter):
    def recompute_iupac(glycan):
        glycan.unset('IUPAC')
        glycan.unset('composition_str')
        glycan.reset_glycan_errors()
        YOLO_Glycan().find_objects(glycan)

class GlycanOrientationFilter(GlycanModifierFilter):
    def __init__(self, mode="ERROR", cutoff=5, majority=0.75):
        assert mode in ("ERROR","CORRECT")
        self.mode = mode
        self.cutoff = cutoff
        self.majority = majority

    def filter(self, glycans):
        freq = defaultdict(int)
        for glycan in glycans:
            if not glycan.has_glycan_errors() and \
                glycan.get('IUPAC',""):            
                    orientation = glycan.get('orientation')
                    if orientation is not None:
                        freq[orientation] += 1
        
        total = sum(freq.values())
        if total <= self.cutoff:
            # do nothing
            return glycans, []
        maxor = max(freq, key=freq.get, default=None)
        if freq[maxor]/total < self.majority:
            # do nothing
            return glycans, []

        # use the majority orientation to flag or correct 
        # glycans likely to be misoriented
        for i,glycan in enumerate(glycans):
            if not glycan.has_glycan_errors() and \
                glycan.get('IUPAC',""):
                    if glycan.get('orientation') != maxor:
                        if self.mode == "ERROR":
                            glycan.add_glycan_error(f"Orientation mismatch: {glycan.get('orientation')} vs majority {maxor}")
                        elif self.mode == "CORRECT":
                            glycan.log(f"setting correct orientation from {glycan.get('orientation')} to {maxor}")
                            # potentially, change the redend and allow the 
                            # IUPAC computation to fix the orientation
                            # and call self.recompute_iupac(glycan) to 
                            # update all derived results.
                            glycan.set('orientation',maxor)
        
        return glycans, []

class GlycanMinConfidence(Figure_Postprocessor):
    filters = [ MinConfidence(0.5) ]

class GlycanOrientation(Figure_Postprocessor):
    filters = [ GlycanOrientationFilter(mode="ERROR") ]

