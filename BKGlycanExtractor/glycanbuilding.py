# -*- coding: utf-8 -*-
"""
build glycan structure from monosaccharide information

all classes need a build method which takes a GlycanMonoInfo object
returns a glycoCT
"""

import logging

from .pygly3.MonoFactory import MonoFactory
from .pygly3.Monosaccharide import Anomer
from .pygly3.Glycan import Glycan
from .pygly3.Monosaccharide import Linkage

class GlycanBuilder:
    def __call__(self, data, **kw):
        return self.build(data, **kw)
    
    def __init__(self):
        pass
    
    def build(self, data, **kw):
        raise NotImplementedError
        
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.glycanbuilding')

# recursive building from root -> children -> children etc        
class RootedTreeBuilder(GlycanBuilder):
    def build(self, mono_info):
        mf = MonoFactory()
        
        monos_list = mono_info.get_monos()
        
        root_id, root_idx = None, None
        
        for i, mono in enumerate(monos_list):
            if mono.is_root():
                root_id = mono.get_ID()
                root_idx = i
                break
        if root_id is None:
            return None
        
        root_type = "".join([char for char in root_id if char.isalpha()])
        root_node = mf.new(root_type)
        
        fail_safe = 0
        root_node = self.build_tree(
            monos_list, root_idx, root_node, fail_safe
            )[2]
        
        if root_node is not None:
            root_node.set_anomer(Anomer.missing)
            g = Glycan(root_node)
            glycoCT = g.glycoct()
            printstr = f"GlycoCT:\n{glycoCT}"
            self.logger.info(printstr)
        else:
            glycoCT = None
        return glycoCT
        
    def build_tree(self, monos_list, root_idx, root_node, fail_safe):
        # print("full mono list:", [mono.get_ID() for mono in monos_list])
        fail_safe += 1
        mf = MonoFactory()
        
        root_mono = monos_list[root_idx]
        # print("root:", root_mono.get_ID())
        
        # print(root_mono)
        
        child_list = list(set(root_mono.get_linked_monos()))
        
        # print(child_list)
        
        if fail_safe > len(monos_list) - 1:
            return None, None, None, fail_safe
        
        if child_list == []:
            child_mono = mf.new("".join([
                char for char in root_mono.get_ID() if char.isalpha()
                ]))
            child_mono.set_anomer(Anomer.missing)
            root_node.add_child(
                child_mono, parent_type=Linkage.oxygenPreserved, 
                child_pos=1, child_type=Linkage.oxygenLost
                )
            return monos_list, root_idx, root_node, fail_safe
        
        else:
            for child in child_list:
                for i, mono in enumerate(monos_list):
                    if root_mono.get_ID() in mono.get_linked_monos():
                        mono.remove_linked_mono(root_mono.get_ID())
                    if mono.get_ID() == child:
                        name_temp = "".join([
                            char for char in child if char.isalpha()
                            ])
                        child_mono = mf.new(name_temp)
                        child_mono.set_anomer(Anomer.missing)
                        if mono.get_linked_monos() != []:
                            _,_, child_mono, fail_safe = self.build_tree(
                                monos_list, i, child_mono, fail_safe
                                )
                        if ((fail_safe > len(monos_list) - 1)
                            or (child_mono is None) 
                            or (root_node is None)):
                            return None, None, None, fail_safe
                        if name_temp in ("NeuAc"):
                            root_node.add_child(
                                child_mono, 
                                parent_type=Linkage.oxygenPreserved, 
                                child_pos=2, child_type=Linkage.oxygenLost
                                )
                        else:
                            root_node.add_child(
                                child_mono, 
                                parent_type=Linkage.oxygenPreserved, 
                                child_pos=1, child_type=Linkage.oxygenLost
                                )
                        break    # once we find the child in mono_info.monos
        if ((fail_safe > len(monos_list) - 1)
            or (child_mono is None)
            or (root_node is None)):
            self.logger.info("fail safe activated")
            return None, None, None, fail_safe
        else:
            return monos_list, root_idx, root_node, fail_safe