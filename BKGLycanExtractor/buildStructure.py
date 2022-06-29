from .pygly3.MonoFactory import MonoFactory
from .pygly3.Monosaccharide import Anomer
from .pygly3.Glycan import Glycan
from .pygly3.Monosaccharide import Linkage

class GlycanStructBuilder:
    def __call__(self,**kw):
        return self.build(**kw)
    def __init__(self):
        pass
    def build(self,**kw):
        raise NotImplementedError

class CurrentBuilder(GlycanStructBuilder):
    def build(self,mono_dict = None):
        mf = MonoFactory()
        #print(mono_dict)
        #aux_list = [(mono_id, mono_dict[mono_id][4:5]) for mono_id in mono_dict.keys()]
        try:
            #print([mono_id for mono_id in mono_dict.keys()])
            root_id = [mono_id for mono_id in mono_dict.keys() if mono_dict[mono_id][5] == "root"][0]
        except IndexError:
            return None
        #print("aux_list", aux_list)

        root_type = "".join([c for c in root_id if c.isalpha()])
        #print(f"root id: {root_id}, type: {root_type}, child = {mono_dict[root_id][4]}")
        root_node = mf.new(root_type)

        #child_list = mono_dict[root_id][4]
        #print("##########################")
        # need stop recursion here #####################
        fail_safe=0
        root_node=self.build_tree(mono_dict,root_id,root_node, fail_safe)[2]
        #unknonw root properties

        if root_node != None:
            #print(root_node)
            root_node.set_anomer(Anomer.missing)
            #root_node.set_ring_start(None)
            #root_node.set_ring_end(None)
            g = Glycan(root_node)
            glycoCT =g.glycoct()
        elif root_node ==None:
            #print("Error in glycan structure")
            glycoCT = None

        return glycoCT
    def build_tree(self,mono_dict,root, root_node, fail_safe):
        # mono_dict[mono id] = {contour, point at center, radius, bounding rect, linkages, root or child}
        # variables:
        fail_safe+=1
        #print(f"Current:{fail_safe} " + root, end=" ")
        mf = MonoFactory()
        #current_type = "".join([c for c in root if c.isalpha()])
        child_list = list(set(mono_dict[root][4]))
        #print("Child_list", child_list)
        # case 0: no child at all return build mono
        if fail_safe > len(mono_dict.values())-1:
            return None, None,None,fail_safe
        if child_list==[]:
            child_mono = mf.new("".join([c for c in root if c.isalpha()]))
            child_mono.set_anomer(Anomer.missing)
            #print(f"adding leaves: {root}")
            root_node.add_child(child_mono,parent_type=Linkage.oxygenPreserved,child_pos=1,
                      child_type=Linkage.oxygenLost)

            return mono_dict, root,root_node,fail_safe

        # case 1: there are child, do for loop
        if child_list != []:


            for child_id in child_list:
                # remove parent link from child
                if root in mono_dict[child_id][4]:
                    mono_dict[child_id][4].remove(root)
                name_temp = "".join([c for c in child_id if c.isalpha()])
                child_mono = mf.new(name_temp)
                child_mono.set_anomer(Anomer.missing)
                if mono_dict[child_id][4] != []:
                    _,_,child_mono,fail_safe = self.buildtree(mono_dict,child_id,child_mono,fail_safe)
                if fail_safe > len(mono_dict.values())-1 or child_mono ==None or root_node ==None:
                    return None, None,None,fail_safe
                if name_temp in ("NeuAc"):#("Glc", "Gal", "GlcNAc"):
                    root_node.add_child(child_mono,parent_type=Linkage.oxygenPreserved,child_pos=2,
                          child_type=Linkage.oxygenLost)
                else:
                    root_node.add_child(child_mono, parent_type=Linkage.oxygenPreserved, child_pos=1,
                                        child_type=Linkage.oxygenLost)
        if fail_safe > len(mono_dict.values())-1 or child_mono == None or root_node == None:
            print("fail safe activated")
            return None, None,None,fail_safe
        else:
            return mono_dict, root, root_node,fail_safe