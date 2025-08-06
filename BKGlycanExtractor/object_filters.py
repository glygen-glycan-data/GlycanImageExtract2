from collections import defaultdict

from BKGlycanExtractor import CompareBoxes

class ObjectFilter:
    '''        
    Filters must:
    - Not mutate the input list in place.
    - Not mutate the objects inside the list.
    - Return (accepted, rejected) where rejected items can contain reasons for rejection.
    '''

    def filter(self, objlist):
        '''
        objlist is always sorted according to confidence scores of detected boxes 
        '''
        raise NotImplementedError


# Not sure if this is a good design strategy to use?
# I have created a Base class for Monos and Roots - because they share similar code and alternatives are handled based on iou (overlapping detections).
# Only difference is how the nested for loop is handled - for monos --> all the detected items can be considered as the primary mono (unless we find a overlap).
# --> for root: only the first detected item (highest confidece) will eb the primary root....and if they are other overlapping items - they will be considered as alternatives/rejected.
# Concern: def is_primary() - is this method clear and easy to understand/logical?
class OverlapFilterBase(ObjectFilter):
    '''
    Base Class meant for monos and root.
    Filters boxes and adds them to the rejected list if they overlap with the primary detected(selected) items
    '''


    def __init__(self):
        self.cb = CompareBoxes()


    def rejected_metadata(self, primary, alternative):
        '''
        Used to format metadata for items which overlap with the primary mono/root 
        (basically these overlapping items will be added to the rejected list)
        '''
        raise NotImplementedError

    
    def is_primary(self,accepeted):
        '''
        used to check if the detected box could be a potential primary mono/root or not
        '''
        raise NotImplementedError


    def filter(self,objlist):
        accepted = []
        rejected = []

        removed = set()


        for i1 in range(0,len(objlist)):
            if i1 in removed or not self.is_primary(i1):
                continue
            m1 = objlist[i1]
            accepted.append(m1)

            for i2 in range(i1+1,len(objlist)):
                if i2 in removed:
                    continue
                m2 = objlist[i2]

                # SOLVE ERROR: AttributeError: 'NoneType' object has no attribute 'get'
                # 127.0.0.1 - - [25/Jul/2025 11:44:33] "GET /get_job_status/nkvtypuo3i HTTP/1.1" 200 -
                # 127.0.0.1 - - [25/Jul/2025 11:44:34] "GET /retrieve?list_ids=["nkvtypuo3i"] HTTP/1.1" 200 -
                if self.cb.have_intersection(m1.get('box'),m2.get('box')):
                    # obj.monosaccharide(m2['id'])['iou'] = self.cb.iou(m1.get('box'),m2.get('box'))
                    # obj.make_alternative_mono(m1['id'],m2['id'])

                    # add m2 as an alternative to m1
                    # and add m2 to rejected
                    # m2['iou'] = self.cb.iou(m1.get('box'),m2.get('box'))
                    # m1['alternative_mono'] = m2 

                    rejected.append(self.rejected_metadata(m1, m2))
                    removed.add(i2)

        # print("\nrejected",rejected)
        return accepted, rejected



class FilterAlternativeMonos(OverlapFilterBase):
    '''
    check for overlaps, necessarily with different classes, keep
    highest confidence as primary - do not expect bad
    cases, predictions are not expected to partially overlap
    '''

    def is_primary(self, mono_count):
        # all the monos in the list of detected monos can be primary unless overlapping.
        return True



    def rejected_metadata(self, primary, alterative):

        # print("\nalterative",alterative)
        return {
            'alternative': alterative, 
            'primary_mono_id': primary.get('id'),
            'iou': self.cb.iou(primary.get('box'),alterative.get('box')), 
            'reason': 'This mono overlaps with the primary selected mono'
        }
    
            

class FilterAlternativeRoots(OverlapFilterBase):
    '''
    This class is meant for root.
    We might receive one or multiple potential roots in the objlist.
    Aim is to the select the best highest confidence prediction as the root (added to the accepted list)
    For the rest of the predictions (if they exist), we will check if those boxes overlap with the primary root and
    if true - they will be added to the rejected list
    '''

    def is_primary(self, mono_count):
        # for root - the first detected item is always considered the primary root
        return mono_count == 0


    def rejected_metadata(self, primary, alterative):

        return {
            'alternative': alterative, 
            'primary_root_id': primary.get('id'),
            'iou': self.cb.iou(primary.get('box'),alterative.get('box')), 
            'reason': 'This mono overlaps with the primary selected root'
        }



# # make generic for root and links
# # iou in __init__
# class FilterAlternativeMonos(ObjectFilter):
#     '''
#     check for overlaps, necessarily with different classes, keep
#     highest confidence as primary - do not expect bad
#     cases, predictions are not expected to partially overlap
#     '''
#     def __init__(self):
#         self.cb = CompareBoxes()

#     def filter(self,objlist):

#         accepted = []
#         rejected = []

#         # accepted should contain everything from objlist - and if anything from the objlist is supposed
#         # to be an alternative - it should be removed from the accepeted and added to the rejected?
#         # and at the same time be added as an alterative to the accepted - what is a good way to do this?
#         # if theres an overlap with m2 - then add that as alt in accepted and add it to rejected
#         # add it to removed also - so that we dont have to deal with it again

#         # accepted should contain all those monos which are the primary monos
#         # rejected should contain all the alterative monos - if true

#         removed = set()
#         for i1 in range(0,len(objlist)):
#             if i1 in removed:
#                 continue
#             m1 = objlist[i1]
#             accepted.append(m1)

#             for i2 in range(i1+1,len(objlist)):
#                 if i2 in removed:
#                     continue
#                 m2 = objlist[i2]
#                 if self.cb.have_intersection(m1.get('box'),m2.get('box')):
#                     # obj.monosaccharide(m2['id'])['iou'] = self.cb.iou(m1.get('box'),m2.get('box'))
#                     # obj.make_alternative_mono(m1['id'],m2['id'])

#                     # add m2 as an alternative to m1
#                     # and add m2 to rejected
#                     # m2['iou'] = self.cb.iou(m1.get('box'),m2.get('box'))
#                     # m1['alternative_mono'] = m2 

#                     iou = self.cb.iou(m1.get('box'),m2.get('box'))
#                     rejected.append({
#                         **m2, 
#                         'primary_mono_id': m1['id'],
#                         'iou': iou, 
#                         'reason': 'This mono overlaps with the primary selected mono'
#                     })
#                     removed.add(i2)

#         return accepted, rejected


# class FilterAlternativeRoots(ObjectFilter):
#     '''
#     This class is meant for root.
#     We might receive one or multiple potential roots in the objlist.
#     Aim is to the select the best highest confidence prediction as the root (added to the accepted list)
#     The rest of the predictions will be the alternative roots which will be added to the rejected list
#     '''
    
#     def filter(self, objlist):
#         '''
#         objlist is always sorted based on confidence in descending order
#         '''

#         accepted = []
#         rejected = []

#         if objlist:
#             accepted.append(objlist[0])     # first item has highest confidence - so most likely to be the root
#             m1 = objlist[0]

#             for i in range(1,len(objlist)):
#                 m2 = objlist[i]
#                 if self.cb.have_intersection(m1.get('box'),m2.get('box')):
                    
#                     iou = self.cb.iou(m1.get('box'),m2.get('box'))

#                     rejected.append({
#                         **m2, 
#                         'primary_root_id': m1['id'],
#                         'iou': iou, 
#                         'reason': 'This mono overlaps with the primary selected root'
#                     })


#         return accepted, rejected

            
# should take data lists and return data lists
# not pull things out, make changes and then add changes back to it
class FilterRepeatedLinks(ObjectFilter):
    '''
    This class is meant for link finders only.
    Used to track which monosaccharides are linked together, so that the model doesnt
    create a duplicate link while mapping monos present in a detected link box.

    The mono_ids link mappings are performed based on a greedy confidence scores strategy.
    '''

    def filter(self, objlist):
        '''
        objlist contains a list of links (mono_id pairs)
        goal is to eliminate the repeated pairs (so if a mono_id pair is taken - do not take it again)
        The objlist has links arranged according to the confidence (in descending order)...
        '''
        accepted = []
        rejected = []

        id_added = defaultdict(set)  # Track used mono pairs

        for link in objlist:
            id1, id2 = link.get('mono_ids')

            if id2 in id_added[id1]:
                rejected.append({'link':link,'reason':'duplicate link/mono_ids pairs found.'})
                continue
            
            accepted.append(link)
            
            id_added[id1].add(id2)
            id_added[id2].add(id1)

        return accepted, rejected


# should take data lists and return data list
# not pull things out, make changes and then add changes back to it
class FilterTreeLinks(ObjectFilter):   
    """
    Derieved class - used to eliminate extra links (which shouldnt be part of the tree) 
    and links that form cycles in the glycan structure.
    """

    def filter(self,objlist):
        """
        Accepts links in descending order of confidence, but skips any link that
        forms a cycle. This builds a maximum-confidence spanning tree (Kruskals algo).
        Logs the skipped/cycle causing links in semantics["non_tree_links"].
        """
        accepted = []
        rejected = []

        # links = objlist
        mono_ids_set = {m_id for link in objlist for m_id in link.get("mono_ids")}
        num_nodes = len(mono_ids_set)

        parent = {node: node for node in mono_ids_set}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x == root_y:
                return False  # Cycle!
            parent[root_y] = root_x
            return True

        for link in objlist:
            if len(accepted) == num_nodes - 1:
                # Tree is complete, any remaining edges are rejected
                # this mutates/adds "reason" to the objlist - is that okay? - I am leaving
                # it here for now for debugging, but if we want to maintain obj_list as a 
                # clean raw list - then take it out and maintain the reason in the rejected list only
                # link["reason"] = "Tree is already complete. Extra edge dropped."
                # obj.link_cycle(link)
                rejected.append({'link':link, 'reason': 'Tree is already complete. Extra edge dropped.'})
                continue

            u, v = link.get("mono_ids")
            if union(u, v):
                accepted.append(link)
            else:
                # this mutates/adds "reason" to the objlist - is that okay?
                # link["reason"] = "Cycle detected. Dropped the edge."
                # rejected.append(link)
                rejected.append({'link': link,'reason':'Cycle detected. Dropped the edge.'})
                # obj.link_cycle(link)

        return accepted, rejected

        # ObjectFilters should not stash anything in the semantics data-object
        # (but you can mark (and keep)) the objects that are not accepted - mark them as rejected
        # Or perhaps get, and return two lists? --> accepted and rejected
        # objlist[:] = accepted

        # instead pass two lists in and pass out two list - with accpeted and rejected
        # 1) all items and empty list
        # 2) returns the list that you keep, and returns the list that you removed

        # filters just need an (in and out loop)

        # Removing a repeated pair of ids becomes a filter - this is an objectFilter based on Confidence


        # Update links in the object
        # obj.set_undirected_links(accepted)

        # Disjointed tree error
        # if len(objlist) < num_nodes - 1:
        #     obj.glycan_error("Unable to build the structure due to missing edges")