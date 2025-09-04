from collections import defaultdict

from BKGlycanExtractor import CompareBoxes

class ObjectFilter:
    '''        
    Filters must:
    - Not mutate the input list in place.
    - Not mutate the objects inside the list.
    - Return (accepted, rejected) where rejected items can contain reasons for rejection.
    '''


    def make_rejection(self, obj, reason, **metadata):
        return {
            'object': obj,
            'reason': reason,
            **metadata
        }

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
class FilterOverlaps(ObjectFilter):
    '''
    Base class to check overlapping boxes and filter them into accepted and rejected lists.
    Filter can be used for Mono
    '''

    def __init__(self):
        self.cb = CompareBoxes()


    def filter(self, objlist):
        '''
        Considers objects with the highest confidence (stored in accepeted list). 
        Any other objects which intersect (based on IOU) with the chosen highest confidence object is stored in the rejected list.
        '''
        accepted = []
        rejected = []
        removed = set()

        # sort the object list based on confidence in reverse order
        for i1 in range(len(objlist)):
            if i1 in removed:
                continue
            m1 = objlist[i1]
            accepted.append(m1)
            for i2 in range(i1 + 1, len(objlist)):
                if i2 in removed:
                    continue
                m2 = objlist[i2]
                if self.cb.have_intersection(m1.box(), m2.box()):
                    # add confidence for both the boxes and add box for reference isnetad of primary_id
                    rejected.append(
                        self.make_rejection(
                            m2,
                            confidece=m2.get['confidence'],
                            reason="This object overlaps with the primary selected object",
                            primary=box,
                            iou=self.cb.iou(m1.box(), m2.box())
                        )
                    )
                    removed.add(i2)

        return accepted, rejected

class DiscardClass(ObjectFilter):
    def __init__(self,tokeep=None,todiscard=None):
        self.tokeep = tokeep
        self.todiscard = todiscard
        assert self.tokeep is None or self.todiscard is None

    def filter(self, objlist):

        accepted,rejected = [],[]
        for obj in objlist:
            if self.tokeep is not None and obj.classlabel() in self.tokeep:
                accepted.append(obj)
            if self.todiscard is not None and obj.classlabel() not in self.todiscard:
                accepted.append(obj)

        return accepted,rejected

class SingleBest(ObjectFilter):
    def filter(self, objlist):
        if len(objlist) == 0:
            return [],[]
        return objlist[0:1],objlist[1:]

# make this a seperate class
class RootFilter(ObjectFilter):
    '''
    Specialized filter for root objects.
    Keeps only one root with highest confidence, rest go to rejected.
    (rejected includes both root and non-root objects).

    Considers objects with the highest confidence (stored in accepeted list). 
    Any other objects which intersect (based on IOU) with the chosen highest confidence object is stored in the rejected list.
        
    '''

    def filter(self, objlist):

        # Uses ParentClass for the below filtering logic:
        # Considers objects with the highest confidence (stored in accepeted list). 
        # Any other objects which intersect (based on IOU) with the chosen highest confidence object is stored in the rejected list.
        accepted, rejected = super().filter(objlist)

        # Filter accepted roots (classid == 0)
        roots = [obj for obj in accepted if obj.get('classlabel') == 0]
        non_roots = [obj for obj in accepted if obj.get('classlabel') != 0]

        if roots:
            # Keep only highest confidence root in accepted
            roots = sorted(roots, key=lambda x: x.get('confidence', 0), reverse=True)
            accepted_root = roots[0]

            # Move all other root objects to rejected list with reason
            for r in roots[1:]:
                rejected.append(
                    self.make_rejection(
                        r,
                        reason="Multiple roots found; this root rejected",
                        primary_id=accepted_root.get('id')
                    )
                )
            accepted = [accepted_root]
        else:
            accepted = []

        # If no roots at all, accepted could be empty; that's fine.
        # Return a single root object or None
      
        return accepted, rejected


class RemoveNonRootsFilter(FilterOverlaps):
    '''
    Specialized filter for root objects.

    Inherits FilterOverlaps to get accepted/rejected after overlap filtering.
    Then ensures accepted contains only one root (highest confidence),
    and rejected gets all other root objects that were accepted but not chosen.
    Non-root objects remain untouched in accepted and not moved to rejected here.
    '''

    def filter(self, objlist):

        # Uses ParentClass for the below filtering logic:
        # Considers objects with the highest confidence (stored in accepeted list). 
        # Any other objects which intersect (based on IOU) with the chosen highest confidence object is stored in the rejected list.
        accepted, rejected = super().filter(objlist)

        # clear the rejected list
        rejected = []

        # Extract roots and non-roots from accepted
        roots = [obj for obj in accepted if obj.get('classid') == 0]
        non_roots = [obj for obj in accepted if obj.get('classid') != 0]

        if roots:
            # Keep only highest confidence root in accepted
            roots = sorted(roots, key=lambda x: x.get('confidence', 0), reverse=True)
            accepted_root = roots[0]

            # Move all other root objects to rejected list with reason
            for r in roots[1:]:
                rejected.append(
                    self.make_rejection(
                        r,
                        reason="Multiple roots found; this root rejected",
                        primary_id=accepted_root.get('id')
                    )
                )
            accepted = [accepted_root]
        else:
            accepted = []

        # If no roots at all, accepted could be empty; that's fine.
        # Return a single root object or None
        if accepted:
            # Return single root object instead of list
            return accepted, rejected
        else:
            return None, rejected
            
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