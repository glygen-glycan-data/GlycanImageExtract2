import matplotlib
# matplotlib.use('tkagg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os
import re
import copy
import cv2
from collections import defaultdict
import multiprocessing
import sys
import time
import queue
from . import Image_Manager
from .semantics import Figure_Semantics, Glycan_Semantics
from .bbox import BoundingBox
from .compareboxes import CompareBoxes
from .debug_methods import DebugMode

from .build_pipeline import BuildPipeline

from .glycanannotator import Config_Manager


class Utility:

    @staticmethod
    def floor_precision(value, precision):
        scale = 10 ** precision
        return math.floor(value * scale) / scale


    def update_metrics(results,threshold, TP, FP, FN):
        prec_threshold = str(Utility.floor_precision(threshold, 8))
        results.setdefault((prec_threshold), {"TP": 0, "FP": 0, "FN": 0})
        results[prec_threshold]["TP"] += TP
        results[prec_threshold]["FP"] += FP
        results[prec_threshold]["FN"] += FN


    @staticmethod
    def build_adjacency_list(mono_semantics):
        adj = defaultdict(list)
        monos = mono_semantics.semantics['monos']

        for mono_id, mono_data in monos.items():
            for link_data in mono_data['links']:
                if isinstance(link_data, list):  # Handle case with confidence value - for pred data case
                    linked_id, conf = link_data
                    try:
                        adj[mono_data['box']].append([monos[linked_id]['box'], Utility.floor_precision(conf,8)])
                    except KeyError as k:
                        print("Key doesnt exist:", k)
                else:  # Handle case without confidence - for known data case
                    linked_id = link_data
                    adj[mono_data['box']].append(monos[linked_id]['box'])

        return adj


class BoxCompare:
    def __init__(self, iou):
        self.iou = iou
        self.compare_boxes = CompareBoxes(**dict(detection_threshold=self.iou,overlap_threshold=self.iou,containment_threshold=self.iou))


    def filtered_data(self,pred_boxes,known_boxes):
        edges = []
        confidence_scores = []

        # Compute all potential matches between known and detected boxes using IOU as a constraint
        for dbox_id, dbox in enumerate(pred_boxes):
            matched = False
            for tbox_id, tbox in enumerate(known_boxes):
                computed_iou = self.compare_boxes.iou(tbox, dbox) if self.compare_boxes.have_intersection(tbox, dbox) else 0.0

                if computed_iou >= self.iou:
                    edges.append(dict(tbox=tbox_id,dbox=dbox_id,iou=computed_iou,conf=Utility.floor_precision(dbox.get('confidence'),8),cls=(tbox.get('classid'),dbox.get('classid'))))
            
            confidence_scores.append(Utility.floor_precision(dbox.get('confidence'),8))

        return edges, set(confidence_scores)
            


    def compare(self, pred_boxes, known_boxes, **kwargs):

        results = {}
        edges = []

        edges, confidence_scores = self.filtered_data(pred_boxes,known_boxes)

        
        for threshold in sorted(confidence_scores):
            TP, FP, FN = 0, 0, 0

            matched_gt = set()  # Set of matched ground truth IDs
            matched_pred = set()  # Set of matched predicted box IDs

            # Greedy matching based on Confidence
            for item in sorted(edges, key=lambda x: (-x['conf'], -float(x['iou']))):
                if item['conf'] < threshold:
                    break
                if item['tbox'] in matched_gt:
                    continue
                if item['dbox'] in matched_pred:
                    continue

                matched_gt.add(item['tbox'])
                matched_pred.add(item['dbox'])

                if item['cls'][0] == item['cls'][1]:
                    TP += 1
                else:
                    FP += 1
                    FN += 1

            FN += len(known_boxes) - len(matched_gt)
            FP += len([pred_box for pred_box in pred_boxes if Utility.floor_precision(pred_box.get('confidence'),8) >= threshold]) - len(matched_pred)

            Utility.update_metrics(results,threshold,TP,FP,FN)
            
        last_threshold = 1.00000001
        Utility.update_metrics(results,last_threshold,0,0,FN+1)

        return results

   


class MonosCompare:

    def __init__(self, radius_threshold):
        self.radius_threshold = radius_threshold


    def filtered_predictions(self,pred_data,threshold):

        for id, item in pred_data.semantics['monos'].copy().items():
            # condition to remove data if it is below the threshold
            if Utility.floor_precision(item['box'].get('confidence'), 8) < threshold:
                del pred_data.semantics['monos'][id] 


    def filtered_data(self,pred_data,known_data):
        edges = []
        confidence_scores = []

        for pred_id, pred_item in enumerate(pred_data.monosaccharides()):

            for known_id, known_item in enumerate(known_data.monosaccharides()):
                distance = Evaluator.euclidean_distance(known_item['box'], pred_item['box'])
                proximity = self.radius_threshold * min(known_item['box'].w, known_item['box'].h)

                if distance <= proximity:
                    edges.append(dict(tbox=known_id,dbox=pred_id,distance=distance,conf=Utility.floor_precision(pred_item['box'].get('confidence'),8),cls=(known_item.get('classid'),pred_item.get('classid'))))

            confidence_scores.append(Utility.floor_precision(pred_item['box'].get('confidence'),8))

        return edges, set(confidence_scores)

    def confidence_data(self, pred_data):
        confidence_scores = [
            Utility.floor_precision(pred_item['box'].get('confidence'), 8)
            for pred_item in pred_data.monosaccharides()
            ]

        return set(confidence_scores)


    def compare(self, pred_data, known_data, **kwargs):
        results = {}
        edges = []

        whole_glycan = kwargs.get('whole_glycan',False)

        # For Root Compare - if a root is not detected --> it is treated as a FN
        if kwargs.get('p_root') and kwargs.get('p_root') == -1:
            FN = 1
            Utility.update_metrics(results,0.0,0,0,FN)

        else:
            edges, confidence_scores = self.filtered_data(pred_data,known_data)
                

            # Iterate over each critical point and filter predictions based on current threshold
            for threshold in sorted(confidence_scores):

                TP, FP, FN = 0, 0, 0

                matched_gt = set()  # Set of matched ground truth IDs
                matched_pred = set()  # Set of matched predicted box IDs


                for item in sorted(edges,key=lambda x: (-x['conf'], -float(x['distance']))):
                    if item['conf'] < threshold:
                        break
                    if item['tbox'] in matched_gt:
                        continue
                    if item['dbox'] in matched_pred:
                        continue

                    matched_gt.add(item['tbox'])
                    matched_pred.add(item['dbox'])

                    if item['cls'][0] == item['cls'][1]:
                        TP += 1
                    else:
                        FP += 1
                        FN += 1

                FN += len(known_data.monosaccharides()) - len(matched_gt)
                FP += len([item for item in pred_data.monosaccharides() if Utility.floor_precision(item['box'].get('confidence'),8) >= threshold]) - len(matched_pred)


                Utility.update_metrics(results,threshold,TP,FP,FN)

                # if DebugMode.debug:
                #     data = {}
                #     if len(set(matches)) > 1:
                #         data['incorrect_confidence'] = [confidence_threshold]
                #         DebugMode.log_data(DebugMode.curr_image, data)  

        
        last_threshold = 1.00000001
        Utility.update_metrics(results,last_threshold,0,0,FN+1)

        return results


        
class RootCompare:
    def __init__(self, radius_threshold):
        self.mono_compare = MonosCompare(radius_threshold)


    def filtered_predictions(self,pred_data, threshold):

        root_id = pred_data.root()

        pred_semantics = pred_data.semantics['monos'].copy()

        if root_id in pred_semantics:
            if Utility.floor_precision(pred_semantics[root_id]['box'].get('confidence'), 8) < threshold:
                del pred_semantics[root_id]



    def confidence_data(self, pred_data):

        root_id = pred_data.root()

        if root_id == -1:
            return [1.0]

        root_data = pred_data.semantics['monos'][root_id]
        return [Utility.floor_precision(root_data['box'].get('confidence'),8)]


    def root_data(self, data):
        root_id = data.root()

        if root_id != -1:
            mono_data = data.monosaccharide(root_id)

            box = mono_data.get('box')
            if not hasattr(box, 'confidence'):
                setattr(box, 'confidence', 0.0)  # Set default confidence if missing

            data.clear_monos()
            args = dict(id=root_id)
            data.add_mono(mono_data.get('classid'), mono_data.get('symbol'), box, **args)

        return data, root_id

    def compare(self, pred_data, known_data, **kwargs):
        predicted = copy.deepcopy(pred_data)
        known = copy.deepcopy(known_data)

        # Prepare root data - which can be used by the monos comparator
        pred_root_data, p_root = self.root_data(predicted)
        known_root_data, k_root = self.root_data(known)

        args = dict(p_root= p_root, k_root= k_root)
        return self.mono_compare.compare(pred_root_data, known_root_data,**args)



class LinksCompare:
    def __init__(self, radius_threshold):
        self.radius_threshold = radius_threshold

    
    def filtered_predictions(self,pred_data,threshold):


        pred_adj = self.build_adjacency_list(pred_data)

        links_to_delete = set()

        for pred_box, linked_items in pred_adj.items():
            for linked_pred_box, conf in linked_items:
                if Utility.floor_precision(conf, 8) < threshold:
                    links_to_delete.add(pred_box.get('id'))
                    links_to_delete.add(linked_pred_box.get('id'))


        for id, item in pred_data.semantics['monos'].copy().items():
            # condition to remove data if it is below the threshold and iterating the list in reverse so that we do not get index errors
            for i in range(len(item['links'])-1, -1, -1):
                
                if item['links'][i][0] in links_to_delete:
                    item['links'].pop(i)

        

    def confidence_data(self, pred_data):
        confidence_scores = []

        pred_adj = self.build_adjacency_list(pred_data)

        for pred_box, linked_items in pred_adj.items():
            for linked_pred_box, conf in linked_items:
                confidence_scores.append(float(Utility.floor_precision(conf,8)))

        return set(confidence_scores)



    def filtered_data(self,pred_adj, known_adj):

        edges = []
        confidence_scores = []
        matched_pred_ids = set()

        # pred_adj = self.build_adjacency_list(pred_data)

        for pred_box, linked_items in pred_adj.items():
            for linked_pred_box, conf in linked_items:
                for known_box, linked_known_boxes in known_adj.items():
                    for linked_known_box in linked_known_boxes:
                        dist1 = Evaluator.euclidean_distance(known_box, pred_box)
                        dist2 = Evaluator.euclidean_distance(linked_known_box, linked_pred_box)
                        proximity1 = self.radius_threshold * min(known_box.w, known_box.h)
                        proximity2 = self.radius_threshold * min(linked_known_box.w, linked_known_box.h)

                        if dist1 <= proximity1 and dist2 <= proximity2:       
                            known_id_pair = tuple(sorted((known_box.get('id'), linked_known_box.get('id'))))
                            pred_id_pair = tuple(sorted((pred_box.get('id'),linked_pred_box.get('id'))))

                            if pred_id_pair not in matched_pred_ids:
                                edges.append(dict(tbox=known_id_pair,dbox=pred_id_pair,dist1=dist1,dist2=dist2,conf=Utility.floor_precision(conf,8),cls=(sorted((known_box.get('classid'), linked_known_box.get('classid'))),sorted((pred_box.get('classid'), linked_pred_box.get('classid'))))))
                                matched_pred_ids.add(pred_id_pair)


                confidence_scores.append(float(Utility.floor_precision(conf,8)))

        return edges, set(confidence_scores)


    def compare(self, pred_data, known_data, **kwargs):

        results = {}

        other_adj = self.build_adjacency_list(pred_data)
        known_adj = self.build_adjacency_list(known_data)

        # matched_pred_ids = set()

        edges, confidence_scores = self.filtered_data(other_adj, known_adj)

        # confidence_scores = set(confidence_scores)

        for threshold in sorted(confidence_scores):
            TP, FP, FN = 0, 0, 0

            matched_gt = set()  # Set of matched ground truth IDs
            matched_pred = set()  # Set of matched predicted box IDs

            count = 0
            for item in sorted(edges, key=lambda x: (-x['conf'], -x['dist1'], -x['dist2'])):
                if item['conf'] < threshold:
                    break
                if item['tbox'] in matched_gt:
                    continue
                if item['dbox'] in matched_pred:
                    continue

                matched_gt.add(item['tbox'])
                matched_pred.add(item['dbox'])

                if item['cls'][0] == item['cls'][1]:
                    count += 1
                    TP += 1
                else:
                    FP += 1
                    FN += 1

            # No. of links = no. of monos - 1
            FN += len(known_data.monosaccharides()) - 1 - len(matched_gt)

            unmatched_pred_boxes = 0
            pred_id_pair = set()
            for pred_box, linked_items in other_adj.items():
                for linked_pred_box, conf in linked_items:
                    id_pair = tuple(sorted((pred_box.get('id'),linked_pred_box.get('id'))))
                    if id_pair not in pred_id_pair and Utility.floor_precision(conf,8) >= threshold:
                        pred_id_pair.add(id_pair)
                        unmatched_pred_boxes += 1

            FP += unmatched_pred_boxes - len(matched_pred)
            Utility.update_metrics(results,threshold,TP,FP,FN)


        last_threshold = 1.00000001
        Utility.update_metrics(results,last_threshold,0,0,FN+1)

        return results


    def build_adjacency_list(self, mono_semantics):
        adj = defaultdict(list)
        monos = mono_semantics.semantics['monos']

        for mono_id, mono_data in monos.items():
            for link_data in mono_data['links']:
                if isinstance(link_data, list):  # Handle case with confidence
                    linked_id, conf = link_data
                    adj[mono_data['box']].append([monos[linked_id]['box'], Utility.floor_precision(conf,8)])
                else:  # Handle case without confidence - known data case
                    linked_id = link_data
                    adj[mono_data['box']].append(monos[linked_id]['box'])

        return adj


 

class SemanticGlycanCompare:

    def __init__(self, base_pipeline, known_pipeline, radius_threshold):
        self.base_pipeline = base_pipeline
        self.known_pipeline = known_pipeline
        self.radius_threshold = radius_threshold

    # Data structure: Considering pipeline_name incase we want to include multiple pipelines in the future
    # {pipeline_name: image1: {TP: 1, FP:2, FN: 3}, image2: {TP: 1, FP:2, FN: 3}}
    def runall(self,images):

        start_time = time.time()

        observations = {}

        # for loop for all the different Pipelines
        observations[self.base_pipeline.name] = {}

        # self.critical_values = []

        selected_critical_value = 0.77913584
        for idx, image in enumerate(images):
            print("image:",idx, image)

            pred_semantics = self.base_pipeline.run(image)
            glycan = pred_semantics.glycans()[0]

            known_semantics = self.known_pipeline.run(image)
            known_glycan = known_semantics.glycans()[0]

            results = self.compare(glycan,known_glycan,selected_critical_value)

            observations[self.base_pipeline.name][os.path.basename(image)] = results

        # print("\n------>>>>>>>observations",observations)
        # print("\n critical vals",sorted(self.critical_values))

        end_time = time.time()
        Evaluator.plotprecisionrecall(observations, 'Whole_Glycan', **dict(sort_results=False))

        execution_time = end_time - start_time
        print(f"\nExecution Time {execution_time} seconds")



    # get one minimum confidence value for all images
    def compare(self, pred_data, known_data, threhsold):
        compare_classes = [MonosCompare, RootCompare, LinksCompare]

        # known IUPAC
        k_monos, k_root_id = known_data.semantics['monos'], known_data.semantics['root']
        known_IUPAC = Glycan_Semantics.IUPAC(k_monos, k_root_id)


        # comment this for loop - it is only for experimentation to find best confidence threhsold
        # for class_name in compare_classes:      
        #     results = class_name(self.radius_threshold).confidence_data(pred_data)
        #     self.critical_values.append(min(results))
            # print("\nresults:",class_name, results)
        

        # Save the original state of pred_data
        original_pred_data = copy.deepcopy(pred_data)

        TP, FP, FN = 0, 0, 0
        # for conf in sorted(self.critical_values[:1]):
        # print("\nconf",conf)

        # Reset pred_data to its original state
        pred_data = copy.deepcopy(original_pred_data)

        # pred_data is filtered based on threshold for the different predictors (compare_classes)
        for class_name in compare_classes:
            class_name(self.radius_threshold).filtered_predictions(pred_data, threhsold)
        

        # monos and root_id is derived after the filtering process was done using a threshold value
        monos, root_id = pred_data.semantics['monos'], pred_data.semantics['root']


        # before building IUPAC - put checks about:
        # if root exists
        # if no.of links = monos - 1
        # are all monos reachable from the link
        # if all the above checks are true - build IUPAC 

        if root_id == -1:
            FN += 1
            print("Log: Root doesn't exist")
        elif self.link_count(pred_data) != len(k_monos) - 1:
            FN += 1
            print("Log: Insufficient Links")
        elif self.all_monos_reachable(pred_data) != len(k_monos):
            FN += 1
            print("Log: Cannot traverse all nodes")
        else:
            try:
                pred_IUPAC = Glycan_Semantics.IUPAC(monos, root_id)

                if pred_IUPAC and pred_IUPAC == known_IUPAC:
                    TP += 1
                else:
                    FP += 1
                    FN += 1

                    print("Log: The known and pred sequence's dont match")
                    print("PRED SEQ", pred_IUPAC)
                    print("KNOWN SEQ",known_IUPAC)
                
            except Exception as e:
                print("EXCEPTION OCCURED",e)

        return {'TP': TP, 'FP': FP, 'FN': FN}


    
    def link_count(self, pred_data):

        # no. of links = no. of monos - 1
        num_links = set()

        links_adj = Utility.build_adjacency_list(pred_data)


        for box, linked_boxes in links_adj.items():
            # print("\n--->>",box.get('id'))
            for link_box in linked_boxes:
                # print(link_box[0].get('id'))
                
                link = tuple(sorted((box.get('id'), link_box[0].get('id'))))
                num_links.add(link)

        # print("--->>",len(links_adj), len(num_links))

        return len(num_links)

    
    def all_monos_reachable(self, pred_data):
        links_adj = Utility.build_adjacency_list(pred_data)
        visited = set()
        source = next(iter(links_adj))
        self.DFS(links_adj,visited,-1,source)
        return len(visited)


    def DFS(self,adj,visited,parent,u):
        visited.add(u.get('id'))

        for v,conf in adj[u]:
            if v.get('id') == parent:
                continue
            elif v.get('id') in visited:
                return True
            
            elif self.DFS(adj,visited,u.get('id'),v):
                return True
        
        return False




class Worker:
    def __init__(self, index, tasks, results, predictors, evaluation_method, eval_param):
        self.index = index
        self.tasks = tasks
        self.results = results  # Queue to send results to the main process
        self.predictors = predictors
        self.loaded_pipeline = None
        self.end_known_step = None
        self.evaluation_method = evaluation_method
        self.eval_param = eval_param


    def worker(self):
        batch_results = []  # Collect results in a batch
        batch_size = 10 # set this appropriately

        try:
            while True:
                task = self.tasks.get()  # Fetch tasks from the queue
                if task is None:  # Check for the sentinel value to exit
                    # Put remaining results in the queue before exiting
                    if batch_results:
                        self.results.put(batch_results, timeout=5)
                    self.results.put(None)
                    print(f"Worker {self.index}: Received sentinel, exiting.")
                    break

                try:
                    for image in task:

                        loaded_pipeline, end_known_step, predict = Worker.prediction_compare(
                            image=image,
                            predictors=self.predictors,
                            loaded_pipeline=self.loaded_pipeline,
                            end_known_step=self.end_known_step,
                            evaluation_method=self.evaluation_method,
                            eval_param=self.eval_param,
                            serial=False,
                        )

                        # Update cache for pipeline and known step
                        self.loaded_pipeline = loaded_pipeline
                        self.end_known_step = end_known_step

                        # Collect results in the batch
                        batch_results.append((image, predict))

                        # If batch is full, send it to the main process
                        if len(batch_results) >= batch_size:
                            self.results.put(batch_results, timeout=5)
                            batch_results = []  # Reset batch

                except Exception as img_error:
                    print(f"Worker {self.index}: Error processing image {task}: {img_error}")

            print(f"Worker {self.index}: Exiting Finally.")
        except Exception as e:
            print(f"Worker {self.index} encountered an error: {e}")


    @staticmethod
    def prediction_compare(
            image,
            predictors,
            loaded_pipeline = None,
            end_known_step = None,
            compare_strategy = None,
            known_data = None,
            evaluation_method = None,
            eval_param = None,
            serial = True,
        ):  

        predict = {}
        figure_semantics = None

        for pred_name, pred in predictors.items():

            # loaded_pipeline is the base_pipeline (excludes the end_step) which needs to be created only
            # once - and then the same can be used for all images.
            if loaded_pipeline is None:
                loaded_pipeline, end_known_step = Worker.build_pipeline('SingleGlycanImage-YOLOFinders', pred_name)

            if figure_semantics is None:
                figure_semantics = loaded_pipeline.run(image)
                figure_semantics = figure_semantics.glycans()[0]
                
            pred_data, known_data, compare_strategy = evaluation_method(
                figure_semantics, end_known_step, pred, eval_param, compare_strategy, known_data)

            results = compare_strategy.compare(pred_data, known_data)

            predict[pred_name] = results

        return loaded_pipeline, end_known_step, predict

    

    @staticmethod
    def build_pipeline(pipeline_name,finder_name):
        base_pipeline = BuildPipeline(pipeline_name, finder_name)
        base_pipeline, end_known_step = base_pipeline.load_pipeline()
        return base_pipeline, end_known_step


    @staticmethod
    def box_eval(
        figure_semantics, 
        end_known_step,
        end_pred_step, 
        eval_param = 0.5,
        compare_strategy = None,
        known_data = None
        ):

        pred_boxes = end_pred_step.find_boxes(figure_semantics.image()) 

        if compare_strategy is None:
            compare_strategy = end_pred_step.box_components(eval_param)

        if known_data is None:
            known_data = end_known_step.find_boxes(figure_semantics.image_path())

        return pred_boxes, known_data, compare_strategy


    @staticmethod
    def semantic_eval(
        figure_semantics, 
        end_known_step, 
        end_pred_step,
        eval_param = 0.25,
        compare_strategy = None,
        known_data = None
        ):
        semantics = copy.deepcopy(figure_semantics)
        pred_data = end_pred_step.find_objects(semantics)

        if compare_strategy is None:
            compare_strategy = end_pred_step.semantic_components(eval_param)

        if known_data is None:
            known_data = end_known_step.find_objects(figure_semantics)

        return pred_data, known_data, compare_strategy


        
class Evaluator:

    def __init__(self, predictors, **kwargs):
        self.predictors = predictors

        self.semantics = kwargs.get('semantics',False)
        self.parallel_process = kwargs.get('parallel', True)

        if self.semantics:
            self.evaluation_method = Worker.semantic_eval
            self.eval_param = kwargs.get('proximity',0.25)
        else:
            self.evaluation_method = Worker.box_eval
            self.eval_param = kwargs.get('iou',0.5)

    @staticmethod
    def check_data_monotonicity(predict, **kwargs):
        for pred_name, data in predict.items():

            if kwargs.get('sort_data',True):
                data = {k: v for k, v in sorted(data.items(), key=lambda x: float(x[0]))}

            prev_conf = 0.0
            TP = float('inf')
            FP = float('inf')
            FN = -1
            for conf, pairs in data.items():
                if float(conf) >= prev_conf:
                    prev_conf = float(conf)

                    if pairs['TP'] <= TP:
                        TP = pairs['TP']
                    else:
                        print("-->>TP error:",conf, TP, pairs['TP'])

                    if pairs['FP'] <= FP:
                        FP = pairs['FP']
                    else:
                        print("-->>FP error:",conf,FP,pairs['FP'])

                    if pairs['FN'] >= FN:
                        FN = pairs['FN']
                    else:
                        print("-->>FN error",conf)
                else:
                    print("CONFIDENCE IS NOT ORDERED")



    def runall(self, image_folder):
        collected_results = defaultdict(lambda: defaultdict(dict))
        
        image_data = Image_Manager(image_folder,pattern="*.png,*.jpg")

        print("\nProcessing the images...")
        
        start_time = time.time()

        # Serial Processing
        if not self.parallel_process:
            # cv2.setNumThreads(1)    # uncommenting this will disable cv2 multi core processing
            loaded_pipeline = None
            end_known_step = None
            compare_strategy = None

            for image in image_data:
                print("\nimage:",image)
                known_data = None   # reset known_data for each image

                loaded_pipeline, end_known_step, predict =  Worker.prediction_compare(
                    image = image,
                    predictors = self.predictors,
                    loaded_pipeline = loaded_pipeline,    # pass cached pipeline
                    end_known_step = end_known_step,     # pass cached end step
                    compare_strategy = compare_strategy,
                    known_data = known_data,
                    evaluation_method = self.evaluation_method,
                    eval_param = self.eval_param,
                    serial = True,
                )

                for pred_name, content in predict.items():
                    collected_results[pred_name][os.path.basename(image)] = content

                Evaluator.check_data_monotonicity(predict)
                        
        # Parallel processing  
        else:
            ncpus = multiprocessing.cpu_count()
            if ncpus > 4:
                ncpus = 4
                
            # print("No of CPUS:",ncpus)
            batch_size = len(image_data.images)//ncpus

            tasks = multiprocessing.Queue()
            results = multiprocessing.Queue(maxsize=1000)  # Limit size to avoid indefinite blocking
            procs = []

            # Create workers
            for i in range(ncpus):
                worker = Worker(i, tasks, results, self.predictors, self.evaluation_method, self.eval_param)
                proc = multiprocessing.Process(target=worker.worker)
                procs.append(proc)
                proc.start()


            images_batch = []
            for img in image_data:
                images_batch.append(img)

                if len(images_batch) == batch_size:
                    tasks.put(list(images_batch))
                    images_batch = []

            # Pass any remaining images
            if images_batch:
                tasks.put(list(images_batch))

            # Signal workers to exit
            for _ in range(ncpus):
                tasks.put(None)

            worker_done_count = 0
            while worker_done_count < ncpus:
                try:
                    result = results.get(timeout=5)  # Adjust timeout as needed
                    if result is None:
                        worker_done_count += 1  # One worker has finished
                        print(f"Worker finished. Remaining: {ncpus - worker_done_count}")
                    else:
                        # print("\nresult",result)
                        for image, data in result:
                            for pred_name, content in data.items():
                                collected_results[pred_name][os.path.basename(image)] = data[pred_name]
                except queue.Empty:
                    # print("Timeout: No results received. Waiting for workers to finish...")
                    pass


            for proc in procs:
                proc.join(timeout=10)  # Wait for worker to exit
                if proc.is_alive():
                    print(f"Worker {proc.pid} did not terminate. Forcing termination.")
                    proc.terminate()

        final_structure = self.process_results(collected_results)        

        end_time = time.time()
        
        # Plotting the results
        Evaluator.plotprecisionrecall(final_structure, self.evaluation_method.__name__)

        execution_time = end_time - start_time
        print(f"\nExecution Time in {'Parallel' if self.parallel_process else 'Serial'} mode: {execution_time} seconds")



    @staticmethod
    def process_results(collected_results,pipeline_name=None):
        aggregated_results = {}

        all_confidences = set()
        for pred_name, data in collected_results.items():
            for image_name, confidence_data in data.items():
                all_confidences.update(map(float, confidence_data.keys()))

        sorted_confidences = sorted(all_confidences)

        # print("\ncollected_results",collected_results)
        for pred_name in collected_results.keys():
            aggregated_results[pred_name] = {
                str(conf): {'TP': 0, 'FP': 0, 'FN': 0} for conf in sorted_confidences
            }

        # aggregated_results = {conf: {'TP': 0, 'FP': 0, 'FN': 0} for conf in sorted_confidences}

        for conf in sorted_confidences:
            # print("\nconf",conf)
            for pred_name, data in collected_results.items():
                for image_name, results in data.items():
                    relevant_confs = [float(c) for c in results.keys() if float(c) >= conf]
                    if relevant_confs:
                        nearest_conf = min(relevant_confs)
                        metrics = results[str(nearest_conf)]
                        # Aggregate metrics into the corresponding confidence level
                        aggregated_results[pred_name][str(conf)]['TP'] += metrics['TP']
                        aggregated_results[pred_name][str(conf)]['FP'] += metrics['FP']
                        aggregated_results[pred_name][str(conf)]['FN'] += metrics['FN']
                    else:
                        print("NOT RELEVANT")

        # print("\naggregated_results",aggregated_results)

        Evaluator.check_data_monotonicity(aggregated_results, **dict(sort_data=False))   # aggregated data should already be in sorted format

        return aggregated_results


    @staticmethod
    def plotprecisionrecall(observations, evaluator_type, **kwargs):

        sort_results = kwargs.get('sort_results', True)

        # directory = os.getcwd() + '/output_plots'
        directory = os.getcwd() + '/PR_curves'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.figure(1) 
        plt.figure(2)


        for pipeline_name, result_data in observations.items():
            
            print("pipeline name:",pipeline_name)
            # collect = defaultdict(list)


            precision = []
            recall = []

            # Use sorted only if sort_results is True - condition so that the
            # same function if useful for Finders and Pipelines both
            data_iterator = (
                sorted(result_data.items(), key=lambda x: float(x[0]))
                if sort_results
                else result_data.items()
            )
            
            for confidence, results in data_iterator:
                # print("confidence",confidence)
                tp = results['TP']
                fp = results['FP']
                fn = results['FN']

                # Calculate precision and recall for each threshold
                pos = tp + fp  # Total positive predictions
                tpfn = tp + fn  # Total ground truth positives

                try:
                    prec = tp / pos if pos != 0 else 0
                except ZeroDivisionError:
                    prec = 0
                
                try:
                    rec = tp / tpfn if tpfn != 0 else 0
                except ZeroDivisionError:
                    rec = 0

                precision.append(prec)
                recall.append(rec)


            precision = list(precision)
            recall = list(recall)


            # remove non-monotonic values...
            filtered_recall = []
            filtered_precision = []
            for i in range(len(recall)):
                if len(filtered_recall) == 0:
                    filtered_recall.append(recall[i])
                    filtered_precision.append(precision[i])
                elif precision[i] > filtered_precision[-1]:
                    filtered_recall.append(recall[i])
                    filtered_precision.append(precision[i])

            # print("\nfilter prec",filtered_precision)
            # print("\nfilter recall", filtered_recall)

            # and make step-based...
            step_recall = [] 
            step_precision = []
            step_recall.append(filtered_recall[0])
            step_precision.append(0)
            step_recall.append(filtered_recall[0])
            step_precision.append(filtered_precision[0])
            for i in range(1,len(filtered_recall)):
                step_recall.append(filtered_recall[i])
                step_precision.append(filtered_precision[i-1])
                step_recall.append(filtered_recall[i])
                step_precision.append(filtered_precision[i])
            step_recall.append(0)
            step_precision.append(step_precision[-1])

            # print("\nstep_prec",step_precision)
            # print("\nstep_recall",step_recall)


            # Plot on figure 1
            plt.figure(1)
            plt.plot(step_recall, step_precision, ".-", label=f"{pipeline_name}")
            # plt.plot(recall, precision, "r.",)

            plt.figure(2)
            plt.plot(step_recall, step_precision, ".-", label=f"{pipeline_name}")
            plt.plot(recall, precision, "r.",)

        # Plot figure 1
        plt.figure(1)
        plt.title(f'{evaluator_type} PR Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.xlim([0.0, 1.1])
        plt.ylim([0.0, 1.1])
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=1, color='k', linestyle='--', alpha=0.3)
        plt.legend(loc="best")

        # Zoomed-in graph (figure 2)
        plt.figure(2)
        plt.title(f'{evaluator_type} Precision-Recall Curve')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.xlim([0.5, 1.1])
        plt.ylim([0.5, 1.1])
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=1, color='k', linestyle='--', alpha=0.3)
        plt.legend(loc="best")

        pr = plt.figure(1)
        pr_zoom = plt.figure(2)

        files = os.listdir(directory)
        pattern = re.compile(rf"{re.escape(evaluator_type)}(\d+)\.png")

        numbers = [int(match.group(1)) for file in files if (match := pattern.search(file))]

        plot_no1 = max(numbers, default=0) + 1
        plot_no2 = plot_no1 + 1

        pr.savefig(directory + '/' + evaluator_type + str(plot_no1) + '.png')
        pr_zoom.savefig(directory + '/' + evaluator_type + str(plot_no2) + '.png')

        
        return pr, pr_zoom



    @staticmethod
    def euclidean_distance(box1,box2):
        bx1_cen_x, bx1_cen_y = box1.center()
        bx2_cen_x, bx2_cen_y = box2.center()
        return math.sqrt((bx1_cen_x - bx2_cen_x)**2 + (bx1_cen_y - bx2_cen_y)**2)   


    # @staticmethod
    # def critical_value_graph(all_boxes):

    #     directory = os.getcwd() + '/output_plots'
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)


    #     confidences = [round(box.get('confidence'),5) for box in all_boxes]
    #     confidences.sort()

    #     # Indices as x-axis points
    #     x_values = list(range(len(confidences)))  

    #     plt.figure(figsize=(16, 6))
    #     plt.plot(x_values, confidences, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)

    #     # Highlight each critical point with text
    #     for i, confidence in enumerate(confidences):
    #         plt.text(x_values[i], confidence, str(confidence), ha='right', va='bottom', rotation=90)

    #     plt.xlabel("Point Index")
    #     plt.ylabel("Confidence Value")
    #     plt.title("Critical Points Graph")

    #     plt.grid(True)

    #     plt.savefig(directory + '/critical_points.png') 

