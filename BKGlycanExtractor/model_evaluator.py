# correct RootSemantics
# Links compare_data() --> can be used from the common function


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
# import multiprocessing
import sys
import time
import queue
import importlib

from . import Image_Manager
from .semantics import Figure_Semantics, Glycan_Semantics
from .bbox import BoundingBox
from .compareboxes import CompareBoxes
from .debug_methods import DebugMode
from .build_pipeline import BuildPipeline
from .glycanannotator import Config_Manager
from .distproc import DistributedProcessing as dp

class CompareBase(object):
    def __init__(self,precision=8,verbose=False,whole_image=False,restrict_class=None,**kwargs):
        self.verbose = verbose
        self.whole_image = whole_image
        self.precision = precision
        self.scale = (10 ** precision)
        self.scaled_onepluseps = (self.scale+1)
        while self.float_trunc_conf(self.scaled_onepluseps) <= 1.0:
            self.scaled_onepluseps += 1
        self.classrestriction = None
        if restrict_class is not None:
            self.classrestriction = set(restrict_class)

    def scaled_trunc_conf(self,value):
        return int(math.floor(value*self.scale))

    def float_trunc_conf(self,value):
        return value/self.scale

    def str_trunc_conf(self,value):
        return "%.*f"%(self.precision,value/self.scale)

    def valid_assignemnt(self, known_box, pred_box):
        """ Default comparison function (to be overridden in derived classes) """
        raise NotImplementedError

    def secondary_sorting_criteria(self, x):
        """Subclasses can override this method to define their secondary sorting behavior.
        Sorting criteria for matches between known and predicted data.
        Primary sorting criteria is confidence"""
        return 0

    def _update_metrics(self,results,confidence,TP,FP,FN):
        if self.verbose:
            print(self.str_trunc_conf(confidence),"TP",TP,"FP",FP,"FN",FN,file=sys.stderr)
        results[confidence] = dict(TP=TP,FP=FP,FN=FN)        

    def update_metrics(self,results,confidence,nTRUE,TP,FP,FN):
        if self.whole_image:
            if FP + TP >= nTRUE:
                if FP == 0:
                    self._update_metrics(results,confidence,1,0,0)
                elif TP == nTRUE:
                    self._update_metrics(results,confidence,1,1,0)
                else:
                    self._update_metrics(results,confidence,0,1,0)
            else:
                self._update_metrics(results,confidence,0,0,1)
        else:
            self._update_metrics(results,confidence,TP,FP,FN)

    def compare(self,pred_objs,known_objs,**kwargs):
        if self.classrestriction is not None:
            pred_objs = [ obj for obj in pred_objs if obj.get('classlabel') in self.classrestriction ]
            known_objs = [ obj for obj in known_objs if obj.get('classlabel') in self.classrestriction ]
        edges, confidence_scores = self.matched_data(pred_objs, known_objs, **kwargs)
        return self.compare_data(known_objs, pred_objs, edges,confidence_scores)

    def matched_data(self,pred_objs,known_objs, **kwargs):
        """ Returns sorted edges (sort optionally based on condition of IOU/proximity) and sorted confidence values """
        
        edges = []
        confidence_scores = set()

        pred_semantics = kwargs.get('pred_semantics')
        known_semantics = kwargs.get('known_semantics')

        for p_id, pred_obj in enumerate(pred_objs):
            scaled_trunc_conf = self.scaled_trunc_conf(pred_obj.get('confidence'))
            
            for k_id, known_obj in enumerate(known_objs):
                match_info = {}
                if self.valid_assignemnt(known_obj,pred_obj,match_info):
                    edge = {
                        'known': known_obj,
                        'pred': pred_obj,
                        'conf': scaled_trunc_conf,
                        'classlabel': (known_obj.get('classlabel'),pred_obj.get('classlabel')),
                        'ids': (k_id, p_id),
                        **match_info,    # adds (key,val) - iou/proximity
                    }   
                    edges.append(edge)

            confidence_scores.add(scaled_trunc_conf)

        edges = sorted(edges, key=lambda x: (-x['conf'], self.secondary_sorting_criteria(x)))
        return edges, sorted(confidence_scores)


    def compare_data(self, known_data, pred_data, edges, confidence_scores):
        results = {}

        gt_count = len(known_data)
        
        for threshold in confidence_scores:
            TP, FP, FN = 0, 0, 0

            matched_gt = set()  # Set of matched ground truth IDs
            matched_pred = set()  # Set of matched predicted box IDs

            # FP's includes all the pred_data which is above the threshold
            accepted_pred_count = sum( 
                1 for data in pred_data if self.scaled_trunc_conf(data.get('confidence')) >= threshold
            )

            # Greedy matching based on Confidence
            # if edges are not sorted  - algo breaks
            for item in edges:
                if item['conf'] < threshold:
                    break

                known_id, pred_id = item['ids']

                if known_id in matched_gt:
                    continue
                if pred_id in matched_pred:
                    continue

                matched_gt.add(known_id)
                matched_pred.add(pred_id)

                # print("item['classlabel'][0],item['classlabel'][1]",item['classlabel'][0],item['classlabel'][1])
                if item['classlabel'][0] == item['classlabel'][1]:
                    TP += 1
                else:
                    FP += 1
                    FN += 1

            FN += gt_count - len(matched_gt)
            FP += accepted_pred_count - len(matched_pred)

            self.update_metrics(results,threshold,gt_count,TP,FP,FN)

        last_threshold = self.scaled_onepluseps
        self.update_metrics(results,last_threshold,gt_count,0,0,gt_count)
        return results


class BoxCompare(CompareBase):
    def __init__(self, iou=0.5, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou
        
    def valid_assignemnt(self, known_box, pred_box, match_info={}):
        iou = CompareBoxes.iou(known_box, pred_box)
        match_info['iou'] = iou
        return iou >= self.iou_threshold

    def secondary_sorting_criteria(self, x):
        '''Sorting criteria for matches between known and predicted data.
        Primary sorting criteria is confidence'''
        return -x['iou']  # Descending order


class MonosCompare(CompareBase):

    def __init__(self, proximity=0.25, **kwargs):
        super().__init__(**kwargs)
        self.proximity_threshold = proximity

    def valid_assignemnt(self, known_obj, pred_obj, match_info={}):
        proximity = CompareBoxes.proximity(known_obj, pred_obj)
        match_info['proximity'] = proximity
        return proximity <= self.proximity_threshold

    def secondary_sorting_criteria(self, x):
        '''Sorting criteria for matches between known and predicted data.
        Primary sorting criteria is confidence'''
        return x['proximity']  # Ascending order
        

class RootCompare(CompareBase):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def valid_assignemnt(self, known_obj, pred_obj, match_info={}):
        return known_obj['mono_id'] == pred_obj['mono_id']


class LinksCompare(CompareBase):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def valid_assignemnt(self, known_obj, pred_obj, match_info={}):
        return pred_obj['mono_ids'] == known_obj['mono_ids']


# ignore the below class - need to work on this 
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

        selected_critical_value = 0.77913584    # chosen after analyzing confidence values geenrated from all images after using their predictors
        for idx, image in enumerate(images):
            print("image:",idx, image)

            pred_semantics = self.base_pipeline.run(image)
            glycan = pred_semantics.glycans()[0]

            known_semantics = self.known_pipeline.run(image)
            known_glycan = known_semantics.glycans()[0]

            results = self.compare(glycan,known_glycan,selected_critical_value)

            observations[self.base_pipeline.name][os.path.basename(image)] = results


        end_time = time.time()
        self.plotprecisionrecall(observations, 'Whole_Glycan', **dict(sort_results=False))

        execution_time = end_time - start_time
        print(f"\nExecution Time {execution_time} seconds")



    # get one minimum confidence value for all images
    def compare(self, pred_data, known_data, threshold):
        compare_classes = [MonosCompare, RootCompare, LinksCompare]

        # known IUPAC
        k_monos, k_root_id = known_data.semantics['monos'], known_data.semantics['root']
        known_IUPAC = Glycan_Semantics.IUPAC(k_monos, k_root_id)


        # comment this for loop - it is only for experimentation to find best confidence threshold
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
            class_name(self.radius_threshold).filtered_predictions(pred_data, threshold)
        

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



        
class Evaluator:

    def __init__(self, known_pipeline, prediction_pipelines, compare_strategies, workers=None, boxeval=False, verbose=False):
        # prediction_pipelines and compare_strategies are dictionaries, providing a name as the key
        self.known_pipeline = known_pipeline
        self.pred_pipelines = prediction_pipelines
        self.compare = compare_strategies
        self.workers = workers
        self.boxeval = boxeval
        self.verbose = verbose
        self.final_structure = None    # Data structure metrics: (TP, FP, FN) - created after all images are analyzed


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


    def isboxeval(self):
        return self.boxeval

    # # remove this
    # def pred_items(self,*args):
    #     return list(args[0])

    # # remove this
    # def known_items(self,*args):
    #     return list(args[0])

    def process_image(self, image, **kwargs):

        results = dict()
        # known_results = self.known_pipeline.run_evaluation(image,self.isboxeval())
        # known_items = self.known_items(*known_results)
        known_items, known_semantics = self.known_pipeline.run_evaluation(image,self.isboxeval())

        i = 1
        j = 1
        for prname,pl in self.pred_pipelines.items():
            # pred_results = pl.run_evaluation(image,self.isboxeval())
            # pred_items = self.pred_items(*pred_results)
            pred_items, pred_semantics = pl.run_evaluation(image,self.isboxeval())
            k = 1
            for cpname,cmp in self.compare.items():
                results[(i,prname,j,cpname,k)] = cmp.compare(pred_items, known_items, **{'pred_semantics':pred_semantics,'known_semantics':known_semantics})
                i += 1
                k += 1
            j += 1

        return image,results

    def runall(self, images):

        collected_results = defaultdict(lambda: defaultdict(dict))
        start_time = time.time()

        for result in dp.process(workers=self.workers,target=self.process_image,
                                 tasks=images,verbose=self.verbose):
            for pred_name, content in result[1].items():
                collected_results[pred_name][os.path.basename(result[0])] = content

        self.final_structure = self.process_results(collected_results)        

        end_time = time.time()
        
        # Plotting the results
        # self.plotprecisionrecall(final_structure)

        execution_time = end_time - start_time
        print(f"\nExecution Time: {execution_time} seconds")

    def process_results(self,collected_results):
        aggregated_results = {}

        all_confidences = set()
        for pred_name, data in collected_results.items():
            for image_name, confidence_data in data.items():
                all_confidences.update(confidence_data.keys())

        sorted_confidences = sorted(all_confidences)

        # print("\ncollected_results",collected_results)
        for pred_name in collected_results.keys():
            aggregated_results[pred_name] = {
                conf: {'TP': 0, 'FP': 0, 'FN': 0} for conf in sorted_confidences
            }

        # aggregated_results = {conf: {'TP': 0, 'FP': 0, 'FN': 0} for conf in sorted_confidences}

        for conf in sorted_confidences:
            # print("\nconf",conf)
            for pred_name, data in collected_results.items():
                for image_name, results in data.items():
                    relevant_confs = [c for c in results.keys() if c >= conf]
                    if relevant_confs:
                        nearest_conf = min(relevant_confs)
                        metrics = results[nearest_conf]
                        # Aggregate metrics into the corresponding confidence level
                        aggregated_results[pred_name][conf]['TP'] += metrics['TP']
                        aggregated_results[pred_name][conf]['FP'] += metrics['FP']
                        aggregated_results[pred_name][conf]['FN'] += metrics['FN']
                    else:
                        print("NOT RELEVANT")
        
        for k,v in aggregated_results.items():
            if self.verbose:
                print("-",k,file=sys.stderr)
                for k1,v1 in v.items():
                    print("-->>",k1,v1,file=sys.stderr)
             
        # print("\naggregated_results",aggregated_results)

        Evaluator.check_data_monotonicity(aggregated_results, sort_data=False)   # aggregated data should already be in sorted format

        return aggregated_results


    def plotprecisionrecall(self, **kwargs):
        # title, other plot keywords?
        """
        Plots Precision-Recall curves with full customization.

        Parameters:
        - dir (str): Directory to save plots (default: "PR_curves")
        - filename (str): Custom filename (default: auto-generated)
        - title (str): Title of the plot
        - figsize (tuple): Size of the figure (width, height)
        - legend_loc (str): Legend location (e.g., 'best', 'upper right')
        - xlim (tuple): X-axis limits (default: (0.0, 1.1))
        - ylim (tuple): Y-axis limits (default: (0.0, 1.1))
        - grid (bool): Whether to show grid lines (default: True)
        - remove - sort_results (bool): Whether to sort results by confidence (default: True)
        """

        # user defined options, with default behaviour
        directory = kwargs.get('dir', os.path.join(os.getcwd(), 'PR_curves'))
        custom_filename = kwargs.get('filename', None)
        title = kwargs.get('title', "Precision-Recall Curve")
        label = kwargs.get('label',"%(predictor)s, %(comparitor)s")
        figsize = kwargs.get('figsize', (8, 6))  # Default size if not provided
        legend_loc = kwargs.get('legend_loc', 'best')
        xlim = kwargs.get('xlim', (0.0, 1.1))
        ylim = kwargs.get('ylim', (0.0, 1.1))
        grid = kwargs.get('grid', True)

        sort_results = kwargs.get('sort_results', True)

        # directory = os.getcwd() + '/output_plots'
        # directory = os.getcwd() + '/PR_curves'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.figure(1,figsize=figsize) 
        plt.figure(2,figsize=figsize)


        for pipeline_details, result_data in self.final_structure.items():
            
            # print("pipeline name:",pipeline_name)
            # collect = defaultdict(list)
            # pipeline_name, compare_type, id = pipeline_details
            # print("compare_type",compare_type)
            fields = "curve_index,predictor,predictor_index,comparitor,comparitor_index".split(',')
            details = dict(zip(fields,pipeline_details))

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
            plt.plot(step_recall, step_precision, ".-", label=label%details)
            # plt.plot(recall, precision, "r.",)

            plt.figure(2)
            plt.plot(step_recall, step_precision, ".-", label=label%details)
            # plt.plot(recall, precision, "r.",)

        # Plot figure 1
        plt.figure(1)
        plt.title(title%details)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        xlim = (xlim[0], min(1.1, xlim[1] + 0.1))
        ylim = (ylim[0], min(1.1, ylim[1] + 0.1))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=1, color='k', linestyle='--', alpha=0.3)
        plt.legend(loc=legend_loc)
        if grid:
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

        # Zoomed-in graph (figure 2)
        plt.figure(2)
        plt.title(title%details)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.xlim([0.5, 1.1])
        plt.ylim([0.5, 1.1])
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=1, color='k', linestyle='--', alpha=0.3)
        plt.legend(loc=legend_loc)
        if grid:
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)

        pr = plt.figure(1)
        pr_zoom = plt.figure(2)

        # Auto-generate filename if not provided
        if not custom_filename:

            files = os.listdir(directory)
            pattern = re.compile(rf"{re.escape(compare_type)}(\d+)\.png")

            numbers = [int(match.group(1)) for file in files if (match := pattern.search(file))]

            plot_no1 = max(numbers, default=0) + 1
            plot_no2 = plot_no1 + 1

            custom_filename = f"{compare_type}{plot_no1}"

        # save plots
        # pr.savefig(directory + '/' + compare_type + str(plot_no1) + '.png')
        # pr_zoom.savefig(directory + '/' + compare_type + str(plot_no2) + '.png')
        
        pr.savefig(f"{directory}/{custom_filename}.png")
        pr_zoom.savefig(f"{directory}/{custom_filename}_zoom.png")

        
        return pr, pr_zoom



    

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

            
