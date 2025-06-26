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
from functools import partial

from . import Image_Manager
from .semantics import Figure_Semantics, Glycan_Semantics
from .bbox import BoundingBox
from .compareboxes import CompareBoxes
from .debug_methods import DebugMode
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
            # print(self.str_trunc_conf(confidence),"TP",TP,"FP",FP,"FN",FN,file=sys.stderr)
            print(self.str_trunc_conf(confidence),"TP",TP,"FP",FP,"FN",FN)
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

        # print("-----",pred_objs)
        # print("kno",known_objs)
        for p_id, pred_obj in enumerate(pred_objs):
            # print("pred_obj",pred_obj)
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

                # print("\nknown",item['classlabel'][0])
                # print("pred",item['classlabel'][1])
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


# class GlycanCompare(CompareBase): 
#     def __init__(self, iou=0.5, **kwargs):
#         super().__init__(**kwargs)
#         self.iou_threshold = iou

#     def valid_assignemnt(self, known_obj, pred_obj, match_info={}):
#         iou = CompareBoxes.iou(known_obj, pred_obj)
#         match_info['iou'] = iou
#         return iou >= self.iou_threshold

#     def secondary_sorting_criteria(self, x):
#         '''Sorting criteria for matches between known and predicted data.
#         Primary sorting criteria is confidence'''
#         return x['iou']  # Ascending order

# ----proximity--- (Note: this can be adapted for IOU as)
class GlycanCompare(CompareBase):

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
        

        
class Evaluator:

    def __init__(self, pipelines, compares, workers=None, boxeval=False, verbose=False):
        # pipelines and compare are dictionaries, providing a name as the key
        self.pipelines = pipelines
        self.compares = compares
        self.workers = workers
        self.boxeval = boxeval
        self.verbose = verbose

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


    def process_image(self, image, **kwargs):

        results = dict()
        i = 0
        prnames = list(self.pipelines)
        cmpnames = list(self.compares)
        for j,prname in enumerate(prnames):
            prpl,knpl = self.pipelines[prname]
            pred_items, pred_semantics = prpl.run_evaluation(image,self.isboxeval())
            known_items, known_semantics = knpl.run_evaluation(image,self.isboxeval())
            # sometimes no root is detected - so pred_items could be None
            if pred_items[0] is not None:
                for k,cmpname in enumerate(cmpnames):
                    cmp = self.compares[cmpname]
                    i += 1
                    results[(i,prname,j+1,cmpname,k+1)] = cmp.compare(pred_items, known_items, 
                                                                    pred_semantics=pred_semantics, 
                                                                    known_semantics=known_semantics)
        return image,results

    def runall(self, images):
        
        collected_results = defaultdict(lambda: defaultdict(dict))
        start_time = time.time()

        for result in dp.process(workers=self.workers,target=self.process_image,
                                 tasks=images,verbose=self.verbose):
            for pred_name, content in result[1].items():
                collected_results[pred_name][os.path.basename(result[0])] = content

        self.process_results(collected_results)        

        end_time = time.time()        
        execution_time = end_time - start_time
        print(f"\nExecution Time: {execution_time} seconds")
        

    def process_results(self,collected_results):
        # print("all_pipelines",self.pipelines)
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
                # print("pred_name",pred_name)
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
            
        if self.verbose:
            for k,v in aggregated_results.items():
                print("-",k,file=sys.stderr)
                for k1,v1 in v.items():
                    print("-->>",k1,v1,file=sys.stderr)
             
        # print("\naggregated_results",aggregated_results)

        Evaluator.check_data_monotonicity(aggregated_results, sort_data=False)   # aggregated data should already be in sorted format

        self.final_structure = aggregated_results

    @staticmethod
    def plotprecisionrecall(results, **kwargs):
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

        for pipeline in results:
            for pipeline_details, result_data in pipeline.final_structure.items():


            
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
                # data_iterator = (
                #     sorted(result_data.items(), key=lambda x: float(x[0]))
                #     if sort_results
                #     else result_data.items()
                # )
                
                for confidence, results in result_data.items():
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

            
def runall_evaluators(evaluators, images, workers=None, verbose="TQDM"):

    # ensure evaluators is in a deterministic order
    evaluators = list(evaluators)
    proc = dp.stage_process_init(workers,[None]+[eval.process_image for eval in evaluators],verbose)

    for i,eval in enumerate(evaluators):

        start_time = time.time()

        collected_results = defaultdict(dict)
        for result in proc.stage_process(i+1,images):
            for pred_name, content in result[1].items():
                collected_results[pred_name][os.path.basename(result[0])] = content

        eval.process_results(collected_results)        

        end_time = time.time()
        execution_time = end_time - start_time
        if verbose == True:
            print(f"Stage {i+1}: Execution Time: {execution_time} sec.")

    proc.stage_process_finish()
