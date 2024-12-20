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
from .semantics import Figure_Semantics
from .bbox import BoundingBox
from .compareboxes import CompareBoxes
from .debug_methods import DebugMode

from .build_pipeline import BuildPipeline

from .glycanannotator import Config_Manager


class BoxCompare:
    def __init__(self, iou):
        self.iou = iou
        self.compare_boxes = CompareBoxes(**dict(detection_threshold=self.iou,overlap_threshold=self.iou,containment_threshold=self.iou))



    def compare(self, pred_boxes, known_boxes):

        confidence_scores = sorted({box.get('confidence') for box in pred_boxes})
        # print("confidence_scores",confidence_scores)

        matches = []
        for threshold in confidence_scores:
            FN, FP, TP = 0,0,0

            # Filter predictions based on the current confidence threshold
            boxes = [box for box in pred_boxes if box.get('confidence') >= threshold]
            compare_dict = {}
            t_visited = set()
            d_visited = set()

            iou_box_pairs = defaultdict(list)
            
            # should we take dbox and then tboxes or do the opposite? b/c rn a single dbox can have many tboxes ---> but that shouldnt happen --> one dbox box should have
            # only one tbox and all should be unique pairs ---> rn because of multiples ---> we are getting too many TP's
            for idx,dbox in enumerate(boxes):
                dbox.set('id',idx)
                compare_dict[dbox.get('id')] = (dbox, None)
                max_int = 0
                for tbox in known_boxes:
                    if self.compare_boxes.have_intersection(tbox,dbox):
                        iou = self.compare_boxes.iou(tbox,dbox)
                        if iou > max_int:
                            max_int = iou

                            compare_dict[dbox.get('id')] = (dbox,tbox)

                            iou_box_pairs[dbox].append([tbox,iou])
                    else:
                        continue

            for dbox, pairs in iou_box_pairs.items():
                iou_box_pairs[dbox] = sorted(pairs, key=lambda x: -x[1])

                for training_box,iou in iou_box_pairs[dbox]:
                    if training_box not in t_visited:
                        compare_dict[dbox.get('id')] = (dbox,training_box)

                        t_visited.add(training_box)

                        break

            for tbox in known_boxes:
            
                found = False
                for dbox in boxes:
                    if self.compare_boxes.have_intersection(tbox,dbox):
                        found = True
                        break
                if found:
                    continue
                else:                                  
                    FN += 1
                    
            for key,boxpair in compare_dict.items():
                dbox = boxpair[0]
                tbox = boxpair[1]

                assert dbox.get('id') == key
                if tbox is None:
                    FP += 1
                else:
                    # links do not have classes
                    if tbox.has('classid') and dbox.has('classid') and not self.compare_boxes.compare_class(tbox,dbox):
                        FP += 1
                        FN += 1
                    else:
                        t_area = tbox.area()
                        d_area = dbox.area()
                        inter = self.compare_boxes.intersection_area(tbox,dbox)
                        if inter == 0:
                            FP += 1
                        elif inter == t_area:
                            if self.compare_boxes.training_contained(tbox,dbox):
                                TP += 1
                            else:
                                FP += 1
                                FN += 1
                        elif inter == d_area:
                            if self.compare_boxes.detection_sufficient(tbox,dbox):
                                TP += 1
                            else:
                                FN += 1
                                FP += 1
                        else:
                            if self.compare_boxes.is_overlapping(tbox,dbox):
                                TP += 1
                            else:
                                FP += 1
                                FN += 1   

            matches.append([
                str(round(threshold,5)),
                {
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                }
            ])


        return matches



class MonosCompare:

    def __init__(self, radius_threshold):
        self.radius_threshold = radius_threshold

    # if classes dont match - it should be a FP and a FN
    def compare(self, pred_data, known_data, **kwargs):
        matches = []

        # Extract critical points (unique confidence scores)
        confidence_scores = sorted({item['box'].get('confidence') for item in pred_data.monosaccharides()})
        # confidence_scores = [i/10000 for i in range(9800,10000,10)]


        # Iterate over each critical point and filter predictions based on current threshold
        for confidence_threshold in confidence_scores:
            # confidence_threshold = confidence_scores[0]
            pred_items = [item for item in pred_data.monosaccharides() if item['box'].get('confidence') >= confidence_threshold]


            visited_ids = set()  # Track matched ground truth items
            TP, FP, FN = 0, 0, 0


            # get all possible distance mappings b/w known_item and pred_items when the distance < proximity
            for pred_item in pred_items:
                best_match = None
                closest_mono = float('inf')

                for known_item in known_data.monosaccharides():
                    if known_item.get('id') in visited_ids:
                        continue

                    distance = Evaluator.euclidean_distance(known_item['box'], pred_item['box'])
                    proximity = self.radius_threshold * min(known_item['box'].w, known_item['box'].h)

                    if distance <= proximity and distance < closest_mono:
                        best_match = known_item
                        closest_mono = distance

                if best_match:
                    visited_ids.add(best_match.get('id'))

                    # Compare classes to determine TP or FP/FN
                    if pred_item.get('classid') == best_match.get('classid'):
                        TP += 1  # True Positive
                    else:
                        # print("class mismatch:",best_match.get('classid'), pred_item.get('classid'))
                        FP += 1  # False Positive (wrong class)
                        FN += 1  # False Negative (missed correct class)
                else:
                    # print("NO match for prediction")
                    FP += 1  # False Positive (no match for prediction)


            # Count False Negatives for unmatched ground truth items
            unmatched_known = [
                known_item for known_item in known_data.monosaccharides()
                if known_item.get('id') not in visited_ids
            ]
            FN += len(unmatched_known)

            matches.append([
                str(round(confidence_threshold, 5)),
                {
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                }
            ])


        # if DebugMode.debug:
        #     data = {}
        #     if len(set(matches)) > 1:
        #         data['incorrect_confidence'] = [confidence_threshold]
        #         DebugMode.log_data(DebugMode.curr_image, data)  

        return matches


        
class RootCompare:
    def __init__(self, radius_threshold):
        self.mono_compare = MonosCompare(radius_threshold)


    def root_data(self, data):
        root_id = data.root()
        mono_data = data.monosaccharide(root_id)

        box = mono_data.get('box')
        # Add confidence to box if it doesn't exist
        if not hasattr(box, 'confidence'):
            setattr(box, 'confidence', 0.0)  # Set default confidence if missing

        data.clear_monos()
        args = dict(id=root_id)
        data.add_mono(mono_data.get('classid'), mono_data.get('symbol'), box, **args)

        return data

    def compare(self, pred_data, known_data, **kwargs):
        # Deep copy to prevent modifying original data
        predicted = copy.deepcopy(pred_data)
        known = copy.deepcopy(known_data)

        # Prepare root data
        pred_root_data = self.root_data(predicted)
        known_root_data = self.root_data(known)

        # Extract critical confidence points from predictions
        confidence_scores = sorted({item['box'].get('confidence', 1.0) for item in pred_root_data.monosaccharides()})

        # Pass the data to MonosCompare without filtering confidence
        return self.mono_compare.compare(pred_root_data, known_root_data)



class LinksCompare:
    def __init__(self, radius_threshold):
        self.radius_threshold = radius_threshold


    def compare(self, pred_data, known_data):
        matches = []

        other_adj = self.build_adjacency_list(pred_data)
        known_adj = self.build_adjacency_list(known_data)

        # Extract critical points (unique confidence scores)
        confidence_scores = sorted({
            conf
            for box, links in other_adj.items()
            for _, conf in links
        })

        for confidence_threshold in confidence_scores:

            # bidirectional links exist
            filtered_other_adj = {
                box: [linked_box for linked_box, conf in linked_items if conf >= confidence_threshold]
                for box, linked_items in other_adj.items()
            }


            # Track matched pairs to prevent double counting
            visited_pairs = set()  # Track matched ground truth pairs
            items_distance_pair = defaultdict(list)

            # Match predictions to known data
            for pred_box, linked_pred_boxes in filtered_other_adj.items():
                for linked_pred_box in linked_pred_boxes:
                    closest_distances = (float('inf'), float('inf'))
                    best_match = None

                    for known_box, linked_known_boxes in known_adj.items():
                        for linked_known_box in linked_known_boxes:
                            dist1 = Evaluator.euclidean_distance(known_box, pred_box)
                            dist2 = Evaluator.euclidean_distance(linked_known_box, linked_pred_box)
                            proximity1 = self.radius_threshold * min(known_box.w, known_box.h)
                            proximity2 = self.radius_threshold * min(linked_known_box.w, linked_known_box.h)

                            if dist1 <= proximity1 and dist2 <= proximity2 and (dist1, dist2) < closest_distances:
                                closest_distances = (dist1, dist2)
                                best_match = [pred_box, linked_pred_box, known_box, linked_known_box, dist1, dist2]

                        if best_match:
                            # since links are bi-directional - we want to store data only once - hence sort the id pairs and save them
                            pair_id = tuple(sorted((pred_box.get('id'), linked_pred_box.get('id'))))
                            if pair_id not in visited_pairs: 
                                items_distance_pair[pair_id].append(best_match)
                                visited_pairs.add(pair_id)


            compare_dict = {
                pair_id: min(mappings, key=lambda x: (x[-2], x[-1]))[:4]
                for pair_id, mappings in items_distance_pair.items()
            }

            TP, FP, FN = 0, 0, 0


            # Evaluate matched predictions
            for pred_id, mapping in compare_dict.items():
                pred_box, linked_pred_box, known_box, linked_known_box = mapping

                if known_box is None or linked_known_box is None:
                    FP += 1
                elif (
                    pred_box.get('classid') != known_box.get('classid') and
                    linked_pred_box.get('classid') != linked_known_box.get('classid')
                ):  
                    FP += 1
                    FN += 1
                else:
                    TP += 1

            # Check for unmatched ground truth (False Negatives)
            for known_box, linked_known_boxes in known_adj.items():
                # Check if known_box has already matched in `compare_dict`
                known_box_id = known_box.get('id')

                for link_k_box in linked_known_boxes:
                    link_k_box_id = link_k_box.get('id')

                    
                    matched = any(
                        (known_box_id == pair_id[0] or known_box_id == pair_id[1]) and (link_k_box_id == pair_id[0] or link_k_box_id == pair_id[1])
                        for pair_id in compare_dict.keys()
                    )

                    if not matched:
                        FN += 1

            matches.append([
                str(round(confidence_threshold, 5)),
                {"TP": TP, "FP": FP, "FN": FN}
            ])

       

        return matches

    # def compare(self, pred_data, known_data):
    #     matches = []

    #     # Extract critical points (unique confidence scores)
    #     confidence_scores = sorted({
    #         conf
    #         for box, links in self.build_adjacency_list(pred_data).items()
    #         for _, conf in links if isinstance(links[0], list)
    #     })

    #     for confidence_threshold in confidence_scores:
    #         other_adj = self.build_adjacency_list(pred_data)

    #         # Filter predictions based on confidence threshold
    #         filtered_other_adj = {
    #             box: [linked_box for linked_box, conf in linked_items if conf >= confidence_threshold]
    #             for box, linked_items in other_adj.items()
    #         }

    #         known_adj = self.build_adjacency_list(known_data)


    #         # Track matched pairs to prevent double counting
    #         visited_pairs = set()  # Track matched ground truth pairs
    #         items_distance_pair = defaultdict(list)

    #         # Match predictions to known data
    #         for pred_box, linked_pred_boxes in filtered_other_adj.items():
    #             for linked_pred_box in linked_pred_boxes:
    #                 closest_distances = (float('inf'), float('inf'))
    #                 best_match = None

    #                 for known_box, linked_known_boxes in known_adj.items():
    #                     for linked_known_box in linked_known_boxes:
    #                         dist1 = Evaluator.euclidean_distance(known_box, pred_box)
    #                         dist2 = Evaluator.euclidean_distance(linked_known_box, linked_pred_box)
    #                         proximity1 = self.radius_threshold * min(known_box.w, known_box.h)
    #                         proximity2 = self.radius_threshold * min(linked_known_box.w, linked_known_box.h)

    #                         if dist1 <= proximity1 and dist2 <= proximity2 and (dist1, dist2) < closest_distances:
    #                             closest_distances = (dist1, dist2)
    #                             best_match = (pred_box, linked_pred_box, known_box, linked_known_box)

    #                 if best_match:
    #                     pair_id = tuple(sorted((pred_box.get('id'), linked_pred_box.get('id'))))
    #                     if pair_id not in visited_pairs:
    #                         items_distance_pair[pair_id].append((*best_match, closest_distances[0], closest_distances[1]))
    #                         visited_pairs.add(pair_id)


    #         compare_dict = {
    #             pair_id: min(mappings, key=lambda x: (x[4], x[5]))[:4]
    #             for pair_id, mappings in items_distance_pair.items()
    #         }

    #         TP, FP, FN = 0, 0, 0

    #         # Evaluate matched predictions
    #         for pred_id, mapping in compare_dict.items():
    #             pred_box, linked_pred_box, known_box, linked_known_box = mapping

    #             if known_box is None or linked_known_box is None:
    #                 FP += 1
    #             elif (
    #                 pred_box.get('classid') != known_box.get('classid') or
    #                 linked_pred_box.get('classid') != linked_known_box.get('classid')
    #             ):
    #                 FP += 1
    #                 FN += 1
    #             else:
    #                 TP += 1


    #         # Check for unmatched ground truth (False Negatives)
    #         for known_box, linked_known_boxes in known_adj.items():
    #             # Check if known_box has already matched in `compare_dict`
    #             known_box_id = known_box.get('id')
                
    #             matched = any(
    #                 known_box_id == pair_id[1]
    #                 for pair_id in compare_dict.keys()
    #             )

    #             if not matched:
    #                 FN += 1

    #         matches.append([
    #             str(round(confidence_threshold, 5)),
    #             {"TP": TP, "FP": FP, "FN": FN}
    #         ])

    #     return matches



    def build_adjacency_list(self, mono_semantics):
        adj = defaultdict(list)
        monos = mono_semantics.semantics['monos']

        for mono_id, mono_data in monos.items():
            for link_data in mono_data['links']:
                if isinstance(link_data, list):  # Handle case with confidence
                    linked_id, conf = link_data
                    # print("pred--monos",linked_id, mono_data)
                    adj[mono_data['box']].append([monos[linked_id]['box'], conf])
                else:  # Handle case without confidence - known data case
                    linked_id = link_data
                    # print("known--monos",linked_id, mono_data)
                    adj[mono_data['box']].append(monos[linked_id]['box'])

        return adj


 

class SemanticGlycanCompare:

    def __init__(self, radius_threshold):
        self.radius_threshold = radius_threshold


    def compare(self, pred_data, known_data, pipeline_name):
        matches = []

        compare_classes = [MonosCompare, RootCompare, LinksCompare]

        for class_name in compare_classes:
            matches.extend(class_name(self.radius_threshold).compare(pred_data, known_data))

        return matches



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

                        # print(f"Worker {self.index}: Processing task {task}")
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
                        batch_results.append(predict)

                        # If batch is full, send it to the main process
                        if len(batch_results) >= batch_size:
                            self.results.put(batch_results, timeout=5)
                            batch_results = []  # Reset batch

                except Exception as img_error:
                    print(f"Worker {self.index}: Error processing image {task}: {img_error}")

            print(f"Worker {self.index}: Exiting Finally.")
        except Exception as e:
            print(f"Worker {self.index} encountered an error: {e}")



    # if something is shared and doesnt depend on instance - you need to pass it externally through methods
    # things that need to be cached can be stored externally in the init as self.
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



    def runall(self, image_folder):
        collected_results = []
        image_data = Image_Manager(image_folder,pattern="*.png,*.jpg")

        print("\nProcessing the images...")
        
        start_time = time.time()

        # Serial Processing
        if not self.parallel_process:
            cv2.setNumThreads(1)    # disable cv2 multi core processing for serial mode
            loaded_pipeline = None
            end_known_step = None
            compare_strategy = None

            for image in image_data:
                print("image:",image)
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
                collected_results.append(predict)

        # Parallel processing  
        else:
            ncpus = multiprocessing.cpu_count()
            # if ncpus > 4:
                # ncpus = 4
                
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
                        # print("final result", result)
                        collected_results.extend(result)
                except queue.Empty:
                    # print("Timeout: No results received. Waiting for workers to finish...")
                    pass


            for proc in procs:
                proc.join(timeout=10)  # Wait for worker to exit
                if proc.is_alive():
                    print(f"Worker {proc.pid} did not terminate. Forcing termination.")
                    proc.terminate()

            
        final_structure = self.process_results(collected_results)
        print("\nfinal_structure",final_structure)

        end_time = time.time()
        
        # Plotting the results
        Evaluator.plotprecisionrecall(final_structure, self.evaluation_method.__name__)

        execution_time = end_time - start_time
        print(f"\nExecution Time in {'Parallel' if self.parallel_process else 'Serial'} mode: {execution_time} seconds")



    @staticmethod
    def process_model_data(model_name, model_data, aggregated_data):
        for confidence, matches in model_data:
            aggregated_data[model_name][confidence]['TP'] += matches['TP']
            aggregated_data[model_name][confidence]['FP'] += matches['FP']
            aggregated_data[model_name][confidence]['FN'] += matches['FN']


    @staticmethod
    def process_results(collected_results,pipeline_name=None):
        # Initialize aggregated_data as a nested defaultdict where inner values are dictionaries
        aggregated_data = defaultdict(lambda: defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0}))


        for data in collected_results: 
            if isinstance(data,dict):
                for model_name, model_data in data.items():
                    Evaluator.process_model_data(model_name, model_data, aggregated_data)

            else:
                Evaluator.process_model_data(pipeline_name, [data], aggregated_data)

        return aggregated_data



    @staticmethod
    def plotprecisionrecall(observations, evaluator_type):

        # directory = os.getcwd() + '/output_plots'
        directory = os.getcwd() + '/PR_curves'
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.figure(1) 
        plt.figure(2)

        for pipeline_name, result_data in observations.items():
            precision = []
            recall = []

            for confidence, results in result_data.items():
                tp = results['TP']
                fp = results['FP']
                fn = results['FN']

                pos = tp + fp  # Total positive predictions
                tpfn = tp + fn  # Total ground truth positives

                # Calculate precision and recall
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

            # Sort the recall and precision for plotting
            recall, precision = zip(*sorted(zip(recall, precision)))

            # Plot on figure 1
            plt.figure(1)
            plt.plot(recall, precision, ".-", label=f"{pipeline_name}")

            plt.figure(2)
            plt.plot(recall, precision, ".-", label=f"{pipeline_name}")

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

