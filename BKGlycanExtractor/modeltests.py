# -*- coding: utf-8 -*-
"""
Various tests for YOLO models
"""

import logging

class ModelTest:               
    def compare(self, boxes, training, comparison_alg, threshold=0.0):
        for box in boxes:
            if box.get_confidence() < threshold:
                boxes.remove(box)
        [box.to_four_corners() for box in boxes]
        compare_dict = {}
        for idx,dbox in enumerate(boxes):

            dbox.set_name(str(idx))
            compare_dict[dbox.name] = (dbox, None)
            max_int = 0
            for tbox in training:
                if comparison_alg.have_intersection(tbox, dbox):
                    intersect = comparison_alg.intersection_area(tbox, dbox)
                    if intersect > max_int:
                        max_int = intersect
                        compare_dict[dbox.name] = (dbox, tbox)
                else:
                    continue
        results = []
        self.logger.info("Training box checks:")
        for tbox in training:
            self.logger.info(str(tbox))
            found = False
            for dbox in boxes:
                if comparison_alg.have_intersection(tbox, dbox):
                    found = True
                    break
            if found:
                self.logger.info(
                    "Training box intersects with detected box(es)."
                    )
                continue
            else:                                       
                results.append('FN')
                self.logger.info("FN, training box not detected.")
        
        self.logger.info("Detected box checks:")
        for key, boxpair in compare_dict.items():
            dbox = boxpair[0]
            tbox = boxpair[1]
            assert dbox.name == key
            self.logger.info(str(dbox))
            if tbox is None:
                results.append("FP")
                self.logger.info(
                    "FP, detection does not intersect with training box."
                    )
            else:
                if not comparison_alg.compare_class(tbox, dbox):
                    results.append("FP")
                    results.append("FN")
                    self.logger.info("FP/FN, incorrect class")
                else:
                    t_area = tbox.area()
                    d_area = dbox.area()
                    inter = comparison_alg.intersection_area(tbox, dbox)
                    if inter == 0:
                        results.append("FP")
                        self.logger.info(
                            "FP, detection does not intersect with training box."
                            )
                    elif inter == t_area:
                        if comparison_alg.training_contained(tbox, dbox):
                            results.append("TP")
                            self.logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN")
                            self.logger.info("FP/FN, detection area too large.")
                    elif inter == d_area:
                        if comparison_alg.detection_sufficient(tbox, dbox):
                            results.append("TP")
                            self.logger.info("TP")
                        else:
                            results.append("FN")
                            results.append("FP")
                            self.logger.info("FP/FN, detection area too small.")
                    else:
                        if comparison_alg.is_overlapping(tbox, dbox):
                            results.append("TP")
                            self.logger.info("TP")
                        else:
                            results.append("FP")
                            results.append("FN")
                            self.logger.info("FP/FN, not enough overlap.")
        return results
    
    def compare_one(self, box, trainingbox, comparison_alg, threshold=0.0):
        if box.get_confidence() < threshold:
            self.logger.info(
                f'FN, training box not detected at confidence {threshold}.'
                )
            return False
        box.to_four_corners()
        if not comparison_alg.have_intersection(trainingbox, box):
            self.logger.info('FN, training box not detected.')
            return False
        else:                                       
            self.logger.info("Training box intersects with detected box.")
        if not comparison_alg.compare_class(trainingbox, box):
            self.logger.info("FP/FN, incorrect class")
            return False
        else:
            t_area = trainingbox.area()
            d_area = box.area()
            inter = comparison_alg.intersection_area(trainingbox, box)
            if inter == 0:
                self.logger.info(
                    "FP, detection does not intersect with training box."
                    )
                return False
            elif inter == t_area:
                if comparison_alg.training_contained(trainingbox, box):
                    self.logger.info("TP")
                    return True
                else:
                    self.logger.info("FP/FN, detection area too large.")
                    return False
            elif inter == d_area:
                if comparison_alg.detection_sufficient(trainingbox, box):
                    self.logger.info("TP")
                    return True
                else:
                    self.logger.info("FP/FN, detection area too small.")
                    return False
            else:
                if comparison_alg.is_overlapping(trainingbox, box):
                    self.logger.info("TP")
                    return True
                else:
                    self.logger.info("FP/FN, not enough overlap.")
                    return False
    
    def set_logger(self, logger_name=''):
        self.logger = logging.getLogger(logger_name+'.modeltests')
        
class GlycanContaining(ModelTest):
    def is_glycan(self, glycans, threshold=0.5):
        for glycan in glycans:
            confidence = glycan.get_confidence()
            if confidence >= threshold:
                return True
        return False
    
class CompositionCheck(ModelTest):
    def check_composition(self, known_comp_dict, model_comp_dict):
        for mono in known_comp_dict.keys():
            if int(known_comp_dict[mono]) != int(model_comp_dict[mono]):
                return False
        return True