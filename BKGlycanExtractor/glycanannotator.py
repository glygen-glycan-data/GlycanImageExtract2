# -*- coding: utf-8 -*-
"""
Class to work with the glycan annotation pipeline, 
read in configs, set up annotation methods, etc
"""

import configparser
import json
import logging
import os
import shutil
import sys
import time
import copy


import cv2
import fitz
import numpy as np
import pdfplumber
import importlib

from .config import getfromgdrive
from .pygly3.GlycanFormatter import GlycoCTFormat, GlycoCTParseError

from . import glycanfinding
from . import monosaccharideid
from . import glycanconnections
from . import rootmonofinding
from . import glycanbuilding
from . import glycansearch

from BKGlycanExtractor.semantics import Image_Semantics, Figure_Semantics, Glycan_Semantics

class Annotator():
    def __init__(self,config):
        self.config = config
    
    def get_finder(self,key,default=None):
        instance = self.config.get_finder(key,default)
        if instance:
            return instance
        return default
        
    def run(self,image):
        figure_semantics = Figure_Semantics(image)

        image_steps = self.get_finder('image_steps')[0]
        glycan_steps = self.get_finder('glycan_steps')
        # search_steps = self.get_finder('search_steps')

        # * Handle any errors
        image_steps.execute(figure_semantics)
        for step in glycan_steps:
            for gly_obj in figure_semantics.glycans():
                step.execute(obj=gly_obj)

        return figure_semantics.semantics
    
class Config_Manager():
    # reads/gets configs from configs.ini
    def __init__(self):
        self.config_folder = './BKGlycanExtractor/config'
        self.config_file_name = 'configs.ini'
        self.config_file_path = os.path.join(self.config_folder, self.config_file_name)
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file_path)
        self.pipeline_name = None

    def get(self,pipeline_name, **kw):
        if isinstance(pipeline_name,list) is False:
            pipeline_name = pipeline_name.split(',')
        
        pipeline_configs = {}
        for name in pipeline_name:
            pipeline = 'Pipeline:' + name
            config = Config(pipeline)

            glycan_steps = config.get_steps('glycan_steps')
            step_instances = config.get_finder('glycan_steps')

            pipeline_configs[pipeline] = {
                'steps': glycan_steps,
                'instantiated_finders': step_instances
            }

        return pipeline_configs
        
    def get_pipeline(self,pipeline_name,**kw):
        self.pipeline_name = "Pipeline:" + pipeline_name
        config = Config(self.pipeline_name)
        pipeline = Annotator(config=config)
        return pipeline
        
    def read_pipeline_configs(self,finder_type):
        methods = {}
        method_descriptions = {
            "glycan_finder": {"prefix": "glycanfinding."},
            "mono_finder": {"prefix": "monosaccharideid."},
            "link_finder": {"prefix": "glycanconnections."},
            "root_finder": {"prefix": "rootmonofinding."},
            "sequence_builder": {"prefix": "glycanbuilding."},
            "sequence_lookup": {"prefix": "glycansearch."},
        }

        if finder_type in method_descriptions:
            if finder_type in method_descriptions:
                prefix = method_descriptions[finder_type]['prefix']
                return prefix
            else:
                print(f"The mentioned step {finder_type} does not exist. For more information check the configs.ini file...")

class Config():
    def __init__(self,section_name):
        self.section_name = section_name
        self.config_manager = Config_Manager()

    # eg. key - glycan_finder
    def get_str(self,section,key,default=None):
        # Retrieve string value for key from the relevant section
        return self.config_manager.config[section].get(key,default)

    # def get_int(key,default=None):

    # def get_float(key,default=None):

    def get_config_filename(self,key,default=None):
        # Get the full pathname for a key
        filename = self.config[self.section_name].get(key,default)
        if filename:
            name = '/'.join(filename.split('.')[:-1])
            return os.path.join(os.getcwd(), name + '.py')
        return default

    def get_steps(self,key,default=None):
        # list of keys of Finder instances to execute, CSV?
        steps = self.config_manager.config[self.section_name].get(key).split(',')
        return steps

    def get_finder(self,key,default=None):
        # print("---->>> 2 Config",self.section_name)
        steps = self.get_steps(key,default)
        instance_list = []
        for step in steps:
            step = step.strip()
            prefix = self.config_manager.read_pipeline_configs(step)
            config_finder = self.get_str(self.section_name,step)
            finder_section = "Finder:" + config_finder
            finder_method = self.get_str(finder_section, "class")

            module = importlib.import_module('BKGlycanExtractor.'+prefix[:-1])
    
            finder_class = getattr(module, finder_method)
            instance = finder_class(config=self.config_manager.config)

            instance_list.append(instance)
        return instance_list if instance_list is not None else default