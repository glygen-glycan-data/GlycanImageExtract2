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

import importlib

from . semantics import Figure_Semantics, Glycan_Semantics
from . distproc import DistributedProcessing as dp

class GlycanExtractorPipeline():
    
    pipeline_stages = ['figure','glycan', 'clean']

    defaults = {
        'figure_steps': [],
        'glycan_steps': [],
        'clean_steps': []
    }

    def __init__(self,**kwargs):
        self.steps = {}
        for stage in self.pipeline_stages:
            if stage == 'clean':
                data = Config.get_param(stage+'_steps', Config.IMAGE_STEPS, kwargs, self.defaults) 
                if data:
                    self.steps[stage] = data
            else:
                self.steps[stage] = Config.get_param(stage+'_steps', Config.STEPS, kwargs, self.defaults)

    # step should be a finder instance
    def add_step(self,stage,step):
        assert stage in ("figure","glycan","clean_image"), "Bad stage specification: "+stage
        self.steps[stage].append(step)
            
    def get_steps(self,stage):
        assert stage in ("figure","glycan","clean_image"), "Bad stage specification: "+stage
        return self.steps[stage]
            
    # steps should be a list of finder instances, shallow copy!
    def set_steps(self,stage,steps):
        assert stage in ("figure","glycan","clean_image"), "Bad stage specification: "+stage
        self.steps[stage] = list(steps)

    # Shallow clone, finders should be stateless
    def clone(self):
        gep = GlycanExtractorPipeline()
        for stage in self.pipeline_stages:
            gep.set_steps(self,stage,self.get_steps(stage))
        return gep
    
    def run(self,image):
        # empty figure semantics
        figure_semantics = Figure_Semantics(image)
        
        for figstep in self.steps['figure']:
            figstep.execute(figure_semantics)

        for glycan_semantics in figure_semantics.glycans():
            for glystep in self.steps['glycan']:
                glystep.execute(glycan_semantics)

        return figure_semantics

    def dorun(self,image,**kwargs):
        return self.run(image)

    def runall(self,images,workers=None,verbose=False):
        return dp.process(workers=workers,target=self.dorun,
                          tasks=images,verbose=verbose)

    def run_evaluation(self,image,boxesonly=False):

        figure_semantics = Figure_Semantics(image)
        
        if len(self.steps['glycan']) == 0:

            # special case for testing glycan finders
            assert boxesonly == True

            for figstep in self.steps['figure'][:-1]:
                figstep.execute(figure_semantics)

            final_step = self.steps['figure'][-1]

            result = final_step.execute(figure_semantics,boxesonly=boxesonly)
            return result,figure_semantics
        
        # typical case
        
        for figstep in self.steps['figure']:
            figstep.execute(figure_semantics)

        assert len(figure_semantics.glycans()) == 1

        final_step = self.steps['glycan'][-1]

        result =  None
        for glycan_semantics in figure_semantics.glycans():
            for glystep in self.steps['glycan'][:-1]:
                glystep.execute(glycan_semantics)
            result = (final_step.execute(glycan_semantics,boxesonly=boxesonly),glycan_semantics)

        return result

class Config_Manager(object):

    default_config_folder = os.path.join(os.path.split(__file__)[0],"config")
    config_filename = "configs.ini"

    def __init__(self, config_folder=default_config_folder):
        self.config_folder = config_folder
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(self.config_folder,self.config_filename))

    def has(self, section, key):
        return key in self.config[section]
    
    def get(self, section, key, default):
        return self.config[section].get(key,default)
    
    def get_config(self, instance_name):
        return Config(self,instance_name)

    def get_pipeline(self, pipeline_name):
        conf = self.get_config("Pipeline:" + pipeline_name)
        return GlycanExtractorPipeline(__config__=conf)

    def get_finder(self, finder_name):
        module = importlib.import_module(".pipeline",package="BKGlycanExtractor")
        conf = self.get_config("Finder:" + finder_name)
        assert conf.has("class"), "Finder %s: class not specified"
        findercls = getattr(module,conf.get("class"))
        try:
            return findercls(__config__=conf)
        except:
            # For DefaultOrientationRootFinder - it doesn't take any configs
            return findercls()

    def get_image_finder(self, finder_name):
        res = {}
        conf = self.get_config("Image:" + finder_name)
        res['crop_image'] = conf.get_bool('crop_image', False)
        res['clean_image'] = conf.get_bool('clean_image', False)

        return res


class Config(object):
    def __init__(self,config_manager,section_name):
        assert config_manager.config.has_section(section_name), "Configuration has no section: "+section_name
        self.section_name = section_name
        self.config_manager = config_manager

    def has(self,key):
        return self.config_manager.has(self.section_name,key)

    def get(self,key,default=None):
        # Retrieve string value for key from the relevant section
        return self.config_manager.get(self.section_name,key,default).strip()

    def step_names(self,key,default=None):
        if self.has(key):
            steps = [ s.strip() for s in self.get(key).split(',') ]
            return steps
        return default

    def get_steps(self,key,default=None):
        if self.has(key):
            steps = [ s.strip() for s in self.get(key).split(',') ]
            other_steps = [ self.config_manager.get_finder(name) for name in steps ]
            return [ self.config_manager.get_finder(name) for name in steps ]
        return default

    def get_image_steps(self,key,default=None):
        if self.has(key):
            name = self.get(key).strip()
            step = self.config_manager.get_image_finder(name)
            return step
        return default


    def get_int(self,key,default=None):
        if self.has(key):
            return int(self.get(key))
        return default

    def get_float(self,key,default=None):
        if self.has(key):
            return float(self.get(key))
        return default

    def get_bool(self,key,default=None):
        if self.has(key):
            return self.get(key).lower() in ('true','yes','1')
        return default

    def get_config_filename(self,key,default=None):
        return os.path.join(self.config_manager.config_folder,self.get(key,default))

    # Implement a multi-stage strategy for getting parameters from
    # class defaults, then configuration, then keyword arguments

    BOOL = 'get_bool'
    CONFIGFILE = 'get_config_filename'
    STR = 'get'
    INT = 'get_int'
    FLOAT = 'get_float'
    STEPS = 'get_steps'
    IMAGE_STEPS = 'get_image_steps'

    @staticmethod
    def get_param(key,datatype,kwargs={},defaults={}):
        value = copy.copy(defaults.get(key))
        config = kwargs.get('__config__')
        if config:
            value = getattr(config,datatype)(key,value)
        return kwargs.get(key,value)

    @staticmethod
    def get_finder_name(kwargs={}):
        config = kwargs.get('__config__')
        if config:
            finderstring,name = config.section_name.split(":",1)
            assert finderstring == "Finder"
            return name
        return None
