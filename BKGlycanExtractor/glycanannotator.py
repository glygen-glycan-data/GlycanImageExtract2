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

from . semantics import FigureSemantics
from . distproc import DistributedProcessing as dp

class GlycanExtractorPipeline():
    
    pipeline_stages = ['figure','glycan']

    # defaults = {
    #     'figure_steps': [],
    #     'glycan_steps': [],
    # }

    def __init__(self,**kwargs):

        # shifted it inside - so that a new instance of defaults 
        # is created with ever new class initailization - otherwise the same class variable was getting modfied
        self.defaults = {
            'figure_steps': [],
            'glycan_steps': [],
        }

        self.steps = {}
        for stage in self.pipeline_stages:
            self.steps[stage] = Config.get_param(stage+'_steps', Config.STEPS, kwargs, self.defaults)

    # step should be a finder instance
    def add_step(self,stage,step):
        assert stage in ("figure","glycan"), "Bad stage specification: "+stage
        self.steps[stage].append(step)
            
    def get_steps(self,stage):
        assert stage in ("figure","glycan"), "Bad stage specification: "+stage
        return self.steps.get(stage, None)
            
    # steps should be a list of finder instances, shallow copy!
    def set_steps(self,stage,steps):
        assert stage in ("figure","glycan"), "Bad stage specification: "+stage
        self.steps[stage] = list(steps) if steps else []

    # Shallow clone, finders should be stateless
    def clone(self):
        gep = GlycanExtractorPipeline()
        for stage in self.pipeline_stages:
            gep.set_steps(stage,self.get_steps(stage))
        return gep

    def clear(self):
        for figstep in self.steps['figure']:
            figstep.clear()
        for glystep in self.steps['glycan']:
            glystep.clear()
    
    def run(self,image,progress_callback=None):
        # empty figure semantics
        figure_semantics = FigureSemantics(image_path=image)
        
        if progress_callback:
            progress_callback(stage="PIPELINE",checkpoint="START")
            progress_callback(stage="FIGURE",checkpoint="START")

        for figstep in self.steps['figure']:
            figstep.execute(figure_semantics)

        nglycan = len(figure_semantics.glycans())
        if progress_callback:
            progress_callback(stage="FIGURE",checkpoint="DONE",nglycan=nglycan)

        for i,glycan_semantics in enumerate(figure_semantics.glycans()):
            if progress_callback:
                progress_callback(stage="GLYCAN",checkpoint="START",index=i+1,nglycan=nglycan)
            for glystep in self.steps['glycan']:
                glystep.execute(glycan_semantics)
            if progress_callback:
                progress_callback(stage="GLYCAN",checkpoint="DONE",index=i+1,nglycan=nglycan)

        if progress_callback:
            progress_callback(stage="PIPELINE",checkpoint="DONE",nglycan=nglycan)

        return figure_semantics

    def dorun(self,image,**kwargs):
        return self.run(image)

    def runall(self,images,workers=None,verbose='TQDM'):
        return dp.process(workers=workers,target=self.dorun,
                          tasks=images,verbose=verbose)

    def run_evaluation(self,image,boxesonly=False):

        figure_semantics = FigureSemantics(image_path=image)
        
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

        final_step = self.steps['glycan'][-1]

        result =  None
        for glycan_semantics in figure_semantics.glycans():
            for glystep in self.steps['glycan'][:-1]:
                glystep.execute(glycan_semantics)
            result = (final_step.execute(glycan_semantics,boxesonly=boxesonly),glycan_semantics)
            break

        return result

class Config_Manager(object):

    default_config_folder = os.path.join(os.path.split(__file__)[0],"config")
    default_config_filename = "configs.ini"

    # init can accept a different config_filename and if it located outside the default folder - the custom folder can also be specified
    def __init__(self, config_filename=None, config_folder=None, config_fullpath=None):
        self.config_folder = config_folder or self.default_config_folder
        config_filename = config_filename or self.default_config_filename

        if config_fullpath is None:
            config_fullpath = os.path.join(self.config_folder,config_filename)
        else:
            config_fullpath = config_fullpath

        self.config = configparser.ConfigParser()
        self.config.read(config_fullpath)

    def config_filename(self,filename):
        return os.path.join(self.config_folder,filename)

    def has(self, section, key):
        return key in self.config[section]
    
    def get(self, section, key, default=None):
        return self.config[section].get(key,default)
    
    def get_config(self, instance_name):
        return Config(self,instance_name)

    def get_pipeline(self, pipeline_name):
        conf = self.get_config("Pipeline:" + pipeline_name)
        return GlycanExtractorPipeline(__config__=conf)

    def list(self, prefix):
        names = []
        for sec in self.config.sections():
            if sec.startswith(prefix+":"):
                names.append(sec.split(":",1)[1])
        return sorted(names)

    def list_finders(self):
        return self.list("Finder")

    def list_pipelines(self):
        return self.list("Pipeline")

    # added kwargs here
    def get_finder(self, finder_name):
        module = importlib.import_module(".pipeline",package="BKGlycanExtractor")
        conf = self.get_config("Finder:" + finder_name)
        assert conf.has("class"), "Finder %s: class not specified"
        findercls = getattr(module,conf.get("class"))
        new_conf = copy.deepcopy(conf)
        return findercls(__config__=new_conf)

    # add get_finders - for comma seperated finders
    def get_finders(self,finder_names):
        finder_names = [f.strip() for f in finder_names.split(',') if f.strip()]

        finders = [self.get_finder(finder_name) for finder_name in finder_names]

        return finders

    def get_one_finder(self):
        return self.get_finder(self.list_finders()[0])

class Config(object):
    def __init__(self,config_manager,section_name):
        if not config_manager.config.has_section(section_name):
            raise LookupError("Configuration has no section: "+section_name)
        self.section_name = section_name
        self.config_manager = config_manager

    def has(self,key):
        return self.config_manager.has(self.section_name,key)

    def get(self,key,default=None):
        # Retrieve string value for key from the relevant section
        val = self.config_manager.get(self.section_name,key,default)
        if val:
            return val.strip()
        return val

    def step_names(self,key,default=None):
        if self.has(key):
            steps = [ s.strip() for s in self.get(key).split(',') if s.strip() ]
            return steps
        return default

    # def get_steps(self,key,default=None):
    #     if self.has(key):
    #         steps = [ s.strip() for s in self.get(key).split(',') ]
    #         other_steps = [ self.config_manager.get_finder(name) for name in steps ]
    #         return [ self.config_manager.get_finder(name) for name in steps ]
    #     return default

    def get_steps(self, key, default=None):
        if self.has(key):
            steps = [s.strip() for s in self.get(key).split(',') if s.strip() ]
            return [self.config_manager.get_finder(name) for name in steps]
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

    # @staticmethod
    # def get_param(key,datatype,kwargs={},defaults={}):
    #     # if kwrags were provided by a user - it takes precedence over the parameters in the config file
    #     if key in kwargs:
    #         return kwargs[key]
    #     value = copy.copy(defaults.get(key))
    #     config = kwargs.get('__config__')
    #     if config:
    #         # print("key",key,value,datatype)
    #         value = getattr(config,datatype)(key,value)
    #     # return kwargs.get(key,value)
    #     return value

    @staticmethod
    def get_param(key, datatype, kwargs=None, defaults=None):
        kwargs = kwargs or {}
        defaults = defaults or {}

        # Step 1: kwargs takes highest precedence
        if key in kwargs:
            return kwargs[key]

        # Step 2: start with default
        value = copy.copy(defaults.get(key))

        # Step 3: check primary config (__config__)
        config = kwargs.get('__config__')
        if config and config.has(key):
            value = getattr(config, datatype)(key, value)
            return value  # primary config wins over secondary

        # Step 4: check secondary config if present
        secondary_config = kwargs.get('__secondary_config__')
        if secondary_config and secondary_config.has(key):
            value = getattr(secondary_config, datatype)(key, value)

        return value

    @staticmethod
    def get_finder_name(kwargs={}):
        config = kwargs.get('__config__')
        if config:
            finderstring,name = config.section_name.split(":",1)
            assert finderstring == "Finder"
            return name
        return None
