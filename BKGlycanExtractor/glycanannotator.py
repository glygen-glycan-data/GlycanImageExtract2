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

class GlycanExtractorPipeline():
    
    pipeline_stages = ['figure','glycan']

    defaults = {
        'figure_steps': [],
        'glycan_steps': []
    }

    def __init__(self,**kwargs):
        self.steps = {}
        for stage in self.pipeline_stages:
            self.steps[stage] = Config.get_param(stage+'_steps', Config.STEPS, kwargs, self.defaults)

    # step should be a finder instance
    def add_step(self,stage,step):
        assert stage in ("figure","glycan"), "Bad stage specification: "+stage
        self.steps[stage].append(step)
            
    def get_steps(self,stage):
        assert stage in ("figure","glycan"), "Bad stage specification: "+stage
        return self.steps[stage]
            
    # steps should be a list of finder instances, shallow copy!
    def set_steps(self,stage,steps):
        assert stage in ("figure","glycan"), "Bad stage specification: "+stage
        self.steps[stage] = list(steps)

    # Shallow clone, finders should be stateless
    def clone(self):
        gep = GlycanExtractorPipeline()
        for stage in self.pipeline_stages:
            gep.set_steps(self,stage,self.get_steps(stage))
        return gep
            
    def run(self,image):
        figure_semantics = Figure_Semantics(image)
        
        for figstep in self.steps['figure']:
            figstep.execute(figure_semantics)
        
        for glycan_semantics in figure_semantics.glycans():
            for glystep in self.steps['glycan']:
                glystep.execute(glycan_semantics)

        return figure_semantics
    
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

    def get_steps(self,key,default=None):
        if self.has(key):
            steps = [ s.strip() for s in self.get(key).split(',') ]
            return [ self.config_manager.get_finder(name) for name in steps ]
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

    @staticmethod
    def get_param(key,datatype,kwargs={},defaults={}):
        value = defaults.get(key)
        config = kwargs.get('__config__')
        if config:
            value = getattr(config,datatype)(key,value)
        return kwargs.get(key,value)
