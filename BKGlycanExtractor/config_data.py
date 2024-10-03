'''
The class accepts the configs.ini reference passed from a Finder class and the Finder class_name.
Provides file location for - weights/config/color_range.
Ensures that the Finder and ConfigData class doesn't require the configs.ini file which is part of the design
'''

class ConfigData:
    def __init__(self,config,defaults,class_name):
        self.config = config
        self.class_name = class_name
        self.defaults = defaults
        self.get_config()

    def get_config(self):
        for section in self.config.sections():
            if section.startswith('Finder') and self.config[section]['class'] == self.class_name:
                self.weights = self.config[section].get('weights',None)
                self.config_net = self.config[section].get('config',None)
                self.color_range = self.config[section].get('color_range',None)

    def get_param(self,param_name=None,**kwargs):
        '''
        Get weights/config/color_range from the self.config referernce.
        Helper function to prioritize values from kwargs > config > defaults.
        '''

        location = './BKGlycanExtractor/config/'

        # kwargs
        if kwargs.get(param_name,None) is not None:
            return kwargs.get(param_name)

        # configs.ini
        if param_name == 'weights':
            value = self.weights
        elif param_name == 'config':
            value = self.config_net
        elif param_name == 'color_range':
            value = self.color_range
        
        if value is not None:
            return location + value

        # default
        return self.defaults.get(param_name) #defaults
    