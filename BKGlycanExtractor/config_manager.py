import os
import configparser
import sys
# gdrive needs to be imported

from . import glycanfinding
from . import monosaccharideid
from . import glycanconnections
from . import rootmonofinding
from . import glycanbuilding
from . import glycansearch
from . import glycanannotator as ga


class Config_Manager():
    def __init__(self):
        # add code to search for the correct path of the config file and check if the file really exists
        self.config_folder = './BKGlycanExtractor/config'
        self.config_file_name = 'configs.ini'
        self.config_file_path = os.path.join(self.config_folder, self.config_file_name)
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file_path)


    def get(self,pipeline_name, **kw):
        # config = configparser.ConfigParser()
        # self.config.read(self.config_file_path)
        pipelines = []
        known_monos = []
        for key, value in self.config.items():
            if value.get("sectiontype") == "annotator":
                pipelines.append(key)
            elif value.get("sectiontype") == "knownmonos":
                known_monos.append(key)

        if kw.get('training_data',False) is True:
            try:
                known_mono_methods = self.config[pipeline_name]
            except KeyError:
                print(pipeline_name,"is not a valid pipeline.")
                print("Valid pipelines:", known_monos)
                sys.exit(1)
            return known_mono_methods
        else:
            try:
                annotator_methods = self.config[pipeline_name]
            except KeyError:
                print(pipeline_name,"is not a valid pipeline.")
                print("Valid pipelines:", pipelines)
                sys.exit(1)
            return annotator_methods

    def get_pipeline(self,pipeline_name,**kw):
        annotator_methods = self.get(pipeline_name,**kw)
        pipeline = ga.Annotator(annotator_methods)
        return pipeline



    def read_pipeline_configs(self, annotator_methods):
        methods = {}
        
        method_descriptions = {
            "glycanfinder": {"prefix": "glycanfinding.", "multiple": False},
            "mono_id": {"prefix": "monosaccharideid.", "multiple": False},
            "connector": {"prefix": "glycanconnections.", "multiple": False},
            "rootfinder": {"prefix": "rootmonofinding.", "multiple": False},
            "builder": {"prefix": "glycanbuilding.", "multiple": False},
            "search": {"prefix": "glycansearch.", "multiple": True},
            }
        for method, desc in method_descriptions.items():
            if desc.get("multiple"):
                all_methods = annotator_methods.get(method) 
                if all_methods is not None:
                    method_names = annotator_methods.get(method).split(",")
                    methods[method] = []
                    for method_name in method_names:
                        methods[method].append(self.setup_method(
                            self.config, desc.get("prefix"), self.config_folder, method_name
                            ))
            else:
                method_name = annotator_methods.get(method)
                if method_name is not None:
                    # print(method_name)
                    methods[method] = self.setup_method(
                        self.config, desc.get("prefix"), self.config_folder, method_name
                        )

        execution_order = annotator_methods.get('execution_order', None)
        if execution_order is not None:
            methods['execution_order'] = execution_order

        return methods


    def setup_method(
            self, configparserobject, prefix, directory, method_name
            ):
        gdrive_dict = {
            "coreyolo.cfg":
                "1M2yMBkIB_VctyH01tyDe1koCHT0U8cwV",
            "Glycan_300img_5000iterations.weights":
                "1xEeMF-aJnVDwbrlpTHkd-_kI0_P1XmVi",
            "largerboxes_plusindividualglycans.weights":
                "16-AIvwNd-ZERcyXOf5G50qRt1ZPlku5H",
            "monos2.cfg":
                "15_XxS7scXuvS_zl1QXd7OosntkyuMQuP",
            "orientation_redo.weights":
                "1KipiLdlUmGSDsr0WRUdM0ocsQPEmNQXo",
            "orientation.cfg":
                "1AYren1VnmB67QLDxvDNbqduU8oAnv72x",
            "orientation_flipped.cfg":
                "1YXkSWjqjbx5_GkCrOdkIHrSocTAqu9WX",
            "orientation_flipped.weights":
                "1PQH6_JPpE_1o9WdhKAIGJdmOF5fI39Ew",
            "yolov3_monos_new_v2.weights":
                "1h-QiO2FP7fU7IbvZjoF7fPf55N0DkTPp",
            "yolov3_monos_random.weights": 
                "1m4nJqxrJLl1MamIugdyzRh6td4Br7OMg",
            "yolov3_monos_largerboxes.weights":
                "1WQI9UiJkqGx68wy8sfh_Hl5LX6q1xH4-",
            "rootmono.cfg":
                "1RSgCYxkNvrPYann5MG7WybyBZS2UA5v0",
            "yolov3_rootmono.weights":
                "1dUTFbPA7XV-HztWeM5uto2mF_xo5F-3Z"
        }
        
        method_values = configparserobject[method_name]
        method_class = method_values.pop("class")
        method_configs = {}
        for key, value in method_values.items():
            filename = os.path.join(directory,value)
            if os.path.isfile(filename):
                method_configs[key] = filename
            else:
                gdrive_id = gdrive_dict.get(value)
                if gdrive_id is None:
                    raise FileNotFoundError(
                        value + 
                        "was not found in configs directory or Google Drive"
                        )
                getfromgdrive.download_file_from_google_drive(
                    gdrive_id, filename
                    )
                method_configs[key] = filename
        if not method_configs:
            return eval(prefix+method_class+"()")
        return eval(prefix+method_class+"(method_configs)")


