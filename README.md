# GlycanImageExtract2

New mainprogram_example.py is compatible with all updates, old example programs may not be. 
It expects 3 command line arguments: a file to extract glycans from, the name of the pipeline you want to use (as named in configs.ini), and the configuration file directory.
Configuration file directory is optional; if not given it defaults to BKGlycanExtractor/config.
Pipeline name is optional, if not given it defaults to YOLOMonosAnnotator. A pipeline name is required to give a configuration file directory argument; if a directory is given without a pipeline name it defaults to using BKGlycanExtractor/config.

All configuration files (YOLO weights and .cfg files, etc) can be downloaded at https://drive.google.com/drive/u/1/folders/1cK7xwAKl5jwezDBZRUDyYVltVHv1NsRf and placed in the BKGlycanExtractor/config directory with the configs.ini file.
If they are not manually downloaded, the program will attempt to download on request.
You can create a new configuration directory if desired; place configs.ini inside this directory with all other configuration files and give the new directory as the third argument to mainprogram_example.py.

The image extractor should be run in a virtual environment. requirements.txt outlines the packages sufficient for setup of the virtual environment.

Training data used to train the YOLO models provided in the linked Google Drive is available in the TrainingData folder of that same Google Drive. This is provided for examples of the training images and classes used. The classes.txt file within the zip file lists classes used.
* glycan.zip: Glycan_300img_5000iterations.weights
* plusindividuals.zip: largerboxes_plusindividualglycans.weights
* orientation.zip: orientation_redo.weights
* monos.zip: yolov3_monos_new_v2.weights

[LabelImg](https://github.com/heartexlabs/labelImg) was used to view and create YOLO training boxes.

