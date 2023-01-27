# GlycanImageExtract2

New mainprogram_example.py is compatible with all updates, old example programs may not be. It expects 2 command line arguments: a file to extract glycans from and the configuration file directory.

All configuration files (YOLO weights and .cfg files, etc) can be downloaded at https://drive.google.com/drive/u/1/folders/1cK7xwAKl5jwezDBZRUDyYVltVHv1NsRf and placed in the BKGLycanExtractor/configs directory with the configs.ini file.
If they are not manually downloaded, the program will attempt to download on request.
You can create a new configuration directory if desired; place configs.ini inside this directory with all other configuration files and give the new directory as the second argument to mainprogram_example.py.

The image extractor should be run in a virtual environment running python 3.7. requirements.txt outlines the packages and versions sufficient for setup of the virtual environment.