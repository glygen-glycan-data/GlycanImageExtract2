### Steps to generate random SVG images
* In the terminal run: `svglinkout.sh` or `bash svglinkout.sh`
* A folder called glycan_images will be generated which will contain 100 SVG images

### Running the main file
* The `main_file.py` will convert all the SVG images into PNG images and text files required for processing (if they are not already present)
* The main_file accepts various command_line arguments - remember to provide the folder name which contains all the images
* This file generates: Image Semantics, IUPAC sequence's and Annotates Images (populated in /annotated_images directory)

### Plotting PR Curves
* PR curves can be created for various different components of a Glycan: Monosaccharides, Roots, Links using box predictions or semantics.

* `boxes_pr.py` - generates PR curves for YOLO box predictions
* `semantics_pr.py` - generates PR curves for Semantics

* `glycan_compare.py` - generates PR curves for the Entire Glycan


### Running the WebApplication
* Switch to the WebApplication directory. 

* You can run the Flask App using either of the two method's:
    - Running the scripts file:  `GlyImageExtractor.sh start/stop <pipeline_name>` or `bash GlyImageExtractor.sh start/stop <pipeline_name>` 

    - Running the python file: `GlyImageExtractor.py --p <pipeline_name>`

<p>

#### Note:
* Refer to `BKGlycanExtractor/config/configs.ini` for pipeline names and predictor name's.
* Use a python virtual environment to run all the python files.