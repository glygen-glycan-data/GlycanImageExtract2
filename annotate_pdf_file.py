#!.venv/bin/python
import os
import sys
import argparse
import logging
import fitz  # PyMuPDF
import shutil
import json
import time

from BKGlycanExtractor import annotate

parser = argparse.ArgumentParser(description="Annotate PDF")

# TODO - if url path is not valid for PDF - state a warning

parser.add_argument(
    '--pdf',
    type = str,
    nargs = "+",
    # required = True,
    help = 'PDF Manuscript(s), Accpets pdf paths and directories.'
)

parser.add_argument(
    '--pmid',
    type = str,
    nargs = "+",
    # required = True,
    help = 'Pubmed id (Note that the Pubmed resources should be open access).'
)

parser.add_argument(
    '--json',
    type = str,
    nargs = "+",
    default = None,
    help = 'JSON format extractor result(s).'
)

parser.add_argument(
    '--taskid',
    type = str,
    default = None,
    nargs = "+",
    help = 'Task ID(s).'
)

parser.add_argument(
    '--extractorurl',
    type = str,
    default = 'https://extractor.glyomics.org/',
    help = 'Extractor URL.'
)

parser.add_argument(
    '--resubmit',
    action = 'store_true',
    default = False,
    help = 'Resubmit analysis, even if results JSON is present.'
)

# TODO - maybe create an output_dir arg?

args = parser.parse_args()

annotate(
    pdf = args.pdf,
    pmid = args.pmid,
    json_file = args.json,
    taskid = args.taskid,
    extractorurl = args.extractorurl,
    resubmit = args.resubmit,
)




# '''
# Storing the XREF in the figures annotation - because XREF is a figure property and not an individual
# annotations (eg. glycan) property.
# Eg. for a figure with/without glycan annotations -  will still need xref (if present, so that the 
# image can be extracted in its original format without having to specify a fixed dpi) information during
# extract_annotations stage to save the image (or else a default dpi will be used) and a good place to store this information would be in the
# figures annotations information itself.
# The TSV file generated only stores information about the glycan (monos, root, links) annotations, so the xref can be tracked via
# the figure annotation in the pdf and this ensures that the dimensions of the figure remain consistent during any extraction activity.
# '''