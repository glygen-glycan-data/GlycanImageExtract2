#!.venv/bin/python
import os
import sys
import argparse
import logging
import fitz  # PyMuPDF
import shutil
import json
import time

from BKGlycanExtractor.glyomicsclient import *
from BKGlycanExtractor.bbox import BoundingBox, PDFBoundingBox, PDFConversionContext

parser = argparse.ArgumentParser(description="Annotate PDF")

parser.add_argument(
    '--pdf',
    type = str,
    nargs = "+",
    # required = True,
    help = 'PDF Manuscript(s). Required.'
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

class InputItem:
    '''
    Stores the input item and its metadata (so that user can submit both pmid and pdf's at the same time via cmd line args)
    Note: can also be extended to support file_url's - API framework supports the upload_file request.

    This class helps in recognising the type of submission while using APIFramework/glyomics client.
    '''

    def __init__(self, input_type, value, index):
        # validate type
        if input_type not in ('pdf', 'pmid'):
            raise ValueError(f"Invalid type: {input_type}. Must be 'pdf' or 'pmid'")

        self.type = input_type
        self.value = value
        self.index = index

        # get basename only for pdf's
        if self.type == 'pdf':
            self.basename = os.path.splitext(value)[0]
        else:
            self.basename = f'PMID-{value}'

    def is_pdf(self):
        return self.type == 'pdf'

    def is_pmid(self):
        return self.type == 'pmid'

    def get_annotated_filename(self):
        '''Returns the expected annotated PDF filename for this input item.'''
        if self.type == 'pdf':
            return f'{self.basename}.annotated.pdf'
        elif self.type == 'pmid':
            return f'{self.basename}.annotated.pdf'
        return None

def build_input_items(pdf_list=None, pmid_list=None):
    '''Build InputItem list from PDF and PMID lists.'''
    
    input_items = []
    
    if pdf_list:
        if len(pdf_list) != len(set(pdf_list)):
            print("Provided PDF's are not unique")
            sys.exit(1)
        for pdf in pdf_list:
            input_items.append(InputItem('pdf', pdf, len(input_items)))
    
    if pmid_list:
        if len(pmid_list) != len(set(pmid_list)):
            print("Provided PMID's are not unique")
            sys.exit(1)
        for pmid in pmid_list:
            input_items.append(InputItem('pmid', pmid, len(input_items)))
    
    return input_items

args = parser.parse_args()

# Build unified input items list
input_items = build_input_items(args.pdf, args.pmid)

total_inputs = len(input_items)
if args.json is not None:
    assert len(args.json) == total_inputs, f"Number of JSON files ({len(args.json)}) must match number of inputs ({total_inputs})"
if args.taskid is not None:
    assert len(args.taskid) == total_inputs, f"Number of task IDs ({len(args.taskid)}) must match number of inputs ({total_inputs})"
if args.resubmit:
    assert not args.json, "Cannot use --resubmit with --json"
    assert not args.taskid, "Cannot use --resubmit with --taskid"

client = ExtractorClient(apiurl=args.extractorurl)

needsresults = set()
all_json_data = {}
resultfilename = {}

for i, item in enumerate(input_items):
    if item.is_pdf():
        if not os.path.exists(item.value):
            print(f"Error: PDF file not found: {item.value}")
            sys.exit(1)
            # assert os.path.exists(pdf)
    if item.is_pmid():
        # TODO: validate if pmid is Open Source, else skip and notify user
        pass
    
    if args.json:
        resultfilename[i] = args.json[i]
        assert os.path.exists(resultfilename[i])
    else:
        if item.is_pdf():
            resultfilename[i] = item.basename+".results.json"
        elif item.is_pmid():   # pmid
            resultfilename[i] = f"PMID-{item.value}.results.json"

    if not os.path.exists(resultfilename[i]) or args.resubmit:
        if args.taskid:
            taskid = args.taskid[i]
        elif item.is_pmid():
            print(f"PMID {item.value} submitted for analysis")
            taskid = client.submit_pmid(item.value, curation_task=True)
        else:
            print(os.path.split(item.value)[1],"PDF submitted for analysis.")
            taskid = client.submit_manuscript_file(item.value, curation_task=True)

        json_data = client.retrieve_once(taskid,asis=True)
        with open(resultfilename[i],'w') as f:
            json.dump(json_data,f,indent=2)
    else:
        with open(resultfilename[i], 'r') as f:
            json_data = json.load(f)
        if not json_data.get('finished',False):
            tmp_json_data = client.retrieve_once(json_data['id'],asis=True)
            if 'submission_detail' not in tmp_json_data:
                # result file has non-existent taskid
                if item.is_pmid():
                    print(f"PMID {item.value} resubmitted for analysis (bad taskid).")
                    taskid = client.submit_pmid(item.value)
                else:
                    print(os.path.split(item.value)[1],"resubmitted for analysis (bad taskid).")
                    taskid = client.submit_manuscript_file(item.value, curation_task=True)

                json_data = client.retrieve_once(taskid,asis=True)
                with open(resultfilename[i],'w') as f:
                    json.dump(json_data,f,indent=2)
    all_json_data[i] = json_data
    if not json_data.get('finished',False):
        needsresults.add(i)

completed = set()
while True:
    for i in sorted(needsresults):
        if i in completed:
            continue
        taskid = all_json_data[i]['id']
        json_data = client.retrieve_once(taskid,asis=True)
        input_item = json_data['submission_detail']['original_file_name']
        if json_data.get('finished',False):
            all_json_data[i] = json_data
            completed.add(i)
            if json_data['state'] == "Complete":
                print(input_item,"analysis complete.")
            elif json_data['state'] == "Error":
                print(input_item,"analysis error.")
            # basename = os.path.splitext(args.pdf[i])[0]
            with open(resultfilename[i],'w') as wh:
                wh.write(json.dumps(json_data))
            print("Wrote results JSON:",resultfilename[i])
        else:
            if json_data.get('status'):
                print(input_item,"analysis in progress:",json_data['status'])
            elif json_data['state'] == "Running":
                print(input_item,"analysis in progress.")
            else:
                pass # print(pdf,"analysis queued.")
    if completed == needsresults:
        break
    time.sleep(15)

for i,input_item in enumerate(input_items):

    if all_json_data[i].get('state') == "Error":
        print(input_item.value,"skipping due to analysis error.")
        continue

    original_filepath = all_json_data[i]['result']['abs_original_filepath']
    doc = fitz.open(original_filepath)
    basename = input_item.basename

    if os.path.exists(basename + ".annotated.pdf") or \
        os.path.exists(basename + ".annotated.tsv"):
        print(f'{basename}.pdf,skipping due to presence of output files.')
        continue

    image_data = []

    anyvotes = False
    for result in all_json_data[i]['result']['figure_result']:
        fig_num = result["image_number"]
        taskid = all_json_data[i]['id']

        # page_num - 1, because semantics counts page number starting from 1
        # but fitz accesses page numbers starting from 0
        page = doc[result["page_number"]-1]   

        # add figure boxes on the pdf with a fig: <fig_number> comment
        try:

            fig_annot = page.add_rect_annot(result["pdf_fig_bbox"])
            fig_annot.set_colors(stroke=(0, 0, 1)) 
            fig_annot.set_border(width=0.5) 
                        
            # set fig id
            fig_annot.set_info(content=f"fig:{result['image_count']}")
            fig_annot.update()

            pdf_context_instance = PDFConversionContext.from_result_dict(result)

            for glycan in result["glycans"]:
                pdf_gly_box = pdf_context_instance.to_pdf_bbox(glycan["bbox"])
                gly_annot = page.add_rect_annot(pdf_gly_box.bbox())

                gid = f"G{fig_num}.{glycan['fig_glycan_count']}"
                url = client.url() + f"/result/{taskid}#glycan-{fig_num}-{glycan['fig_glycan_count']}"
                content = (
                    f"id: {gid}\n"
                    f"url: {url}\n"
                )

                gly_annot.set_info(content=content)
                gly_annot.set_border(width=0.5) 
                gly_annot.update()

                votes = glycan.get('upvotes',0)-glycan.get('downvotes',0)
                if votes != 0:
                    anyvotes = True

                image_data.append({
                    "ID": gid,
                    "xref": result.get("xref"),
                    "page_num": result["page_number"],
                    "fig_num": fig_num,
                    "accession": glycan.get('accession', ''),
                    "iupac": glycan.get('IUPAC', ''),
                    "composition": glycan.get('composition_str', ''),
                    'wurcs': glycan.get('WURCS', ''),
                    'votes': votes,
                    "url": url,
                    # uncomment the below if we decide to use bounding box info from the semnatics -
                    # for now it is decided to use info that is extracted during the extarct_figures step
                    # using the fitz model to extract co-ordinates of the annotations on the pdf
                    # "_gly_bbox": glycan.get("bbox"),  
                    # "_fig_width": result['width'],     
                    # "_fig_height": result['height']    
                })

        except Exception as e:
            print(f"\nException occured while drawing bounding box on pdf: {e}")

    doc.save(basename + ".annotated.pdf")
    doc.close()
    print("Wrote annotated PDF:",basename + ".annotated.pdf")   

    wh = open(basename + ".annotated.tsv",'w')
    headers = "ID xref page_num fig_num accession iupac composition wurcs votes url".split()
    if not anyvotes:
        headers.remove("votes")    
    print("\t".join(headers),file=wh)
    for row in image_data:
        print("\t".join(map(str,map(row.get,headers))),file=wh)
    wh.close()
    print("Wrote annotation table:",basename + ".annotated.tsv")