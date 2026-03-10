import os
import sys
# import argparse
import logging
import fitz  # PyMuPDF
import shutil
import json
import time

from .image_manager import Image_Manager
from .bbox import PDFConversionContext

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

def annotate_from_webapp(json_path, pdf_path, extractorurl, output_dir):
    '''
    WebApp use case: Build pdf annotations and TSV directly from JSON files.
    No submission or polling needed.
    The output files will be stored in annotated_files folder (located within the webapp's static files)
    '''
    
    # Load JSON data and create input items
    all_json_data = {}
    input_items = []

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Original PDF file not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Create input item and supporting json - for this PDF
    # TODO - remove index values from InputItem (json dict and input_items later), if needed --> need to check the code to verify if this change should be made or not
    input_items.append(InputItem('pdf', pdf_path, 0))
    all_json_data[0] = json_data

    # Build annotations
    build_annotations(input_items, all_json_data, extractorurl, output_dir=output_dir)

def annotate(pdf, pmid, json_file, taskid, extractorurl, resubmit):
    '''
    Script/CLI use case: Submit, poll for results, then annotate.
    
    Args:
        pdf: List of PDF file paths or directories
        pmid: List of PubMed IDs
        json_file: Optional list of JSON result files (if results already exist)
        taskid: Optional list of task IDs to retrieve
        resubmit: Force resubmission even if results exist
    '''

    from .glyomicsclient import ExtractorClient

    # Expand PDF directories
    if pdf:
        pdf_manager = Image_Manager(pdf, pattern='*.pdf', exclude='*.annotated.pdf')
        pdf = pdf_manager.images
    
    # Build unified input items list
    input_items = build_input_items(pdf, pmid)
    total_inputs = len(input_items)
    
    # Validate inputs
    if json_file is not None:
        if len(json_file) != total_inputs:
            raise ValueError(f"Number of JSON files ({len(json_file)}) must match number of inputs ({total_inputs})")
    if taskid is not None:
        if len(taskid) != total_inputs:
            raise ValueError(f"Number of task IDs ({len(taskid)}) must match number of inputs ({total_inputs})")
    if resubmit:
        if json_file:
            raise ValueError("Cannot use --resubmit with --json")
        if taskid:
            raise ValueError("Cannot use --resubmit with --taskid")
    
    # Validate PDF files exist
    for item in input_items:
        if item.is_pdf():
            if not os.path.exists(item.value):
                raise FileNotFoundError(f"PDF file not found: {item.value}")
    
    client = ExtractorClient(apiurl=extractorurl)
    # Submit/retrieve results
    all_json_data, resultfilename = submit_and_retrieve(
        input_items, json_file, taskid, resubmit, client
    )
    
    # Poll for incomplete results
    needsresults = {i for i, data in all_json_data.items() if not data.get('finished', False)}
    if needsresults:
        poll_for_completion(needsresults, all_json_data, resultfilename, client)
    
    # Build annotations
    build_annotations(input_items, all_json_data, client.url())

def submit_and_retrieve(input_items, json_file, taskid, resubmit, client):
    '''
    Submit PDFs/PMIDs or retrieve existing results.
    
    Returns:
        (all_json_data dict, resultfilename dict)
    '''
    all_json_data = {}
    resultfilename = {}
    
    for i, item in enumerate(input_items):
        # Determine result filename
        if json_file:
            resultfilename[i] = json_file[i]
            if not os.path.exists(resultfilename[i]):
                raise FileNotFoundError(f"JSON file not found: {resultfilename[i]}")
        else:
            if item.is_pdf():
                resultfilename[i] = item.basename + ".results.json"
            else:  # pmid
                resultfilename[i] = f"PMID-{item.value}.results.json"
        
        # Submit or load existing results
        if not os.path.exists(resultfilename[i]) or resubmit:
            if taskid:
                current_taskid = taskid[i]
            elif item.is_pmid():
                print(f"PMID {item.value} submitted for analysis")
                current_taskid = client.submit_pmid(item.value, curation_task=True)
            else:
                print(f"{os.path.split(item.value)[1]} PDF submitted for analysis.")
                current_taskid = client.submit_manuscript_file(item.value, curation_task=True)
            
            json_data = client.retrieve_once(current_taskid, asis=True)
            with open(resultfilename[i], 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            # Load existing results
            with open(resultfilename[i], 'r') as f:
                json_data = json.load(f)
            
            # Check if existing result is still valid
            if not json_data.get('finished', False):
                tmp_json_data = client.retrieve_once(json_data['id'], asis=True)
                if 'submission_detail' not in tmp_json_data:
                    # Bad taskid, resubmit
                    if item.is_pmid():
                        print(f"PMID {item.value} resubmitted for analysis (bad taskid).")
                        current_taskid = client.submit_pmid(item.value, curation_task=True)
                    else:
                        print(f"{os.path.split(item.value)[1]} resubmitted for analysis (bad taskid).")
                        current_taskid = client.submit_manuscript_file(item.value, curation_task=True)
                    
                    json_data = client.retrieve_once(current_taskid, asis=True)
                    with open(resultfilename[i], 'w') as f:
                        json.dump(json_data, f, indent=2)
        
        all_json_data[i] = json_data
    
    return all_json_data, resultfilename
    
def poll_for_completion(needsresults, all_json_data, resultfilename, client):
    '''
    Poll for incomplete analysis results until all are finished.
    
    Args:
        needsresults: Set of indices that need polling
        all_json_data: Dict of all JSON data (modified in place)
        resultfilename: Dict mapping indices to result filenames
    '''
    completed = set()
    
    while needsresults:
        for i in sorted(needsresults):
            if i in completed:
                continue
            
            current_taskid = all_json_data[i]['id']
            json_data = client.retrieve_once(current_taskid, asis=True)
            
            input_item_name = json_data.get('submission_detail', {}).get('filename', f'Item {i}')
            
            if json_data.get('finished', False):
                all_json_data[i] = json_data
                completed.add(i)
                
                if json_data.get('state') == "Complete":
                    print(f"{input_item_name} analysis complete.")
                elif json_data.get('state') == "Error":
                    print(f"{input_item_name} analysis error.")
                
                with open(resultfilename[i], 'w') as wh:
                    json.dump(json_data, wh, indent=2)
                print(f"Wrote results JSON: {resultfilename[i]}")
            else:
                if json_data.get('status'):
                    print(f"{input_item_name} analysis in progress: {json_data['status']}")
                elif json_data.get('state') == "Running":
                    print(f"{input_item_name} analysis in progress.")
        
        if completed == needsresults:
            break
        time.sleep(15)

def build_annotations(input_items, all_json_data, base_url, output_dir=None):
    """
    Build annotated PDFs and TSV files from JSON results.

    - Idea is to create temporary files (annotated pdf and tsv) with the newest version of updates (json).

    - Once completed - remove the old files and rename these temporary files to the original file name.
    
    - incase error/exception - remove the temporary files, clean up step.

    This ensures that the users see either the old complete file or the new complete file.
    They never see partially written files - because write happens to a temp file and then it is renamed to the original file name.

    If process dies/exceptions midway, only the temp files are the bad/corrupted - which will be cleaned up in the finally block.

    Args:
        input_items: List of InputItem objects
        all_json_data: Dict mapping indices to JSON data
        output_dir: Absolute dir to store annotated pdf + tsv.
    """
    for i, input_item in enumerate(input_items):
        json_data = all_json_data[i]

        # Skip error / missing-result / missing-PDF cases
        if json_data.get('state') == "Error":
            print(f"{input_item.value} skipping due to analysis error.")
            continue

        result = json_data.get('result')
        if not result:
            print(f"{input_item.value} skipping: no result data.")
            continue

        original_filepath = input_item.value
        if not original_filepath or not os.path.exists(original_filepath):
            print(f"{input_item.value} skipping: original PDF not found.")
            continue

        if output_dir:
            save_dir = output_dir
        else:
            # save in the same location where the json file exists
            base_dir = os.path.dirname(input_item.basename)
            if base_dir:
                save_dir = base_dir
            else:
                save_dir = os.getcwd()

        os.makedirs(save_dir, exist_ok=True)

        # pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]

        # paths for the annotated pdf and tsv file
        pdf_basename = os.path.splitext(os.path.basename(original_filepath))[0]
        annotated_pdf_path = os.path.join(save_dir, pdf_basename + ".annotated.pdf")
        tsv_path = os.path.join(save_dir, pdf_basename + ".annotated.tsv")

        # each process has its own PID, so this kind of file naming avoid temp file naming collisions
        # accross processes (although all files/submissions are stored in a unique folder which is named using a hash and avoids collision)
        # Maybe the below step is not necessary??
        pid = os.getpid()
        temp_pdf = os.path.join(save_dir, f"{pdf_basename}.annotated.pdf.tmp.{pid}")
        temp_tsv = os.path.join(save_dir, f"{pdf_basename}.annotated.tsv.tmp.{pid}")

        doc = None
        try:
            doc = fitz.open(original_filepath)
            image_data = []

            figure_results = result.get('figures', [])
            taskid = json_data.get('id')

            for result_item in figure_results:
                annotate_figure(doc, result_item, taskid, image_data, base_url)

            # Write PDF to temp file
            doc.save(temp_pdf)
            doc.close()
            doc = None

            # Write TSV to temp file
            anyvotes = any(row.get('votes', 0) != 0 for row in image_data)
            write_tsv(temp_tsv, image_data, anyvotes)

            # Replace old final files with new ones (this avoids partial writes)
            # So remove old files, rename the temp files which will serve as the annoated pdf and tsv
            if os.path.exists(annotated_pdf_path):
                os.remove(annotated_pdf_path)
            if os.path.exists(tsv_path):
                os.remove(tsv_path)

            os.rename(temp_pdf, annotated_pdf_path)
            os.rename(temp_tsv, tsv_path)

            print(f"Wrote annotated PDF: {annotated_pdf_path}")
            print(f"Wrote annotation table: {tsv_path}")

        except Exception as e:
            print(f"Error during PDF/TSV generation: {e}", file=sys.stderr)
            raise

        finally:
            if doc is not None:
                try:
                    doc.close()
                except:
                    pass

            # Clean up any leftover temp files (in case of error before rename)
            for tmp in (temp_pdf, temp_tsv):
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except:
                        pass

def annotate_figure(doc, result_item, taskid, image_data, base_url):
    '''Draw figure box and glycan boxes / labels on a pdf page'''

    image_count = result_item.get("image_count")
    page_num = result_item.get("page_number", 1)

    # Page indices in fitz are 0 based
    try:
        page = doc[page_num - 1]
    except IndexError:
        print(f"Warning: Page {page_num} not found in PDF, skipping figure {image_count}")
        return 

    try:
        # Figure bounding box
        pdf_fig_bbox = result_item.get("pdf_fig_bbox")
        if pdf_fig_bbox:
            fig_annot = page.add_rect_annot(pdf_fig_bbox)
            fig_annot.set_colors(stroke=(0, 0, 1))
            fig_annot.set_border(width=0.5)

            content = f"fig:{image_count}\n"
            xref = result_item.get("xref")
            if xref is not None and xref > 0:
                content += f"xref: {xref}\n"

            fig_annot.set_info(content=content)
            fig_annot.update()

        pdf_context_instance = PDFConversionContext.from_result_dict(result_item)
        glycans = result_item.get("glycans", [])

        for glycan in glycans:
            try:
                bbox = glycan.get("bbox")
                if not bbox:
                    continue

                pdf_gly_box = pdf_context_instance.to_pdf_bbox(bbox)
                gly_annot = page.add_rect_annot(pdf_gly_box.bbox())

                gid = f"G{image_count}.{glycan.get('fig_glycan_count', '?')}"

                url = (
                    f"{base_url}/result/{taskid}"
                    f"#glycan-{image_count}-{glycan.get('fig_glycan_count', '?')}"
                )

                content = f"id: {gid}\nurl: {url}\n"

                gly_annot.set_info(content=content)
    
                votes = glycan.get('upvotes', 0) - glycan.get('downvotes', 0)
                color = (0, 0, 1)
                if votes > 0:
                    color = (0, 1, 0)       # green
                elif votes < 0:
                    color = (1, 0, 0)         # red

                gly_annot.set_colors(stroke=color)

                gly_annot.set_border(width=0.5)
                gly_annot.update()

                votes = glycan.get('upvotes', 0) - glycan.get('downvotes', 0)

                image_data.append({
                    "ID": gid,
                    "xref": result_item.get("xref"),
                    "page_num": page_num,
                    "fig_num": image_count,     # fig_num was used as a key in the past tsv file, so keep the same key for backward compatability
                    "accession": glycan.get('accession', ''),
                    "iupac": glycan.get('IUPAC', ''),
                    "composition": glycan.get('composition_str', ''),
                    "wurcs": glycan.get('WURCS', ''),
                    "votes": votes,
                    "url": url,
                })
            except Exception as e:
                print(f"Exception occurred while processing glycan: {e}")
                continue

    except Exception as e:
        print(f"Exception occurred while drawing bounding box on PDF: {e}")

def write_tsv(tsv_path, image_data, anyvotes):
    ''' Write the annotation TSV for the PDF'''

    headers = "ID xref page_num fig_num accession iupac composition wurcs votes url".split()
    if not anyvotes:
        headers.remove("votes")

    with open(tsv_path, 'w') as wh:
        print("\t".join(headers), file=wh)
        for row in image_data:
            print("\t".join(map(str, [row.get(h, '') for h in headers])), file=wh)

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
