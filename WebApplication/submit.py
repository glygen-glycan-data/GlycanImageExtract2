import sys
import time
import json
import requests

from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError

from BKGlycanExtractor import BatchAPIFrameworkClient

default_dev_email="nje5+extractor@georgetown.edu"

def _create_batch_client(service_name, baseurl=None, devemail=None, delay=1, maxretry=10):
    """
    Factory function to create batch clients with consistent defaults.
    """

    service_urls = {
        'glylookup': 'https://glylookup.glyomics.org',
        'glymage': 'https://glymage.glyomics.org',
        'subsumption': 'https://subsumption.glyomics.org',
    }
    
    return BatchAPIFrameworkClient(
        apiurl=service_urls.get(service_name, baseurl),
        developer_email=devemail or default_dev_email,
        request_interval=delay,
        max_retrieve_wait=maxretry * delay
    )

def sendToGNOme(*seqs, baseurl=None, devemail=None):
    """
    Submit sequences to GNOme and return Structure Browser URL.
    
    Returns: GNOme Structure Browser URL with ondemandtaskid parameter
    """
    client = _create_batch_client('subsumption', baseurl, devemail)
    
    tasks = [{"seq": seq.strip() if seq else ""} for seq in seqs]
    jobids = client.submit_batch(tasks)
    
    task_id = jobids[0]
    return f"https://gnome.glyomics.org/StructureBrowser.html?ondemandtaskid={task_id}"

def searchGlyImage(*seqs, orientations=None, default_orientation='RL', display='normal',
                   image_format='svg', accession=False, baseurl=None, devemail=None,
                   delay=1, maxretry=10):
    """
    Get images for a batch of sequences or accessions.
    """
    # print("BATCH REQUEST IMAGES", len(seqs))
    
    client = _create_batch_client('glymage', baseurl, devemail, delay, maxretry)
    
    # Validate orientations
    if orientations is not None:
        if len(orientations) != len(seqs):
            raise ValueError("orientations list must have same length as seqs")
        orient_list = orientations
    else:
        orient_list = [default_orientation] * len(seqs)
    
    # Build tasks
    key = 'acc' if accession else 'seq'
    tasks = [
        {
            key: s.strip() if s else "",
            "orientation": orient_list[i],
            "display": display,
            "image_format": image_format
        }
        for i, s in enumerate(seqs)
    ]
    
    # Submit and retrieve
    jobids = client.submit_batch(tasks)
    results = client.retrieve_batch(jobids)
    
    retval = []
    base_url = (baseurl or "https://glymage.glyomics.org").rstrip('/')

    urls = []
    for job in results:
        result_path = job.get('result', '')
        if not result_path:
            urls.append(None)
            continue
        if not result_path.startswith('/'):
            result_path = '/' + result_path
        urls.append(f"{base_url}{result_path}")

    return urls[0] if len(seqs) == 1 else urls

def searchGlyLookup(*seqs, baseurl=None, devemail=None, delay=1, maxretry=10):
    """
    Lookup accessions and WURCS for a batch of IUPAC sequences.
    
    *seqs: Variable number of IUPAC sequence strings
    
    returns: a list of tuples - (accession, wurcs)
    """
    # print("BATCH REQUEST LOOKUP", len(seqs))
    
    client = _create_batch_client('glylookup', baseurl, devemail, delay, maxretry)
    
    # Build tasks
    tasks = [{"seq": seq.strip() if seq is not None else ""} for seq in seqs]
    
    # Submit and retrieve
    jobids = client.submit_batch(tasks)
    results = client.retrieve_batch(jobids)
    
    retval = []
    for job in results:
        result = (None, None)
        for res in job.get("result", []):
            wurcs = None
            for seqrec in res.get("sequences", []):
                if seqrec['format'] == 'WURCS':
                    wurcs = seqrec['seq']
                    break
            result = (res['accession'], wurcs)
            break
        retval.append(result)
    
    return retval[0] if len(seqs) == 1 else retval
