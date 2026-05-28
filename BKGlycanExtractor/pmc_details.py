# TODO - add better custom expections for the API's

import requests, urllib, traceback, sys, unicodedata
import xml.etree.ElementTree as ET
import os
import re
import shutil
import tarfile
import traceback
from PIL import Image
import json
import threading
import time


# create a PMID validation cache - in order to avoid hitting pmid to pmcid conversion API twice
# so now when the UI request is made - the API will be used and response will be cached, to avoid calling
# the API again during the download pmid resoources stage.
_PMID_CACHE = {}
_PMID_CACHE_LOCK = threading.Lock()
_PMID_CACHE_TTL = 1800       # 30 mins

def _normalize_pmid(pmid):
    pmid = str(pmid).strip()
    if pmid.lower().endswith(".pdf"):
        pmid = pmid.rsplit(".", 1)[0]
    return pmid

def cache_pmid_validation(pmid, body, status):
    if status != 200 or not body.get("valid"):
        return
    key = _normalize_pmid(pmid)
    with _PMID_CACHE_LOCK:
        _PMID_CACHE[key] = (time.time() + _PMID_CACHE_TTL, body, status)

def get_cached_pmid_validation(pmid):
    key = _normalize_pmid(pmid)
    with _PMID_CACHE_LOCK:
        entry = _PMID_CACHE.get(key)
        if not entry:
            return None
        expires_at, body, status = entry
        if time.time() > expires_at:
            del _PMID_CACHE[key]
            return None
        return body, status

class PMCData:
    '''
    basic stuff here
    general stuff for OLD PMID based extraction.
    pmid validation - add those method here, and url mapping (GET/POST) can be done in GlyImageExtractor because these are specific --> which needs to be 
    linked to APIFramework -- need to seperate the concerns of having all the mappings in APIFramework
    
    '''

    devemail = 'nje5%2bextractor@georgetown.edu'      # works
    # devemail = 'nje5+extractor@georgetown.edu'

    @staticmethod
    def validate_pmid(pmid):
        '''
        Validates is the given PMID has a PMCID and that the resources for the PMCID are Open Access (check if zip file can be retrieved)
        Returns (dict, http_status)
        '''
        # developer_email = "nje5%2bextractor@georgetown.edu"
        # pmid = pmid.strip()
        # if pmid.endswith(".pdf"):
        #     pmid = pmid.rsplit('.', 1)[0]

        pmid = _normalize_pmid(pmid)

        pmid_to_pmc_converter_api = f'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={pmid}&tool=extract&email={PMCData.devemail}&idtype=pmid&format=json'

        try:
            resp = requests.get(pmid_to_pmc_converter_api, timeout=10)
        except requests.exceptions.RequestException as e:
            return {"valid": False, "error": "PubMed Central ID service unreachable: %s" % e}, 503

        if not resp.ok:
            return {
                "valid": False,
                "error": "ID service returned HTTP %s" % resp.status_code,
            }, 502

        text = (resp.text or "").strip()
        if not text:
            return {"valid": False, "error": "ID service returned an empty response"}, 502

        try:
            resp_json = resp.json()
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "error": "ID service returned invalid JSON (try again)",
            }, 502

        # API-level errors from idconv (dont rely on raise_for_status)
        if resp_json.get("status") == "error":
            error_message = "; ".join(
                e.get("message", str(e)) for e in resp_json.get("errors", [])
            )
            http_status = resp_json.get("http_status", 400)
            return {"valid": False, "error": error_message or "ID conversion failed"}, http_status
        records = resp_json.get("records") or []
        if not records:
            return {"valid": False, "error": f"PMID {pmid} is not in PubMed Central"}, 400

        pmcid = records[0].get("pmcid")
        if not pmcid:
            return {"valid": False, "error": f"PMID {pmid} is not in PubMed Central"}, 400

        body, status = PMCData.validate_pmcid_resources(pmid, pmcid)
        cache_pmid_validation(pmid, body, status)  # only caches valid 200
        return body, status
    
    @staticmethod
    def validate_pmcid_resources(pmid, pmcid):
        pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        try:
            r = requests.get(pmc_api, timeout=10)
        except requests.exceptions.RequestException as e:
            return {
                "valid": False,
                "error": "PMC Open Access service unreachable: %s" % e,
            }, 503
        if not r.ok:
            return {
                "valid": False,
                "error": "PMC Open Access service returned HTTP %s" % r.status_code,
            }, 502
        text = (r.text or "").strip()
        if not text:
            return {
                "valid": False,
                "error": "PMC Open Access service returned an empty response",
            }, 502
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return {
                "valid": False,
                "error": "PMC Open Access service returned invalid XML (try again)",
            }, 502
        link = root.find(".//link[@format='tgz']")
        if link is None:
            return {
                "valid": False,
                "error": f"PMID {pmid} is not Open Access in PubMed Central",
            }, 400
        return {
            "valid": True,
            "success": f"Given PMCID: {pmcid} is Open Access",
            "resource": {
                "href": link.get("href"),
                "format": link.get("format"),
                "pmcid": pmcid,
            },
        }, 200


    # Generated by Gemini to solve the problem of embedded HTML 
    # itaics markers <i> </i> in manuscript titles.
    @staticmethod
    def get_unicode_italic(text):
        """Converts standard A-Z and a-z characters to Unicode Mathematical Italics."""
        if not text:
            return ""
        
        italic_map = {}
        # Map Uppercase A-Z to Unicode Mathematical Italic block
        for i, char_code in enumerate(range(65, 91)):
            italic_map[chr(char_code)] = chr(0x1D434 + i)
            
        # Map Lowercase a-z 
        for i, char_code in enumerate(range(97, 123)):
            if chr(char_code) == 'h':
                # 'h' is a Unicode exception (Planck constant)
                italic_map['h'] = '\u210E' 
            else:
                italic_map[chr(char_code)] = chr(0x1D44E + i)

        # Translate the characters, leaving non-alphabet characters untouched
        return "".join(italic_map.get(c, c) for c in text)

    @staticmethod
    def toascii(s):
        nfkd_form = unicodedata.normalize('NFKD', s)
        ascii_text = nfkd_form.encode('ascii', 'ignore')
        return ascii_text.decode('utf-8')

    @staticmethod
    def extract_full_text(element):
        """Reconstructs the full text from an ElementTree element, italicizing <i> tags."""
        # 1. Start with the text before any child tags
        text_parts = [element.text or ""]
        
        # 2. Iterate through child elements
        for child in element:
            if child.tag == 'i':
                # Convert text inside <i> to Unicode italics
                italicized_text = PMCData.get_unicode_italic(child.text)
                text_parts.append(italicized_text)
            else:
                # Handle other tags normally
                text_parts.append(child.text or "")
            
            # 3. Always append the tail (text immediately following the child tag)
            text_parts.append(child.tail or "")
            
        return "".join(text_parts)

    @staticmethod
    def citation_details(pmid):
        baseurl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        query = dict(db="pubmed",id=pmid)
        fullurl = baseurl + "?" + urllib.parse.urlencode(query)
        # print(fullurl)
        try:
            resp = requests.get(fullurl, timeout=5)
        except (requests.exceptions.Timeout,requests.exceptions.ReadTimeout,requests.exceptions.RequestException) as e:
            traceback.format_exc()
            resp = None
        root = ET.fromstring(resp.content)
        articles = list(root.iter('Article'))
        if len(articles) == 0:
            return None
        article = articles[0]
        volume = article.findtext('Journal/JournalIssue/Volume')
        issue = article.findtext('Journal/JournalIssue/Issue')
        articledate = article.find("ArticleDate")
        if articledate is not None:
            year = articledate.findtext('Year')
        else:
            pubdate = article.find("Journal/JournalIssue/PubDate")
            if pubdate is not None:
                year = pubdate.findtext('Year')
        journal = article.findtext('Journal/Title')
        isoabbrev = article.findtext('Journal/ISOAbbreviation')
        
        if isoabbrev == "bioRxiv" and not volume and not issue:
            for elid in article.findall('ELocationID'):
                if elid.attrib.get('EIdType') == "pii":
                    volume = elid.text

        doi = None
        for elid in article.findall("ELocationID"):
            if elid.attrib.get('EIdType') == "doi":
                doi = elid.text

        # Deal with italicized text in the titles
        title = PMCData.extract_full_text(article.find("ArticleTitle"))
        ascii_title = PMCData.toascii(title)

        page = article.findtext("Pagination/MedlinePgn")
        authors = []
        for author in article.findall("AuthorList/Author"):
            lastname = author.findtext("LastName")
            forename = author.findtext("ForeName")
            initials = author.findtext("Initials")
            collective = author.findtext("CollectiveName")
            authordict = dict(lastname=lastname,forename=forename,initials=initials,collective=collective)
            for k in list(authordict):
                if not authordict.get(k):
                    authordict[k] = None
                else:
                    authordict['ascii_'+k] = PMCData.toascii(authordict[k])
            authors.append(authordict)

        details = dict(title=title,ascii_title=ascii_title,
                    journal=journal,journal_abbrev=isoabbrev,
                    issue=issue,volume=volume,year=year,
                    authors=authors,page=page,pmid=pmid,doi=doi)

        citation = PMCData.format_citation(details)
        details['citation'] = citation
        details['ascii_citation'] = PMCData.toascii(citation)

        for k in list(details):
            if not details.get(k):
                details[k] = None
        return details

    @staticmethod
    def format_citation(d):
        author_list = []
        for au in d['authors']:
            if au['lastname'] and au['initials']:
                author_list.append(au['lastname'] + " " + au['initials'])
            elif au['lastname']:
                author_list.append(au['lastname'])
            elif au['collective']:
                author_list.append(au['collective'])

        authors = ", ".join(author_list)
        title = d['title'].rstrip('.')
        rest = [ d['journal_abbrev'] ]
        if d['volume'] and d['issue']:
            rest.append("%(year)s;%(volume)s(%(issue)s)"%d)
        elif d['volume']:
            rest.append("%(year)s;%(volume)s"%d)
        else:
            rest.append("%(year)s"%d)
        if d['page']:
            rest[-1] += ":%(page)s"%d
        rest = " ".join(rest)
        return ". ".join([authors,title,rest]) + "."

    @staticmethod
    def lookup(doi=None,pmid=None,pmcid=None):
        baseurl = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
        query = dict(tool="extractor",email=PMCData.devemail,format="json")
        if pmid:
            query.update(dict(ids=pmid,idtype="pmid"))
        elif pmcid:
            query.update(dict(ids=pmcid,idtype="pmcid"))
        elif doi:
            query.update(dict(ids=doi,idtype="doi"))
        fullurl = baseurl + "?" + urllib.parse.urlencode(query)
        try:
            resp = requests.get(fullurl, timeout=5).json()
        except (requests.exceptions.Timeout,requests.exceptions.ReadTimeout,requests.exceptions.RequestException) as e:
            traceback.format_exc()
            resp = None
        if not resp or 'records' not in resp or len(resp['records']) != 1:
            return None
        record = resp['records'][0]
        if record.get('status') == 'error':
            return None
        return dict(doi=record['doi'],pmid=str(record['pmid']),pmcid=record['pmcid'])


class PMCTarFile:
    '''
    class extract items from tar file and check things and adding them to the correct spots
    with file names

    pmid validation - in base class --> the api can be used to download and store Tar file in the correct path
    
    tar extraction and parsing for file - can be done here (remove logic from processjob)
    '''

    NAMESPACES = {
        'nlm': 'http://dtd.nlm.nih.gov/2.0/xsd/archivearticle',
        'xlink': 'http://www.w3.org/1999/xlink',
        'mml': 'http://www.w3.org/1998/Math/MathML'
    }
    XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

    def __init__(self, tar_filepath):
        self.tar_filepath = tar_filepath
        self.root = None
        self.figure_info_by_basename = {}


    @staticmethod
    def download_and_prepare_pmid_data(pmid, file_dir, pdf_filename):
        
        # validate if PMCID resources are Open Access before proceeding
        # pmc_resp_json, pmc_status = PMCData.validate_pmid(pmid)
        # check cache for pmid validation
        cached = get_cached_pmid_validation(pmid)
        if cached:
            pmc_resp_json, pmc_status = cached
        else:
            pmc_resp_json, pmc_status = PMCData.validate_pmid(pmid)
        
        if pmc_status != 200 or not pmc_resp_json.get('valid'):
            return pmc_resp_json, pmc_status
                    
        resource = pmc_resp_json.get("resource") or {}
        href = resource.get("href")
        pmcid = resource.get("pmcid")
        if not pmcid or not href:
            return {"valid": False, "error": "OA response missing pmcid or download href"}, 400

        # 1) get citation from json response - if available
        # pmc_publication - is the PMC publication information obtained
        # from hittin the PMC API (useful when publication information is only partially present in the xml document provided by pmc)
        # pmc_publication = resource.get("pmc_publication")
        expected_pdf_path = os.path.join(file_dir, pdf_filename)
        try:
            # 2) extract the href link, which is in ftp (NCBI supports both ftp and https protocols)
            # temporary PMC deprecation fix...
            href = href.replace('pub/pmc/','pub/pmc/deprecated/')
            # Convert FTP to HTTPS
            download_url = href.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")

            # 3) Download the zipped file to the input folder
            tar_filepath = os.path.join(file_dir, f"PMID-{pmid}.tar.gz")
            with open(tar_filepath, "wb") as f:
                with requests.get(download_url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)

            # 4) Extract data from tar file
            # Note: The zip file may contain multiple pdf's, so the main pdf filename is same
            # as the xml filename - the below code tracks and finds the correct pdf to use
            with tarfile.open(tar_filepath, "r:gz") as tar:
                # Single pass: collect nxml files and their corresponding PDFs
                nxml_files = []
                pdf_files = {}
                for member in tar.getmembers():
                    base = os.path.basename(member.name).lower()
                    ext = os.path.splitext(base)[1]
                    if ext == '.nxml':
                        nxml_basename = os.path.splitext(base)[0]
                        nxml_files.append((member, nxml_basename))
                    elif ext == '.pdf':
                        pdf_basename = os.path.splitext(base)[0]
                        pdf_files[pdf_basename] = member
                # Now extract the main pdf and rename it as PMID-<PMID>.pdf
                for nxml_member, nxml_basename in nxml_files:
                    if nxml_basename in pdf_files:
                        pdf_member = pdf_files[nxml_basename]
                        tar.extract(pdf_member, file_dir, filter="data")
                        # Rename the PDF
                        old_pdf_path = os.path.join(file_dir, pdf_member.name)
                        if os.path.exists(old_pdf_path):
                            os.rename(old_pdf_path, expected_pdf_path)
                        else:
                            print(f"File not found: {old_pdf_path}")
        except requests.exceptions.RequestException as e:
            return {"valid": False, "error": f"PMC download failed: {e}"}, 502
        except (tarfile.TarError, EOFError) as e:
            return {"valid": False, "error": f"Invalid PMC archive: {e}"}, 422
        except OSError as e:
            return {"valid": False, "error": f"File error: {e}"}, 500
        finally:
            # cleanup step to remove residual archieved paths, the original zipped file is preserved.
            pmc_folder_path = os.path.join(file_dir, pmcid)
            try:
                shutil.rmtree(pmc_folder_path)
            except FileNotFoundError:
                pass

        # if loop finishes with no PDF renamed:
        if not os.path.exists(expected_pdf_path):
            return {"valid": False, "error": "No PDF found in PMC package"}, 404

        return {'valid': True}, 200

    # TODO - are the method parameters fine??
    # this needs some parameters - base_path, pmid, 
    # prepare ready tp use figures and metadata here
    def extract_figures_from_tar(self, figures_dir):
        '''
        Extract PMID tar file from folder: input/task_id, 
        parse NXML metadata, 
        extract figure images,
        and return sorted image filenames + figure metadata map.
        '''

        # base_path = os.path.dirname(os.path.abspath(__file__))
        # pmc_publication_info = self.task_detail.get("pmc_publication")
        # figures_src = os.path.join(base_path, "input", self.id, f"PMID-{self.pmid}.tar.gz")
        fig_to_label_map = {}
        image_files = []
        self.figure_info_by_basename = {}
        figure_info_map = {}

        try:
            with tarfile.open(self.tar_filepath, "r:gz") as tar:
                # pass 1: parse nxml
                for member in tar.getmembers():
                    if not member.name.lower().endswith(".nxml"):
                        continue
                    file_obj = tar.extractfile(member)
                    if not file_obj:
                        continue
                    nxml_content = file_obj.read().decode("utf-8", errors="ignore")
                    # xml_obj = PMCData(nxml_content)
                    try:
                        # self.document_metadata = {k: v for k, v in xml_data.items() if k != "figure_info"}
                        # self.document_metadata["citation"] = xml_data.get("citation", None)
                        self.root = ET.fromstring(nxml_content)
                        self.figure_info_by_basename = self._extract_figure_info() 
                        for basename, info in self.figure_info_by_basename.items():
                            fig_to_label_map[basename] = info.get("figure_number", "")
                    except Exception as e:
                        # self.log_file.write(f"Warning: Could not parse nxml: {e}\n")
                        traceback.print_exc()
                    break
                # pass 2: extract image files
                seen_basenames = set()
                for member in tar.getmembers():
                    filename = os.path.basename(member.name)
                    base_name, ext = os.path.splitext(filename)
                    ext = ext.lower()
                    if ext not in (".jpg", ".jpeg", ".png"):
                        continue
                    if base_name not in fig_to_label_map:
                        continue
                    if base_name in seen_basenames:
                        continue
                    file_obj = tar.extractfile(member)
                    if not file_obj:
                        continue
                    renamed_file_path = os.path.join(figures_dir, filename)
                    with open(renamed_file_path, "wb") as f:
                        f.write(file_obj.read())
                    image_files.append(filename)
                    seen_basenames.add(base_name)
                    fig_info = self.figure_info_by_basename.get(base_name, {}).copy()
                    fig_info["figure_number"] = fig_to_label_map[base_name]
                    figure_info_map[filename] = fig_info
        except Exception as e:
            # self.log_file.write(f"Error: opening tar {self.tar_filepath}: {e}\n")
            traceback.print_exc()
            print()
            return [], {}

        # TODO image/figures variables are not consistent
        # image_files.sort(key=self._pmid_image_sort_key, figure_info_map)
        image_files.sort(key=lambda imfn: self._pmid_image_sort_key(imfn, figure_info_map))

        return image_files, figure_info_map

    def _pmid_image_sort_key(self, imfn, figure_info_map):
        fn = figure_info_map.get(imfn, {}).get("figure_number", "")
        if fn == "":
            return (0, 0)
        try:
            return (0, int(fn))
        except Exception:
            pass
        return (ord(fn[0]), int(fn[1:]) if fn[1:].isdigit() else 0) 

    def _extract_figure_info(self):
        '''Extract figure captions'''
        if self.root is None:
            raise ValueError("Parse NXML and set self.root before extract_figure_info()")
        XLINK_NS = "http://www.w3.org/1999/xlink"
        XLINK_HREF = f"{{{XLINK_NS}}}href"
        figures = {}
        for fig in self.root.findall('.//fig', self.NAMESPACES):
            fig_info = {}
            label_elem = fig.find('label', self.NAMESPACES)
            if label_elem is not None:
                label = (label_elem.text or "").strip()
                if label:
                    m = re.search(r'^\s*\w+\.?\s*(\w?\d+)\.?\s*$', label)
                    if m:
                        fig_info['figure_number'] = m.group(1)
            caption_elem = fig.find('caption', self.NAMESPACES)
            if caption_elem is not None:
                title = caption_elem.find('title', self.NAMESPACES)
                if title is not None:
                    fig_info['caption'] = title.text
                else:
                    caption = []
                    for p in caption_elem.findall('.//p', self.NAMESPACES):
                        text = ''.join(p.itertext()).strip()
                        text = " ".join(text.split())
                        if text:
                            caption.append(text)
                    fig_info['caption'] = ' '.join(caption) if caption else None
            filename = None
            for tag in ('graphic', 'inline-graphic'):
                graphic_elem = fig.find(tag, self.NAMESPACES)
                if graphic_elem is not None:
                    href = graphic_elem.get('href') or graphic_elem.get(XLINK_HREF)
                    if href:
                        filename = os.path.basename(href)
                        break
            if filename:
                fig_info['filename'] = filename
                figures[filename] = fig_info
        return figures

    def figures_metadata(self, figures_dir, **kwargs) -> list[dict]:

        if kwargs.get('input_dir'):         # useful for re-analyze
            self.copy_pmidtar_to_input(kwargs['input_dir'])

        image_files, figure_info_map = self.extract_figures_from_tar(figures_dir)

        if not image_files:
            return

        metadata = []

        # citation is added via xml data parsing during the load/extract pmid figures stage
        for image_count, fig_name in enumerate(image_files, 1):
            image_path = os.path.join(figures_dir, fig_name)

            with Image.open(image_path) as img:
                width, height = img.size

                base_fig_name = fig_name.rsplit('.', 1)[0]

                # look up XML metadata for this renamed figure (if any)
                fig_info = figure_info_map.get(fig_name, {})
                # if the figure_number is empty, can we assume Graphical Abstract?
                caption = fig_info.get("caption","")
                figure_number = fig_info.get("figure_number", "")

                image = {
                    "image_path": image_path,
                    "fig_bbox": [0, 0, width, height],
                    "image_count": image_count,
                    # XML-derived metadata (keys match XMLParser output)
                    "caption": fig_info.get("caption",""),
                    "figure_number": fig_info.get("figure_number", ""),
                    "pmid_job": True, 
                }
                if not image["figure_number"] and not image["caption"]:
                    image["caption"] = "Graphical Abstract"

                metadata.append(image)

                # add status message here or use the ones in process job

        return metadata

    def copy_pmidtar_to_input(self, input_dir):
        # base_path = os.path.dirname(os.path.abspath(__file__))
        # figures_src = os.path.join(base_path, "input", self.id, f"PMID-{self.pmid}.tar.gz")
        if os.path.isfile(self.tar_filepath):
            try:
                shutil.copy2(self.tar_filepath, input_dir)
                return True
            except Exception as e:
                # self.log_file.write(f"Warning: tar copy failed ({figures_src}): {e}\n")
                return False
        # self.log_file.write(f"Warning: tar not found for copy: {figures_src}\n")
        return False


class PMCFiles:
    '''
    after pmid validation.
    class to extract individual files based on requirement and add them to the correct spots.
    '''
    pass


if __name__ == "__main__":
    for doi in sys.argv[1:]:
        pmid = PMCData.lookup(doi=doi)
        if pmid:
            print(doi,pmid['pmid'])
            pmid = pmid['pmid']
        else:
            pmid = doi
        print(PMCData.citation_details(pmid))


    # # XML parser
    # xml_file = '/home/nmathias/GlycanImageExtract2/xml_files/369_330.nxml'

    # xml_obj = XMLParser(xml_file)

    # parsed_data = xml_obj.parse()

    # print(parsed_data)