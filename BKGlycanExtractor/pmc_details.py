import requests, urllib, traceback, sys, unicodedata
import xml.etree.ElementTree as ET
import os
import re
import shutil
import tarfile
from PIL import Image
import json
import threading
import time
from urllib.parse import urlparse

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

        cached = get_cached_pmid_validation(pmid)
        if cached:
            return cached


        pmid_to_pmc_converter_api = f'https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={pmid}&tool=extract&email={PMCData.devemail}&idtype=pmid&format=json'

        try:
            resp = requests.get(pmid_to_pmc_converter_api, timeout=10)
        except requests.exceptions.RequestException as e:
            return {"valid": False, "error": "PubMed Central ID service unreachable: %s" % e}, 503

        if resp.status_code == 429:
            return {
                "valid": False,
                "error": "PubMed Central is busy (rate limited). Please wait a few seconds and try again.",
            }, 429

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
    def get_validated_pmc_resource(pmid):
        """
        Shared validation for PMCTarFile and PMCFiles download paths.
        Returns (pmcid, body, status). pmcid is None on failure.
        """
        body, status = PMCData.validate_pmid(pmid)
        if status != 200 or not body.get("valid"):
            return None, body, status
        resource = body.get("resource") or {}
        pmcid = resource.get("pmcid")
        if not pmcid:
            return None, {"valid": False, "error": "OA response missing pmcid"}, 400
        return pmcid, body, status
    
    @staticmethod
    def validate_pmcid_resources(pmid, pmcid):
        # pmc_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        pmc_api = f"https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:{pmcid[3:]}&metadataPrefix=pmc_fm"
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
        ns = {"art":"https://jats.nlm.nih.gov/ns/archiving/1.4/"}
        meta = root.find('.//art:custom-meta[art:meta-name="pmc-prop-open-access"]',ns)
        isoa = ((meta is not None) and \
                (meta.find("./art:meta-value",ns) is not None) and \
                (meta.find("./art:meta-value",ns).text == "yes"))
        if not isoa:
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
            if child.tag in ('i','italic'):
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


class PMCFigureExtractor:

    IMAGE_EXTS = (".jpg", ".jpeg", ".png")

    def __init__(self):
        self.root = None
        self.figure_info_by_basename = {}

    def _local_tag(self, tag):
        return tag.rsplit('}', 1)[-1] if '}' in tag else tag

    def _find_child(self, parent, local_name):
        for child in parent:
            if self._local_tag(child.tag) == local_name:
                return child
        return None

    def _find_descendants(self, root, local_name):
        for elem in root.iter():
            if self._local_tag(elem.tag) == local_name:
                yield elem

    def _extract_figure_info(self):
        '''Extract figure captions'''
        if self.root is None:
            raise ValueError("Parse NXML and set self.root before extract_figure_info()")
        XLINK_NS = "http://www.w3.org/1999/xlink"
        XLINK_HREF = f"{{{XLINK_NS}}}href"
        figures = {}
        figure_seq = 0

        # for fig in self.root.findall('.//fig', self.NAMESPACES):
        for fig in self._find_descendants(self.root, 'fig'):
            fig_info = {}
            # m = re.search(r'^[a-zA-Z]+(\d+)$',fig.attrib["id"])
            # assert m, f"Can't match figure element id string: {fig.attrib["id"]}."
            # fig_info['figure_id'] = int(m.group(1))

            # label_elem = fig.find('label', self.NAMESPACES)
            label_elem = self._find_child(fig, 'label')
            if label_elem is not None:
                label = (label_elem.text or "").strip()
                if label:
                    m = re.search(r'^\s*(\w+(\s+\w+)*)\.?\s*(\w?\d+)\.?\s*$', label)
                    if m:
                        fig_info['figure_number'] = m.group(3)
                        fig_info['figure_label'] = m.group(1)
            # caption_elem = fig.find('caption', self.NAMESPACES)
            caption_elem = self._find_child(fig, 'caption')
            if caption_elem is not None:
                # title = caption_elem.find('title', self.NAMESPACES)
                title = self._find_child(caption_elem, 'title') if caption_elem is not None else None
                if title is not None:
                    fig_info['caption'] = PMCData.extract_full_text(title)
                else:
                    caption = []
                    # for p in caption_elem.findall('.//p', self.NAMESPACES):
                    for p in self._find_descendants(caption_elem, 'p') if caption_elem is not None else []:
                        text = PMCData.extract_full_text(p).strip()
                        text = " ".join(text.split())
                        if text:
                            caption.append(text)
                    fig_info['caption'] = ' '.join(caption) if caption else None
                if fig_info.get('caption'):
                    fig_info['ascii_caption'] = PMCData.toascii(fig_info['caption'])
            filename = None
            for tag in ('graphic', 'inline-graphic'):
                # graphic_elem = fig.find(tag, self.NAMESPACES)
                graphic_elem = self._find_child(fig, tag)
                if graphic_elem is not None:
                    href = graphic_elem.get('href') or graphic_elem.get(XLINK_HREF)
                    if href:
                        filename = os.path.basename(href)
                        break
            if filename:
                figure_seq += 1
                fig_info['figure_id'] = figure_seq
                fig_info['filename'] = filename
                base_name = os.path.splitext(filename)[0]
                figures[base_name] = fig_info  
        return figures

    def _save_figure_to_dir(self, figures_dir, filename, source):
        '''source: bytes, or path to copy from.'''
        dest_path = os.path.join(figures_dir, filename)
        if isinstance(source, (bytes, bytearray)):
            with open(dest_path, "wb") as f:
                f.write(source)
        else:
            shutil.copy2(source, dest_path)

    def _map_figure_info(self, filename, base_name, fig_to_label_map,
                                   image_files, seen_basenames, figure_info_map):
        image_files.append(filename)
        seen_basenames.add(base_name)
        fig_info = self.figure_info_by_basename.get(base_name, {}).copy()
        fig_info["figure_id"] = fig_to_label_map.get(base_name, "")
        figure_info_map[filename] = fig_info

    def _sorted_image_files(self, image_files, figure_info_map):

        def pmid_image_sort_key(imfn, figure_info_map):
            fn = figure_info_map.get(imfn, {}).get("figure_id", "")
            if fn == "":
                return (0, 0)
            try:
                return (0, int(fn))
            except Exception:
                pass
            return (ord(fn[0]), int(fn[1:]) if fn[1:].isdigit() else 0) 

        image_files.sort(key=lambda imfn: pmid_image_sort_key(imfn, figure_info_map))
        return image_files, figure_info_map

    def _parse_nxml_content(self, nxml_content):
        '''Parse NXML string
        set self.root and figure maps. 
        Returns fig_to_label_map.'''
        fig_to_label_map = {}
        self.figure_info_by_basename = {}
        try:
            self.root = ET.fromstring(nxml_content)
            self.figure_info_by_basename = self._extract_figure_info()
            for base_name, info in self.figure_info_by_basename.items():
                fig_to_label_map[base_name] = info.get("figure_id", "")
        except Exception:
            traceback.print_exc()
            self.figure_info_by_basename = {}
            return {}
        return fig_to_label_map

    def collect_figures(self, figures_dir, fig_to_label_map, image_sources):
        """Save matching images and build sorted metadata map."""
        image_files = []
        figure_info_map = {}
        seen_basenames = set()
        for filename, source in image_sources:
            base_name, ext = os.path.splitext(filename)
            if ext.lower() not in self.IMAGE_EXTS:
                continue
            if base_name not in fig_to_label_map or base_name in seen_basenames:
                continue
            self._save_figure_to_dir(figures_dir, filename, source)
            self._map_figure_info(
                filename, base_name, fig_to_label_map,
                image_files, seen_basenames, figure_info_map,
            )
        return self._sorted_image_files(image_files, figure_info_map)

    def build_figures_metadata(self, figures_dir, image_files, figure_info_map):
        """Turn extracted images into the metadata list processjob expects."""
        metadata = []
        for image_count, fig_name in enumerate(image_files, 1):
            image_path = os.path.join(figures_dir, fig_name)
            with Image.open(image_path) as img:
                width, height = img.size
                fig_info = figure_info_map.get(fig_name, {})
                image = {
                    "image_path": image_path,
                    "fig_bbox": [0, 0, width, height],
                    "image_count": image_count,
                    "caption": fig_info.get("caption", ""),
                    "ascii_caption": fig_info.get("ascii_caption", ""),
                    "figure_number": fig_info.get("figure_number", ""),
                    "figure_id": fig_info.get("figure_id", ""),
                    "figure_label": fig_info.get("figure_label", ""),
                    "pmid_job": True,
                }
                if not image["figure_number"] and not image["caption"]:
                    image["caption"] = "Graphical Abstract"
                    image["ascii_caption"] = "Graphical Abstract"
                metadata.append(image)
        return metadata
    
    
class PMCTarFile:
    '''
    class extract items from tar file and check things and adding them to the correct spots
    with file names

    pmid validation - in base class --> the api can be used to download and store Tar file in the correct path
    
    tar extraction and parsing for file - can be done here (remove logic from processjob)
    '''

    def __init__(self, tar_filepath):
        self.tar_filepath = tar_filepath
        self._figureExtractor = PMCFigureExtractor()


    @staticmethod
    def download_and_prepare_pmid_data(pmid, file_dir, pdf_filename):
        
        # validate if PMCID resources are Open Access before proceeding
        # pmc_resp_json, pmc_status = PMCData.validate_pmid(pmid)
        # check cache for pmid validation
        pmcid, pmc_resp_json, pmc_status = PMCData.get_validated_pmc_resource(pmid)
        if not pmcid:
            return pmc_resp_json, pmc_status
                    
        resource = pmc_resp_json.get("resource") or {}
        href = resource.get("href")
        if not href:
            return {"valid": False, "error": "OA response missing download href"}, 400

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


    def _read_nxml_from_tar(self, tar):
        """Return nxml content string from tar, or None."""
        for member in tar.getmembers():
            if not member.name.lower().endswith(".nxml"):
                continue
            file_obj = tar.extractfile(member)
            if not file_obj:
                continue
            return file_obj.read().decode("utf-8", errors="ignore")
        return None

    def _image_sources_from_tar(self, tar, fig_to_label_map):
        """Return [(filename, bytes), ...] for figure images in tar."""
        image_sources = []
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            base_name, ext = os.path.splitext(filename)
            if ext.lower() not in PMCFigureExtractor.IMAGE_EXTS:
                continue
            if base_name not in fig_to_label_map:
                continue
            file_obj = tar.extractfile(member)
            if not file_obj:
                continue
            image_sources.append((filename, file_obj.read()))
        return image_sources

    def extract_figures_from_tar(self, figures_dir):
        '''
        Extract PMID tar file from folder: input/task_id, 
        parse NXML metadata, 
        extract figure images,
        '''
        try:
            with tarfile.open(self.tar_filepath, "r:gz") as tar:
                nxml_content = self._read_nxml_from_tar(tar)
                if not nxml_content:
                    return [], {}
                fig_to_label_map = self._figureExtractor._parse_nxml_content(nxml_content)
                image_sources = self._image_sources_from_tar(tar, fig_to_label_map)
                return self._figureExtractor.collect_figures(
                    figures_dir, fig_to_label_map, image_sources
                )
        except Exception:
            traceback.print_exc()
            return [], {}

    def figures_metadata(self, figures_dir, **kwargs):
        image_files, figure_info_map = self.extract_figures_from_tar(figures_dir)
        if not image_files:
            return []
        return self._figureExtractor.build_figures_metadata(
            figures_dir, image_files, figure_info_map
        )

class PMCFiles:
    '''
    This PMC API provides json files/data - which contains metadata pointing to links of media_url, pdf_url, xml_url
    and other details about the publication. eg. https://pmc-oa-opendata.s3.amazonaws.com/metadata/<pmcid>/<version_no>.json.
    '''

    PMC_S3_HTTPS = "https://pmc-oa-opendata.s3.amazonaws.com"
    NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    
    def __init__(self, pmc_files_dir):
        # pmc_files_dir contains unpacked NXML, PDF, and image files (no tar)
        # pmc_files_dir: input/task_id/ --> is the dirextory which contains all the files downloaded after the PMID was validated.
        # this directory is required so that items from here can be copied to static/files/task_id/input_dir during extract figures_metadata phases
        self.pmc_files_dir = pmc_files_dir
        self._figureExtractor = PMCFigureExtractor()

    @staticmethod
    def list_pmc_versions(pmcid):
        '''
        The new API provides different versions for its data (including author manuscripts)
        So get all the different version numbers that exist for a PMCID.
        '''
        # List all versions belonging to a PMCID
        try:
            r = requests.get(
                f"{PMCFiles.PMC_S3_HTTPS}/",
                params={
                    "list-type": "2",
                    "prefix": f"{pmcid}.",
                    "delimiter": "/",
                },
                timeout=60,
            )
        except requests.RequestException as e:
            return {"valid": False, "error": f"Failed to list PMC versions: {e}"}, 502

        if r.status_code != 200:
            return {
                "valid": False,
                "error": f"Failed to list PMC versions for {pmcid} (HTTP {r.status_code})",
            }, r.status_code

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            return {"valid": False, "error": f"Invalid PMC listing response: {e}"}, 502

        versions = []
        for cp in root.findall(".//s3:CommonPrefixes/s3:Prefix", PMCFiles.NS):
            prefix = cp.text.rstrip("/")          # PMC10009416.1
            versions.append(int(prefix.rsplit(".", 1)[1]))
        
        if not versions:
            return {"valid": False,"error": f"No PMC versions found for {pmcid}",}, 404

        return {"valid": True, "versions": versions}, 200

    @staticmethod
    def get_latest_pmc_metadata(pmcid):
        '''
        Since different version numbers for the data exists for the same PMCID, 
        Goal is to get the latest version - which is not an author manuscript version.
        This function makes sure to get the latest published version.
        '''

        pmc_version_result, status = PMCFiles.list_pmc_versions(pmcid)
        if status != 200 or not pmc_version_result.get("valid"):
            return pmc_version_result, status

        pmc_versions = pmc_version_result.get("versions")
        for version in sorted(pmc_versions, reverse=True):

            url = f"{PMCFiles.PMC_S3_HTTPS}/metadata/{pmcid}.{version}.json"

            try:
                r = requests.get(url, timeout=60)
            except requests.RequestException as e:
                return {"valid": False,"error": f"Failed to fetch metadata for {pmcid} v{version}: {e}",}, 502

            if r.status_code != 200:
                continue  # try next lower version
            
            try:
                meta = r.json()
            except ValueError as e:
                return {"valid": False, "error": f"Invalid metadata JSON for {pmcid} v{version}: {e}"}, 502

            if meta and not meta.get("is_manuscript"): # ensure that it is not an author manuscript
                return {"valid": True, "metadata": meta, "version": version}, 200
        return {"valid": False, "error": f"No metatdata found for pmcid {pmcid}"}, 404

    @staticmethod
    def s3_uri_to_https(s3_url):
        if not s3_url.startswith("s3://"):
            return s3_url
        # s3://pmc-oa-opendata/PMC10009416.1/PMC10009416.1.pdf?md5=...
        path = s3_url.replace("s3://pmc-oa-opendata/", "", 1)
        return f"{PMCFiles.PMC_S3_HTTPS}/{path}"

    # # TODO check this method
    @staticmethod
    def filename_from_url(url, response=None, pmid=None):
        name = os.path.basename(urlparse(url).path.split("?")[0])
        if not name:
            return None
        ext = os.path.splitext(name)[1].lower()
        if pmid and ext in (".pdf", ".xml"):
            return f"PMID-{pmid}{ext}"
        return name

    @staticmethod
    def get_https_urls(metadata, pmid=None):
        metadata_urls = ['pdf_url', 'xml_url', 'media_urls']     # media_urls can contain supplimetary files as well, so limit the media urls to images only
        url_list = []

        image_urls = []

        for url_type in metadata_urls:
            if not metadata.get(url_type):
                return {"valid": False, "error": f"No {url_type} in metadata for pmid: {pmid}"}, 400

            # TODO -  # media_urls can contain supplimetary files as well, 
            # so limit the media urls to images only

            # TODO - improve these loops
            if url_type == 'media_urls':
                image_urls = [
                    PMCFiles.s3_uri_to_https(uri)
                    for uri in PMCFiles.filter_media_image_urls(metadata['media_urls'])
                ]
            else:
                https_url = PMCFiles.s3_uri_to_https(metadata[url_type])
                url_list.append(https_url)

        url_list.extend(image_urls)

        return {"valid": True, "urls": url_list}, 200

    @staticmethod
    def _basename_and_ext_from_s3_uri(uri):
        # drop query string, then get filename
        path = urlparse(uri).path  # /PMC.../NPR2-43-85-g001.jpg
        filename = os.path.basename(path)
        base, ext = os.path.splitext(filename)
        return filename, base, ext.lower()

    @staticmethod
    def filter_media_image_urls(media_urls):
        seen_basenames = set()
        image_urls = []
        for uri in media_urls:
            filename, base_name, ext = PMCFiles._basename_and_ext_from_s3_uri(uri)
            if ext not in PMCFigureExtractor.IMAGE_EXTS:
                continue
            if base_name in seen_basenames:
                continue
            image_urls.append(uri)
            seen_basenames.add(base_name)
        return image_urls

    @staticmethod
    def download_pmc_files(metadata, dest_dir, pmid=None):
        url_result, status = PMCFiles.get_https_urls(metadata, pmid=pmid)

        if status != 200 or not url_result.get("valid"):
            return url_result, status

        for url in url_result["urls"]:
            try:
                with requests.get(url, stream=True, timeout=120) as resp:
                    if resp.status_code != 200:
                        return {"valid": False, "error":f"Failed to download {url} (HTTP {resp.status_code})"}, resp.status_code
                        
                    filename = PMCFiles.filename_from_url(url, resp, pmid=pmid)

                    if not filename:
                        return {"valid": False , "error":f"Could not determine filename for {url}"}, 502

                    dest_path = os.path.join(dest_dir, filename)
                    with open(dest_path, "wb") as f:
                        for chunk in resp.iter_content(1 << 20):
                            if chunk:
                                f.write(chunk)
            except requests.RequestException as e:
                return {"valid": False, "error":f"Failed to download {url}: {e}"} , 502
            except OSError as e:
                return {"valid": False, "error":f"Failed to write {dest_path}: {e}"}, 500
                                
        return {"valid": True}, 200

    @staticmethod
    def download_and_prepare_pmid_data(pmid, file_dir, pdf_filename):
        '''
        There can be multiple versions of journal data - published manuscript, author manuscript, latest revisions, etc (mostly only version 1 exists for most publications).
        So, list all the exisiting version nos, and then check each versions json file's metadata which indicates if its an author manuscript or not.
        Pick the highest version which is not an author manuscript.

        After obtaining the json metadata of selected version - parse the data to get all the required URL's and
        download the files into the input/task_id folder
        '''

        pmcid, pmc_resp_json, pmc_status = PMCData.get_validated_pmc_resource(pmid)
        if not pmcid:
            return pmc_resp_json, pmc_status

        pmc_metadata_result, status = PMCFiles.get_latest_pmc_metadata(pmcid)
        if status != 200:
            return pmc_metadata_result, status
                    
        os.makedirs(file_dir, exist_ok=True)
        download_result, status = PMCFiles.download_pmc_files(
            pmc_metadata_result["metadata"], file_dir, pmid=pmid
        )

        if status != 200:
            return download_result, status
        return {"valid": True}, 200

    def _read_nxml(self):
        """Return xml/nxml content from downloaded files, or None."""
        for dirpath, _, filenames in os.walk(self.pmc_files_dir):
            for name in filenames:
                if not name.lower().endswith((".nxml", ".xml")):
                    continue
                with open(os.path.join(dirpath, name), encoding="utf-8", errors="ignore") as f:
                    return f.read()
        return None

    def _image_sources_from_dir(self, fig_to_label_map):
        """Return [(filename, filepath), ...] for figure images on disk."""
        image_sources = []
        for dirpath, _, filenames in os.walk(self.pmc_files_dir):
            for name in filenames:
                filename = os.path.basename(name)
                base_name, ext = os.path.splitext(filename)
                if ext.lower() not in PMCFigureExtractor.IMAGE_EXTS:
                    continue
                if base_name not in fig_to_label_map:
                    continue
                image_sources.append((filename, os.path.join(dirpath, name)))
        return image_sources

    def extract_figures(self, figures_dir):
        '''
        Read NXML from self.pmc_files_dir,
        link figure images to captions and figure_number,
        copy matching images into figures_dir (original filenames),
        '''
        try:
            nxml_content = self._read_nxml()
            if not nxml_content:
                return [], {}
            fig_to_label_map = self._figureExtractor._parse_nxml_content(nxml_content)
            image_sources = self._image_sources_from_dir(fig_to_label_map)
            return self._figureExtractor.collect_figures(
                figures_dir, fig_to_label_map, image_sources
            )
        except Exception:
            traceback.print_exc()
            return [], {}

    def figures_metadata(self, figures_dir, **kwargs):
        '''
        the main file in the submission_detail (task) - is already copied to static/files/task_id/input
        during processjob.

        In the input/task_id folder there are images, xml, pdf files
        parse the xml to get metadata information about each image (captions, figure_name, figure_number, etc)
        Copy the images to static/files/task_id/extracted_figures/figures and return image_files and corresposnding metadata derived from xml.

        Cleanup the input/task_id folder - to keep the original pdf file only (remove the figures, xml, json)
        '''

        image_files, figure_info_map = self.extract_figures(figures_dir)
        if not image_files:
            return []
        return self._figureExtractor.build_figures_metadata(
            figures_dir, image_files, figure_info_map
        )
       

if __name__ == "__main__":

    pmc = PMCFiles(sys.argv[1])
    print(json.dumps(pmc.figures_metadata('.')))
    
    # for doi in sys.argv[1:]:
    #     pmid = PMCData.lookup(doi=doi)
    #     if pmid:
    #         print(doi,pmid['pmid'])
    #         pmid = pmid['pmid']
    #     else:
    #         pmid = doi
    #     print(PMCData.citation_details(pmid))


    # # XML parser
    # xml_file = '/home/nmathias/GlycanImageExtract2/xml_files/369_330.nxml'

    # xml_obj = XMLParser(xml_file)

    # parsed_data = xml_obj.parse()

    # print(parsed_data)