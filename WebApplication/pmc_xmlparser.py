import xml.etree.ElementTree as ET
import os
import re

class XMLParser:
    '''Parser for NLM/NCBI XML files. (.nxml format)'''

    NAMESPACES = {
        'nlm': 'http://dtd.nlm.nih.gov/2.0/xsd/archivearticle',
        'xlink': 'http://www.w3.org/1999/xlink',
        'mml': 'http://www.w3.org/1998/Math/MathML'
    }

    def __init__(self, xml_source, pmc_publication=None):
        self.xml_source = xml_source
        self.tree = None
        self.root = None
        self.pmc_publication = pmc_publication
    
    def parse(self):
        
        # Handle file-like objects (like from tar.extractfile())
        if hasattr(self.xml_source, 'read'):
            # It's a file-like object - read it first
            content = self.xml_source.read()
            if isinstance(content, bytes):
                self.root = ET.fromstring(content)
            else:
                self.root = ET.fromstring(content)
        elif isinstance(self.xml_source, ET.Element):
            # Already parsed
            self.root = self.xml_source
        elif isinstance(self.xml_source, bytes):
            # XML as bytes
            self.root = ET.fromstring(self.xml_source)
        elif isinstance(self.xml_source, str):
            if os.path.exists(self.xml_source):
                # File path
                self.tree = ET.parse(self.xml_source)
                self.root = self.tree.getroot()
            else:
                # XML as string
                self.root = ET.fromstring(self.xml_source)
        else:
            raise ValueError(f"Unsupported XML source type: {type(self.xml_source)}")

        # Verify root was set
        if self.root is None:
            print("ERROR: xml root is None after parsing!")
            return {}
                
        parsed_data = {
            # **self._extract_pubmed_ids(),
            # 'title': self._extract_title(),
            # 'authors': self._extract_authors(),
            # 'journal': self._extract_journal(),
            # 'publication_date': self._extract_publication_date(),
            # 'doi': self._extract_doi(),
            # 'url': self._extract_article_url(),
            'figure_info': self._extract_figure_info(),  # figure label, title, caption, figure filename
            # **self._extract_metadata(),
            'citation': self.format_citation(),
            'pmid_url': self._extract_article_url()['pmid_url'],
            'pmid_job': True
        }
        
        return parsed_data
        

    def format_citation(self):
        authors = self._extract_authors()
        title = self._extract_title()
        journal = self._extract_journal()
        year_vol_pages, has_vol_or_pages = self._format_year_volume_pages()

        def to_str(x):
            if x is None:
                return None
            if isinstance(x, dict):
                return (x.get("title") or x.get("name") or "").strip() or None
            s = str(x).strip()
            return s if s else None

        authors = to_str(authors)
        title = to_str(title)
        journal = to_str(journal)
        year_vol_pages = to_str(year_vol_pages) if year_vol_pages is not None else None

        trailing_parts = [p for p in (journal, year_vol_pages) if p]
        trailing = " ".join(trailing_parts) if trailing_parts else None

        trailing_complete = bool(journal and year_vol_pages and has_vol_or_pages)
        if not trailing_complete and self.pmc_publication:
            trailing = self.pmc_publication

        parts = [authors, title, trailing]
        citation = " ".join(p for p in parts if p).strip()

        if citation and not citation.endswith("."):
            citation += "."

        return citation


    def _format_year_volume_pages(self):
        pub_date_elem = self.root.find('.//pub-date[@pub-type="ppub"]', self.NAMESPACES)
        if pub_date_elem is None:
            pub_date_elem = self.root.find('.//pub-date[@pub-type="epub"]', self.NAMESPACES)

        year = None
        if pub_date_elem is not None:
            year_el = pub_date_elem.find('year', self.NAMESPACES)
            year = year_el.text.strip() if year_el is not None and year_el.text else None

        article_meta = self.root.find('.//article-meta', self.NAMESPACES)
        volume = None
        issue = None
        fpage = None
        lpage = None
        if article_meta is not None:
            vol_el = article_meta.find('volume', self.NAMESPACES)
            volume = vol_el.text.strip() if vol_el is not None and vol_el.text else None
            iss_el = article_meta.find('issue', self.NAMESPACES)
            issue = iss_el.text.strip() if iss_el is not None and iss_el.text else None
            fp_el = article_meta.find('fpage', self.NAMESPACES)
            fpage = fp_el.text.strip() if fp_el is not None and fp_el.text else None
            lp_el = article_meta.find('lpage', self.NAMESPACES)
            lpage = lp_el.text.strip() if lp_el is not None and lp_el.text else None

        if volume is not None and issue is not None:
            vol_issue = f"{volume}({issue})"
        elif volume is not None:
            vol_issue = volume
        elif issue is not None:
            vol_issue = f"({issue})"
        else:
            vol_issue = None

        if fpage and lpage:
            pages = f"{fpage}-{lpage}"
        elif fpage:
            pages = fpage
        elif lpage:
            pages = lpage
        else:
            pages = None

        has_vol_or_pages = bool(volume or issue or fpage or lpage)

        if not (year or vol_issue or pages):
            return None, has_vol_or_pages

        seg = ""
        if year:
            seg += year
        if vol_issue:
            seg += (";" if seg else "") + vol_issue
        if pages:
            if vol_issue:
                seg += f":{pages}"
            else:
                seg += (":" if seg else "") + pages

        if not seg:
            return None, has_vol_or_pages

        return seg + ".", has_vol_or_pages

    
    def _extract_pubmed_ids(self):
        pubmed_info = {}
        for article_id in self.root.findall('.//article-id', self.NAMESPACES):
            if article_id.get('pub-id-type') == 'pmid':
                pubmed_info['pmid'] = article_id.text

            if article_id.get('pub-id-type') == 'pmc':
                pubmed_info['pmcid'] = article_id.text
            
        return pubmed_info

    def _extract_pmcid(self):
        for article_id in self.root.findall('.//article-id', self.NAMESPACES):
            if article_id.get('pub-id-type') == 'pmc':
                return article_id.text
        return None   

    def _extract_title(self):
        title_elem = self.root.find('.//article-title', self.NAMESPACES)
        if title_elem is not None:
            return ''.join(title_elem.itertext()).strip()
        return None

    def _extract_initials(self, given_names):
        """
        Convert given names like 'John P. Doe' -> 'JP'.
        given_names: str (not an Element).
        """
        if given_names is None or not isinstance(given_names, str):
            return None
        given_names = given_names.strip()
        if not given_names:
            return None
        cleaned = re.sub(r"[^A-Za-z\s]", " ", given_names)
        parts = [p for p in cleaned.split() if p]
        if not parts:
            return None
        return "".join(p[0].upper() for p in parts)


    def _extract_authors(self):
        authors = []
        for contrib in self.root.findall('.//contrib', self.NAMESPACES):
            name_elem = contrib.find('.//name', self.NAMESPACES)
            if name_elem is None:
                continue

            surname_el = name_elem.find('surname', self.NAMESPACES)
            given_el = name_elem.find('given-names', self.NAMESPACES)

            surname = (surname_el.text or "").strip() if surname_el is not None and surname_el.text else None
            given_text = (given_el.text or "").strip() if given_el is not None and given_el.text else None

            initials = self._extract_initials(given_text) if given_text else None

            if surname and initials:
                authors.append(f"{surname} {initials}")
            elif surname:
                authors.append(surname)
            elif initials:
                authors.append(initials)

        if not authors:
            return None
        return ", ".join(authors) + "."
    
    def _extract_journal(self):
        journal_elem = self.root.find('.//journal-meta', self.NAMESPACES)

        if journal_elem is not None:
            journal_info = {}

            # Journal title
            journal_title = journal_elem.find('.//journal-title-group', self.NAMESPACES)
            if journal_title is not None:
                journal_info['title'] = journal_title.find('journal-title', self.NAMESPACES).text or ''
            
            # issn needed?

            # Publisher
            publisher_elem = journal_elem.find(".//publisher", self.NAMESPACES)
            if publisher_elem is not None:
                journal_info['publisher'] = publisher_elem.find('publisher-name', self.NAMESPACES).text or ''
            return journal_info
        return None
    
    def _extract_publication_date(self):
        pub_date_elem = self.root.find('.//pub-date[@pub-type="ppub"]', self.NAMESPACES)
        
        if pub_date_elem is None:
            pub_date_elem = self.root.find('.//pub-date[@pub-type="epub"]', self.NAMESPACES)
        if pub_date_elem is not None:
            date_info = {}
            year = pub_date_elem.find('year', self.NAMESPACES)
            month = pub_date_elem.find('month', self.NAMESPACES)
            day = pub_date_elem.find('day', self.NAMESPACES)
            if year is not None:
                date_info['year'] = year.text
            if month is not None:
                date_info['month'] = month.text
            if day is not None:
                date_info['day'] = day.text
            return date_info
        return None
   
    def _extract_figure_info(self):
        '''Extract figure captions'''
        XLINK_NS = "http://www.w3.org/1999/xlink"
        XLINK_HREF = f"{{{XLINK_NS}}}href"

        figures = {}
        for fig in self.root.findall('.//fig', self.NAMESPACES):
            fig_info = {}

            label_elem = fig.find('label', self.NAMESPACES)
            if label_elem is not None:
                label = label_elem.text
                m = re.search(r'^\s*\w+\.?\s*(\d+)\.?\s*$', label)
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
                        if text:
                            caption.append(text)
                    fig_info['caption'] = ' '.join(caption) if caption else None

            # Extract filename from graphic or inline-graphic elements
            filename = None
            for tag in ('graphic', 'inline-graphic'):
                graphic_elem = fig.find(tag, self.NAMESPACES)
                if graphic_elem is not None:
                    # Checks both regular href and xlink:href
                    href = graphic_elem.get('href') or graphic_elem.get(XLINK_HREF)
                    if href:
                        filename = os.path.basename(href)
                        break
            if filename:
                fig_info['filename'] = filename
                figures[filename] = fig_info

        return figures
    
    def _extract_metadata(self):
        # extracts volume, issue, fpage, lpage
        metadata = {}
        article_meta = self.root.find('.//article-meta', self.NAMESPACES)

        if article_meta is None:  # Added None check
            return metadata

        volume = article_meta.find('volume', self.NAMESPACES)
        metadata['volume'] = volume.text if volume is not None else None

        issue = article_meta.find('issue', self.NAMESPACES)  # FIXED: was 'volume', now 'issue'
        metadata['issue'] = issue.text if issue is not None else None

        fpage = article_meta.find('fpage', self.NAMESPACES)
        metadata['fpage'] = fpage.text if fpage is not None else None

        lpage = article_meta.find('lpage', self.NAMESPACES)
        metadata['lpage'] = lpage.text if lpage is not None else None

        return metadata

    def _extract_doi(self):
        '''Extract DOI from article-id with pub-id-type='doi'.'''
        for article_id in self.root.findall('.//article-id', self.NAMESPACES):
            if article_id.get('pub-id-type') == 'doi' and article_id.text:
                return article_id.text.strip()
        return None

    def _extract_article_url(self):
        '''Article URL: self-uri, or doi.org, or PMC link.'''

        url_data = {}

        for article_id in self.root.findall('.//article-id', self.NAMESPACES):
            if article_id.get('pub-id-type') == 'pmc' and article_id.text:
                pmcid = article_id.text.strip()
                url_data["pmc_url"] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"

            if article_id.get('pub-id-type') == 'pmid' and article_id.text:
                pmid = article_id.text.strip()
                url_data["pmid_url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return url_data
        

if __name__ == "__main__":
    xml_file = '/home/nmathias/GlycanImageExtract2/xml_files/369_330.nxml'

    xml_obj = XMLParser(xml_file)

    parsed_data = xml_obj.parse()

    print(parsed_data)
