import xml.etree.ElementTree as ET
import os

class XMLParser:
    '''Parser for NLM/NCBI XML files. (.nxml format)'''

    NAMESPACES = {
        'nlm': 'http://dtd.nlm.nih.gov/2.0/xsd/archivearticle',
        'xlink': 'http://www.w3.org/1999/xlink',
        'mml': 'http://www.w3.org/1998/Math/MathML'
    }

    def __init__(self, xml_source):
        self.xml_source = xml_source
        self.tree = None
        self.root = None
    
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
            **self._extract_pubmed_ids(),
            'title': self._extract_title(),
            'authors': self._extract_authors(),
            'journal': self._extract_journal(),
            'publication_date': self._extract_publication_date(),
            'doi': self._extract_doi(),
            'url': self._extract_article_url(),
            'figure_info': self._extract_figure_info(),  # figure label, title, caption, figure filename
            **self._extract_metadata(),
        }
        
        return parsed_data
    
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
   
    def _extract_authors(self):
        authors = []

        for contrib in self.root.findall('.//contrib', self.NAMESPACES):
            author_info = {}

            # for author_details in contrib:
            name_elem = contrib.find('.//name', self.NAMESPACES)

            if name_elem is not None:
                surname = name_elem.find('surname', self.NAMESPACES)
                given_names = name_elem.find('given-names', self.NAMESPACES)

                if surname is not None:
                    author_info['surname'] = surname.text or ''
                if given_names is not None:
                    author_info['given_names'] = given_names.text or ''

                author_info['full_name'] = f"{author_info.get('given_names', '')} {author_info.get('surname', '')}".strip()
            
            if author_info:
                authors.append(author_info)
        return authors
    
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
            fig_info = {
                'id': fig.get('id', ''),
                'label': None,
                'caption_text': None,
            }

            label_elem = fig.find('label', self.NAMESPACES)
            if label_elem is not None:
                fig_info['label'] = label_elem.text

            caption_elem = fig.find('caption', self.NAMESPACES)
            if caption_elem is not None:
                title = caption_elem.find('title', self.NAMESPACES)
                if title is not None:
                    fig_info['caption_text'] = title.text 
                else:
                    caption_text = []
                    for p in caption_elem.findall('.//p', self.NAMESPACES):
                        text = ''.join(p.itertext()).strip()
                        if text:
                            caption_text.append(text)
                    fig_info['caption_text'] = ' '.join(caption_text) if caption_text else None

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
