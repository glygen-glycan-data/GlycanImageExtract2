
import requests, urllib, traceback, sys
import xml.etree.ElementTree as ET

devemail = 'nje5+extractor@georgegetown.edu'

def lookup(doi=None,pmid=None,pmcid=None):
    baseurl = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    query = dict(tool="extractor",email=devemail,format="json")
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
    return dict(doi=record['doi'],pmid=record['pmid'],pmcid=record['pmcid'])

def citation_details(pmid):
    baseurl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    query = dict(db="pubmed",id=pmid)
    fullurl = baseurl + "?" + urllib.parse.urlencode(query)
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
    volume = article.find('Journal/JournalIssue/Volume')
    if volume is not None:
        volume = volume.text
    issue = article.find('Journal/JournalIssue/Issue')
    if issue is not None:
        issue = issue.text
    articledate = article.find("ArticleDate")
    if articledate is not None:
        year = articledate.find('Year')
        if year is not None:
            year = year.text
    else:
        pubdate = article.find("Journal/JournalIssue/PubDate")
        if pubdate is not None:
            year = pubdate.find('Year')
            if year is not None:
                year = year.text
    journal = article.find('Journal/Title')
    if journal is not None:
        journal = journal.text
    isoabbrev = article.find('Journal/ISOAbbreviation')
    if isoabbrev is not None:
        isoabbrev = isoabbrev.text
    title = article.find("ArticleTitle")
    if title is not None:
        title = title.text
    page = article.find("Pagination/MedlinePgn")
    if page is not None:
        page = page.text
    authors = []
    for author in article.findall("AuthorList/Author"):
        lastname = author.find("LastName")
        if lastname is not None:
            lastname = lastname.text
        forename = author.find("ForeName")
        if forename is not None:
            forename = forename.text
        initials = author.find("Initials")
        if initials is not None:
            initials = initials.text
        authors.append(dict(lastname=lastname,forename=forename,initials=initials))
    details = dict(title=title,journal=journal,journal_abbrev=isoabbrev,issue=issue,volume=volume,year=year,authors=authors,page=page)
    citation = format_citation(details)
    details['citation'] = citation
    return details

def format_citation(d):
    author_list = []
    for au in d['authors']:
        author_list.append(au['lastname'] + " " + au['initials'])
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

if __name__ == "__main__":
    for doi in sys.argv[1:]:
        pmid = lookup(doi=doi)
        if pmid:
            print(doi,pmid['pmid'])
            pmid = pmid['pmid']
        else:
            pmid = doi
        print(citation_details(pmid))
