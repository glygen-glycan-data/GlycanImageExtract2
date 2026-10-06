#!.venv/bin/python

import os, os.path, copy, shutil, argparse, csv

from BKGlycanExtractor.semantics import ManuscriptSemantics
from BKGlycanExtractor.glyomicsclient import GlyLookupClient
from BKGlycanExtractor.glycansemantics import YOLO_Glycan
from BKGlycanExtractor.pdf_image_metadata import ImageSearch


_mono_finder = None
_mono_finder_failed = False


def _get_mono_finder():
    global _mono_finder, _mono_finder_failed
    if _mono_finder is not None or _mono_finder_failed:
        return _mono_finder
    try:
        from BKGlycanExtractor import Config_Manager
        _mono_finder = Config_Manager().get_finder('YOLOMono_58')
    except Exception as e:
        print(f"Warning: could not load YOLOMono_58 finder: {e}")
        _mono_finder_failed = True
    return _mono_finder


def find_uncleaned_monos(glycan, raw_img):
    finder = _get_mono_finder()
    if finder is None:
        return None, None
    g = copy.deepcopy(glycan)
    g.set_image(raw_img)
    g.reset_monos()
    try:
        finder.find_objects(g)
    except Exception as e:
        print(f"Warning: YOLO mono finder failed on uncleaned crop: {e}")
        return None, None
    return g, g.compstr()


_EXTRA_IOU_THRESHOLD = 0.3
_COMP_LABELS = ("GlcNAc", "NeuAc", "Fuc", "Man", "GalNAc", "Gal", "Glc", "NeuGc", "Xyl")


def extra_monos(glycan, yolo_uncleaned_glycan, iou_threshold=_EXTRA_IOU_THRESHOLD):
    """Monos found on the uncleaned crop that don't overlap any mono in glycan."""
    cleaned_boxes = [m.box() for m in glycan.monos()]
    result = []
    for mu in yolo_uncleaned_glycan.monos():
        bu = mu.box()
        if all(bu.iou(bc) < iou_threshold for bc in cleaned_boxes):
            result.append(mu)
    return result


def compstr_from_monos(monos):
    from collections import Counter
    counts = Counter(m.get('symbol') for m in monos)
    return "".join(f"{sym}({counts[sym]})" for sym in _COMP_LABELS if counts.get(sym, 0) > 0)


def recompute_iupac(glycan):
    glycan.unset('IUPAC')
    glycan.unset('composition_str')
    glycan.reset_glycan_errors()
    YOLO_Glycan().find_objects(glycan)


def update_tsv_row(tsvresults, glycan_gid, glycan, glylookup):
    seq = glycan.get('IUPAC')
    compstr = glycan.get('composition_str')
    acc, wurcs = "", ""
    if seq and not glycan.has_glycan_errors():
        try:
            acc, wurcs = glylookup.get_wurcs(seq)
        except Exception:
            print("Warning: glylookup unavailable, accession/WURCS not updated.")
    tsvresults[glycan_gid]['accession'] = acc
    tsvresults[glycan_gid]['iupac'] = seq
    tsvresults[glycan_gid]['composition'] = compstr
    tsvresults[glycan_gid]['wurcs'] = wurcs


def write_files(results, jsonfile, tsvfilename, tsvresults, tsvfieldnames):
    for filepath in (jsonfile, tsvfilename):
        if os.path.exists(filepath):
            fnparts = filepath.rsplit('.', 1)
            origfile = ".".join([fnparts[0], "orig", fnparts[1]])
            if not os.path.exists(origfile):
                shutil.copy(filepath, origfile)
    with open(jsonfile, 'w') as wh:
        wh.write(results.tojson())
    with open(tsvfilename, 'w') as wh:
        writer = csv.DictWriter(wh, fieldnames=tsvfieldnames,
                                dialect="excel-tab", extrasaction='ignore')
        writer.writeheader()
        writer.writerows(tsvresults.values())
    print(f"Written: {jsonfile} and {tsvfilename}")


def _normalize_img_ext(filename):
    base, ext = os.path.splitext(filename)
    if ext.lower() == '.jpg':
        ext = '.jpeg'
    return base + ext


def dedupe_fieldnames(fieldnames):
    seen, out = set(), []
    for name in fieldnames:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def ensure_field(fieldnames, name, after):
    if name in fieldnames:
        return
    idx = fieldnames.index(after) + 1 if after in fieldnames else len(fieldnames)
    fieldnames.insert(idx, name)


def move_field(fieldnames, name, after):
    while name in fieldnames:
        fieldnames.remove(name)
    ensure_field(fieldnames, name, after)


_URL_PREFIX_REWRITES = (
    ("http://0.0.0.0:10981", "https://extractor.glyomics.org"),
    ("http://0.0.0.0:10982", "https://edwardslab.bmcb.georgetown.edu/tandem10982"),
)


def fix_result_urls(tsvresults):
    for row in tsvresults.values():
        url = row.get('url') or ''
        for old, new in _URL_PREFIX_REWRITES:
            if old in url:
                url = url.replace(old, new)
                row['url'] = url


def wire_figure_images(results, pdffilename, figuresdir, image_search_strategy):
    image_path_dict = {}
    bbox_override_dict = {}
    for f in results.figures():
        ic = f.get('image_count')
        ip = f.get('image_path')
        if ic is not None and ip is not None:
            image_path_dict[ic] = os.path.join(
                figuresdir, _normalize_img_ext(os.path.split(ip)[1]))
        bbox = f.get('pdf_fig_bbox')
        if ic is not None and bbox is not None:
            bbox_override_dict[ic] = bbox

    if not os.path.isdir(figuresdir):
        os.makedirs(figuresdir)
    strategy = ImageSearch.search_method(image_search_strategy)
    strategy.get_metadata(pdffilename, figuresdir,
                          image_path_dict=image_path_dict or None,
                          use_annotations=False,
                          bbox_override_dict=bbox_override_dict or None)

    for f in results.figures():
        ic = f.get('image_count')
        ip = f.get('image_path')
        if ip is not None:
            f.set_image_path(os.path.join(
                figuresdir, _normalize_img_ext(os.path.split(ip)[1])))
        elif ic is not None:
            f.set_image_path(os.path.join(figuresdir, f"fig{ic}.png"))


def parse_vote(row):
    try:
        return int(row.get('votes', ''))
    except (TypeError, ValueError):
        return None


def process(jsonfile):
    assert jsonfile.endswith(".json") and os.path.exists(jsonfile)
    tsvfilename = jsonfile.replace('.json', '.tsv')
    pdffilename = jsonfile.replace('.json', '.pdf')
    figuresdir = jsonfile.replace('.json', '.figs')
    assert os.path.exists(tsvfilename), f"TSV not found: {tsvfilename}"
    assert os.path.exists(pdffilename), f"PDF not found: {pdffilename}"

    glylookup = GlyLookupClient()

    with open(tsvfilename) as fh:
        tsvreader = csv.DictReader(fh, dialect="excel-tab")
        tsvresults = {row['ID']: row for row in tsvreader}
        tsvfieldnames = list(tsvreader.fieldnames)

    tsvfieldnames = dedupe_fieldnames(tsvfieldnames)
    ensure_field(tsvfieldnames, 'uncleaned_composition', after='composition')
    ensure_field(tsvfieldnames, 'extra_composition', after='uncleaned_composition')
    move_field(tsvfieldnames, 'error_count', after='votes')
    fix_result_urls(tsvresults)

    results = ManuscriptSemantics.read_json(jsonfile)
    wire_figure_images(results, pdffilename, figuresdir,
                       results.get('image_search_strategy'))

    for row in tsvresults.values():
        if parse_vote(row) != 2:
            row['uncleaned_composition'] = ''
            row['extra_composition'] = ''

    n_glycans = n_vote2 = n_errors = 0
    for figure in results.figures():
        for glycan in figure.glycans():
            gid = glycan.get('GID')
            if gid is None or gid not in tsvresults:
                continue
            n_glycans += 1
            row = tsvresults[gid]
            vote = parse_vote(row)
            prev_iupac = row.get('iupac') or ''
            recompute_iupac(glycan)
            if vote in (1, 2):
                new_iupac = glycan.get('IUPAC') or ''
                if prev_iupac != new_iupac:
                    print(f"Warning: {gid}: IUPAC changed on recompute (vote {vote})")
                    print(f"    old: {prev_iupac}")
                    print(f"    new: {new_iupac}")
            update_tsv_row(tsvresults, gid, glycan, glylookup)
            if vote == 2:
                compstr = None
                yolo_uncleaned = None
                try:
                    raw_img = glycan.box().crop(figure.image())
                    yolo_uncleaned, compstr = find_uncleaned_monos(glycan, raw_img)
                except Exception as e:
                    print(f"Warning: {gid}: raw crop failed: {e}")
                row['uncleaned_composition'] = compstr or ''
                if yolo_uncleaned is not None:
                    row['extra_composition'] = compstr_from_monos(
                        extra_monos(glycan, yolo_uncleaned))
                else:
                    row['extra_composition'] = ''
                if compstr:
                    n_vote2 += 1
            else:
                row['uncleaned_composition'] = ''
                row['extra_composition'] = ''

            err_n = len(glycan.glycan_errors())
            row['error_count'] = err_n
            if err_n:
                n_errors += 1

            print(f"[{n_glycans}] {gid}  vote={vote}  errors={err_n}")
            if row.get('iupac'):
                print(f"    IUPAC: {row['iupac']}")
            if row.get('composition'):
                print(f"    composition: {row['composition']}")
            if vote == 2:
                print(f"    uncleaned_composition: {row['uncleaned_composition'] or '(none)'}")
                print(f"    extra_composition: {row['extra_composition'] or '(none)'}")
            if row.get('accession'):
                print(f"    accession: {row['accession']}")
            for err in glycan.glycan_errors():
                print(f"    error: {err}")

    for row in tsvresults.values():
        row.setdefault('error_count', '')

    write_files(results, jsonfile, tsvfilename, tsvresults, tsvfieldnames)
    print(f"Summary: {n_glycans} glycans, {n_vote2} vote-2 with "
          f"uncleaned_composition, {n_errors} with glycan errors.")


parser = argparse.ArgumentParser(
    description="Automated cleanup/reprocessing of glycan results TSV+JSON.")
parser.add_argument('--json', type=str, required=True,
                    help='JSON format extractor result file.')
args = parser.parse_args()

process(args.json)
