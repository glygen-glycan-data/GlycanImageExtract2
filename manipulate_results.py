#!.venv/bin/python

import sys, os, os.path, copy, shutil
import numpy as np
import argparse
from BKGlycanExtractor.bbox import BoundingBox
from BKGlycanExtractor.semantics import ManuscriptSemantics, UndirectedLinkSemantics, MonoSemantics
from BKGlycanExtractor.glyomicsclient import ExtractorClient, GlymageClient, GlyLookupClient
from BKGlycanExtractor.glycansemantics import YOLO_Glycan

parser = argparse.ArgumentParser(description="Edit results JSON, and reprocess")

parser.add_argument(
    '--json',
    type = str,
    default = None,
    help = 'JSON format extractor result file.'
)

parser.add_argument(
    '--extractorurl',
    type = str,
    default = 'https://extractor.glyomics.org/',
    help = 'Extractor URL.'
)

parser.add_argument(
    '--glycan',
    type = str,
    required=True,
    help = "Glycan identifier"
)

parser.add_argument(
    '--set_redend',
    type = int,
    default=None,
    help = "New reducing end monosaccharide."
)

parser.add_argument(
    '--add_mono',
    type = str,
    default = None,
    help = "New monosaccharide - <origin-monoid>:<direction>:<scale>:<label>."
)

parser.add_argument(
    '--set_monolabel',
    type = str,
    default=None,
    help = "New monosaccharide label - <monoid>:<label>."
)

parser.add_argument(
    '--delete_mono',
    type = int,
    default=None,
    help = "Delete monosaccharide."
)

parser.add_argument(
    '--delete_link',
    type = str,
    default=None,
    help = "Delete link - <monoid>:<monoid>."
)

parser.add_argument(
    '--recover_link',
    type = str,
    default=None,
    help = "Recover link - <monoid>:<monoid>."
)

parser.add_argument(
    '--add_link',
    type = str,
    default=None,
    help = "Add link - <monoid>:<monoid>."
)

parser.add_argument(
    '--set_orientation',
    type = str,
    default=None,
    help = "New orientation."
)

parser.add_argument(
    '--force',
    action='store_true',
    default=False,
    help = "Force recomputation of IUPAC sequence."
)

parser.add_argument(
    '--outputfile',
    type = str,
    default=None,
    help = "Filename for modified JSON document."
)

parser.add_argument(
    '--write',
    action='store_true',
    default=False,
    help = "Write modified JSON to file instead of displaying."
)

def recompute_iupac(glycan):
    YOLO_Glycan(ignore_errors=True).find_objects(glycan)

args = parser.parse_args()

# assume we can get images from the extractor based on the taskid
client = ExtractorClient(apiurl=args.extractorurl)
glymage = GlymageClient(image_format="png")
glylookup = GlyLookupClient()

results = ManuscriptSemantics.read_json(args.json)
task_id = results.get('id')

glycan = None
figure = None
for f in results.figures():
    figure_path = f.get('image_path')
    figure_url = client.makeresulturl(task_id=task_id,path=figure_path,location=results.get('location'))
    f.set_image_url(figure_url)
    for g in f.glycans():
        if g.get("GID") == args.glycan:
            glycan = g
            figure = f
        # remove when crop fix is deployed
        g.set('height',g.height()-1)
        g.set('width',g.width()-1)

modified = False

if args.add_mono is not None:
    mid,dirn,scale,label = args.add_mono.split(':')
    mid = int(mid)
    assert dirn in ("UP","DOWN","LEFT","RIGHT")
    scale = float(scale)
    if glycan.has_mono(mid):
        dims = []
        n = glycan.mono_count()
        for m in glycan.monos():
            dims.append(m.width())
            dims.append(m.height())
        # print(sorted(dims)); print(np.median(dims))
        median_dim = (dims[n-1]+dims[n])/2
        m = glycan.mono(mid)
        mcent = m.center()
        if dirn == "UP":
            newcent = (mcent[0],mcent[1]-median_dim*scale)
        elif dirn == "DOWN":
            newcent = (mcent[0],mcent[1]+median_dim*scale)
        elif dirn == "LEFT":
            newcent = (mcent[0]-median_dim*scale,mcent[1])
        elif dirn == "RIGHT":
            newcent = (mcent[0]+median_dim*scale,mcent[1])
        x1 = int(round(newcent[0]-median_dim/2,0))
        x2 = int(round(newcent[0]+median_dim/2,0))
        y1 = int(round(newcent[1]-median_dim/2,0))
        y2 = int(round(newcent[1]+median_dim/2,0))
        bbox = BoundingBox(x1=x1,x2=x2,y1=y1,y2=y2)
        newm = MonoSemantics(symbol=label,classlabel=label,box=bbox,confidence=1.0)
        glycan.add_mono(newm)
        newmid = newm.id()
        glycan.add_undirected_link(UndirectedLinkSemantics(mono_id1=mid,mono_id2=newmid,classlabel="link",confidence=1.0))
        modified = True

if args.delete_mono is not None and glycan.has_mono(args.delete_mono):
    glycan.delete_mono_and_ulinks(args.delete_mono)
    modified = True

if args.delete_link is not None:
    mid1,mid2 = args.delete_link.split(':')
    mid1 = int(mid1); mid2 = int(mid2)
    assert glycan.remove_undirected_link(mid1,mid2), "No such link to delete"
    modified = True

if args.recover_link is not None:
    mid1,mid2 = args.recover_link.split(':')
    mid1 = int(mid1); mid2 = int(mid2)
    assert glycan.recover_rejected_undirected_link(mid1,mid2), "No such rejected link to recover"
    modified = True

if args.add_link is not None:
    mid1,mid2 = args.add_link.split(':')
    mid1 = int(mid1); mid2 = int(mid2)
    assert not glycan.has_undirected_link(mid1,mid2), "Link is already an undirected link"
    assert not glycan.has_rejected_undirected_link(mid1,mid2), "Link is a rejected undirected link"
    glycan.add_undirected_link(UndirectedLinkSemantics(mono_id1=mid1,mono_id2=mid2,classlabel="link",confidence=1.0))
    modified = True

if args.set_monolabel is not None:
    mid,label = args.set_monolabel.split(":")
    mid = int(mid); label = label.strip()
    m = glycan.mono(mid)
    if m:
        m.set_label(label)
        m.set_symbol(label)
        modified = True

if args.set_redend is not None and glycan.has_mono(args.set_redend):
    m = glycan.mono(args.set_redend)
    glycan.swap_roots(m)
    modified = True

if modified or args.force or args.write:
    recompute_iupac(glycan)

if args.set_orientation is not None:
    glycan.set("orientation",args.set_orientation)

if not args.write:

    seq = glycan.get('IUPAC')
    if seq:
        print("IUPAC:",seq)
    compstr = glycan.get('composition_str')
    if compstr:
        print("Composition:",compstr)

    if seq:
        task_id = glymage.submit_glymage(seq=seq,redend=True,orientation=glycan.get('orientation'))
        result = glymage.retrieve(task_id)
        glymageurl = glymage.url() + result.get('result')

        acc,wurcs = glylookup.get_wurcs(seq)
        print("Accession:",acc)
        print("WURCS:",wurcs)

    glycan.set_image(glycan.box().crop(figure.image()))
    glycan.scaleimg(factor=4.0)
    # label="INDEX"
    glycan.annotate_monos(label="INDEX",textanchor="CENTER",font_scale=1.0)
    if seq:
        glycan.show_image(title="Annotated Glycan",extraimageurl=glymageurl)
    else:
        glycan.show_image(title="Annotated Glycan")

else:

    if args.outputfile:
        outputfile = args.outputfile
    else:
        outputfile = args.json
        if os.path.exists(outputfile):
            fnparts = args.json.rsplit('.',1)
            origfile = ".".join([ fnparts[0], "orig", fnparts[1] ])
            if not os.path.exists(origfile):
                shutil.copy(outputfile,origfile)

    with open(outputfile,'w') as wh:
        wh.write(results.tojson())
