#!.venv/bin/python

import sys, os, os.path, copy
import argparse
from BKGlycanExtractor.semantics import ManuscriptSemantics, RootSemantics
from BKGlycanExtractor.glyomicsclient import ExtractorClient, GlymageClient
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
    '--output',
    type = str,
    default=None,
    help = "Output modified JSON document."
)

args = parser.parse_args()

# assume we can get images from the extractor based on the taskid
client = ExtractorClient(apiurl=args.extractorurl)
glymage = GlymageClient(image_format="png")
results = ManuscriptSemantics.read_json(args.json)
task_id = results.get('id')

glycan = None
figure = None
for f in results.figures():
    figure_path = f.get('image_path')
    figure_url = client.makeresulturl(task_id=task_id,path=figure_path)
    f.set_image_url(figure_url)
    for g in f.glycans():
        if g.get("GID") == args.glycan:
            glycan = g
            figure = f
        # remove when crop fix is deployed
        g.set('height',g.height()-1)
        g.set('width',g.width()-1)

modified = False
if args.set_redend is not None and glycan.mono(args.set_redend) is not None:

    m = glycan.mono(args.set_redend)
    glycan.swap_roots(m)
    modified = True

if modified:
    YOLO_Glycan().find_objects(glycan)

if not args.output:

    seq = glycan.get('IUPAC')
    if seq:
        task_id = glymage.submit_glymage(seq=seq,redend=True,orientation=glycan.get('orientation'))
        result = glymage.retrieve(task_id)
        glymageurl = glymage.url() + result.get('result')

    glycan.set_image(glycan.box().crop(figure.image()))
    glycan.scaleimg(factor=4.0)
    glycan.annotate_monos(label="INDEX",textanchor="TR",font_scale=1.0)
    if seq:
        glycan.show_image(title="Annotated Glycan",extraimageurl=glymageurl)
    else:
        glycan.show_image(title="Annotated Glycan")

else:

    with open(args.output,'w') as wh:
        wh.write(results.tojson())
