#!.venv/bin/python
from __future__ import print_function

import sys, os, random, time, re, shutil, traceback
from collections import defaultdict
import findpygly
from pygly.GlycanImage import GlycanImage
from pygly.GlycanResource import GlyTouCan, GlyCosmos
from pygly.GlycanFormatter import IUPACLinearFormat
from pygly.manipulation import Topology
from pygly.CompositionTable import Composition

from BKGlycanExtractor.image_manager import Image_Data

import argparse

parser = argparse.ArgumentParser(description="Randomized glycan image generation")
parser.add_argument("-n", "--nimages", type=int, help="Number of images. Default: 100.", default=100)
# parser.add_argument("-B", "--batchsize", type=int, help="Number of images with each randomly selected set of options. Default: 10.", default=10)
# parser.add_argument("-f", "--format", type=str, help="Image format. One of \"png\" or \"svg\". Default: png.", default='png')
parser.add_argument("-o", "--outdir", type=str, help="Ouput directory. Default: current directory.", default=None)
parser.add_argument("-c", "--clear", action='store_true', help="Clear output directory first.", default=False)
parser.add_argument("-F", "--force", action='store_true', help="Force re-download of GlyTouCan accessions and sequences", default=False)
parser.add_argument("-s", "--skip", type=str, help="File of accessions to skip. Default: None.", default=None)
parser.add_argument("-r", "--random", type=str, help="Randomization mode. One of uniform accessions (uniform), biased accessions (biased), random monosaccharides (mono), random monosaccharides + baised accessions (biasmono). Default: uniform.", default="uniform")
parser.add_argument("-A", "--accessions", type=str, help="Limit to specific accessions by regular expression or prefix. Default: No restriction.", default=None)
parser.add_argument("-L", "--linkage", action='store_true', help="Require glycosydic linkage information (display: normalinfo). Default: compact, normal, normainfo. ", default=False)
parser.add_argument("--no_links", action='store_true', help="Do not display glycosydic linkage information (display: normal, compact). Default: compact, normal, normainfo.", default=False)
parser.add_argument("--writelinks", type=str, help="overwrite all links to the specification [anomer,carbon#]", default=False)

args = parser.parse_args()
imagenum = args.nimages
mode = "svg"
args.batchsize = 1
cachemode = 'c'
if args.force:
    cachemode = 'n'
accregex = None
if args.accessions:
    accregex = args.accessions
    if not accregex.startswith('^'):
        accregex = "^"+accregex
    if not accregex.endswith('$'):
        accregex = accregex+".*$"
    accregex = re.compile(args.accessions)
assert mode in ("png","svg")
output_folder = args.outdir
if output_folder:
    if args.clear:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    assert os.path.isdir(output_folder)

badaccfile = args.skip
if badaccfile:
    assert os.path.isfile(badaccfile)
randmode = args.random
assert randmode in ("uniform","biased","mono","biasmono")

print("Start randimg...")

batch = args.batchsize
iterations = imagenum//batch
scale_options = [ 0.5, 1.0, 2.0, 4.0, ]
redend_options = [ True, False ]
orient_options = [ "RL", "LR", "TB", "BT" ]
notation_options = [ "snfg", "cfg", "snfglink", "cfglink" ]
display_options = [ "normal", "normalinfo", "compact" ]
if args.linkage:
    display_options = [ "normal", "compact" ] + 18*[ "normalinfo" ]
    # display_options = [ "normalinfo" ]
    notation_options = [ "snfg", "cfg" ]
if args.no_links:
    display_options = [ "normal", "compact" ]
    notation_options = [ "snfg", "cfg" ]
if args.writelinks:
    display_options = ["normalinfo"]
    notation_options = [ "snfg", "cfg" ]
opaque_options = [ True, False ]

valid_monos_str = """
Glc Gal Man
NeuAc NeuGc
Fuc Xyl
GlcNAc GalNAc
"""
valid_monos = valid_monos_str.split()
valid_subst = defaultdict(set)
for l in valid_monos_str.splitlines():
    monos = l.split()
    for m1 in monos:
        for m2 in monos:
            valid_subst[m1].add(m2)
for k in valid_subst:
    valid_subst[k] = list(valid_subst[k])

ip = IUPACLinearFormat()
topo = Topology()

print("GlyCosmos archived...",file=sys.stderr)
start = time.time()
gco = GlyCosmos(verbose=False,usecache=True,cachemode=cachemode)
archived = set(map(lambda d: d['accession'],gco.archived()))
print("GlyCosmos archived complete. (%s secs.)"%(time.time()-start,),file=sys.stderr)

print("GlyTouCan accessions...",file=sys.stderr)
start = time.time()
gtc = GlyTouCan(verbose=False,usecache=True,cachemode=cachemode)
accs = list(filter(lambda acc: acc not in archived,gtc.allaccessions()))
dummy = gtc.getseq('G00912UN','wurcs')
print("GlyTouCan accessions complete. (%s secs.)"%(time.time()-start,),file=sys.stderr)

if accregex:
    accs = list(filter(lambda acc: accregex.search(acc),accs))
    imagenum = min(imagenum,len(accs))

# accessions your model was trained on, to avoid testing on them
trained_accessions = set()
if badaccfile is not None:
    trained_accessions = set(open(badaccfile).read().split())

monofreq = Composition()
monofreq.set(*valid_monos,value=1)
monofreq['Count'] = len(valid_monos)

imageData = Image_Data(valid_monos)

outputcount = 0
seen = trained_accessions
for j in range(iterations):
    imageWriter = GlycanImage()
    imageWriter.set('scale',random.choice(scale_options))
    imageWriter.set('reducing_end',random.choice(redend_options))
    imageWriter.set('orientation',random.choice(orient_options))
    imageWriter.set('notation',random.choice(notation_options))
    imageWriter.set('display',random.choice(display_options))
    #imageWriter.set('opaque',random.choice(opaque_options))
    imageWriter.set('format',mode)
    imageWriter.force(True)
    # imageWriter.verbose(True)

    count = 0
    while count < batch and len(accs) > len(seen):
        acc = random.choice(accs)
        if acc in seen:
            continue
        print("random choice:",acc,file=sys.stderr)
        seen.add(acc)
        acc1 = acc
        if 'mono' in randmode:
            acc1 = "R%07d"%(outputcount + 1,)
        outfile = os.path.join(output_folder, acc1 + "." + mode)
        pngfile = os.path.join(output_folder, acc1 + ".png")
        if os.path.exists(outfile) or os.path.exists(pngfile):
            continue
        seq = gtc.getseq(acc,format='wurcs')
        if not seq:
            continue
        gly = gtc.getGlycan(acc,format='wurcs')
        if not gly:
            continue
        if gly.undetermined():
            continue
        if not gly.has_root():
            continue
        if gly.repeated():
            continue
        comp = gly.iupac_composition(floating_substituents=False,
                                     aggregate_basecomposition=False)
        if comp['Count'] < 3:
            continue
        bad = False
        for k,v in comp.items():
            if k in valid_monos or k == "Count":
                continue
            if v <= 0:
                continue
            bad = True
            break
        for l in gly.all_links():
            pp = l.parent_pos() 
            if pp != None and len(pp) > 1:
                bad = True
                break
            if pp != None and list(pp)[0] not in (2,3,4,6,8):
                bad = True
                break
        if bad:
            continue
        if randmode in ("mono","biasmono"):
            gly_iupac = ip.toStr(gly)
            gly1 = ip.toGlycan(gly_iupac)
            # print(gly_iupac)
            l = re.split(r'([A-Za-z]+)',gly_iupac)
            # print(l)
            for i,tok in enumerate(l):
                if tok in valid_monos:
                    l[i] = random.choice(valid_subst[tok])
                elif tok[:-1] in valid_monos and tok[-1] in 'ab':
                    l[i] = random.choice(valid_subst[tok[:-1]]) + tok[-1]
                elif re.search(r'[A-Za-z]',tok):
                    raise RuntimeError("Bad IUPAC split")
            # print(l)
            gly_iupac = "".join(l)
            # print(gly_iupac)
            gly = ip.toGlycan(gly_iupac)
            comp = gly.iupac_composition(floating_substituents=False,
                                         aggregate_basecomposition=False)
            seq = gly.glycoct()
            
        if randmode in("biased","biasmono"):
            orig_freq = [ monofreq[m]/monofreq['Count'] for m in valid_monos ]
            new_freq = [ (monofreq[m]+comp[m])/(monofreq['Count']+comp['Count']) for m in valid_monos ]
            minf = 1e+20
            # print(orig_freq)
            # print(new_freq)
            good1 = False
            good2 = False
            for i,(of,nf) in enumerate(zip(orig_freq,new_freq)):
                if of <= min(orig_freq) and nf > of:
                    good1 = True
                    # print(valid_monos[i],round(of,2),"<",round(nf,2))
                elif of >= max(orig_freq) and nf < of:
                    good2 = True
                    # print(valid_monos[i],round(of,2),">",round(nf,2))
            if not good1 or not good2:
                # print()
                continue

        imageWriter.writeImage(seq,outfile)
        mapfile = None
        try:
            mapfile = imageData.generate_image(outfile, overwrite_links=args.writelinks) #campbell
        except (ValueError,FileNotFoundError):
            if os.path.exists(pngfile):
                os.unlink(pngfile)
            if os.path.exists(outfile):
                os.unlink(outfile)
            if mapfile is not None and os.path.exists(mapfile):
                os.unlink(mapfile)
            continue

        # If mapfile is None, skip the rest of the code
        if mapfile is None:
            print(f"Error: mapfile is None for {outfile}. Skipping...")
            continue

        h = open(mapfile)
        mapfiledata = list(h.read().splitlines())
        h.close()
        comp1 = Composition()
        for l in mapfiledata:
            sl = l.split()
            if sl[0] != 'm':
                continue
            comp1[sl[2]] += 1
        bad = False
        for m in valid_monos:
            if comp1[m] != comp[m]:
                bad = True
        if bad:
            os.unlink(outfile)
            os.unlink(pngfile)
            os.unlink(mapfile)
            continue
        if imageWriter.get('display') not in ("normalinfo",):
            for i in range(len(mapfiledata)):
                sl = mapfiledata[i].split()
                if sl[0] == "m":
                   sl[3] = "?"
                elif sl[0] == "l":
                   sl[2] = "?"
                   sl[3] = "?"
                mapfiledata[i] = "\t".join(sl)
        if imageWriter.get('reducing_end') not in (True,):
            for i in range(len(mapfiledata)):
                sl = mapfiledata[i].split()
                if sl[0] == "m":
                   sl[3] = "?"
                   mapfiledata[i] = "\t".join(sl)
                   break
        wh = open(mapfile,'w')
        if acc1 != acc:
            print("# orig_accession:",acc,file=wh)
        for k in ('scale','reducing_end','orientation','notation','display','opaque'):
            print("# "+k+":",imageWriter.get(k),file=wh)
        print("# composition:",comp,file=wh)
        gly_iupac = ip.toStr(gly)
        print("# iupac:",gly_iupac,file=wh)
        topo_iupac = ip.toStr(topo(gly))
        print("# topo:",topo_iupac,file=wh)
        wh.write("\n".join(mapfiledata))
        wh.close()
        print(outputcount,acc1,file=sys.stderr)
        monofreq.add(comp)
        os.unlink(outfile)
        count += 1
        outputcount += 1

if outputcount > 0:
    print(monofreq)
