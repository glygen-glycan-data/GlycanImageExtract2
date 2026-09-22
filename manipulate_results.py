#!.venv/bin/python

import sys, os, os.path, copy, shutil
import readline, queue, threading
import numpy as np
import argparse, cmd, csv
from BKGlycanExtractor.bbox import BoundingBox
from BKGlycanExtractor.semantics import ManuscriptSemantics, UndirectedLinkSemantics, MonoSemantics, SideBySideImage
from BKGlycanExtractor.glyomicsclient import ExtractorClient, GlymageClient, GlyLookupClient
from BKGlycanExtractor.glycansemantics import YOLO_Glycan
from BKGlycanExtractor.pdf_image_metadata import ImageSearch


def _parse(arg, *types):
    parts = arg.split()
    if len(parts) != len(types):
        raise ValueError(f"Expected {len(types)} argument(s), got {len(parts)}")
    return [t(p) for t, p in zip(types, parts)]


def recompute_iupac(glycan):
    glycan.unset('IUPAC')
    glycan.unset('composition_str')
    YOLO_Glycan(ignore_errors=True).find_objects(glycan)


_VALID_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "UPLEFT", "UPRIGHT", "DOWNLEFT", "DOWNRIGHT")
_DIAG_SCALE = 1 / 2**0.5  # place diagonal at same Euclidean distance as cardinal

def op_add_mono(glycan, mid, dirn, scale, label):
    if dirn not in _VALID_DIRECTIONS:
        raise ValueError(f"Direction must be one of {'/'.join(_VALID_DIRECTIONS)}, got {dirn!r}")
    if not glycan.has_mono(mid):
        raise ValueError(f"No monosaccharide with id {mid}")
    m = glycan.mono(mid)
    ref_w, ref_h = m.width(), m.height()
    mcent = m.center()
    dx_step = ref_w * scale
    dy_step = ref_h * scale
    offsets = {
        "UP":        (0,                      -dy_step),
        "DOWN":      (0,                       dy_step),
        "LEFT":      (-dx_step,                0),
        "RIGHT":     ( dx_step,                0),
        "UPLEFT":    (-dx_step * _DIAG_SCALE, -dy_step * _DIAG_SCALE),
        "UPRIGHT":   ( dx_step * _DIAG_SCALE, -dy_step * _DIAG_SCALE),
        "DOWNLEFT":  (-dx_step * _DIAG_SCALE,  dy_step * _DIAG_SCALE),
        "DOWNRIGHT": ( dx_step * _DIAG_SCALE,  dy_step * _DIAG_SCALE),
    }
    dx, dy = offsets[dirn]
    newcent = (mcent[0] + dx, mcent[1] + dy)
    x = int(round(newcent[0] - ref_w / 2))
    y = int(round(newcent[1] - ref_h / 2))
    bbox = BoundingBox(x=x, y=y, w=ref_w, h=ref_h)
    newm = MonoSemantics(symbol=label, classlabel=label, box=bbox, confidence=1.0)
    glycan.add_mono(newm)
    newmid = newm.id()
    glycan.add_undirected_link(UndirectedLinkSemantics(mono_id1=mid, mono_id2=newmid, classlabel="link", confidence=1.0))


def op_delete_mono(glycan, mid):
    if not glycan.has_mono(mid):
        raise ValueError(f"No monosaccharide with id {mid}")
    glycan.delete_mono_and_ulinks(mid)


def op_adjust_box(glycan, mid, dim, delta):
    if dim not in ('x', 'y', 'w', 'h'):
        raise ValueError(f"dim must be x/y/w/h, got {dim!r}")
    m = glycan.mono(mid)
    if not m:
        raise ValueError(f"No monosaccharide with id {mid}")
    box = m.box()
    cur = getattr(box, dim)
    new_val = cur + delta
    if dim in ('w', 'h') and new_val < 1:
        raise ValueError(f"Resulting {dim}={new_val} would be < 1")
    box.update_bbox(**{dim: new_val})


def op_delete_link(glycan, mid1, mid2):
    if not glycan.remove_undirected_link(mid1, mid2):
        raise ValueError(f"No such link to delete: {mid1}-{mid2}")


def op_recover_link(glycan, mid1, mid2):
    if not glycan.recover_rejected_undirected_link(mid1, mid2):
        raise ValueError(f"No such rejected link to recover: {mid1}-{mid2}")


def op_add_link(glycan, mid1, mid2):
    if glycan.has_undirected_link(mid1, mid2):
        raise ValueError(f"Link {mid1}-{mid2} already exists")
    if glycan.has_rejected_undirected_link(mid1, mid2):
        raise ValueError(f"Link {mid1}-{mid2} is a rejected undirected link")
    glycan.add_undirected_link(UndirectedLinkSemantics(mono_id1=mid1, mono_id2=mid2, classlabel="link", confidence=1.0))


def op_set_monolabel(glycan, mid, label):
    m = glycan.mono(mid)
    if not m:
        raise ValueError(f"No monosaccharide with id {mid}")
    m.set_label(label)
    m.set_symbol(label)


def op_set_redend(glycan, mid):
    if not glycan.has_mono(mid):
        raise ValueError(f"No monosaccharide with id {mid}")
    glycan.swap_roots(glycan.mono(mid))


def op_set_orientation(glycan, orientation):
    glycan.set("orientation", orientation)


def update_tsv_row(tsvresults, glycan_gid, glycan, glylookup):
    seq = glycan.get('IUPAC')
    compstr = glycan.get('composition_str')
    acc, wurcs = None, None
    if seq:
        try:
            acc, wurcs = glylookup.get_wurcs(seq)
        except Exception:
            print("Warning: glylookup unavailable, accession/WURCS not updated.")
    tsvresults[glycan_gid]['accession'] = acc
    tsvresults[glycan_gid]['iupac'] = seq
    tsvresults[glycan_gid]['composition'] = compstr
    tsvresults[glycan_gid]['wurcs'] = wurcs
    return seq, compstr, acc, wurcs


class _PersistentDisplay:
    """Single persistent Tk window updated in place; driven by update() calls from cmdloop."""
    def __init__(self):
        self._win = SideBySideImage(title="Glycan Editor")

    def update_images(self, cv_image, glymageurl=None):
        self._win.update(cv_image, glymageurl)

    def update(self):
        try:
            self._win.root.update()
        except Exception:
            pass


def update_annotated(display, glycan, figure, seq, glymage,
                     img_scale=4.0, font_scale=1.0, label_style="INDEX", text_anchor="CENTER"):
    display_glycan = copy.deepcopy(glycan)
    display_glycan.set_image(display_glycan.box().crop(figure.image()))
    display_glycan.scaleimg(factor=img_scale)
    display_glycan.annotate_monos(label=label_style, textanchor=text_anchor, font_scale=font_scale)
    glymageurl = None
    if seq:
        try:
            task_id = glymage.submit_glymage(seq=seq, redend=True, orientation=glycan.get('orientation'))
            result = glymage.retrieve(task_id)
            glymageurl = glymage.url() + result.get('result')
        except Exception:
            print("Warning: glymage unavailable, reference image not shown.")
    display.update_images(display_glycan.image(), glymageurl)


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
        writer = csv.DictWriter(wh, fieldnames=tsvfieldnames, dialect="excel-tab")
        writer.writeheader()
        writer.writerows(tsvresults.values())
    print(f"Written: {jsonfile} and {tsvfilename}")


class GlycanEditor(cmd.Cmd):
    prompt = "glycan> "
    intro = "Glycan editor. Type 'help' for commands, 'quit' to exit."

    def __init__(self, results, jsonfile, tsvresults, tsvfilename, tsvfieldnames, glymage, glylookup):
        super().__init__()
        self.results = results
        self.jsonfile = jsonfile
        self.tsvresults = tsvresults
        self.tsvfilename = tsvfilename
        self.tsvfieldnames = tsvfieldnames
        self.glymage = glymage
        self.glylookup = glylookup
        self.glycan = None
        self.figure = None
        self.glycan_gid = None
        self.votes_overridden = False
        self.modified = False
        self.any_modified = False
        self._dirty_gids = set()
        self.next_filter = 'bad'  # 'bad', 'all', or int (specific votes value)
        self._quit_pending = False
        self._display = _PersistentDisplay()
        self.img_scale = 4.0
        self.font_scale = 1.0
        self.label_style = "INDEX"
        self.text_anchor = "CENTER"

    def _after_modify(self):
        try:
            recompute_iupac(self.glycan)
        except Exception as e:
            print(f"Warning: IUPAC recomputation failed: {e}")
        seq, compstr, acc, wurcs = update_tsv_row(
            self.tsvresults, self.glycan_gid, self.glycan, self.glylookup)
        if not self.votes_overridden:
            self.tsvresults[self.glycan_gid]['votes'] = 11
        self.modified = True
        self.any_modified = True
        self._dirty_gids.add(self.glycan_gid)
        print("GID:", self.glycan_gid)
        if seq:
            print("IUPAC:", seq)
        if compstr:
            print("Composition:", compstr)
        if acc:
            print("Accession:", acc)
        if wurcs:
            print("WURCS:", wurcs)
        print("Votes:", self.tsvresults[self.glycan_gid].get('votes'))
        update_annotated(self._display, self.glycan, self.figure, seq, self.glymage,
                         self.img_scale, self.font_scale, self.label_style, self.text_anchor)

    def _display_current(self):
        seq = self.glycan.get('IUPAC')
        compstr = self.glycan.get('composition_str')
        row = self.tsvresults.get(self.glycan_gid, {})
        acc = row.get('accession')
        wurcs = row.get('wurcs')
        print("GID:", self.glycan_gid)
        if seq:
            print("IUPAC:", seq)
        if compstr:
            print("Composition:", compstr)
        if acc:
            print("Accession:", acc)
        if wurcs:
            print("WURCS:", wurcs)
        print("Votes:", row.get('votes'))
        update_annotated(self._display, self.glycan, self.figure, seq, self.glymage,
                         self.img_scale, self.font_scale, self.label_style, self.text_anchor)

    def cmdloop(self, intro=None):
        self.preloop()
        if intro is None:
            intro = self.intro
        if intro:
            self.stdout.write(str(intro) + "\n")
            self.stdout.flush()

        q = queue.Queue()
        ready = threading.Event()

        def _reader():
            while True:
                ready.wait()
                ready.clear()
                try:
                    q.put(input(self.prompt))
                except EOFError:
                    q.put(None)
                    return
                except KeyboardInterrupt:
                    q.put('')

        threading.Thread(target=_reader, daemon=True).start()

        # Map the window and fire the deferred render before showing the prompt
        if self._display:
            for _ in range(3):
                self._display.update()
        ready.set()

        stop = False
        while not stop:
            if self._display:
                self._display.update()
            try:
                raw = q.get(timeout=0.05)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                self.stdout.write('\n')
                self.stdout.flush()
                ready.set()
                continue
            if self._quit_pending:
                self._quit_pending = False
                self.prompt = type(self).prompt
                ans = ('' if raw is None else raw).strip().lower()
                if ans == 'y':
                    self.do_write('')
                    stop = True
                elif ans == 'n':
                    stop = True
                else:
                    print("Enter y (save and quit) or n (quit without saving).")
                    self._quit_pending = True
                    self.prompt = "Save? [y/N] "
                    ready.set()
                continue
            line = 'EOF' if raw is None else raw
            line = self.precmd(line)
            stop = self.onecmd(line)
            stop = self.postcmd(stop, line)
            if not stop:
                ready.set()
        self.postloop()

    def _require_glycan(self):
        if self.glycan is None:
            print("No glycan selected. Use: glycan <GID>")
            return False
        return True

    def do_glycan(self, arg):
        """glycan <GID>  — Select which glycan to edit."""
        try:
            [gid] = _parse(arg, str)
        except ValueError as e:
            print(f"Error: {e}")
            return
        if self.modified and self.glycan_gid is not None and not self.votes_overridden:
            ans = input(f"Votes for {self.glycan_gid} [1]: ").strip()
            votes = int(ans) if ans else 1
            self.tsvresults[self.glycan_gid]['votes'] = votes
            self.votes_overridden = True
            self._dirty_gids.add(self.glycan_gid)
        for f in self.results.figures():
            for g in f.glycans():
                if g.get("GID") == gid:
                    self.glycan = g
                    self.figure = f
                    self.glycan_gid = gid
                    self.votes_overridden = False
                    self.modified = False
                    recompute_iupac(self.glycan)
                    update_tsv_row(self.tsvresults, self.glycan_gid, self.glycan, self.glylookup)
                    self._display_current()
                    return
        print(f"GID not found: {gid}")

    def do_add(self, arg):
        """add mono <mid> <UP|DOWN|LEFT|RIGHT|UPLEFT|UPRIGHT|DOWNLEFT|DOWNRIGHT> <scale> <label> | add link <mid1> <mid2>"""
        if not self._require_glycan():
            return
        parts = arg.split()
        if not parts:
            print("Usage: add mono <mid> <dirn> <scale> <label> | add link <mid1> <mid2>")
            return
        subcmd, rest = parts[0], ' '.join(parts[1:])
        try:
            if subcmd == 'mono':
                mid, dirn, scale, label = _parse(rest, int, str, float, str)
                op_add_mono(self.glycan, mid, dirn, scale, label)
                self._after_modify()
            elif subcmd == 'link':
                mid1, mid2 = _parse(rest, int, int)
                try:
                    op_recover_link(self.glycan, mid1, mid2)
                except ValueError:
                    op_add_link(self.glycan, mid1, mid2)
                self._after_modify()
            else:
                print(f"Unknown subcommand: {subcmd!r}. Use 'mono' or 'link'.")
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_delete(self, arg):
        """delete mono <mid> | delete link <mid1> <mid2>"""
        if not self._require_glycan():
            return
        parts = arg.split()
        if not parts:
            print("Usage: delete mono <mid> | delete link <mid1> <mid2>")
            return
        subcmd, rest = parts[0], ' '.join(parts[1:])
        try:
            if subcmd == 'mono':
                [mid] = _parse(rest, int)
                op_delete_mono(self.glycan, mid)
                self._after_modify()
            elif subcmd == 'link':
                mid1, mid2 = _parse(rest, int, int)
                op_delete_link(self.glycan, mid1, mid2)
                self._after_modify()
            else:
                print(f"Unknown subcommand: {subcmd!r}. Use 'mono' or 'link'.")
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_monolabel(self, arg):
        """monolabel <mid> <label>  — Change the label and symbol of a monosaccharide."""
        if not self._require_glycan():
            return
        try:
            mid, label = _parse(arg, int, str)
            op_set_monolabel(self.glycan, mid, label)
            self._after_modify()
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_redend(self, arg):
        """redend <mid>  — Set the reducing end (root) monosaccharide."""
        if not self._require_glycan():
            return
        try:
            [mid] = _parse(arg, int)
            op_set_redend(self.glycan, mid)
            self._after_modify()
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_adjust(self, arg):
        """adjust <mid> <x|y|w|h> <delta>  — Shift/resize a mono bounding box by delta pixels."""
        if not self._require_glycan():
            return
        try:
            mid, dim, delta = _parse(arg, int, str, int)
            op_adjust_box(self.glycan, mid, dim, delta)
            self._after_modify()
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_orientation(self, arg):
        """orientation <orientation>  — Set display orientation without recomputing IUPAC."""
        if not self._require_glycan():
            return
        try:
            [orientation] = _parse(arg, str)
            op_set_orientation(self.glycan, orientation)
            self.modified = True
            self.any_modified = True
            self._dirty_gids.add(self.glycan_gid)
            self._display_current()
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_imgscale(self, arg):
        """imgscale <factor>  — Set image magnification factor (default 4.0)."""
        try:
            [self.img_scale] = _parse(arg, float)
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_fontscale(self, arg):
        """fontscale <factor>  — Set annotation font scale (default 1.0)."""
        try:
            [self.font_scale] = _parse(arg, float)
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_label(self, arg):
        """label <INDEX|MONO:INDEX>  — Set label style for monosaccharide annotations."""
        try:
            [style] = _parse(arg, str)
            if style not in ("INDEX", "MONO:INDEX"):
                print("Error: label must be INDEX or MONO:INDEX")
                return
            self.label_style = style
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_anchor(self, arg):
        """anchor <TR|CENTER|BR>  — Set label position (top-right, center, bottom-right)."""
        try:
            [anchor] = _parse(arg, str)
            if anchor not in ("TR", "CENTER", "BR"):
                print("Error: anchor must be TR, CENTER, or BR")
                return
            self.text_anchor = anchor
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def _next_gids(self):
        if self.next_filter == 'bad':
            return [gid for gid, row in self.tsvresults.items() if int(row.get('votes', 0)) < 0]
        elif self.next_filter == 'all':
            return list(self.tsvresults.keys())
        else:
            v = self.next_filter
            return [gid for gid, row in self.tsvresults.items() if int(row.get('votes', 0)) == v]

    def do_next(self, arg):
        """next  — Move to the next glycan per current mode (see: mode)."""
        bad = self._next_gids()
        if not bad:
            print(f"No glycans matching current mode ({self._mode_desc()}).")
            return
        if self.glycan_gid in bad:
            idx = (bad.index(self.glycan_gid) + 1) % len(bad)
        else:
            all_gids = list(self.tsvresults.keys())
            cur_pos = all_gids.index(self.glycan_gid) if self.glycan_gid in all_gids else -1
            later = [gid for gid in bad if all_gids.index(gid) > cur_pos]
            idx = bad.index(later[0]) if later else 0
        self.do_glycan(bad[idx])

    do_n = do_next

    def do_prev(self, arg):
        """prev  — Move to the previous glycan per current mode (see: mode)."""
        gids = self._next_gids()
        if not gids:
            print(f"No glycans matching current mode ({self._mode_desc()}).")
            return
        if self.glycan_gid in gids:
            idx = (gids.index(self.glycan_gid) - 1) % len(gids)
        else:
            all_gids = list(self.tsvresults.keys())
            cur_pos = all_gids.index(self.glycan_gid) if self.glycan_gid in all_gids else len(all_gids)
            earlier = [gid for gid in gids if all_gids.index(gid) < cur_pos]
            idx = gids.index(earlier[-1]) if earlier else len(gids) - 1
        self.do_glycan(gids[idx])

    do_p = do_prev

    def _mode_desc(self):
        if self.next_filter == 'bad':
            return 'votes < 0'
        elif self.next_filter == 'all':
            return 'all glycans'
        return f'votes == {self.next_filter}'

    def do_mode(self, arg):
        """mode [bad | all | <votes_value>]  — Set or show what next/n cycles through."""
        arg = arg.strip()
        if not arg:
            print(f"Current mode: {self._mode_desc()}")
            return
        if arg == 'bad':
            self.next_filter = 'bad'
        elif arg == 'all':
            self.next_filter = 'all'
        else:
            try:
                self.next_filter = int(arg)
            except ValueError:
                print("Usage: mode [bad | all | <votes_value>]")
                return
        print(f"Mode set: {self._mode_desc()}")
        gids = self._next_gids()
        if gids:
            self.do_glycan(gids[0])
        else:
            print(f"No glycans matching current mode.")

    def do_show(self, arg):
        """show  — Re-display the current annotated image and IUPAC."""
        if not self._require_glycan():
            return
        self._display_current()

    def do_list(self, arg):
        """list  — Print monosaccharide IDs/labels/boxes and link pairs."""
        if not self._require_glycan():
            return
        print(f"Monosaccharides:")
        print(f"  {'ID':>4}  {'Label':<16} {'x':>6} {'y':>6} {'w':>6} {'h':>6}")
        for m in self.glycan.monos():
            b = m.box()
            print(f"  {m.id():>4}  {m.get('classlabel') or '':16} {b.x:>6} {b.y:>6} {b.w:>6} {b.h:>6}")
        print("Links:")
        for lnk in self.glycan.undirected_links():
            ids = lnk.mono_ids()
            print(f"  {ids[0]} -- {ids[1]}")

    def do_votes(self, arg):
        """votes [<n>]  — Set votes for current glycan, or list vote value counts if no argument."""
        arg = arg.strip()
        if not arg:
            from collections import Counter
            counts = Counter(int(row.get('votes', 0)) for row in self.tsvresults.values())
            for v, c in sorted(counts.items()):
                print(f"  {v:4d}: {c} glycan(s)")
            return
        if not self._require_glycan():
            return
        try:
            [votes] = _parse(arg, int)
            self.tsvresults[self.glycan_gid]['votes'] = votes
            self.votes_overridden = True
            self.modified = True
            self.any_modified = True
            self._dirty_gids.add(self.glycan_gid)
        except (ValueError, KeyError, IndexError) as e:
            print(f"Error: {e}")

    def do_modified(self, arg):
        """modified  — List GIDs with unsaved changes to TSV or JSON."""
        if not self._dirty_gids:
            print("No unsaved changes.")
            return
        print("GIDs with unsaved changes:")
        for gid in self.tsvresults:
            if gid in self._dirty_gids:
                print(f"  {gid}")

    def do_write(self, arg):
        """write  — Save JSON and TSV files (with .orig backups)."""
        try:
            write_files(self.results, self.jsonfile, self.tsvfilename,
                        self.tsvresults, self.tsvfieldnames)
            self.modified = False
            self.any_modified = False
            self._dirty_gids.clear()
        except Exception as e:
            print(f"Error writing files: {e}")

    def do_quit(self, arg):
        """quit  — Exit the editor (prompts to save if there are unsaved changes)."""
        if self.any_modified:
            print("Unsaved modifications.")
            self._quit_pending = True
            self.prompt = "Save? [y/N] "
            return False
        return True

    def do_EOF(self, arg):
        print()
        if self.any_modified:
            print("Unsaved modifications.")
            self._quit_pending = True
            self.prompt = "Save? [y/N] "
            return False
        return True

    do_g = do_glycan
    do_a = do_add
    do_d = do_delete
    do_r = do_redend
    do_l = do_list
    do_s = do_show
    do_v = do_votes
    do_w = do_write
    do_m = do_mode
    do_q = do_quit


parser = argparse.ArgumentParser(description="Interactively edit glycan results JSON")

parser.add_argument(
    '--json',
    type=str,
    required=True,
    help='JSON format extractor result file.'
)

args = parser.parse_args()

assert args.json.endswith(".json") and os.path.exists(args.json)

jsonfile = args.json
tsvfilename = jsonfile.replace('.json', '.tsv')
pdffilename = jsonfile.replace('.json', '.pdf')
figuresdir = jsonfile.replace('.json', '.figs')

assert os.path.exists(tsvfilename), f"TSV not found: {tsvfilename}"
assert os.path.exists(pdffilename), f"PDF not found: {pdffilename}"

client = ExtractorClient(apiurl=args.extractorurl)
glymage = GlymageClient(image_format="png")
glylookup = GlyLookupClient()

tsvreader = csv.DictReader(open(tsvfilename), dialect="excel-tab")
tsvresults = dict((row['ID'], row) for row in tsvreader)
tsvfieldnames = tsvreader.fieldnames

results = ManuscriptSemantics.read_json(jsonfile)
image_search_strategy = results.get('image_search_strategy')

def _normalize_img_ext(filename):
    base, ext = os.path.splitext(filename)
    if ext.lower() == '.jpg':
        ext = '.jpeg'
    return base + ext

image_path_dict = {}
bbox_override_dict = {}
for f in results.figures():
    ic = f.get('image_count')
    ip = f.get('image_path')
    if ic is not None and ip is not None:
        image_path_dict[ic] = os.path.join(figuresdir, _normalize_img_ext(os.path.split(ip)[1]))
    bbox = f.get('pdf_fig_bbox')
    if ic is not None and bbox is not None:
        bbox_override_dict[ic] = bbox

if not os.path.isdir(figuresdir):
    os.makedirs(figuresdir)
strategy = ImageSearch.search_method(image_search_strategy)
fresh_figures = strategy.get_metadata(pdffilename, figuresdir,
                                      image_path_dict=image_path_dict or None,
                                      use_annotations=False,
                                      bbox_override_dict=bbox_override_dict or None)

for f in results.figures():
    ic = f.get('image_count')
    ip = f.get('image_path')
    if ip is not None:
        f.set_image_path(os.path.join(figuresdir, _normalize_img_ext(os.path.split(ip)[1])))
    elif ic is not None:
        f.set_image_path(os.path.join(figuresdir, f"fig{ic}.png"))

def _as_list(v):
    return list(v) if isinstance(v, (tuple, list)) else v

fresh_by_ic = {fig['image_count']: fig for fig in fresh_figures if fig.get('image_count') is not None}
for f in results.figures():
    ic = f.get('image_count')
    if ic is None:
        continue
    path = f.get('image_path')
    label = os.path.basename(path) if path else f"figure {ic}"
    if ic not in fresh_by_ic:
        print(f"Warning: {label}: not found in current PDF metadata")
        continue
    fresh = fresh_by_ic[ic]
    for attr in ('pdf_fig_bbox', 'xref', 'page_number'):
        json_val = _as_list(f.get(attr))
        pdf_val  = _as_list(fresh.get(attr))
        if json_val != pdf_val:
            print(f"Warning: {label} {attr} mismatch: JSON={json_val!r}, PDF={pdf_val!r}")
    exp_w, exp_h = f.get('width'), f.get('height')
    if path and exp_w is not None and exp_h is not None:
        if not os.path.exists(path):
            print(f"Warning: {label}: image file missing")
        else:
            img = f.read_image(path)
            if img is None:
                print(f"Warning: {label}: could not load image")
            else:
                act_h, act_w = img.shape[:2]
                if act_w != exp_w or act_h != exp_h:
                    print(f"Warning: {label} size mismatch: "
                          f"expected {exp_w}x{exp_h}, got {act_w}x{act_h}")

editor = GlycanEditor(results, jsonfile, tsvresults, tsvfilename, tsvfieldnames, glymage, glylookup)
print(editor.intro)

if args.glycan:
    editor.do_glycan(args.glycan)
else:
    editor.do_next('')

editor.cmdloop(intro="")
