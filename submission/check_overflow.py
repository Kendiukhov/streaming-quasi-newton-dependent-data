"""Report every overfull box, and every oversized float, in a LaTeX build.

pdflatex reports overfull boxes without naming the file, which makes them tedious to chase in a
document assembled from twenty \\input fragments.  This walks the log's file stack -- the "(./x.tex"
and ")" markers -- so each report is attributed to the file that produced it.

It also reports "Float too large for page", which is a different warning and is NOT an overfull
box: a float taller than \\textheight is simply allowed to run off the bottom of the page. That is
how Table D.4 came to hang 617 pt off page 44 of the manuscript while this script reported nothing.
Height is not something a \\resizebox width guard can catch, so it has to be checked here -- and
independently on the rendered page, which is what check_geometry.py does.

Usage:  python3 check_overflow.py [manuscript.log ...]   (default: every *.log here)
Exit status is 1 if any box exceeds the threshold, or if any float is too large for its page.
"""
import glob
import os
import re
import sys

THRESHOLD = float(os.environ.get("OVERFULL_PT", "1.0"))


def parse(path):
    """Yield (points_too_wide, file, first_line, last_line, kind) for each overfull box.

    Read with errors="replace" and not through a text-mode grep: pdfTeX echoes the offending
    glyphs inside an overfull-box dump, and with newtx those bytes are not valid UTF-8, which
    makes the file look binary. `grep` then skips it and reports nothing at all -- which is
    exactly how paper/main.log appeared to be free of a warning it in fact contains.
    """
    txt = open(path, errors="replace").read()
    # The log interleaves file-open "(name" / file-close ")" markers with messages.  Track them
    # with a stack, ignoring parentheses inside message text by only accepting "(" immediately
    # followed by something that looks like a path.
    stack, out = [], []
    token = re.compile(
        r"\((?P<open>[^()\s]*\.(?:tex|sty|cls|bst|def|cfg|fd|clo))"
        r"|(?P<close>\))"
        r"|Overfull \\(?P<kind>hbox|vbox) \((?P<pt>[\d.]+)pt too (?:wide|high)\)"
        r"|(?P<float>Float too large for page by (?P<fpt>[\d.]+)pt on input line (?P<fl>\d+))"
        r"(?: in paragraph at lines (?P<l1>\d+)--(?P<l2>\d+)"
        r"| detected at line (?P<l3>\d+)"
        r"| in alignment at lines (?P<l4>\d+)--(?P<l5>\d+))?")
    for m in token.finditer(txt):
        if m.group("open"):
            stack.append(m.group("open"))
        elif m.group("close"):
            if stack:
                stack.pop()
        elif m.group("float"):
            out.append((float(m.group("fpt")), stack[-1] if stack else "?",
                        m.group("fl"), m.group("fl"), "float"))
        else:
            a = m.group("l1") or m.group("l3") or m.group("l4") or "?"
            b = m.group("l2") or m.group("l5") or a
            out.append((float(m.group("pt")), stack[-1] if stack else "?", a, b,
                        m.group("kind")))
    return out


def main():
    # Default to the logs of BOTH documents.  Globbing only this directory silently exempted
    # paper/main.log, which carries the same warnings because it \inputs the same fragments.
    here = os.path.dirname(os.path.abspath(__file__))
    logs = sys.argv[1:] or (sorted(glob.glob(os.path.join(here, "*.log")))
                            + sorted(glob.glob(os.path.join(here, "..", "paper", "main.log"))))
    worst = 0.0
    floats_seen = 0
    for log in logs:
        parsed = parse(log)
        floats_seen += sum(1 for b in parsed if b[4] == "float")
        boxes = [b for b in parsed if b[0] >= THRESHOLD or b[4] == "float"]
        name = os.path.basename(log)
        if not boxes:
            print(f"{name}: no overfull box over {THRESHOLD:g}pt")
            continue
        print(f"\n{name}: {len(boxes)} overfull box(es) over {THRESHOLD:g}pt")
        for pt, f, a, b, kind in sorted(boxes, key=lambda x: -x[0]):
            worst = max(worst, pt)
            where = f"{os.path.basename(f)}:{a}" + (f"-{b}" if b != a else "")
            print(f"  {pt:8.1f}pt  {kind:5s}  {where}")
    print(f"\nworst: {worst:.1f}pt")
    if floats_seen:
        print(f"{floats_seen} float(s) too large for their page -- a float taller than the text "
              f"block runs off the bottom of the page; make it a longtable or split it")
    return 1 if (worst >= THRESHOLD or floats_seen) else 0


if __name__ == "__main__":
    sys.exit(main())
