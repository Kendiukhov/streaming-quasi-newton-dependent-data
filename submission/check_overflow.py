"""Report every overfull box in a LaTeX build, with the source file and line it came from.

pdflatex reports overfull boxes without naming the file, which makes them tedious to chase in a
document assembled from twenty \\input fragments.  This walks the log's file stack -- the "(./x.tex"
and ")" markers -- so each report is attributed to the file that produced it.

Usage:  python3 check_overflow.py [manuscript.log ...]   (default: every *.log here)
Exit status is 1 if any box exceeds --threshold, so it can gate a build.
"""
import glob
import os
import re
import sys

THRESHOLD = float(os.environ.get("OVERFULL_PT", "1.0"))


def parse(path):
    """Yield (points_too_wide, file, first_line, last_line, kind) for each overfull box."""
    txt = open(path, errors="replace").read()
    # The log interleaves file-open "(name" / file-close ")" markers with messages.  Track them
    # with a stack, ignoring parentheses inside message text by only accepting "(" immediately
    # followed by something that looks like a path.
    stack, out = [], []
    token = re.compile(
        r"\((?P<open>[^()\s]*\.(?:tex|sty|cls|bst|def|cfg|fd|clo))"
        r"|(?P<close>\))"
        r"|Overfull \\(?P<kind>hbox|vbox) \((?P<pt>[\d.]+)pt too (?:wide|high)\)"
        r"(?: in paragraph at lines (?P<l1>\d+)--(?P<l2>\d+)"
        r"| detected at line (?P<l3>\d+)"
        r"| in alignment at lines (?P<l4>\d+)--(?P<l5>\d+))?")
    for m in token.finditer(txt):
        if m.group("open"):
            stack.append(m.group("open"))
        elif m.group("close"):
            if stack:
                stack.pop()
        else:
            a = m.group("l1") or m.group("l3") or m.group("l4") or "?"
            b = m.group("l2") or m.group("l5") or a
            out.append((float(m.group("pt")), stack[-1] if stack else "?", a, b,
                        m.group("kind")))
    return out


def main():
    logs = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "*.log")))
    worst = 0.0
    for log in logs:
        boxes = [b for b in parse(log) if b[0] >= THRESHOLD]
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
    return 1 if worst >= THRESHOLD else 0


if __name__ == "__main__":
    sys.exit(main())
