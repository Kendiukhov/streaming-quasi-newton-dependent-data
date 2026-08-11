"""Find ink that leaves the text block on any side of any page of a built PDF.

This exists because two defects got past the checks that came before it.

``check_overflow.py`` reads the LaTeX log, and a log tells you only what LaTeX chose to complain
about: an over-wide line is an "Overfull \\hbox", but a table that is taller than the page is a
"Float too large for page" warning, which that script did not look for -- so Table D.4, a float
617 pt too tall, ran off the bottom of page 44 of the manuscript while every check passed.

An earlier pixel scan measured only the *right* margin, and sampled every other pixel row, which
would also have missed a table whose rules are a single pixel tall.

So this one renders every page and measures the extreme ink position on all four sides, over every
pixel row and column. It needs no configuration and no page-geometry constants: the text block's
own edge is whatever value the extremes take on *most* pages, so the baseline per side is the mode
of the per-page extremes, and a defect is a page that exceeds its side's baseline. That is what
makes it self-calibrating -- it works unchanged for the manuscript (line numbers in the left
margin, a running footer below the text block) and for the preprint (neither), because in each
document those features simply move the baseline.

Usage:  python3 check_geometry.py [file.pdf ...]     (default: every PDF beside this script,
                                                      plus ../paper/main.pdf)
Exit status is 1 if any page exceeds its baseline by more than --fail (default 0.10 in).
"""
import collections
import glob
import os
import subprocess
import sys
import tempfile

DPI = int(os.environ.get("GEOMETRY_DPI", "150"))
WARN_IN = float(os.environ.get("GEOMETRY_WARN_IN", "0.04"))
FAIL_IN = float(os.environ.get("GEOMETRY_FAIL_IN", "0.10"))
INK = int(os.environ.get("GEOMETRY_INK", "175"))       # a pixel this dark or darker is ink

SIDES = ("left", "right", "top", "bottom")


def _extremes(png):
    """Return the extreme ink position on each side of one page, in pixels from that edge."""
    from PIL import Image
    im = Image.open(png).convert("L")
    w, h = im.size
    px = im.load()
    left, right, top, bottom = w, 0, h, 0
    for y in range(h):                                  # every row: a table rule is 1 px tall
        row_min = row_max = None
        for x in range(w):
            if px[x, y] <= INK:
                row_min = x
                break
        if row_min is None:
            continue
        for x in range(w - 1, -1, -1):
            if px[x, y] <= INK:
                row_max = x
                break
        left = min(left, row_min)
        right = max(right, row_max)
        top = min(top, y)
        bottom = max(bottom, y)
    return {"left": left, "right": w - 1 - right, "top": top, "bottom": h - 1 - bottom}


def audit(pdf):
    """Measure every page of one PDF and report pages that exceed their side's baseline."""
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, "p")
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", pdf, stem],
                       check=True, capture_output=True)
        pages = sorted(glob.glob(stem + "*.png"))
        if not pages:
            print(f"{os.path.basename(pdf)}: could not render")
            return 0.0, []
        per_page = {}
        for f in pages:
            n = int(os.path.basename(f).split("-")[-1].split(".")[0])
            per_page[n] = _extremes(f)

    # Baseline per side: the value the extremes take on most pages.  On a side with no margin
    # furniture that is the text block edge; where the document puts something in the margin
    # (line numbers on the left, a footer at the bottom) the baseline absorbs it, which is the
    # point -- we are looking for pages that stick out *relative to the rest of the document*.
    base = {}
    for s in SIDES:
        vals = [v[s] for v in per_page.values()]
        base[s] = collections.Counter(vals).most_common(1)[0][0]

    findings = []
    for n in sorted(per_page):
        for s in SIDES:
            over = (base[s] - per_page[n][s]) / DPI      # inches past the baseline
            if over > WARN_IN:
                findings.append((over, n, s))
    findings.sort(reverse=True)

    name = os.path.basename(pdf)
    print(f"\n{name}: {len(pages)} pages, baseline per side (in from the paper edge) "
          + ", ".join(f"{s}={base[s] / DPI:.2f}" for s in SIDES))
    if not findings:
        print(f"  no page sticks out past its baseline by more than {WARN_IN:.2f} in")
        return 0.0, []
    for over, n, s in findings:
        flag = "  FAIL" if over > FAIL_IN else ""
        print(f"  p.{n:<4d} {s:<6s} {over:+.3f} in ({over * 72:+.1f} pt){flag}")
    return max(f[0] for f in findings), findings


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pdfs = sys.argv[1:] or (sorted(glob.glob(os.path.join(here, "*.pdf")))
                            + [os.path.join(here, "..", "paper", "main.pdf")])
    pdfs = [p for p in pdfs if os.path.exists(p)]
    worst = 0.0
    for p in pdfs:
        w, _ = audit(p)
        worst = max(worst, w)
    print(f"\nworst excursion past a baseline: {worst:.3f} in ({worst * 72:.1f} pt); "
          f"fail threshold {FAIL_IN:.2f} in")
    return 1 if worst > FAIL_IN else 0


if __name__ == "__main__":
    sys.exit(main())
