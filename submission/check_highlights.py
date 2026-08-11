"""Check the Highlights against Information Sciences' limits: 3-5 bullets, <= 85 characters each.

The character limit includes spaces, so it is easy to violate by a word or two while editing.
Run this after any change to highlights.tex or highlights.txt; both must agree.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIMIT = 85
MIN_BULLETS, MAX_BULLETS = 3, 5


def from_tex(path):
    s = open(path).read()
    return [re.sub(r"\s+", " ", m.group(1)).strip()
            for m in re.finditer(r"\\item\s+(.*?)(?=\n\s*(?:\\item|\\end\{itemize\}))", s, re.S)]


def from_txt(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
    return out


def main():
    tex = from_tex(os.path.join(HERE, "highlights.tex"))
    txt = from_txt(os.path.join(HERE, "highlights.txt"))
    ok = True

    if tex != txt:
        print("MISMATCH between highlights.tex and highlights.txt:")
        for a, b in zip(tex + [""] * len(txt), txt + [""] * len(tex)):
            if a != b:
                print(f"  tex: {a!r}\n  txt: {b!r}")
        ok = False

    for i, h in enumerate(tex, 1):
        n = len(h)
        flag = "" if n <= LIMIT else f"  <-- OVER by {n - LIMIT}"
        print(f"  {i}. [{n:>2d}/{LIMIT}] {h}{flag}")
        if n > LIMIT:
            ok = False

    if not MIN_BULLETS <= len(tex) <= MAX_BULLETS:
        print(f"  bullet count {len(tex)} outside [{MIN_BULLETS}, {MAX_BULLETS}]")
        ok = False

    print("highlights OK" if ok else "highlights FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
