#!/bin/sh
# Build the Information Sciences submission PDFs.
#
# The manuscript shares its prose, tables, figures and generated numbers with the preprint in
# ../paper.  TEXINPUTS/BIBINPUTS put that directory on TeX's search path so the shared fragments
# resolve by bare name from here, including the nested \input inside sec_appendix.tex.  Nothing is
# copied, so the two documents cannot drift apart.
#
# Usage:  cd submission && sh build.sh
set -e
cd "$(dirname "$0")"

export TEXINPUTS="../paper:${TEXINPUTS}"
export BIBINPUTS="../paper:${BIBINPUTS}"

for doc in manuscript supplementary highlights cover_letter; do
    [ -f "$doc.tex" ] || continue
    printf '\n=== %s ===\n' "$doc"
    pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
    # the \bibliography command may live in an \input'd file, so test the .aux instead
    if grep -q '\\bibdata' "$doc.aux" 2>/dev/null; then
        bibtex "$doc" >/dev/null 2>&1 || true
    fi
    pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
    python3 - "$doc" <<'PY'
import sys, re, os
doc = sys.argv[1]
log = open(doc + ".log", errors="replace").read()
errs = [l for l in log.split("\n") if l.startswith("!")]
und = len(re.findall(r"(?i)undefined (?:control sequence|reference|citation)", log))
print(f"  {doc}.pdf: errors={len(errs)} undefined={und}", end="")
if os.path.exists(doc + ".pdf"):
    import subprocess
    p = subprocess.run(["pdfinfo", doc + ".pdf"], capture_output=True, text=True)
    pages = [l.split()[-1] for l in p.stdout.split("\n") if l.startswith("Pages:")]
    print(f" pages={pages[0] if pages else '?'}")
else:
    print("  [NO PDF]")
for e in errs[:5]:
    print("   ", e[:120])
PY
done
