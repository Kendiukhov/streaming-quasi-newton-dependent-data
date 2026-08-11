"""Structural checks on the manuscript sources.

These exist because of an accident: a scripted edit once overwrote a section file with the
whole of ``main.tex``, and because the paper still compiled (the stray preamble commands
merely errored in ``nonstopmode``) the damage survived a commit. A three-line check catches
it, so here it is.

Run with:  python tests/test_paper_sources.py
"""
import os
import re
import sys

PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
SECTIONS = ["sec_experiments.tex", "sec_related.tex", "sec_limits.tex",
            "sec_appendix.tex", "sec_appendix_exp.tex"]


def test_only_main_is_a_document():
    """No included section may carry a preamble: that means it was overwritten."""
    for f in SECTIONS:
        txt = open(os.path.join(PAPER, f)).read()
        for bad in (r"\documentclass", r"\begin{document}", r"\usepackage"):
            assert bad not in txt, f"{f} contains {bad}: it has been overwritten"
    print("section files carry no preamble: OK")


def test_sections_are_nonempty():
    for f in SECTIONS:
        n = len(open(os.path.join(PAPER, f)).read())
        assert n > 1500, f"{f} is only {n} characters long"
    print("section files are all non-trivial: OK")


def test_every_data_macro_is_defined():
    """Every \\CovXxx-style macro used in the prose must be defined in numbers.tex.

    make_tables.py emits a visible ``[TBD]`` placeholder for any it cannot fill, so an
    undefined one means the emission logic missed it and the paper would fail to compile.
    """
    nums = os.path.join(PAPER, "numbers.tex")
    if not os.path.exists(nums):
        print("numbers.tex absent (run `make tables` first): skipping")
        return
    defined = set(re.findall(r"(?:newcommand|providecommand)\{\\(\w+)\}",
                             open(nums).read()))
    pat = re.compile(r"^(Cov|Kappa|Width|Time|Rmse|Real|Cost|Flop|Ks|Lrv|Se|Rate|Asym|Wc)"
                     r"[A-Z]")
    used = set()
    for f in SECTIONS + ["main.tex"]:
        used |= set(re.findall(r"\\([A-Za-z]+)", open(os.path.join(PAPER, f)).read()))
    missing = sorted(u for u in used if pat.match(u) and u not in defined)
    assert not missing, f"data macros used but not defined: {missing}"
    print(f"all {len([u for u in used if pat.match(u)])} data macros defined: OK")


def test_bib_keys_resolve():
    """Every \\cite key must exist in refs.bib."""
    keys = set(re.findall(r"^@\w+\{([^,]+),",
                          open(os.path.join(PAPER, "refs.bib")).read(), re.M))
    used = set()
    for f in SECTIONS + ["main.tex"]:
        txt = open(os.path.join(PAPER, f)).read()
        for grp in re.findall(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\{([^}]+)\}", txt):
            used |= {k.strip() for k in grp.split(",")}
    missing = sorted(used - keys)
    assert not missing, f"cited but absent from refs.bib: {missing}"
    print(f"all {len(used)} cited keys resolve in refs.bib ({len(keys)} entries): OK")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print("\npaper source checks passed")
