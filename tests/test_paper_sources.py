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
# Every shared fragment: section bodies, the abstract and the figure blocks.  These are
# \input by BOTH paper/main.tex (preprint) and submission/manuscript.tex (Information
# Sciences), so a stray preamble or a macro used but never emitted breaks two documents.
# Each long section is split into a core fragment, which the body includes, and a "_detail"
# fragment, which the appendix includes; that is what keeps the main text inside its page
# budget without dropping anything.  Both halves are checked here.
SECTIONS = ["sec_intro.tex", "sec_setting.tex", "sec_theory.tex", "sec_lrv.tex",
            "sec_experiments.tex", "sec_related.tex", "sec_limits.tex",
            "sec_conclusions.tex", "sec_appendix.tex", "sec_appendix_exp.tex",
            "sec_setting_detail.tex", "sec_theory_detail.tex",
            "sec_experiments_detail.tex", "sec_related_detail.tex",
            "sec_limits_detail.tex", "sec_data.tex",
            "abstract.tex", "fig_coverage.tex", "fig_lrv.tex", "fig_lrvrate.tex",
            "fig_gap_cost.tex", "fig_ablation.tex", "fig_real.tex", "fig_hard.tex"]


def test_only_main_is_a_document():
    """No included section may carry a preamble: that means it was overwritten."""
    for f in SECTIONS:
        txt = open(os.path.join(PAPER, f)).read()
        for bad in (r"\documentclass", r"\begin{document}", r"\usepackage"):
            assert bad not in txt, f"{f} contains {bad}: it has been overwritten"
    print("section files carry no preamble: OK")


def test_sections_are_nonempty():
    """Guard against the failure mode that motivated this test: a scripted edit truncating or
    overwriting a shared fragment, which pdflatex in nonstopmode will happily ignore.

    The floor is per-file because the fragments are legitimately of very different sizes: a
    figure block is a caption, the abstract is capped at 200 words by the journal, and the
    section bodies are thousands of characters.
    """
    FLOOR = {"abstract.tex": 900, "fig_coverage.tex": 600, "fig_lrv.tex": 600,
             "fig_lrvrate.tex": 600, "fig_gap_cost.tex": 900, "fig_ablation.tex": 500,
             "fig_real.tex": 600, "fig_hard.tex": 600,
             # the core fragments were cut to fit a 15-page main text; these floors are set
             # just under their current sizes, so a truncating edit still trips the test
             "sec_conclusions.tex": 2000, "sec_limits.tex": 3000, "sec_related.tex": 3500,
             "sec_data.tex": 1100}
    for f in SECTIONS:
        n = len(open(os.path.join(PAPER, f)).read())
        floor = FLOOR.get(f, 1500)
        assert n > floor, f"{f} is only {n} characters long (floor {floor})"
    print(f"all {len(SECTIONS)} shared fragments are non-trivial: OK")


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
    # keep this prefix list in sync with emit_macros() in experiments/make_tables.py: the two
    # must agree, or a macro can be used in the text and silently never emitted
    pat = re.compile(r"^(Cov|Kappa|Width|Time|Rmse|Real|Cost|Flop|Ks|Lrv|Se|Rate|Asym|Wc|"
                     r"Stream|Hard|Shift|Gap|Adeq|Psd|Proto|Cond|Warm|Strong)[A-Z]")
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
