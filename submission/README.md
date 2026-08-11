# Information Sciences submission package

Everything needed to submit *Inference, Not Just Optimisation: Streaming Inversion-Free
Quasi-Newton Estimation on Dependent Data Streams* to Elsevier's *Information Sciences*
(ISSN 0020-0255), via Editorial Manager at <https://www.editorialmanager.com/ins>.

Build every PDF with:

```bash
cd submission && sh build.sh
```

## Files

| File | Purpose | Upload as |
|---|---|---|
| `manuscript.tex`, `body.tex` | The manuscript. `elsarticle`, single column, line-numbered. | Manuscript (source) |
| `manuscript.pdf` | Compiled manuscript, 71 pages. | Manuscript (PDF, for review) |
| `supplementary.tex` / `.pdf` | Four figures held out of the main text (see below). | Supplementary material |
| `highlights.tex` / `.txt` / `.pdf` | 5 bullets, each ≤ 85 characters. | Highlights (file name contains "highlights") |
| `cover_letter.tex` / `.pdf` | Cover letter to the Editors-in-Chief. | Cover letter |
| `declaration_of_competing_interest.txt` | Text for Elsevier's declarations tool. | See the note inside: must be the tool's `.docx` |
| `credit_statement.txt` | CRediT roles. Also printed in the manuscript. | Enter in the system |
| `data_availability.txt` | Data statement (journal applies Option C). | Enter in the system |
| `author_biography.txt` | Vitae **template — must be completed by the author**. | Editable file + photograph |
| `suggested_reviewers.md` | Candidate reviewers, with affiliations deliberately omitted. | Enter in the system |
| `check_highlights.py` | Verifies the 3–5 bullets / 85-character limits. | — |
| `build.sh` | Builds all four PDFs. | — |

## How this stays consistent with the preprint

The manuscript shares its prose, tables, figure captions, abstract and generated numbers with the
preprint in `../paper` — nothing is copied. `build.sh` puts `../paper` on `TEXINPUTS`, so
`\input{sec_theory}` and friends resolve to the one and only copy. What lives in this directory is
only what is specific to this venue: the title page, keywords, float selection and ordering, the
declarations the journal requires, and the appendix ordering.

Four figures appear in `supplementary.pdf` rather than in the manuscript, because Information
Sciences suggests at most ten figures plus tables for a theoretical article and the main text uses
exactly ten (4 figures + 6 tables). The shared prose refers to them through the macros
`\RefLrvRate`, `\RefGapFig`, `\RefCostFig` and `\RefAblFig`, which expand to `Supplementary Fig.
S1`–`S4` here and to ordinary cross-references in the preprint.

## Compliance with the guide for authors

Checked against the Guide for Authors on 11 August 2026.

- **Abstract** — 200 words, the journal's stated maximum for this journal. Self-contained, no
  citations, no non-standard abbreviations.
- **Keywords** — 6, within the 1–7 range, none joined by "and" or "of".
- **Highlights** — 5 bullets, longest is 80 characters. Verified by `check_highlights.py`.
- **Sections** — numbered 1, 1.1, 1.1.1; cross-references are by number, never "the text"; the
  abstract is outside the numbering. The required **Conclusions** section is Section 7 and is
  written to translate the results into terms accessible to a non-specialist, as the journal asks
  for theoretical papers.
- **Appendices** — lettered A, B, …; equations and floats inside them numbered `(A.1)`, `Table A.1`
  in the journal's format (handled by `elsarticle`'s `\appendix`).
- **Length** — the main narrative is ≈ 42 double-spaced pages, inside the 45-page guideline for a
  theoretical manuscript; the proofs are in appendices.
- **Figures and tables** — exactly 10 in the main text. All are cited in the text. Figures are
  vector PDFs in `../figures`.
- **Declarations** — competing interest, funding, CRediT, data availability, and the declaration on
  generative AI use all appear as unnumbered sections after the Conclusions, and are also provided
  as separate files here.
- **Research data (Option C)** — satisfied: the code and results are archived on Zenodo with a DOI
  ([10.5281/zenodo.21891313](https://doi.org/10.5281/zenodo.21891313)), cited in the reference list
  as a software reference with creator, title, venue, date, version and identifier as Elsevier's
  software-citation guidance requires, and linked from the Data availability section. The two
  third-party data sets are cited with their own UCI DOIs.
- **References** — the journal does not impose a style at submission and requires only internal
  consistency; `elsarticle-harv` (author–year) is used, with DOIs, preprints marked as preprints,
  and software cited with its repository. All 99 cited works were verified against their published
  sources; the evidence is in `../notes/refs_raw.json`.
- **Peer review** — single anonymized: reviewers are anonymous, authors are not, so the manuscript
  is *not* anonymised and the public code repository is linked in the text.

## Things the author must decide or complete before submitting

1. **Read the generative-AI declaration** in the manuscript (unnumbered section after the
   Conclusions) and in the cover letter. It states that a large language model was used
   substantially — under the author's direction and review — to implement the software, run the
   experiments and draft the manuscript. This is a material disclosure; the wording is the author's
   to confirm or change, and the journal requires that it be accurate.
2. **Complete `author_biography.txt`** and attach a photograph. It is a template on purpose.
3. **Fix the creator field on the Zenodo record.** It currently reads `Kendiukhov` with no given
   name, so Zenodo's own generated citation renders as a surname alone. Edit it to
   `Kendiukhov, Ihor`, add an ORCID iD and the University of Tübingen affiliation, and publish the
   metadata change — the DOI is unaffected by metadata edits. The manuscript's reference list
   already spells the name in full. *(The archive itself is done:
   [10.5281/zenodo.21891313](https://doi.org/10.5281/zenodo.21891313), version `release1`,
   MIT, 11 August 2026; concept DOI
   [10.5281/zenodo.21891312](https://doi.org/10.5281/zenodo.21891312).)*
4. **Verify the suggested reviewers'** current affiliations, and remove any with a conflict.
5. **Produce the competing-interest declaration** with Elsevier's declarations tool as a `.docx`;
   the `.txt` here is reference text, not an acceptable upload.
6. Decide whether to post the preprint on SSRN during submission (offered free, no effect on the
   editorial process).
