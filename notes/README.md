# notes/

Working notes kept in the repository because they are part of the provenance of the paper,
not because they are polished.

| file | what it is |
|---|---|
| `design.md` | the design decisions taken before any code was written, including the one that matters most: *serial dependence in the covariates alone does not change the long-run covariance* — the score has to be serially correlated — which is why every simulated design pairs dependent covariates with a dependent error or latent process. |
| `positioning.md` | the binding list of what this paper may and may not claim, compiled from a systematic read of the closest prior work. Each item names the paper and the specific theorem that owns the claim. The paper's "What we do not claim" paragraph and Section 6 are written against this list. |
| `review_response.md` | the adversarial review the manuscript and code were put through before finalising, and what each finding changed. Two of the findings changed reported numbers, and both had been biasing results in the paper's favour; those are listed first. |
| `review_findings.txt`, `review_raw.json` | the raw findings and per-finding verification verdicts. |
| `refs_raw.json` | **verification evidence for the bibliography.** One record per reference with the URL that was actually loaded to confirm it, its status (`verified` / `corrected`), and a note saying exactly which fields were corrected. Every entry in `paper/refs.bib` traces to a record here. |

`paper_draft.txt` (a plain-text rendering of the compiled PDF, used for proofreading) and the
base paper's own PDF are build/reference artefacts and are not tracked.
