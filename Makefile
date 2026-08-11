# Reproduce every number, figure and table in the paper.
#
#   make all            everything (a few CPU-hours; safe to interrupt and resume)
#   make test           validation of every closed-form population quantity
#   make paper          rebuild the preprint PDF from whatever results exist
#   make submission     rebuild the Information Sciences submission package
#   make clean-results  delete results (keeps the downloaded data)
#
# Set NJOBS to control parallelism (default 5) and R to shrink the Monte Carlo sizes,
# e.g.  make all NJOBS=8 R=100

PY      ?= python3
EXP     := experiments
RES     := results
NJOBS   ?= 5
export NJOBS

.PHONY: all data test exp figures tables paper submission clean-results clean

all: data test exp figures tables paper submission

data:
	$(PY) $(EXP)/fetch_data.py

test:
	$(PY) tests/test_paper_sources.py
	$(PY) tests/test_oracles.py

exp: $(RES)/exp0_critical_values.json $(RES)/exp1_main.json $(RES)/exp2_coverage_law.json \
     $(RES)/exp3_lrv.json $(RES)/exp4_gapping.json $(RES)/exp5_ablations.json \
     $(RES)/exp6_real.json $(RES)/exp7_cost.json $(RES)/exp8_lrv_rate.json \
     $(RES)/exp9_wellcond.json $(RES)/exp10_hard_design.json

$(RES)/exp%.json:
	$(PY) -u $(EXP)/$(notdir $(basename $@))*.py 2>&1 | tee $(RES)/$(notdir $(basename $@)).log

figures: tables
	$(PY) $(EXP)/make_figures.py

tables:
	$(PY) $(EXP)/make_tables.py

paper: tables
	cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; \
	  bibtex main >/dev/null 2>&1; \
	  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; \
	  pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1; \
	  echo "paper/main.pdf written"

# Information Sciences (Elsevier) submission: manuscript, supplementary material, highlights
# and cover letter.  Shares all prose, tables, figures and generated numbers with paper/.
submission: tables
	sh submission/build.sh
	$(PY) submission/check_highlights.py

clean-results:
	rm -f $(RES)/*.json $(RES)/*.log $(RES)/*.npz

clean: clean-results
	rm -f paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out paper/*.pdf
	rm -f submission/*.aux submission/*.bbl submission/*.blg submission/*.log \
	      submission/*.out submission/*.spl submission/*.pdf
