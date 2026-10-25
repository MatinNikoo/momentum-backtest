# make all       rerun everything from the committed snapshots (tests, notebook, study, paper)
#                the paper needs a LaTeX install with latexmk and pdflatex
# make tearsheet  the example report in reports/example, pdf and html
# make data      download fresh data and rebuild data/snapshots (needs internet, numbers will move a bit)
PY = python3

.PHONY: all test study notebook paper data tearsheet clean

# the notebook writes results/spy_study.json, which the study's paper numbers read, so it goes first
all: test notebook study paper

test:
	$(PY) -m pytest

study:
	nice -n 10 $(PY) run_study.py

notebook:
	nice -n 10 jupyter nbconvert --to notebook --execute --inplace backtest.ipynb

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

tearsheet:
	$(PY) -m trendlab tearsheet configs/us_century_50_200.toml --outdir reports/example --format both

data:
	$(PY) data_sources.py

clean:
	rm -rf results/study.json results/tables figures/*.png paper/tables
	cd paper && latexmk -C
