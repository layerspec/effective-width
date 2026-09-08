# One-command reproduction of every table, figure and the PDF from results/.
#   make reproduce        analysis text, LaTeX tables, figures, main.pdf
#   make analysis         results/checkpoint_analysis.txt + paper/table_*.tex
#   make figures          results/figures/*.pdf
#   make pdf              paper/main.pdf (tectonic)
# Needs python3 with numpy, pandas, scipy, matplotlib (no torch), and tectonic.
PY ?= python3
R  := results
P  := paper

.PHONY: reproduce analysis figures pdf

reproduce: analysis figures pdf

analysis:
	cd code && $(PY) scripts/checkpoint_analysis.py --results ../$(R) \
	  --latex-models ../$(P)/table_models.tex --latex-pooling ../$(P)/table_pooling.tex \
	  --latex-families ../$(P)/table_families.tex --latex-projection ../$(P)/table_projection.tex \
	  --latex-decompose ../$(P)/table_decompose.tex --latex-ablation ../$(P)/table_ablation.tex > ../$(R)/checkpoint_analysis.txt

figures:
	cd code && $(PY) -m layerspec.figures --results ../$(R) --out ../$(R)/figures > /dev/null
	cd code && $(PY) scripts/paper_figures.py --results ../$(R) --out ../$(R)/figures > /dev/null

pdf:
	cd $(P) && tectonic -X compile main.tex
