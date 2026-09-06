# Build the paper.
#
#   make          build main.pdf
#   make watch    rebuild on save
#   make clean    remove build artefacts
#   make wc       rough word count of the prose
#   make todo     list every outstanding TODO
#
# Needs a TeX distribution with IEEEtran (TeX Live: texlive-publishers).
# On macOS: brew install --cask mactex-no-gui   (or use Overleaf instead --
# upload main.tex, refs.bib and the figures/ directory).

MAIN = main

.PHONY: all watch clean wc todo tectonic

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex refs.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode $(MAIN).tex

clean:
	latexmk -C
	rm -f *.bbl *.blg *.aux *.log *.out *.synctex.gz

# Rough prose count: strip comments and macros first.
wc:
	@sed -e 's/%.*//' $(MAIN).tex | detex 2>/dev/null | wc -w || \
	 echo "install detex for a word count"

todo:
	@grep -n 'todo{' $(MAIN).tex | sed 's/^\([0-9]*\).*todo{/\1: /' | cut -c1-100

# Alternative build without a TeX Live install: brew install tectonic
tectonic:
	tectonic -X compile $(MAIN).tex --keep-logs
