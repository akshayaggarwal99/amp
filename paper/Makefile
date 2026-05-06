.PHONY: all clean watch

all: main.pdf

main.pdf: main.tex sections/*.tex references.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main || true
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc main.pdf

watch:
	while true; do make; sleep 5; done
