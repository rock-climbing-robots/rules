FIGS := $(patsubst %.tex,%.pdf,$(wildcard figures/*.tex))

all: rulebook.pdf

rulebook.pdf: rulebook.tex $(FIGS)
	pdflatex -interaction=nonstopmode rulebook.tex
	pdflatex -interaction=nonstopmode rulebook.tex

figures/%.pdf: figures/%.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figures $<

# standalone SVG / PNG exports of the figures
svg: $(FIGS)
	for f in $(FIGS); do inkscape --export-type=svg --export-plain-svg \
	    -o $${f%.pdf}.svg $$f; done

png: $(FIGS)
	for f in $(FIGS); do convert -density 300 $$f -background white \
	    -alpha remove $${f%.pdf}.png; done

clean:
	rm -f *.aux *.log *.out *.toc figures/*.aux figures/*.log

mrproper: clean
	rm -f rulebook.pdf figures/*.pdf figures/*.svg figures/*.png

.PHONY: all svg png clean mrproper
