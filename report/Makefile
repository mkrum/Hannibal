
report.pdf: report.tex report.bib
	rm report.aux  report.bbl  report.blg  report.pdf 
	pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex

show: report.pdf
	zathura report.pdf

clean: 
	rm report.aux  report.bbl  report.blg  report.pdf 
