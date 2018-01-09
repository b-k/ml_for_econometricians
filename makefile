talk.pdf: talk.tex
	xelatex talk.tex

clean:
	@rm talk.{aux,log,nav,out,snm,toc}
	-@rm -r modelfits/__pycache__
	@rm -r modelfits/indata
	@rm -r modelfits/out
	@rm -r modelfits/remit.db
