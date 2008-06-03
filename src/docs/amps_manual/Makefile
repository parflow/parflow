MANUAL=amps_manual
INSTALL_DIR = $(PARFLOW_HELP)/amps_manual

ONLINE_OPTS =\
 -show_section_numbers\
 -bottom_navigation

FILES =\
 manual.tex


all: manual.ps manual/manual.html
	
manual.ps: manual.dvi
	@dvips -o manual.ps manual

manual/manual.html: manual.dvi
	@latex2html $(ONLINE_OPTS) manual.tex
	@./do_online manual

#	bibtex manual
manual.dvi: Makefile $(FILES)
	@latex manual
	@bibtex manual
	@makeindex manual.idx
	@latex manual
	@latex manual
	
install:
	@/bin/rm -fr $(INSTALL_DIR)/$(MANUAL)_html
	@mkdir $(INSTALL_DIR)/$(MANUAL)_html
	@chgrp parflow $(INSTALL_DIR)/$(MANUAL)_html
	@chmod 775  $(INSTALL_DIR)/$(MANUAL)_html
	@cp manual/* $(INSTALL_DIR)/$(MANUAL)_html
	@chgrp parflow $(INSTALL_DIR)/$(MANUAL)_html/*
	@chmod 664 $(INSTALL_DIR)/$(MANUAL)_html/*
	@/bin/rm -fr $(INSTALL_DIR)/$(MANUAL).ps
	@cp manual.ps $(INSTALL_DIR)/$(MANUAL).ps
	@chgrp parflow $(INSTALL_DIR)/$(MANUAL).ps
	@chmod 664 $(INSTALL_DIR)/$(MANUAL).ps
	
clean:
	@/bin/rm -f *.aux *.bbl *.blg *.dvi *.log *.dlog *.toc
	@/bin/rm -f *.idx *.ilg *.ind

veryclean: clean
	@/bin/rm -f  manual.ps
	@/bin/rm -fr manual

