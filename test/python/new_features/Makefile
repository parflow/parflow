##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

SUBDIRS = clm

clean: testclean
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make --no-print-directory clean); \
	done

testclean:
	@rm -f *.pfb*
	@rm -f *.silo*
	@rm -f *.pfsb*
	@rm -f *.log
	@rm -f *.sa
	@rm -f *.out.timing.csv
	@rm -f .hostfile
	@rm -f .amps.*
	@rm -f *.out.pftcl
	@rm -f *.pfidb
	@rm -f *.out.txt
	@rm -fr test.log.*
	@rm -fr core.*
	@rm -f *.nc
	@rm -f *.pfmetadata
	@rm -f *.etrace
	@rm -f *.vtk
	@rm -f inline_input.txt drv_vegm.dat sabino_domain.pfsol
