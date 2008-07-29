##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

default: all

TESTS := \
	test_X.tcl \
	default_single.tcl \
	default_richards.tcl \
	forsyth2.tcl \
	harvey.flow.tcl \
	default_overland.tcl \
	crater2D.tcl



ifeq (${PARFLOW_HAVE_CLM},yes)
TESTS += clm.tcl
endif

check:
	@-for i in $(TESTS); \
	do \
		tclsh $$i; \
	done
clean:
	@rm -f *.pfb*
	@rm -f *.pfsb*
	@rm -f *.log
	@rm -f .hostfile
	@rm -f .amps.*
	@rm -f *.out.pftcl
	@rm -f *.pfidb
	@rm -f *.out.txt

