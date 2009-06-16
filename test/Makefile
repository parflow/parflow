##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

SUBDIRS = clm

default: check

TESTS := \
	test_X.tcl \
	default_single.tcl \
	default_richards.tcl \
	forsyth2.tcl \
	harvey.flow.tcl \
	harvey_flow_pgs.tcl \
	default_overland.tcl \
	crater2D.tcl \
	small_domain.tcl

ifeq (${PARFLOW_HAVE_HYPRE},yes)
TESTS += \
	pfmg.tcl \
	smg.tcl \
	pfmg_octree.tcl
endif

ifeq (${PARFLOW_HAVE_SILO},yes)
TESTS += \
	default_richards_with_silo.tcl \
	indicator_field.tcl

ifeq (${PARFLOW_HAVE_SILO},yes)
TESTS += \
	water_balance_y.tcl \
	water_balance_x.tcl 
endif
endif

check:
	@-for i in $(TESTS); \
	do \
	        make --no-print-directory testclean; \
		echo "*****************************************************************************"; \
		echo "Running $$i"; \
		tclsh $$i; \
	done
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make); \
	done 

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
	@rm -f .hostfile
	@rm -f .amps.*
	@rm -f *.out.pftcl
	@rm -f *.pfidb
	@rm -f *.out.txt
	@rm -fr default_richards.out 
	@rm -fr indicator_field.out
	@rm -fr water_balance.out
	@rm -fr default_overland.out

