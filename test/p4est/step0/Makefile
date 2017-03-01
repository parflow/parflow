##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

default: check_p4est

TESTS := \

ifeq (${PARFLOW_HAVE_P4EST},yes)
TESTS += step0_tests_2d.tcl step0_tests_3d.tcl
endif

check_p4est:
	@-for t in $(TESTS); \
	do \
		echo "*****************************************************************************"; \
		(tclsh $$t); \
	done

clean: testclean

testclean:
	@-rm -rf output_*
