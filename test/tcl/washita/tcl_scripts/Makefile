##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

SUBDIRS =

default: check

test: check

check:
	@echo "Tests should be run with CTest.  In CMake build directory run 'make test'"

clean: testclean
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make --no-print-directory clean); \
	done

testclean:
	@rm -fr Outputs

