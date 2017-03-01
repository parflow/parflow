##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2219 $
## Modified:	$LastChangedDate: 2008-06-12 13:14:48 -0700 (Thu, 12 Jun 2008) $
##
## Description:	makefile for the parflow library and executable
##

include $(PARFLOW_DIR)/config/Makefile.config

default: check_p4est

SUBDIRS := \

ifeq (${PARFLOW_HAVE_P4EST},yes)
SUBDIRS += step0 step1
endif

check_p4est:
	@-for i in $(SUBDIRS); \
	do \
	        (cd $$i; make --no-print-directory check_p4est); \
	done

clean: testclean

testclean:
	@-for i in $(SUBDIRS); \
	do \
		(cd $$i; make --no-print-directory testclean); \
	done
