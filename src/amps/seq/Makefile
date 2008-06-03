#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************
# No Host in the model

include Makefile.$(PF_ARCH).mptopf

PF_LIBNAME = amps.seq

PF_DEPEND = amps.h amps_proto.h

PF_SRC =\
	../common/amps_sfbcast.c\
	../common/amps_invoice.c\
	../common/amps_newpackage.c\
	../common/amps_ffopen.c\
	../common/amps_io.c\
	../common/amps_clock.c\
	../common/amps_clear.c

include Makefile.$(PF_ARCH).pftomp

