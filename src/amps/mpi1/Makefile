#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************


include Makefile.$(PF_ARCH).mptopf

PF_LIBNAME = amps.mpi1.$(PF_ARCH)

all: $(PF_LIBNAME).a

PF_DEPEND = amps.h amps_proto.h

PF_SRC=\
	amps_init.c \
	amps_finalize.c \
	amps_print.c \
	amps_pack.c amps_unpack.c \
	amps_invoice.c \
	amps_send.c \
	amps_clear.c \
	amps_recv.c \
	amps_vector.c \
	amps_sizeofinvoice.c \
	amps_bcast.c \
	../common/amps_find_powers.c \
	amps_allreduce.c \
	amps_irecv.c \
	../common/amps_newhandle.c \
	../common/amps_wait.c \
	amps_test.c \
	amps_exchange.c \
	../common/amps_sfbcast.c \
	../common/amps_sfopen.c \
	../common/amps_sfclose.c \
	../common/amps_io.c \
	../common/amps_fopen.c \
	amps_createinvoice.c \
	../common/amps_clock.c \
	amps_newpackage.c \
	../common/amps_ffopen.c \
	../common/signal.c

include Makefile.$(PF_ARCH).pftomp
