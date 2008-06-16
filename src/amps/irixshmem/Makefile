#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************
include Makefile.$(PF_ARCH).mptopf

PF_LIBNAME = amps.irix

all: $(PF_LIBNAME).a

PF_DEPEND = amps.h amps_proto.h

PF_SRC = \
	../common/amps_io.c \
	../common/amps_clear.c \
	../common/amps_sfbcast.c \
	../common/amps_sfopen.c \
	../common/amps_sfclose.c \
	../common/amps_find_powers.c \
	../common/amps_fopen.c \
	../common/amps_newhandle.c \
	../common/amps_wait.c \
	../common/amps_invoice.c \
	../common/amps_clock.c \
	../common/signal.c \
	../common/amps_ffopen.c \
	amps_allreduce.c \
	amps_bcast.c \
	amps_buffer.c \
	amps_createinvoice.c \
	amps_exchange.c \
	amps_finalize.c \
	amps_init.c \
	amps_irecv.c \
	amps_newpackage.c \
	amps_pack.c \
	amps_print.c \
	amps_recv.c \
	amps_send.c \
	amps_sizeofinvoice.c \
	amps_test.c \
	amps_unpack.c \
	amps_vector.c


include Makefile.$(PF_ARCH).pftomp
