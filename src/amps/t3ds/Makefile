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

PF_LIBNAME = amps.$(PF_ARCH)

PF_DEPEND = amps.h amps_proto.h

PF_SRC =\
	../common/amps_io.c \
	../common/amps_sfbcast.c \
	../common/amps_sfopen.c \
	../common/amps_sfclose.c \
	../common/amps_fopen.c \
	../common/amps_find_powers.c \
	../common/amps_newhandle.c \
	../common/amps_wait.c \
	../common/amps_invoice.c \
	../common/amps_clock.c \
	../common/amps_ffopen.c \
	amps_allreduce.c \
	amps_bcast.c \
	amps_clear.c \
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

#
# Make seems to be busted
#

NODE_OBJ = \
	../common/amps_io.o \
	../common/amps_sfbcast.o \
	../common/amps_sfopen.o \
	../common/amps_sfclose.o \
	../common/amps_fopen.o \
	../common/amps_find_powers.o \
	../common/amps_newhandle.o \
	../common/amps_wait.o \
	../common/amps_invoice.o \
	../common/amps_clock.o \
	../common/amps_ffopen.o \
	amps_allreduce.o \
	amps_bcast.o \
	amps_clear.o \
	amps_createinvoice.o \
	amps_exchange.o \
	amps_finalize.o \
	amps_init.o \
	amps_irecv.o \
	amps_newpackage.o \
	amps_pack.o \
	amps_print.o \
	amps_recv.o \
	amps_send.o \
	amps_sizeofinvoice.o \
	amps_test.o \
	amps_unpack.o \
	amps_vector.o


include Makefile.$(PF_ARCH).pftomp

