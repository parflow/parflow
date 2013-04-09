##
## Copyright:	(c) 1997-2008 Lawrence Livermore National Security, LLC
## Revision:	$LastChangedRevision: 2026 $
## Modified:	$LastChangedDate: 2008-02-28 16:23:19 -0800 (Thu, 28 Feb 2008) $

PARFLOW	      =	../..
SRCDIR	      =	.
SUBDIR        = amps/$(AMPS)

OBJECT        = ../..

default: all

include $(OBJECT)/config/Makefile.config

OBJS = \
	oas_pfl_vardef.o \
        oas_pfl_init.o \
        oas_pfl_finalize.o \
	oas_pfl_define.o \
	oas_pfl_snd.o \
	oas_pfl_rcv.o \
	send_fld2_clm.o \
	receive_fld2_clm.o \
	amps_allreduce.o \
	amps_bcast.o \
	amps_clear.o \
	amps_createinvoice.o \
	amps_exchange.o \
	amps_finalize.o \
	amps_init.o \
	amps_invoice.o \
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

$(PARFLOW_LIB_DIR)/$(AMPS_LIB): $(OBJS)
	ar rus $(PARFLOW_LIB_DIR)/$(AMPS_LIB) $(OBJS)

all: $(PARFLOW_LIB_DIR)/$(AMPS_LIB)

clean:
	@$(PARFLOW_CLEAN)

veryclean: clean
	@$(RM) Makefile

include $(SRCDIR)/Makefile.depend


