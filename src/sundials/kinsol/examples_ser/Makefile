# -----------------------------------------------------------------
# $Revision: 1.16 $
# $Date: 2006/03/15 20:04:21 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/kinsol/LICENSE.
# -----------------------------------------------------------------
# Makefile for KINSOL serial examples
#
# kinsol/examples_ser/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh

srcdir       = .
top_builddir = ../../
builddir     = .
prefix       = /home/smithsg/parflow/pfclm_ms/src/sundials
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib



CPP      = /usr/bin/cc -E
CPPFLAGS = 
CC       = /usr/bin/cc
CFLAGS   = -g -O2 -ffloat-store
LDFLAGS  = 
LIBS     = -lm 

OBJ_EXT = o

COMPILER_PREFIX = 
LINKER_PREFIX   = 

SUNDIALS_INC_DIR = $(includedir)
SUNDIALS_INCS    = -I$(SUNDIALS_INC_DIR) -I$(SUNDIALS_INC_DIR)/kinsol -I$(SUNDIALS_INC_DIR)/sundials

SUNDIALS_LIB_DIR = $(libdir)
SUNDIALS_LIBS    = -lsundials_kinsol -lsundials_nvecserial

EXEC_FILES = kinkryx kindenx1 kindenx2 kinbanx kinkrydem_lin

all: $(EXEC_FILES)

kinkryx: $(srcdir)/kinkryx.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/kinkryx.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/kinkryx $(builddir)/kinkryx.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
kindenx1: $(srcdir)/kindenx1.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/kindenx1.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/kindenx1 $(builddir)/kindenx1.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
kindenx2: $(srcdir)/kindenx2.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/kindenx2.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/kindenx2 $(builddir)/kindenx2.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
kinbanx: $(srcdir)/kinbanx.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/kinbanx.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/kinbanx $(builddir)/kinbanx.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
kinkrydem_lin: $(srcdir)/kinkrydem_lin.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/kinkrydem_lin.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/kinkrydem_lin $(builddir)/kinkrydem_lin.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)

clean:
	rm -rf .libs
	rm -f *.lo
	rm -f *.o

uninstall: clean
	rm -f $(EXEC_FILES)
