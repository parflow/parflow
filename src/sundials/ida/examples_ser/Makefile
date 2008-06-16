# -----------------------------------------------------------------
# $Revision: 1.13 $
# $Date: 2006/03/15 20:04:00 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/ida/LICENSE.
# -----------------------------------------------------------------
# Makefile for IDA serial examples
#
# ida/examples_ser/Makefile.  Generated from Makefile.in by configure.
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
SUNDIALS_INCS    = -I$(SUNDIALS_INC_DIR) -I$(SUNDIALS_INC_DIR)/ida -I$(SUNDIALS_INC_DIR)/sundials

SUNDIALS_LIB_DIR = $(libdir)
SUNDIALS_LIBS    = -lsundials_ida -lsundials_nvecserial

EXEC_FILES = idadenx idabanx1 idakryx idabanx2 idakrydem_lin

all: $(EXEC_FILES)

idadenx: $(srcdir)/idadenx.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/idadenx.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/idadenx $(builddir)/idadenx.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
idabanx1: $(srcdir)/idabanx1.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/idabanx1.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/idabanx1 $(builddir)/idabanx1.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
idakryx: $(srcdir)/idakryx.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/idakryx.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/idakryx $(builddir)/idakryx.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
idabanx2: $(srcdir)/idabanx2.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/idabanx2.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/idabanx2 $(builddir)/idabanx2.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)
idakrydem_lin: $(srcdir)/idakrydem_lin.c
	$(COMPILER_PREFIX) $(CC) $(CPPFLAGS) $(SUNDIALS_INCS) $(CFLAGS) -c $(srcdir)/idakrydem_lin.c
	$(LINKER_PREFIX) $(CC) -o $(builddir)/idakrydem_lin $(builddir)/idakrydem_lin.$(OBJ_EXT) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) $(LIBS)

clean:
	rm -rf .libs
	rm -f *.lo
	rm -f *.o

uninstall: clean
	rm -f $(EXEC_FILES)
