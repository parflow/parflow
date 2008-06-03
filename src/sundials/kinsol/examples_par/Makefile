# -----------------------------------------------------------------
# $Revision: 1.14 $
# $Date: 2006/03/15 20:04:19 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/kinsol/LICENSE.
# -----------------------------------------------------------------
# Makefile for KINSOL parallel examples
#
# kinsol/examples_par/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh

srcdir       = .
top_builddir = ../../
builddir     = .
prefix       = /home/smithsg/parflow/pfclm_ms/src/sundials
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib



CPP         = /usr/bin/cc -E
CPPFLAGS    = 
CC          = /usr/bin/cc
CFLAGS      = -g -O2 -ffloat-store
LDFLAGS     = 
LIBS        = -lm 
MPICC       = /nfs/apps/mpich/1.2.7p1/bin/mpicc
MPI_INC_DIR = /nfs/apps/mpich/1.2.7p1/bin/../include
MPI_LIB_DIR = /nfs/apps/mpich/1.2.7p1/bin/../lib
MPI_LIBS    = 
MPI_FLAGS   = 

OBJ_EXT = o

COMPILER_PREFIX = 
LINKER_PREFIX   = 

SUNDIALS_INC_DIR = $(includedir)
SUNDIALS_INCS    = -I$(SUNDIALS_INC_DIR) -I$(SUNDIALS_INC_DIR)/kinsol -I$(SUNDIALS_INC_DIR)/sundials

SUNDIALS_LIB_DIR = $(libdir)
SUNDIALS_LIBS    = -lsundials_kinsol -lsundials_nvecparallel

EXEC_FILES = kinkryx_p kinkryx_bbd_p

all: $(EXEC_FILES)

kinkryx_p: $(srcdir)/kinkryx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/kinkryx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/kinkryx_p $(builddir)/kinkryx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
kinkryx_bbd_p: $(srcdir)/kinkryx_bbd_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/kinkryx_bbd_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/kinkryx_bbd_p $(builddir)/kinkryx_bbd_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)

clean:
	rm -rf .libs
	rm -f *.lo
	rm -f *.o

uninstall: clean
	rm -f $(EXEC_FILES)
