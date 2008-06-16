# -----------------------------------------------------------------
# $Revision: 1.14 $
# $Date: 2006/03/15 20:03:32 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/cvode/LICENSE.
# -----------------------------------------------------------------
# Makefile for CVODE parallel examples
#
# cvode/examples_par/Makefile.  Generated from Makefile.in by configure.
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
SUNDIALS_INCS    = -I$(SUNDIALS_INC_DIR) -I$(SUNDIALS_INC_DIR)/cvode -I$(SUNDIALS_INC_DIR)/sundials

SUNDIALS_LIB_DIR = $(libdir)
SUNDIALS_LIBS    = -lsundials_cvode -lsundials_nvecparallel

EXEC_FILES = cvnonx_p cvkryx_p cvkryx_bbd_p

all: $(EXEC_FILES)

cvnonx_p: $(srcdir)/cvnonx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvnonx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvnonx_p $(builddir)/cvnonx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvkryx_p: $(srcdir)/cvkryx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvkryx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvkryx_p $(builddir)/cvkryx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvkryx_bbd_p: $(srcdir)/cvkryx_bbd_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvkryx_bbd_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvkryx_bbd_p $(builddir)/cvkryx_bbd_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)

clean:
	rm -rf .libs
	rm -f *.lo
	rm -f *.o

uninstall: clean
	rm -f $(EXEC_FILES)
