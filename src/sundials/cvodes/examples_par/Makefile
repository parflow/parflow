# -----------------------------------------------------------------
# $Revision: 1.17 $
# $Date: 2006/03/15 20:03:46 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/cvodes/LICENSE.
# -----------------------------------------------------------------
# Makefile for CVODES parallel examples
#
# cvodes/examples_par/Makefile.  Generated from Makefile.in by configure.
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
SUNDIALS_INCS    = -I$(SUNDIALS_INC_DIR) -I$(SUNDIALS_INC_DIR)/cvodes -I$(SUNDIALS_INC_DIR)/sundials

SUNDIALS_LIB_DIR = $(libdir)
SUNDIALS_LIBS    = -lsundials_cvodes -lsundials_nvecparallel

EXEC_FILES = cvsnonx_p cvskryx_p cvskryx_bbd_p cvsfwdnonx_p cvsfwdkryx_p cvsadjnonx_p cvsadjkryx_p cvsadjkryx_p3D

all: $(EXEC_FILES)

cvsnonx_p: $(srcdir)/cvsnonx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvsnonx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvsnonx_p $(builddir)/cvsnonx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvskryx_p: $(srcdir)/cvskryx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvskryx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvskryx_p $(builddir)/cvskryx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvskryx_bbd_p: $(srcdir)/cvskryx_bbd_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvskryx_bbd_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvskryx_bbd_p $(builddir)/cvskryx_bbd_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvsfwdnonx_p: $(srcdir)/cvsfwdnonx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvsfwdnonx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvsfwdnonx_p $(builddir)/cvsfwdnonx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvsfwdkryx_p: $(srcdir)/cvsfwdkryx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvsfwdkryx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvsfwdkryx_p $(builddir)/cvsfwdkryx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvsadjnonx_p: $(srcdir)/cvsadjnonx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvsadjnonx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvsadjnonx_p $(builddir)/cvsadjnonx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvsadjkryx_p: $(srcdir)/cvsadjkryx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/cvsadjkryx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvsadjkryx_p $(builddir)/cvsadjkryx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)
cvsadjkryx_p3D: $(srcdir)/cvsadjkryx_p.c
	$(COMPILER_PREFIX) $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) $(SUNDIALS_INCS) -I$(MPI_INC_DIR) $(CFLAGS) -DUSE3D -c $(srcdir)/cvsadjkryx_p.c
	$(LINKER_PREFIX) $(MPICC) -o $(builddir)/cvsadjkryx_p3D $(builddir)/cvsadjkryx_p.$(OBJ_EXT) $(MPI_FLAGS) $(CFLAGS) $(LDFLAGS) -L$(SUNDIALS_LIB_DIR) $(SUNDIALS_LIBS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS)

clean:
	rm -rf .libs
	rm -f *.lo
	rm -f *.o

uninstall: clean
	rm -f $(EXEC_FILES)
