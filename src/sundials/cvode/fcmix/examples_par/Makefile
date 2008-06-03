# -----------------------------------------------------------------
# $Revision: 1.17 $
# $Date: 2006/03/15 20:03:39 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/cvode/LICENSE.
# -----------------------------------------------------------------
# Makefile for FCVODE parallel examples
#
# cvode/fcmix/examples_par/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh

srcdir       = .
top_builddir = ../../../
builddir     = .
prefix       = /home/smithsg/parflow/pfclm_ms/src/sundials
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib

INSTALL = /usr/bin/install -c



FFLAGS      = -g -O2 -ffloat-store
LDFLAGS     = 
LIBS        = -lm 
FLIBS       =  -L/usr/lib/gcc/i386-redhat-linux/3.4.6 -L/usr/lib/gcc/i386-redhat-linux/3.4.6/../../.. -lfrtbegin -lg2c -lm -lgcc_s
MPIF77      = /nfs/apps/mpich/1.2.7p1/bin/mpif77
MPICC       = /nfs/apps/mpich/1.2.7p1/bin/mpicc
MPI_INC_DIR = /nfs/apps/mpich/1.2.7p1/bin/../include
MPI_LIB_DIR = /nfs/apps/mpich/1.2.7p1/bin/../lib
MPI_LIBS    = 
MPI_FLAGS   = 
MPIF77_LNKR = $(MPIF77)

OBJ_EXT = o

COMPILER_PREFIX = 
LINKER_PREFIX   = 

SUNDIALS_LIB_DIR = $(libdir)
SUNDIALS_LIBS    = -lsundials_fcvode -lsundials_cvode -lsundials_fnvecparallel -lsundials_nvecparallel

EXEC_FILES = fcvnonx_p fcvkryx_p fcvkryx_bbd_p

all: $(EXEC_FILES)

fortran_update.sh:
	@${INSTALL} ${top_builddir}/config/fortran_update.sh ${builddir}

fcvnonx_p: $(srcdir)/fcvnonx_p.f fortran_update.sh
	@${SHELL} ${builddir}/fortran_update.sh ${srcdir} fcvnonx_p.f
	$(COMPILER_PREFIX) $(MPIF77) $(MPI_FLAGS) -I$(MPI_INC_DIR) $(FFLAGS) -c $(builddir)/fcvnonx_p-updated.f
	@if test "X${MPIF77_LNKR}" = "X${MPIF77}"; then                                                                                                                                                                                      \
	   echo "${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvnonx_p ${builddir}/fcvnonx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvnonx_p ${builddir}/fcvnonx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS} ;        \
	 else                                                                                                                                                                                                                                \
	   echo "${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvnonx_p ${builddir}/fcvnonx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvnonx_p ${builddir}/fcvnonx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS} ;        \
	 fi

fcvkryx_p: $(srcdir)/fcvkryx_p.f fortran_update.sh
	@${SHELL} ${builddir}/fortran_update.sh ${srcdir} fcvkryx_p.f
	$(COMPILER_PREFIX) $(MPIF77) $(MPI_FLAGS) -I$(MPI_INC_DIR) $(FFLAGS) -c $(builddir)/fcvkryx_p-updated.f
	@if test "X${MPIF77_LNKR}" = "X${MPIF77}"; then                                                                                                                                                                                      \
	   echo "${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_p ${builddir}/fcvkryx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_p ${builddir}/fcvkryx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS} ;        \
	 else                                                                                                                                                                                                                                \
	   echo "${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_p ${builddir}/fcvkryx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_p ${builddir}/fcvkryx_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS} ;        \
	 fi

fcvkryx_bbd_p: $(srcdir)/fcvkryx_bbd_p.f fortran_update.sh
	@${SHELL} ${builddir}/fortran_update.sh ${srcdir} fcvkryx_bbd_p.f
	$(COMPILER_PREFIX) $(MPIF77) $(MPI_FLAGS) -I$(MPI_INC_DIR) $(FFLAGS) -c $(builddir)/fcvkryx_bbd_p-updated.f
	@if test "X${MPIF77_LNKR}" = "X${MPIF77}"; then                                                                                                                                                                                              \
	   echo "${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_bbd_p ${builddir}/fcvkryx_bbd_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_bbd_p ${builddir}/fcvkryx_bbd_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS} ;        \
	 else                                                                                                                                                                                                                                        \
	   echo "${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_bbd_p ${builddir}/fcvkryx_bbd_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${MPIF77_LNKR} -o ${builddir}/fcvkryx_bbd_p ${builddir}/fcvkryx_bbd_p-updated.${OBJ_EXT} ${MPI_FLAGS} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} ${FLIBS} ;        \
	 fi

clean:
	rm -rf .libs
	rm -f *.lo
	rm -f *.o
	rm -f *-updated.f
	rm -f fortran_update.sh

uninstall: clean
	rm -f $(EXEC_FILES)
