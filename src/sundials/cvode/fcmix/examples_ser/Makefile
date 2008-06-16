# -----------------------------------------------------------------
# $Revision: 1.17 $
# $Date: 2006/03/15 20:03:42 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/cvode/LICENSE.
# -----------------------------------------------------------------
# Makefile for FCVODE serial examples
#
# cvode/fcmix/examples_ser/Makefile.  Generated from Makefile.in by configure.
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



F77      = /usr/bin/f77
FFLAGS   = -g -O2 -ffloat-store
FLIBS    =  -L/usr/lib/gcc/i386-redhat-linux/3.4.6 -L/usr/lib/gcc/i386-redhat-linux/3.4.6/../../.. -lfrtbegin -lg2c -lm -lgcc_s
CC       = /usr/bin/cc
LDFLAGS  = 
LIBS     = -lm 
F77_LNKR = $(F77)

OBJ_EXT = o

COMPILER_PREFIX = 
LINKER_PREFIX   = 

SUNDIALS_LIB_DIR = $(libdir)
SUNDIALS_LIBS    = -lsundials_fcvode -lsundials_cvode -lsundials_fnvecserial -lsundials_nvecserial

EXEC_FILES = fcvdenx fcvbanx fcvkryx fcvkryx_bp

all: $(EXEC_FILES)

fortran_update.sh:
	@${INSTALL} ${top_builddir}/config/fortran_update.sh ${builddir}

fcvdenx: $(srcdir)/fcvdenx.f fortran_update.sh
	@${SHELL} ${builddir}/fortran_update.sh ${srcdir} fcvdenx.f
	$(COMPILER_PREFIX) $(F77) $(FFLAGS) -c $(builddir)/fcvdenx-updated.f
	@if test "X${F77_LNKR}" = "X${F77}"; then                                                                                                                                           \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvdenx ${builddir}/fcvdenx-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvdenx ${builddir}/fcvdenx-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 else                                                                                                                                                                               \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvdenx ${builddir}/fcvdenx-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvdenx ${builddir}/fcvdenx-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 fi

fcvbanx: $(srcdir)/fcvbanx.f fortran_update.sh
	@${SHELL} ${builddir}/fortran_update.sh ${srcdir} fcvbanx.f
	$(COMPILER_PREFIX) $(F77) $(FFLAGS) -c $(builddir)/fcvbanx-updated.f
	@if test "X${F77_LNKR}" = "X${F77}"; then                                                                                                                                           \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvbanx ${builddir}/fcvbanx-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvbanx ${builddir}/fcvbanx-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 else                                                                                                                                                                               \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvbanx ${builddir}/fcvbanx-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvbanx ${builddir}/fcvbanx-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 fi

fcvkryx: $(srcdir)/fcvkryx.f fortran_update.sh
	@${SHELL} ${builddir}/fortran_update.sh ${srcdir} fcvkryx.f
	$(COMPILER_PREFIX) $(F77) $(FFLAGS) -c $(builddir)/fcvkryx-updated.f
	@if test "X${F77_LNKR}" = "X${F77}"; then                                                                                                                                           \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx ${builddir}/fcvkryx-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx ${builddir}/fcvkryx-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 else                                                                                                                                                                               \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx ${builddir}/fcvkryx-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx ${builddir}/fcvkryx-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 fi

fcvkryx_bp: $(srcdir)/fcvkryx_bp.f fortran_update.sh
	@${SHELL} ${builddir}/fortran_update.sh ${srcdir} fcvkryx_bp.f
	$(COMPILER_PREFIX) $(F77) $(FFLAGS) -c $(builddir)/fcvkryx_bp-updated.f
	@if test "X${F77_LNKR}" = "X${F77}"; then                                                                                                                                                 \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx_bp ${builddir}/fcvkryx_bp-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx_bp ${builddir}/fcvkryx_bp-updated.${OBJ_EXT} ${FFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 else                                                                                                                                                                                     \
	   echo "${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx_bp ${builddir}/fcvkryx_bp-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS}" ; \
	   ${LINKER_PREFIX} ${F77_LNKR} -o ${builddir}/fcvkryx_bp ${builddir}/fcvkryx_bp-updated.${OBJ_EXT} ${CFLAGS} ${LDFLAGS} -L${SUNDIALS_LIB_DIR} ${SUNDIALS_LIBS} ${LIBS} ${FLIBS} ;        \
	 fi

clean:
	rm -rf .libs
	rm -f *.lo
	rm -f *.o
	rm -f *-updated.f
	rm -f fortran_update.sh

uninstall: clean
	rm -f $(EXEC_FILES)
