# -----------------------------------------------------------------
# $Revision: 1.20 $
# $Date: 2006/03/15 20:04:33 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/shared/LICENSE.
# -----------------------------------------------------------------
# Makefile for parallel NVECTOR module
#
# nvec_par/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh



srcdir       = .
builddir     = .
abs_builddir = /home/smithsg/parflow/pfclm_ms/src/sundials/nvec_par
top_builddir = ../
prefix       = /home/smithsg/parflow/pfclm_ms/src/sundials
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib

INSTALL        = /usr/bin/install -c
INSTALL_LIB    = ${INSTALL}
INSTALL_HEADER = ${INSTALL} -m 644

LIBTOOL      = $(SHELL) $(top_builddir)/libtool
LIBTOOL_DEPS = config/ltmain.sh

MPICC       = /nfs/apps/mpich/1.2.7p1/bin/mpicc
MPI_INC_DIR = /nfs/apps/mpich/1.2.7p1/bin/../include
MPI_LIB_DIR = /nfs/apps/mpich/1.2.7p1/bin/../lib
MPI_LIBS    = 
MPI_FLAGS   = 
CPPFLAGS    = 
CFLAGS      = -g -O2 -ffloat-store
LDFLAGS     = 
LIBS        = -lm 

LIB_REVISION = 0:0:0

FCMIX_ENABLED = yes

SHARED_BUILD_DIR = $(top_builddir)/shared/source

MPI_INC_DIR     = /nfs/apps/mpich/1.2.7p1/bin/../include
CONFIG_INC_DIR  = $(top_builddir)/config
SHARED_INC_DIR  = $(srcdir)/../shared/include
NVECPAR_INC_DIR = $(srcdir)

NVECPAR_LIB       = libsundials_nvecparallel.la
NVECPAR_LIB_FILES = nvector_parallel.lo

FNVECPAR_LIB       = libsundials_fnvecparallel.la
FNVECPAR_LIB_FILES = fnvector_parallel.lo

SHARED_LIB_FILES = $(SHARED_BUILD_DIR)/sundials_math.lo

all: $(NVECPAR_LIB_FILES) $(FNVECPAR_LIB_FILES)

install: shared $(NVECPAR_LIB_FILES) $(FNVECPAR_LIB_FILES)
	$(LIBTOOL) --mode=link $(MPICC) $(CFLAGS) $(MPI_FLAGS) -o $(NVECPAR_LIB) $(NVECPAR_LIB_FILES) $(SHARED_LIB_FILES) $(LDFLAGS) -L$(MPI_LIB_DIR) $(MPI_LIBS) $(LIBS) -rpath $(libdir) -version-info $(LIB_REVISION)
	$(LIBTOOL) --mode=install $(INSTALL_LIB) $(NVECPAR_LIB) $(libdir)
	$(INSTALL_HEADER) $(NVECPAR_INC_DIR)/nvector_parallel.h $(includedir)
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then                                                                                                                                                                                            \
	   echo "${LIBTOOL} --mode=link ${MPICC} ${CFLAGS} ${MPI_FLAGS} -o ${FNVECPAR_LIB} ${FNVECPAR_LIB_FILES} ${SHARED_LIB_FILES} ${LDFLAGS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} -rpath ${libdir} -static -version-info ${LIB_REVISION}" ; \
	   ${LIBTOOL} --mode=link ${MPICC} ${CFLAGS} ${MPI_FLAGS} -o ${FNVECPAR_LIB} ${FNVECPAR_LIB_FILES} ${SHARED_LIB_FILES} ${LDFLAGS} -L${MPI_LIB_DIR} ${MPI_LIBS} ${LIBS} -rpath ${libdir} -static -version-info ${LIB_REVISION} ;        \
	   echo "${LIBTOOL} --mode=install ${INSTALL_LIB} ${FNVECPAR_LIB} ${libdir}" ;                                                                                                                                                         \
	   ${LIBTOOL} --mode=install ${INSTALL_LIB} ${FNVECPAR_LIB} ${libdir} ;                                                                                                                                                                \
	 fi

shared:
	@cd ${SHARED_BUILD_DIR} ; \
	 ${MAKE} ;                \
	 cd ${abs_builddir}

clean:
	$(LIBTOOL) --mode=clean rm -f $(NVECPAR_LIB)
	rm -f $(NVECPAR_LIB_FILES)
	rm -f nvector_parallel.o
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then               \
	   echo "${LIBTOOL} --mode=clean rm -f ${FNVECPAR_LIB}" ; \
	   ${LIBTOOL} --mode=clean rm -f ${FNVECPAR_LIB} ;        \
	   echo "rm -f ${FNVECPAR_LIB_FILES}" ;                   \
	   rm -f ${FNVECPAR_LIB_FILES} ;                          \
	   echo "rm -f fnvector_parallel.o" ;                     \
	   rm -f fnvector_parallel.o ;                            \
	 fi

uninstall: clean
	$(LIBTOOL) --mode=uninstall rm -f $(libdir)/$(NVECPAR_LIB)
	rm -f $(includedir)/nvector_parallel.h
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then                             \
	   echo "${LIBTOOL} --mode=uninstall rm -f ${libdir}/${FNVECPAR_LIB}" ; \
	   ${LIBTOOL} --mode=uninstall rm -f ${libdir}/${FNVECPAR_LIB} ;        \
	 fi

nvector_parallel.lo: $(srcdir)/nvector_parallel.c
	$(LIBTOOL) --mode=compile $(MPICC) $(CPPFLAGS) $(MPI_FLAGS) -I$(CONFIG_INC_DIR) -I$(SHARED_INC_DIR) -I$(NVECPAR_INC_DIR) -I$(MPI_INC_DIR) $(CFLAGS) -c $(srcdir)/nvector_parallel.c
fnvector_parallel.lo: $(srcdir)/fnvector_parallel.c
	@if test "X${FCMIX_ENABLED}" = "Xyes"; then                                                                                                                                                      \
	   echo "${LIBTOOL} --mode=compile ${MPICC} ${CPPFLAGS} ${MPI_FLAGS} -I${CONFIG_INC_DIR} -I${SHARED_INC_DIR} -I${NVECPAR_INC_DIR} -I${MPI_INC_DIR} ${CFLAGS} -c ${srcdir}/fnvector_parallel.c" ; \
	   ${LIBTOOL} --mode=compile ${MPICC} ${CPPFLAGS} ${MPI_FLAGS} -I${CONFIG_INC_DIR} -I${SHARED_INC_DIR} -I${NVECPAR_INC_DIR} -I${MPI_INC_DIR} ${CFLAGS} -c ${srcdir}/fnvector_parallel.c ;        \
	 fi

libtool: $(top_builddir)/$(LIBTOOL_DEPS)
	@cd ${top_builddir} ;                 \
	 ${SHELL} ./config.status --recheck ; \
	 cd ${abs_builddir}
