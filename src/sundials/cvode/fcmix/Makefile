# -----------------------------------------------------------------
# $Revision: 1.18 $
# $Date: 2006/03/15 20:03:37 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/cvode/LICENSE.
# -----------------------------------------------------------------
# Makefile for FCVODE module
#
# cvode/fcmix/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh

srcdir       = .
builddir     = .
abs_builddir = /home/smithsg/parflow/pfclm_ms/src/sundials/cvode/fcmix
top_builddir = ../../
prefix       = /home/smithsg/parflow/pfclm_ms/src/sundials
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib

INSTALL        = /usr/bin/install -c
INSTALL_LIB    = ${INSTALL}
INSTALL_HEADER = ${INSTALL} -m 644

LIBTOOL      = $(SHELL) $(top_builddir)/libtool
LIBTOOL_DEPS = config/ltmain.sh

CPP      = /usr/bin/cc -E
CPPFLAGS = 
CC       = /usr/bin/cc
CFLAGS   = -g -O2 -ffloat-store
LDFLAGS  = 
LIBS     = -lm 

LIB_REVISION = 0:0:0

CONFIG_INC_DIR = $(top_builddir)/config
SHARED_INC_DIR = $(srcdir)/../../shared/include
CVODE_INC_DIR  = $(srcdir)/../include
CVODE_SRC_DIR  = $(srcdir)/../source
FCVODE_INC_DIR = $(srcdir)

INCLUDES = -I$(CONFIG_INC_DIR) -I$(SHARED_INC_DIR) -I$(CVODE_INC_DIR) -I$(CVODE_SRC_DIR) -I$(FCVODE_INC_DIR)

FCVODE_LIB       = libsundials_fcvode.la
FCVODE_LIB_FILES = fcvode.lo fcvband.lo fcvdense.lo fcvjtimes.lo fcvpreco.lo fcvbbd.lo fcvbp.lo fcvroot.lo fcvewt.lo
FCVODE_OBJ_FILES = fcvode.o fcvband.o fcvdense.o fcvjtimes.o fcvpreco.o fcvbbd.o fcvbp.o fcvroot.o fcvewt.o

all: $(FCVODE_LIB_FILES)

install: $(FCVODE_LIB_FILES)
	$(LIBTOOL) --mode=link $(CC) $(CFLAGS) -o $(FCVODE_LIB) $(FCVODE_LIB_FILES) -rpath $(libdir) $(LDFLAGS) $(LIBS) -static -version-info $(LIB_REVISION)
	$(LIBTOOL) --mode=install $(INSTALL_LIB) $(FCVODE_LIB) $(libdir)

clean:
	$(LIBTOOL) --mode=clean rm -f $(FCVODE_LIB)
	rm -f $(FCVODE_LIB_FILES)
	rm -f $(FCVODE_OBJ_FILES)

uninstall: clean
	$(LIBTOOL) --mode=uninstall rm -f $(libdir)/$(FCVODE_LIB)

fcvode.lo: $(srcdir)/fcvode.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvode.c
fcvewt.lo: $(srcdir)/fcvewt.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvewt.c
fcvband.lo: $(srcdir)/fcvband.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvband.c
fcvdense.lo: $(srcdir)/fcvdense.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvdense.c
fcvjtimes.lo: $(srcdir)/fcvjtimes.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvjtimes.c
fcvpreco.lo: $(srcdir)/fcvpreco.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvpreco.c
fcvbbd.lo: $(srcdir)/fcvbbd.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvbbd.c
fcvbp.lo: $(srcdir)/fcvbp.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvbp.c
fcvroot.lo: $(srcdir)/fcvroot.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fcvroot.c

libtool: $(top_builddir)/$(LIBTOOL_DEPS)
	@cd ${top_builddir} ;                 \
	 ${SHELL} ./config.status --recheck ; \
	 cd ${abs_builddir}
