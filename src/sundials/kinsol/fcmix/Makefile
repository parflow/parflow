# -----------------------------------------------------------------
# $Revision: 1.15 $
# $Date: 2006/03/15 20:04:23 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/kinsol/LICENSE.
# -----------------------------------------------------------------
# Makefile for FKINSOL module
#
# kinsol/fcmix/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh

srcdir       = .
builddir     = .
abs_builddir = /home/smithsg/parflow/pfclm_ms/src/sundials/kinsol/fcmix
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

CONFIG_INC_DIR  = $(top_builddir)/config
SHARED_INC_DIR  = $(srcdir)/../../shared/include
KINSOL_INC_DIR  = $(srcdir)/../include
KINSOL_SRC_DIR  = $(srcdir)/../source
FKINSOL_INC_DIR = $(srcdir)

INCLUDES = -I$(CONFIG_INC_DIR) -I$(SHARED_INC_DIR) -I$(KINSOL_INC_DIR) -I$(KINSOL_SRC_DIR) -I$(FKINSOL_INC_DIR)

FKINSOL_LIB       = libsundials_fkinsol.la
FKINSOL_LIB_FILES = fkinsol.lo fkinpreco.lo fkinjtimes.lo fkinbbd.lo fkindense.lo
FKINSOL_OBJ_FILES = fkinsol.o fkinpreco.o fkinjtimes.o fkinbbd.o fkindense.o

all: $(FKINSOL_LIB_FILES)

install: $(FKINSOL_LIB_FILES)
	$(LIBTOOL) --mode=link $(CC) $(CFLAGS) -o $(FKINSOL_LIB) $(FKINSOL_LIB_FILES) -rpath $(libdir) $(LDFLAGS) $(LIBS) -static -version-info $(LIB_REVISION)
	$(LIBTOOL) --mode=install $(INSTALL_LIB) $(FKINSOL_LIB) $(libdir)

clean:
	$(LIBTOOL) --mode=clean rm -f $(FKINSOL_LIB)
	rm -f $(FKINSOL_LIB_FILES)
	rm -f $(FKINSOL_OBJ_FILES)

uninstall: clean
	$(LIBTOOL) --mode=uninstall rm -f $(libdir)/$(FKINSOL_LIB)

fkinsol.lo: $(srcdir)/fkinsol.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fkinsol.c
fkinpreco.lo: $(srcdir)/fkinpreco.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fkinpreco.c
fkinjtimes.lo: $(srcdir)/fkinjtimes.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fkinjtimes.c
fkinbbd.lo: $(srcdir)/fkinbbd.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fkinbbd.c
fkindense.lo: $(srcdir)/fkindense.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fkindense.c

libtool: $(top_builddir)/$(LIBTOOL_DEPS)
	@cd ${top_builddir} ;                 \
	 ${SHELL} ./config.status --recheck ; \
	 cd ${abs_builddir}
