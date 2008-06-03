# -----------------------------------------------------------------
# $Revision: 1.6 $
# $Date: 2006/03/15 20:04:02 $
# -----------------------------------------------------------------
# Programmer(s): Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2005, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/ida/LICENSE.
# -----------------------------------------------------------------
# Makefile for FIDA module
#
# ida/fcmix/Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh

srcdir       = .
builddir     = .
abs_builddir = /home/smithsg/parflow/pfclm_ms/src/sundials/ida/fcmix
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
IDA_INC_DIR    = $(srcdir)/../include
IDA_SRC_DIR    = $(srcdir)/../source
FIDA_INC_DIR   = $(srcdir)

INCLUDES = -I$(CONFIG_INC_DIR) -I$(SHARED_INC_DIR) -I$(IDA_INC_DIR) -I$(IDA_SRC_DIR) -I$(FIDA_INC_DIR)

FIDA_LIB       = libsundials_fida.la
FIDA_LIB_FILES = fida.lo fidaband.lo fidabbd.lo fidadense.lo fidajtimes.lo fidapreco.lo fidaewt.lo fidaroot.lo
FIDA_OBJ_FILES = fida.o fidaband.o fidabbd.o fidadense.o fidajtimes.o fidapreco.o fidaewt.o fidaroot.o

all: $(FIDA_LIB_FILES)

install: $(FIDA_LIB_FILES)
	$(LIBTOOL) --mode=link $(CC) $(CFLAGS) -o $(FIDA_LIB) $(FIDA_LIB_FILES) -rpath $(libdir) $(LDFLAGS) $(LIBS) -static -version-info $(LIB_REVISION)
	$(LIBTOOL) --mode=install $(INSTALL_LIB) $(FIDA_LIB) $(libdir)

clean:
	$(LIBTOOL) --mode=clean rm -f $(FIDA_LIB)
	rm -f $(FIDA_LIB_FILES)
	rm -f $(FIDA_OBJ_FILES)

uninstall: clean
	$(LIBTOOL) --mode=uninstall rm -f $(libdir)/$(FIDA_LIB)

fida.lo: $(srcdir)/fida.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fida.c
fidaewt.lo: $(srcdir)/fidaewt.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fidaewt.c
fidaband.lo: $(srcdir)/fidaband.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fidaband.c
fidadense.lo: $(srcdir)/fidadense.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fidadense.c
fidajtimes.lo: $(srcdir)/fidajtimes.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fidajtimes.c
fidapreco.lo: $(srcdir)/fidapreco.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fidapreco.c
fidabbd.lo: $(srcdir)/fidabbd.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fidabbd.c
fidaroot.lo: $(srcdir)/fidaroot.c
	$(LIBTOOL) --mode=compile $(CC) $(CPPFLAGS) $(INCLUDES) $(CFLAGS) -c $(srcdir)/fidaroot.c

libtool: $(top_builddir)/$(LIBTOOL_DEPS)
	@cd ${top_builddir} ;                 \
	 ${SHELL} ./config.status --recheck ; \
	 cd ${abs_builddir}
