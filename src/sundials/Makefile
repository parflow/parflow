# -----------------------------------------------------------------
# $Revision: 1.10 $
# $Date: 2005/06/20 17:24:15 $
# -----------------------------------------------------------------
# Programmer(s): Radu Serban and Aaron Collier @ LLNL
# -----------------------------------------------------------------
# Copyright (c) 2002, The Regents of the University of California.
# Produced at the Lawrence Livermore National Laboratory.
# All rights reserved.
# For details, see sundials/shared/LICENSE.
# -----------------------------------------------------------------
# Main Makefile for SUNDIALS suite
#
# Makefile.  Generated from Makefile.in by configure.
# -----------------------------------------------------------------

SHELL = /bin/sh



srcdir       = .
abs_builddir = /home/smithsg/parflow/pfclm_ms/src/sundials
prefix       = /home/smithsg/parflow/pfclm_ms/src/sundials
exec_prefix  = ${prefix}
includedir   = ${prefix}/include
libdir       = ${exec_prefix}/lib

MODULES         = shared/source cvode/source cvode/fcmix cvodes/source ida/source ida/fcmix kinsol/source kinsol/fcmix
NVEC_MODULES    = nvec_ser nvec_par
EXAMPLE_MODULES =  cvode/examples_ser cvode/examples_par cvode/fcmix/examples_ser cvode/fcmix/examples_par cvodes/examples_ser cvodes/examples_par ida/examples_ser ida/examples_par ida/fcmix/examples_ser ida/fcmix/examples_par kinsol/examples_ser kinsol/examples_par kinsol/fcmix/examples_ser kinsol/fcmix/examples_par

COMPONENTS = $(NVEC_MODULES) $(MODULES)
EXAMPLES   = $(EXAMPLE_MODULES)

FGREP   = grep -F
INSTALL = /usr/bin/install -c
CP_FILE = ${INSTALL} -m 644

all:
	@for i in ${COMPONENTS} ; do  \
	   if test -d $${i} ; then    \
	     echo "Making $${i}..." ; \
	     cd $${i} ;               \
	     ${MAKE} ;                \
	     cd ${abs_builddir} ;     \
	     echo "" ;                \
	   fi ;                       \
	 done

install:
	@for i in ${COMPONENTS} ; do      \
	   if test -d $${i} ; then        \
	     echo "Installing $${i}..." ; \
	     cd $${i} ;                   \
	     ${MAKE} install ;            \
	     cd ${abs_builddir} ;         \
	     echo "" ;                    \
	   fi ;                           \
	 done

clean:
	@for i in ${COMPONENTS} ; do \
	   if test -d $${i} ; then   \
	     cd $${i} ;              \
	     ${MAKE} clean ;         \
	     cd ${abs_builddir} ;    \
	   fi ;                      \
	 done

examples:
	@if test "X${EXAMPLES}" = "Xno" ; then             \
	   echo "Examples were NOT configured" ;           \
	   echo "Rerun configure with --enable-examples" ; \
	 else                                              \
	   for i in ${EXAMPLES} ; do                       \
	     if test -d $${i} ; then                       \
	       echo "Making examples in $${i}..." ;        \
	       cd $${i} ;                                  \
	       ${MAKE} ;                                   \
	       cd ${abs_builddir} ;                        \
	       echo "" ;                                   \
	     fi ;                                          \
	   done ;                                          \
	 fi

examples_clean:
	@if test "X${EXAMPLES}" = "Xno" ; then \
	   : ;                                 \
	 else                                  \
	   for i in ${EXAMPLES} ; do           \
	     if test -d $${i} ; then           \
	       cd $${i} ;                      \
	       ${MAKE} clean ;                 \
	       cd ${abs_builddir} ;            \
	     fi ;                              \
	   done ;                              \
	 fi

examples_uninstall:
	@if test "X${EXAMPLES}" = "Xno" ; then \
	   : ;                                 \
	 else                                  \
	   for i in ${EXAMPLES} ; do           \
	     if test -d $${i} ; then           \
	       cd $${i} ;                      \
	       ${MAKE} uninstall ;             \
	       cd ${abs_builddir} ;            \
	     fi ;                              \
	   done ;                              \
	 fi

distclean: clean examples_clean
	@for i in ${COMPONENTS} ; do           \
	   if test -d $${i} ; then             \
	     rm -f $${i}/Makefile ;            \
	   fi ;                                \
	 done
	@if test "X${EXAMPLES}" = "Xno" ; then \
	   : ;                                 \
	 else                                  \
	   for i in ${EXAMPLES} ; do           \
	     if test -d $${i} ; then           \
	       rm -f $${i}/Makefile ;          \
	     fi ;                              \
	   done ;                              \
	 fi
	@rm -f config.log ;                    \
	 rm -f config.status ;                 \
	 rm -f config.cache ;                  \
	 rm -f config.h                        \
	 rm -f config/sundials_config.h ;      \
	 rm -f config/fortran_update.sh ;      \
	 rm -f Makefile

uninstall: examples_uninstall
	@for i in ${COMPONENTS} ; do \
	   if test -d $${i} ; then   \
	     cd $${i} ;              \
	     ${MAKE} uninstall ;     \
	     cd ${abs_builddir} ;    \
	   fi ;                      \
	 done
