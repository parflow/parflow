#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.42 $
#EHEADER***********************************************************************

#
#  Makefile for parflow tools.
# 
#  This file is machine independent.  Machine dependent information should
#  be in the build script that invokes this makfile
#

#
# You should not modify anything past here
#

#
# Clear default rules
#
.SUFFIXES:
.SUFFIXES: .c .o

all: pfwell_cat pfwell_data gms_tools

.c.o : 
	@$(PFTOOLS_CC) $(PFTOOLS_INCLUDE) -I. $(CFLAGS) -c $<

#
# For building pfwell_cat
#
PFWELL_CAT_HDRS = pfwell_cat.h well.h
PFWELL_CAT_OBJS = pfwell_cat.o well.o
pfwell_cat: Makefile $(PFWELL_CAT_HDRS) $(PFWELL_CAT_OBJS)
	@$(PFTOOLS_CC) $(CFLAGS) -o pfwell_cat $(PFWELL_CAT_OBJS) \
		 $(PFTOOLS_LIBS)

#
# For building pfwell_data
#
PFWELL_DATA_HDRS = pfwell_data.h well.h
PFWELL_DATA_OBJS = pfwell_data.o well.o
pfwell_data: Makefile $(PFWELL_DATA_HDRS) $(PFWELL_DATA_OBJS)
	@$(PFTOOLS_CC) $(CFLAGS) -o pfwell_data $(PFWELL_DATA_OBJS) \
		 $(PFTOOLS_LIBS)


#
# GMS programs
#
GMSSOL2PFSOL_HDRS = general.h geometry.h gms.h
GMSSOL2PFSOL_OBJS = gmsSOL2pfsol.o gmsreadSOL.o geom_sort_vertices.o
gmssol2pfsol: Makefile $(GMSSOL2PFSOL_HDRS) $(GMSSOL2PFSOL_OBJS)
	@$(PFTOOLS_CC) $(CFLAGS) -o gmssol2pfsol $(GMSSOL2PFSOL_OBJS)

GMSTINVERTICES_HDRS = general.h geometry.h gms.h
GMSTINVERTICES_OBJS = gmsTINvertices.o \
	gmsreadTIN.o gmswriteTIN.o geom_sort_xy_vertices.o
gmstinvertices: Makefile $(GMSTINVERTICES_HDRS) $(GMSTINVERTICES_OBJS)
	@$(PFTOOLS_CC) $(CFLAGS) -o gmstinvertices $(GMSTINVERTICES_OBJS)

GMSTRIANGULATE_HDRS = general.h geometry.h gms.h
GMSTRIANGULATE_OBJS = gmstriangulate.o \
	gmsreadTIN.o gmswriteTIN.o geom_sort_xy_vertices.o
gmstriangulate: Makefile $(GMSTRIANGULATE_HDRS) $(GMSTRIANGULATE_OBJS)
	@$(PFTOOLS_CC) $(CFLAGS) -o gmstriangulate $(GMSTRIANGULATE_OBJS)

projecttin: gms.h general.h projecttin.o gmsreadTIN.o
	@$(PFTOOLS_CC) $(CFLAGS) -o projecttin projecttin.o gmsreadTIN.o -lm

bgmsfem2pfsol: Makefile bgmsfem2pfsol.c
	@$(PFTOOLS_CC) -I$(PARFLOW_SRC)/config $(CFLAGS) -o bgmsfem2pfsol bgmsfem2pfsol.c

gms_tools: gmssol2pfsol gmstinvertices gmstriangulate bgmsfem2pfsol projecttin

#
# SLIM programs
#
# Note: this currently only works on the SGI 
slim_tools: Makefile slimtopfsb.c slimtopfsb_io.f
	@$(PFTOOLS_CC) -c slimtopfsb.c
	@$(PFTOOLS_F77) -c slimtopfsb_io.f
	@$(PFTOOLS_CC) -o slimtopfsb slimtopfsb.o slimtopfsb_io.o \
	-lI77 -lF77 -lisam

#
# For cleaning
#
clean:
	@rm -f *.o

veryclean: clean
	@rm -f pfwell_cat pfwell_data gmstin2pfsrf gmssol2pfsol
	@rm -f gmstinvertices projecttin gmstriangulate bgmsfem2pfsol

