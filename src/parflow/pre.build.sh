#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************
cp solver.dcl.c solver.dcl.temp
sort -u solver.dcl.temp > solver.dcl.c
rm solver.dcl.temp
cp solver.free.code.c solver.free.code.temp
sort -u solver.free.code.temp > solver.free.code.c
rm solver.free.code.temp

