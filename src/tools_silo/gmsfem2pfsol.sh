#! /bin/sh
#BHEADER***********************************************************************
# (c) 1995   The Regents of the University of California
#
# See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
# notice, contact person, and disclaimer.
#
# $Revision: 1.1.1.1 $
#EHEADER***********************************************************************


while [ "$*" != "" ]
do
case $1 in
  -h|-help) 
        echo "$0 gms_femwater_file "
        echo 
        echo "Where:"
        echo "gms_femwater_file is the GMS Femwater Super filename "
	echo "  without the .sup extension"
       exit 0;;
    *)
       file=$1
       shift;;
esac
done

NumElements=`grep GE6 $file.3dm | wc -l` 
NumNodes=`grep GN $file.3dm | wc -l` 
NumBC=`grep CB1 $file.3bc | wc -l` 

bgmsfem2pfsol $file.3dm $file.3bc $NumNodes $NumElements $NumBC $file.pfsol $file.mmap

