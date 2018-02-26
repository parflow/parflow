#!/bin/bash
R=1
P=2
Q=2
PROBLEMNAME=default_richards
export DAEMONFLAGS="-s 3G"
export DEBUGFLAGS=""

rm -f *.nc

# run it
$PARFLOW_DIR/bin/parflowvr/doMPI.sh $PROBLEMNAME $P $Q $R


#some tests:
errors=0
$PARFLOW_DIR/bin/parflowvr/compare_nc.py merged.default_richards.out.00000.nc normal.default_richards.out.00000.nc
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py merged.default_richards.out.00001.nc normal.default_richards.out.00001.nc
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py merged.default_richards.out.00002.nc normal.default_richards.out.00002.nc
errors=$errors+$?
$PARFLOW_DIR/bin/parflowvr/compare_nc.py merged.default_richards.out.00003.nc normal.default_richards.out.00003.nc
errors=$errors+$?
# we do not compare the last as we do not know for sure which writer finishes quicker.
#$PARFLOW_DIR/bin/parflowvr/compare_nc.py mergeddefault_richards.out.00004.nc normaldefault_richards.out.00004.nc
errors=$errors+$?

errorcount=`echo $errors | bc`
echo -------- $errorcount errors! -------
echo ------------ END! --------------

exit $errorcount
