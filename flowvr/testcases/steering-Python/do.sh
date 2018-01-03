#!/bin/bash

rm $PROBLEMNAME*.out.*
export P=2
export Q=1
export R=1
export PROBLEMNAME=hillslope_sens
export DEBUGFLAGS="-v -v -v"
export DEBUGFLAGS=""
$PARFLOW_DIR/bin/parflowvr/doMPI.sh
$PARFLOW_DIR/bin/parflowvr/undist.tcl veg_map.pfb

echo .
echo .
echo ================== TESTS: ===================
$PARFLOW_DIR/bin/parflowvr/compare_nc.py hillslope_sens.out.00010.nc good/hillslope_sens.out.00010.nc &&
$PARFLOW_DIR/bin/parflowvr/compare_nc.py hillslope_sens.out.00020.nc good/hillslope_sens.out.00020.nc &&
echo passed! || echo XXXXXXXXXXXXXX: not passed!
