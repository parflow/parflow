#!/bin/bash

rm $PROBLEMNAME*.out.*
P=1
Q=2
R=1
PROBLEMNAME=hillslope_sens
$PARFLOW_DIR/bin/parflowvr/doMPI.sh $PROBLEMNAME $P $Q $R
$PARFLOW_DIR/bin/parflowvr/undist.tcl veg_map.pfb

echo .
echo .
echo ================== TESTS: ===================
$PARFLOW_DIR/bin/parflowvr/compare_nc.py hillslope_sens.out.00010.nc good/hillslope_sens.out.00010.nc &&
$PARFLOW_DIR/bin/parflowvr/compare_nc.py hillslope_sens.out.00020.nc good/hillslope_sens.out.00020.nc &&
echo passed! || echo XXXXXXXXXXXXXX: not passed!
