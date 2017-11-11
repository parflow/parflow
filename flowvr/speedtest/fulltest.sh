#!/bin/bash -x
# Preparations:
RESULTFILE="fulltest.results.csv"
NZ=8

rm $RESULTFILE
# write header:
echo "flowvr: parflow grid(3,2,1) + netcdf + simplestarter" >> $RESULTFILE
echo "normal: parflow grid(4,2,1)" >> $RESULTFILE
echo NX,NY,NZ,flowvr_1000,normal_1000,flowvr_500,normal_500,blocks_per_nc,size_per_nc >> $RESULTFILE

#for N in 10 20 30 40 50 60 70 80; do
for N in 90 100 110 120 130 140 150; do
  echo now calculating: $N
  rm Ns.tcl
  echo pfset ComputationalGrid.NX                      $N >> Ns.tcl
  echo pfset ComputationalGrid.NY                      $N >> Ns.tcl
  echo pfset ComputationalGrid.NZ                       8 >> Ns.tcl
  echo normal...
  tclsh ./normal.tcl 4 2 1
  echo flowvr...
  ./do.sh >log 2>&1
  echo measuring...
  ./measure.js $N $N $NZ >> $RESULTFILE
done


