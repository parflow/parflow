#!/bin/bash -x
# Preparations:
RESULTFILE="warmup_test"
NZ=8


algos[0]="tclsh ./normal.tcl 4 2 1"
algos[1]="./do.sh >log 2>&1"
prepaths[0]="results_normal/normal.out"
prepaths[1]="mpi.out"

rm $RESULTFILE.*.csv
for algo in 0 1; do
  # write header:
  rf=$RESULTFILE.$algo.csv
  echo `git rev-parse HEAD` >> $rf
  echo "flowvr: parflow grid(3,2,1) + netcdf + simplestarter" >> $rf
  echo "normal: parflow grid(4,2,1)" >> $rf
  echo "i=0: warmup run" >> $rf
  echo i,NX,_1000,_500 >> $rf
done

#for N in 10 20 30 40 50 60 70 80; do
#for N in 90 100 110 120 130 140 150; do
for N in 10 20 30 40 50; do

  echo now calculating: $N
  rm Ns.tcl
  echo pfset ComputationalGrid.NX                      $N >> Ns.tcl
  echo pfset ComputationalGrid.NY                      $N >> Ns.tcl
  echo pfset ComputationalGrid.NZ                       8 >> Ns.tcl
  for algo in 0 1; do
    echo ------------
    echo algo: ${algos[algo]}
    echo ------------

    for i in 0 1 2 3; do
      eval ${algos[algo]}
      echo measuring...
      p=${prepaths[algo]}
      dt1000=`./measure.py $p.00999.nc $p.00001.nc`
      dt500=`./measure.py $p.00500.nc $p.00001.nc`
      echo $i,$N,$dt1000,$dt500 >> $RESULTFILE.$algo.csv

    done
  done
done


