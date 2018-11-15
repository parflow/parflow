#!/bin/bash -x

#P=1
P=2
Q=1
R=1
PROBLEMNAME=default_richards
export DEBUGFLAGS=""

# Python as reference:
export ANALYZER=../Python-analyzer-template.py
$PARFLOW_DIR/bin/parflowvr/doMPI.sh $PROBLEMNAME $P $Q $R

rm -rf correct_output
mkdir correct_output
mv *.pfb correct_output

# Compare with C run!
cd ..
gcc -L$PARFLOW_DIR/lib -I../pfanalyzer -I/usr/local/include/fca C-analyzer-template.c -lfca -lpfanalyzer
cd test
export ANALYZER=../a.out
$PARFLOW_DIR/bin/parflowvr/doMPI.sh $PROBLEMNAME $P $Q $R

tclsh test.tcl
