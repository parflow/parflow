#!/bin/bash -x

#export P=1
export P=2
export Q=1
export R=1
export PROBLEMNAME=default_richards
export DEBUGFLAGS=""

# Python as reference:
export ANALYZER=../Python-analyzer-template.py
$PARFLOW_DIR/bin/parflowvr/doMPI.sh

rm -rf correct_output
mkdir correct_output
mv *.pfb correct_output

# Compare with C run!
cd ..
gcc -L$PARFLOW_DIR/lib -I../pfanalyzer -I/usr/local/include/fca C-analyzer-template.c -lfca -lpfanalyzer
cd test
export ANALYZER=../a.out
$PARFLOW_DIR/bin/parflowvr/doMPI.sh

tclsh test.tcl
