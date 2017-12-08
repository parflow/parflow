#!/bin/bash

rm _$PROBLEMNAME*.out.*
export P=2
export Q=1
export R=1
export PROBLEMNAME=hillslope_sens
export DEBUGFLAGS="-v -v -v"
export DEBUGFLAGS=""
../scripts/doMPI.sh
../scripts/undist.tcl veg_map.pfb
